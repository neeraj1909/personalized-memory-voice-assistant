import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from personalized_memory.config import Settings
from personalized_memory.fast_chat import FastChatEngine, FastDecision
from personalized_memory.storage import ChatStore, MemoryDraft


class FakeAgent:
    def __init__(self, result):
        self.result = result
        self.inputs = []

    async def run(self, value, **_kwargs):
        self.inputs.append(value)
        return SimpleNamespace(output=self.result)


def test_fast_engine_preserves_account_isolation_and_evidence(tmp_path: Path):
    store = ChatStore(tmp_path / "memory.db")
    store.initialize()
    store.save_memories("alice", "old", [MemoryDraft(kind="fact", key="sign", value="Aries")])
    store.save_memories("bob", "old", [MemoryDraft(kind="fact", key="sign", value="Taurus")])
    agent = FakeAgent(
        FastDecision(
            response="Aries is a fire sign.",
            language_scope="durable",
            reply_language="Hindi",
            language_evidence='"Talk in Hindi"',
            relevant_memory_ids=[1, 2],
            new_memories=[{"kind": "fact", "key": "name", "value": "Maya", "evidence": "Maya"}],
        )
    )
    engine = FastChatEngine(store, agent, Settings(_env_file=None))
    result = asyncio.run(engine.chat("alice", "new", 'What is Aries? "Talk in Hindi"'))
    assert result.response == "Which language would you like me to use?"
    assert "Taurus" not in agent.inputs[0]
    assert result.context_used == ["sign"]
    assert store.current_language("alice") is None
    assert not any(m.key == "name" for m in store.memory_candidates("alice", 20))


def test_fast_engine_persists_only_explicit_unquoted_language_request(tmp_path: Path):
    store = ChatStore(tmp_path / "memory.db")
    store.initialize()
    agent = FakeAgent(
        FastDecision(
            response="ठीक है।",
            language_scope="durable",
            reply_language="Hindi",
            language_evidence="Speak Hindi",
        )
    )
    engine = FastChatEngine(store, agent, Settings(_env_file=None))
    asyncio.run(engine.chat("alice", "s1", "Speak Hindi"))
    assert store.current_language("alice").value == "Hindi"
    assert store.current_language("bob") is None
    assert len(agent.inputs) == 1


def test_quoted_language_evidence_retries_without_changing_saved_preference(tmp_path: Path):
    store = ChatStore(tmp_path / "memory.db")
    store.initialize()
    store.save_memories(
        "alice", "old", [MemoryDraft(kind="preference", key="language_preference", value="English")]
    )
    invalid = FastDecision(
        response="हिंदी में जवाब दूंगा।",
        language_scope="durable",
        reply_language="Hindi",
        language_evidence="अब से हिंदी में जवाब देना",
        new_memories=[
            {
                "kind": "interest",
                "key": "gardening",
                "value": "gardening",
                "evidence": "I like gardening",
            }
        ],
    )
    valid = FastDecision(response="They asked you to respond in Hindi.")

    class SequenceAgent(FakeAgent):
        async def run(self, value, **kwargs):
            self.result = invalid if not self.inputs else valid
            return await super().run(value, **kwargs)

    agent = SequenceAgent(invalid)
    engine = FastChatEngine(store, agent, Settings(_env_file=None))
    result = asyncio.run(
        engine.chat(
            "alice",
            "new",
            "I like gardening. My colleague said ‘अब से हिंदी में जवाब देना’. What did they mean?",
        )
    )
    assert result.response == "They asked you to respond in Hindi."
    assert store.current_language("alice").value == "English"
    assert [m.key for m in store.memory_candidates("alice", 20)] == ["language_preference"]
    assert len(agent.inputs) == 2
    retry_payload = json.loads(agent.inputs[1])
    assert "language_recovery" in retry_payload
    assert "अब से हिंदी में जवाब देना" not in retry_payload["current_message"]
    assert "अब से हिंदी में जवाब देना" in retry_payload["quoted_content"][0]


def test_quote_clarification_retry_is_bounded_and_preserves_language(tmp_path: Path):
    store = ChatStore(tmp_path / "memory.db")
    store.initialize()
    store.save_memories(
        "alice", "old", [MemoryDraft(kind="preference", key="language_preference", value="English")]
    )
    agent = FakeAgent(FastDecision(response="Please clarify.", language_scope="clarify"))
    engine = FastChatEngine(store, agent, Settings(_env_file=None))
    result = asyncio.run(engine.chat("alice", "new", "What does ‘हिंदी में जवाब देना’ mean?"))
    assert result.response == "Which language would you like me to use?"
    assert store.current_language("alice").value == "English"
    assert len(agent.inputs) == 2


def test_quote_retry_honors_separate_unquoted_language_request(tmp_path: Path):
    store = ChatStore(tmp_path / "memory.db")
    store.initialize()
    store.save_memories(
        "alice", "old", [MemoryDraft(kind="preference", key="language_preference", value="English")]
    )
    uncertain = FastDecision(response="Please clarify.", language_scope="clarify")
    switch = FastDecision(
        response="ठीक है।",
        language_scope="durable",
        reply_language="Hindi",
        language_evidence="Please answer in Hindi from now on",
    )

    class SequenceAgent(FakeAgent):
        async def run(self, value, **kwargs):
            self.result = uncertain if not self.inputs else switch
            return await super().run(value, **kwargs)

    agent = SequenceAgent(uncertain)
    engine = FastChatEngine(store, agent, Settings(_env_file=None))
    asyncio.run(
        engine.chat("alice", "new", "What does ‘hello’ mean? Please answer in Hindi from now on.")
    )
    assert store.current_language("alice").value == "Hindi"
    assert len(agent.inputs) == 2


def test_fast_engine_passes_saved_language_as_context_on_career_question(tmp_path: Path):
    store = ChatStore(tmp_path / "memory.db")
    store.initialize()
    store.save_memories(
        "alice", "old", [MemoryDraft(kind="preference", key="language_preference", value="Hindi")]
    )
    agent = FakeAgent(FastDecision(response="आप UX शोध में जाना चाहते हैं।"))
    engine = FastChatEngine(store, agent, Settings(_env_file=None))
    result = asyncio.run(engine.chat("alice", "new", "What career move am I preparing for?"))
    payload = json.loads(agent.inputs[0])
    assert payload["stored_language"] == "Hindi"
    assert result.context_used == ["language_preference"]
    assert store.current_language("alice").value == "Hindi"


def test_saved_english_remains_available_for_hindi_preference_question(tmp_path: Path):
    store = ChatStore(tmp_path / "memory.db")
    store.initialize()
    store.save_memories(
        "alice", "old", [MemoryDraft(kind="preference", key="language_preference", value="English")]
    )
    agent = FakeAgent(FastDecision(response="Your saved reply preference is English."))
    engine = FastChatEngine(store, agent, Settings(_env_file=None))
    result = asyncio.run(engine.chat("alice", "new", "मेरी saved reply preference क्या है?"))
    assert json.loads(agent.inputs[0])["stored_language"] == "English"
    assert result.context_used == ["language_preference"]
    assert store.current_language("alice").value == "English"


def test_hinglish_turn_uses_saved_preference_and_memory_without_persisting_a_switch(tmp_path: Path):
    store = ChatStore(tmp_path / "memory.db")
    store.initialize()
    store.save_memories(
        "alice",
        "old",
        [
            MemoryDraft(kind="preference", key="language_preference", value="Hinglish"),
            MemoryDraft(kind="goal", key="career_goal", value="become a data analyst"),
        ],
    )
    goal = next(m for m in store.memory_candidates("alice", 20) if m.key == "career_goal")
    agent = FakeAgent(
        FastDecision(
            response="Aap data analyst banna chahte hain; SQL se start kijiye.",
            relevant_memory_ids=[goal.id],
        )
    )
    engine = FastChatEngine(store, agent, Settings(_env_file=None))
    result = asyncio.run(engine.chat("alice", "new", "Mere career ke liye next step kya hai?"))
    payload = json.loads(agent.inputs[0])
    assert payload["stored_language"] == "Hinglish"
    assert any(m["value"] == "become a data analyst" for m in payload["relevant_memories"])
    assert result.context_used == ["language_preference", "career_goal"]
    assert store.current_language("alice").value == "Hinglish"


def test_uncertain_language_request_does_not_replace_saved_preference(tmp_path: Path):
    store = ChatStore(tmp_path / "memory.db")
    store.initialize()
    store.save_memories(
        "alice", "old", [MemoryDraft(kind="preference", key="language_preference", value="Hindi")]
    )
    agent = FakeAgent(FastDecision(response="Please clarify.", language_scope="clarify"))
    engine = FastChatEngine(store, agent, Settings(_env_file=None))
    result = asyncio.run(
        engine.chat("alice", "new", "Aage se kabhi kabhi English bhi chalegi, shayad?")
    )
    assert result.response == "Which language would you like me to use?"
    assert store.current_language("alice").value == "Hindi"


def test_fast_engine_recalls_general_memories_in_new_sessions(tmp_path: Path):
    store = ChatStore(tmp_path / "memory.db")
    store.initialize()
    agent = FakeAgent(
        FastDecision(
            response="Thank you for sharing.",
            new_memories=[
                {
                    "kind": "goal",
                    "key": "career_goal",
                    "value": "move into UX research",
                    "evidence": "I want to move into UX research",
                },
                {
                    "kind": "interest",
                    "key": "gardening_interest",
                    "value": "growing tomatoes",
                    "evidence": "I enjoy growing tomatoes",
                },
            ],
        )
    )
    engine = FastChatEngine(store, agent, Settings(_env_file=None))
    asyncio.run(
        engine.chat("alice", "s1", "I want to move into UX research. I enjoy growing tomatoes.")
    )
    memories = {m.key: m for m in store.memory_candidates("alice", 20)}
    agent.result = FastDecision(
        response="You want to move into UX research.",
        relevant_memory_ids=[memories["career_goal"].id],
    )
    result = asyncio.run(engine.chat("alice", "s2", "What career change did I mention?"))
    assert '"value":"move into UX research"' in agent.inputs[-1]
    assert result.context_used == ["career_goal"]

    agent.result = FastDecision(
        response="You enjoy growing tomatoes.",
        relevant_memory_ids=[memories["gardening_interest"].id],
    )
    result = asyncio.run(engine.chat("alice", "s3", "What do I enjoy growing?"))
    assert result.response == "You enjoy growing tomatoes."
    assert result.context_used == ["gardening_interest"]
