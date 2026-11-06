import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from personalized_memory.config import Settings
from personalized_memory.fast_chat import FastChatEngine, FastDecision
from personalized_memory.main import create_app
from personalized_memory.storage import ChatStore


class Agent:
    def __init__(self):
        self.calls = []

    async def run(self, value, **_kwargs):
        self.calls.append(value)
        if len(self.calls) == 1:
            return SimpleNamespace(
                output=FastDecision(
                    response="Saved",
                    new_memories=[
                        {
                            "kind": "fact",
                            "key": "zodiac_sign",
                            "value": "Aries",
                            "evidence": "My sign is Aries",
                        }
                    ],
                )
            )
        return SimpleNamespace(output=FastDecision(response="Aries", relevant_memory_ids=[1]))


def test_warm_and_turns_use_no_memory_or_history_reads(tmp_path: Path, monkeypatch):
    async def run():
        store = ChatStore(tmp_path / "db.sqlite3")
        store.initialize()
        engine = FastChatEngine(store, Agent(), Settings(_env_file=None))
        async with engine.cache.acquire("alice", "session"):
            pass
        statements = []
        original_connect = store._connect

        def traced_connect():
            connection = original_connect()
            connection.set_trace_callback(statements.append)
            return connection

        monkeypatch.setattr(store, "_connect", traced_connect)

        def forbidden(*args, **kwargs):
            raise AssertionError("warmed turn attempted a memory or history read")

        monkeypatch.setattr(store, "current_language", forbidden)
        monkeypatch.setattr(store, "memory_candidates", forbidden)
        monkeypatch.setattr(store, "recent_history", forbidden)
        await engine.chat("alice", "session", "My sign is Aries")
        result = await engine.chat("alice", "session", "What is my sign?")
        assert result.context_used == ["zodiac_sign"]
        assert "My sign is Aries" in engine.agent.calls[-1]
        assert "Aries" in engine.agent.calls[-1]
        assert not any(sql.lstrip().upper().startswith("SELECT") for sql in statements)

    asyncio.run(run())


def test_preload_endpoint_requires_server_secret(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PMVA_MEMORY_FAST", "1")
    monkeypatch.setenv("PMVA_MEMORY_WARM_TOKEN", "a-long-internal-secret")
    test_model = TestModel(custom_output_args={"response": "Hello"})
    app = create_app(
        settings=Settings(_env_file=None, database_path=tmp_path / "db.sqlite3"),
        context_model=test_model,
        reply_model=test_model,
    )
    with TestClient(app) as client:
        payload = {"user_id": "alice", "session_id": "one"}
        assert client.post("/internal/memory/warm", json=payload).status_code == 401
        assert (
            client.post(
                "/internal/memory/warm", json=payload, headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        response = client.post(
            "/internal/memory/warm",
            json=payload,
            headers={"Authorization": "Bearer a-long-internal-secret"},
        )
        assert response.status_code == 200
        assert response.json() == {"warmed": True}
        assert (
            client.post(
                "/internal/memory/warm",
                json={"user_id": "alice"},
                headers={"Authorization": "Bearer a-long-internal-secret"},
            ).status_code
            == 422
        )


def test_concurrent_same_user_turns_serialize_memory_and_session_history(tmp_path: Path):
    class DelayedAgent(Agent):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def run(self, value, **kwargs):
            if not self.calls:
                self.started.set()
                await self.release.wait()
            return await super().run(value, **kwargs)

    async def run():
        store = ChatStore(tmp_path / "db.sqlite3")
        store.initialize()
        agent = DelayedAgent()
        engine = FastChatEngine(store, agent, Settings(_env_file=None))
        first = asyncio.create_task(engine.chat("alice", "same", "My sign is Aries"))
        await agent.started.wait()
        second = asyncio.create_task(engine.chat("alice", "same", "What is my sign?"))
        await asyncio.sleep(0)
        assert not agent.calls
        agent.release.set()
        await asyncio.gather(first, second)
        assert "My sign is Aries" in agent.calls[1]
        assert '"value":"Aries"' in agent.calls[1]

    asyncio.run(run())


def test_failed_memory_commit_does_not_publish_to_cache(tmp_path: Path, monkeypatch):
    async def run():
        store = ChatStore(tmp_path / "db.sqlite3")
        store.initialize()
        engine = FastChatEngine(store, Agent(), Settings(_env_file=None))
        async with engine.cache.acquire("alice", "session"):
            pass

        def fail_write(*args, **kwargs):
            raise OSError("synthetic disk failure")

        monkeypatch.setattr(store, "insert_memories", fail_write)
        try:
            await engine.chat("alice", "session", "My sign is Aries")
        except OSError:
            pass
        else:
            raise AssertionError("expected write failure")
        async with engine.cache.acquire("alice", "session") as state:
            assert not state.memories

    asyncio.run(run())


def test_eviction_reloads_committed_memory_and_history(tmp_path: Path, monkeypatch):
    import personalized_memory.cache as cache_module

    async def run():
        store = ChatStore(tmp_path / "db.sqlite3")
        store.initialize()
        engine = FastChatEngine(store, Agent(), Settings(_env_file=None))
        monkeypatch.setattr(cache_module, "MAX_USERS", 1)
        await engine.chat("alice", "same", "My sign is Aries")
        async with engine.cache.acquire("bob", "other"):
            pass
        assert "alice" not in engine.cache.users
        result = await engine.chat("alice", "same", "What is my sign?")
        assert result.context_used == ["zodiac_sign"]
        assert "My sign is Aries" in engine.agent.calls[-1]

    asyncio.run(run())


def test_cancellation_releases_user_lock_and_keeps_committed_history(tmp_path: Path):
    class CancellableAgent:
        def __init__(self):
            self.calls = []
            self.started = asyncio.Event()

        async def run(self, value, **_kwargs):
            self.calls.append(value)
            if len(self.calls) == 1:
                self.started.set()
                await asyncio.Event().wait()
            return SimpleNamespace(output=FastDecision(response="Ready"))

    async def run():
        store = ChatStore(tmp_path / "db.sqlite3")
        store.initialize()
        agent = CancellableAgent()
        engine = FastChatEngine(store, agent, Settings(_env_file=None))
        first = asyncio.create_task(engine.chat("alice", "same", "first turn"))
        await agent.started.wait()
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("expected cancellation")
        result = await asyncio.wait_for(engine.chat("alice", "same", "new turn"), timeout=1)
        assert result.response == "Ready"
        assert "first turn" in agent.calls[-1]
        async with engine.cache.acquire("alice", "same") as state:
            assert not state.memories

    asyncio.run(run())
