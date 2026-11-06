"""Bounded, process-local copies of committed per-user memory and session history.

The copied memory service must be the sole writer of its SQLite database.
Evicted entries reload from SQLite; the database remains authoritative.
"""

import asyncio
import json
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from .storage import ChatStore, Memory, MemoryDraft, Message

MAX_USERS = 128
MAX_SESSIONS_PER_USER = 32
MAX_MEMORIES_PER_USER = 10000
MAX_HISTORY_MESSAGES = 10000


class MemoryCacheError(RuntimeError):
    pass


def history_cost(message: Message) -> int:
    return (
        len(json.dumps({"role": message.role, "content": message.content}, ensure_ascii=False)) + 1
    )


@dataclass
class UserState:
    memories: list[Memory] | None = None
    histories: OrderedDict[str, list[Message]] = field(default_factory=OrderedDict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    references: int = 0


class MemoryCache:
    def __init__(self, store: ChatStore, history_budget: int) -> None:
        self.store = store
        self.history_budget = history_budget
        self.users: OrderedDict[str, UserState] = OrderedDict()
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, user_id: str, session_id: str) -> AsyncIterator[UserState]:
        async with self.lock:
            state = self.users.get(user_id)
            if state is None:
                if len(self.users) >= MAX_USERS:
                    victim = next(
                        (key for key, item in self.users.items() if not item.references), None
                    )
                    if victim is None:
                        raise MemoryCacheError("memory cache is busy")
                    del self.users[victim]
                state = UserState()
                self.users[user_id] = state
            self.users.move_to_end(user_id)
            state.references += 1
        try:
            async with state.lock:
                if state.memories is None:
                    memories = await asyncio.to_thread(
                        self.store.memory_candidates, user_id, MAX_MEMORIES_PER_USER + 1
                    )
                    if len(memories) > MAX_MEMORIES_PER_USER:
                        raise MemoryCacheError("memory cache capacity exceeded")
                    state.memories = memories
                if session_id not in state.histories:
                    history = await asyncio.to_thread(
                        self.store.recent_history,
                        user_id,
                        session_id,
                        2**63 - 1,
                        self.history_budget,
                    )
                    if len(history) > MAX_HISTORY_MESSAGES:
                        raise MemoryCacheError("session history cache capacity exceeded")
                    if len(state.histories) >= MAX_SESSIONS_PER_USER:
                        state.histories.popitem(last=False)
                    state.histories[session_id] = history
                state.histories.move_to_end(session_id)
                yield state
        finally:
            async with self.lock:
                state.references -= 1

    def append_history(self, state: UserState, session_id: str, message: Message) -> None:
        history = state.histories[session_id]
        history.append(message)
        total = sum(history_cost(item) for item in history)
        while len(history) > MAX_HISTORY_MESSAGES or (
            total > self.history_budget and len(history) > 1
        ):
            total -= history_cost(history.pop(0))

    def selected_history(self, state: UserState, session_id: str, available: int) -> list[Message]:
        chosen = []
        for message in reversed(state.histories[session_id]):
            cost = history_cost(message)
            if cost <= available:
                chosen.append(message)
                available -= cost
            if available < 25:
                break
        return list(reversed(chosen))

    async def save(
        self, state: UserState, user_id: str, session_id: str, drafts: list[MemoryDraft]
    ) -> None:
        existing = {memory.key: memory for memory in state.memories or []}
        changed = [
            draft
            for draft in drafts
            if draft.key not in existing or existing[draft.key].value != draft.value
        ]
        new_keys = sum(draft.key not in existing for draft in changed)
        if len(existing) + new_keys > MAX_MEMORIES_PER_USER:
            raise MemoryCacheError("memory cache capacity exceeded")
        if not changed:
            return
        # No await between commit and cache publication: cancellation cannot
        # leave an asynchronously committed SQLite write absent from the cache.
        committed = self.store.insert_memories(user_id, session_id, changed)
        for memory in committed:
            existing[memory.key] = memory
        language = existing.pop("language_preference", None)
        state.memories = ([language] if language else []) + sorted(
            existing.values(), key=lambda m: m.id, reverse=True
        )
