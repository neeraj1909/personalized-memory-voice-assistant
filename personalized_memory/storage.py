"""SQLite storage for session messages and user memories."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_user_session_created
    ON conversation_messages (user_id, session_id, created_at, id);
CREATE TABLE IF NOT EXISTS user_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    memory_type TEXT NOT NULL CHECK (memory_type IN ('goal','preference','interest','fact')),
    memory_key TEXT NOT NULL,
    value TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_memories_user ON user_memories (user_id);
CREATE INDEX IF NOT EXISTS idx_user_memories_user_key ON user_memories (user_id, memory_key);
"""


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    role: Literal["user", "assistant"]
    content: str


class Memory(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    kind: Literal["goal", "preference", "interest", "fact"] = Field(alias="memory_type")
    key: str = Field(alias="memory_key")
    value: str


class MemoryDraft(BaseModel):
    kind: Literal["goal", "preference", "interest", "fact"]
    key: str
    value: str


class ChatStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        if str(self.path) == ":memory:":
            raise ValueError("use a file-backed SQLite path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with closing(self._connect()) as db:
            db.executescript(SCHEMA)

    def append_message(
        self, user_id: str, session_id: str, role: Literal["user", "assistant"], content: str
    ) -> int:
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO conversation_messages (user_id,session_id,role,content) "
                "VALUES (?,?,?,?)",
                (user_id, session_id, role, content),
            )
            return int(cursor.lastrowid)

    def current_language(self, user_id: str) -> Memory | None:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT id,memory_type,memory_key,value FROM user_memories "
                "WHERE user_id=? AND memory_key='language_preference' ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        return Memory.model_validate(dict(row)) if row else None

    def memory_candidates(self, user_id: str, limit: int) -> list[Memory]:
        language = self.current_language(user_id)
        memories = [language] if language else []
        seen = {"language_preference"} if language else set()
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT id,memory_type,memory_key,value FROM user_memories "
                "WHERE user_id=? ORDER BY id DESC",
                (user_id,),
            )
            for row in rows:
                key = row["memory_key"]
                if key in seen:
                    continue
                seen.add(key)
                memories.append(Memory.model_validate(dict(row)))
                if len(memories) >= limit:
                    break
        return memories[:limit]

    def save_memories(self, user_id: str, session_id: str, drafts: list[MemoryDraft]) -> None:
        with closing(self._connect()) as db, db:
            for draft in drafts:
                latest = db.execute(
                    "SELECT value FROM user_memories WHERE user_id=? AND memory_key=? "
                    "ORDER BY id DESC LIMIT 1",
                    (user_id, draft.key),
                ).fetchone()
                if latest and latest["value"] == draft.value:
                    continue
                db.execute(
                    "INSERT INTO user_memories "
                    "(user_id,memory_type,memory_key,value,source_session_id) VALUES (?,?,?,?,?)",
                    (user_id, draft.kind, draft.key, draft.value, session_id),
                )

    def insert_memories(
        self, user_id: str, session_id: str, drafts: list[MemoryDraft]
    ) -> list[Memory]:
        """Insert cache-deduplicated drafts atomically and return committed rows."""
        saved = []
        with closing(self._connect()) as db, db:
            for draft in drafts:
                cursor = db.execute(
                    "INSERT INTO user_memories "
                    "(user_id,memory_type,memory_key,value,source_session_id) VALUES (?,?,?,?,?)",
                    (user_id, draft.kind, draft.key, draft.value, session_id),
                )
                saved.append(
                    Memory(
                        id=cursor.lastrowid,
                        memory_type=draft.kind,
                        memory_key=draft.key,
                        value=draft.value,
                    )
                )
        return saved

    def recent_user_messages(self, user_id: str, session_id: str, limit: int) -> list[str]:
        if limit == 0:
            return []
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT content FROM conversation_messages "
                "WHERE user_id=? AND session_id=? AND role='user' "
                "ORDER BY id DESC LIMIT ?",
                (user_id, session_id, limit),
            ).fetchall()
        return [row["content"] for row in reversed(rows)]

    def recent_history(
        self, user_id: str, session_id: str, before_id: int, available_chars: int
    ) -> list[Message]:
        """Choose complete older messages within the operational character budget."""
        chosen: list[Message] = []
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT id,role,content FROM conversation_messages "
                "WHERE user_id=? AND session_id=? AND id<? ORDER BY id DESC",
                (user_id, session_id, before_id),
            )
            for row in rows:
                message = Message.model_validate(dict(row))
                cost = (
                    len(
                        json.dumps(
                            {"role": message.role, "content": message.content}, ensure_ascii=False
                        )
                    )
                    + 1
                )
                if cost <= available_chars:
                    chosen.append(message)
                    available_chars -= cost
                if available_chars < 25:
                    break
        chosen.reverse()
        return chosen
