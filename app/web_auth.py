"""Same-origin browser authentication backed by the local account store."""

from __future__ import annotations

from personalized_memory.compat import getenv

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from typing import Awaitable, Callable

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response

from .accounts import (
    Account,
    AccountStore,
    InvalidAccountInput,
    InvalidCredentials,
    UsernameTaken,
)

SESSION_TTL_SECONDS = 60 * 60
MAX_CREDENTIAL_BODY_BYTES = 4096
COOKIE_NAME = "astro_session"
_COOKIE_TOKEN = re.compile(r"[A-Za-z0-9_-]{40,64}\Z")
log = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class WebIdentity:
    """Server-resolved account and conversation identity for a protected route."""

    user_id: str
    session_id: str
    username: str


def _canonical_origin(value: str, *, configured: bool = False) -> str | None:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("expected a bare HTTP(S) origin")
        scheme = parsed.scheme.casefold()
        hostname = parsed.hostname.casefold()
        port = parsed.port
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = hostname == "localhost"
        if scheme == "http" and not loopback:
            raise ValueError("plain HTTP is allowed only for localhost or loopback")
        default_port = 80 if scheme == "http" else 443
        host = f"[{hostname}]" if ":" in hostname else hostname
        authority = host if port in {None, default_port} else f"{host}:{port}"
        return f"{scheme}://{authority}"
    except ValueError as exc:
        if configured:
            raise ValueError(
                "PMVA_WEB_AUTH_ORIGIN must be HTTPS or a loopback HTTP origin"
            ) from exc
        return None


class WebAuthService:
    """Cookie auth router plus a FastAPI dependency for server-owned identities."""

    def __init__(
        self,
        *,
        accounts: AccountStore,
        session_db_path: str | Path,
        public_origin: str,
        session_ttl_seconds: int = SESSION_TTL_SECONDS,
        cookie_name: str = COOKIE_NAME,
        warm_memory: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        origin = _canonical_origin(public_origin, configured=True)
        if origin is None:
            raise ValueError("A valid public origin is required")
        if not 60 <= session_ttl_seconds <= 24 * 60 * 60:
            raise ValueError("session_ttl_seconds must be between one minute and one day")
        if not cookie_name or any(char in cookie_name for char in "=; \t\r\n"):
            raise ValueError("cookie_name is invalid")

        self.accounts = accounts
        self.session_db_path = Path(session_db_path).expanduser()
        self.origin = origin
        self.secure_cookie = origin.startswith("https://")
        self.session_ttl_seconds = session_ttl_seconds
        self.cookie_name = cookie_name
        self.warm_memory = warm_memory
        self._initialize_session_store()
        self.router = APIRouter(prefix="/api/auth", tags=["browser-auth"])
        self.router.add_api_route("/register", self.register, methods=["POST"])
        self.router.add_api_route("/login", self.login, methods=["POST"])
        self.router.add_api_route("/logout", self.logout, methods=["POST"], status_code=204)
        self.router.add_api_route("/session", self.session, methods=["GET"])

    def _initialize_session_store(self) -> None:
        self.session_db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            fd = os.open(
                self.session_db_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError:
            self.session_db_path.chmod(0o600)
        else:
            os.close(fd)
        with sqlite3.connect(self.session_db_path, timeout=10) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS web_sessions ("
                "cookie_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL UNIQUE, "
                "username TEXT NOT NULL, expires_at INTEGER NOT NULL)"
            )
        self.session_db_path.chmod(0o600)

    @staticmethod
    def _cookie_hash(token: str) -> str:
        return hashlib.sha256(token.encode("ascii")).hexdigest()

    def _check_origin(self, request: Request) -> None:
        origin = _canonical_origin(request.headers.get("origin", ""))
        if origin != self.origin:
            raise HTTPException(status_code=403, detail="Request origin is not allowed")

    async def _read_credentials(self, request: Request) -> tuple[str, str]:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_CREDENTIAL_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request is too large")
            body.extend(chunk)
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail="Invalid credentials request") from None
        if (
            not isinstance(payload, dict)
            or set(payload) != {"username", "password"}
            or not isinstance(payload.get("username"), str)
            or not isinstance(payload.get("password"), str)
            or not 1 <= len(payload["username"]) <= 64
            or not 1 <= len(payload["password"]) <= 1024
        ):
            raise HTTPException(status_code=400, detail="Invalid credentials request")
        return payload["username"], payload["password"]

    def _create_session(self, account: Account) -> tuple[str, str]:
        token = secrets.token_urlsafe(32)
        session_id = secrets.token_urlsafe(24)
        now = int(time.time())
        with sqlite3.connect(self.session_db_path, timeout=10) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM web_sessions WHERE expires_at <= ?", (now,))
            db.execute(
                "INSERT INTO web_sessions (cookie_hash,user_id,session_id,username,expires_at) "
                "VALUES (?,?,?,?,?)",
                (
                    self._cookie_hash(token),
                    account.id,
                    session_id,
                    account.username,
                    now + self.session_ttl_seconds,
                ),
            )
        self.session_db_path.chmod(0o600)
        return token, session_id

    async def _warm_memory(self, user_id: str, session_id: str) -> None:
        if self.warm_memory is None:
            return
        try:
            await self.warm_memory(user_id, session_id)
        except Exception:
            # Authentication is independent of this optional optimization.
            log.warning("Memory preload failed; it will be loaded on demand")

    def _delete_session(self, token: str | None) -> None:
        if not isinstance(token, str) or _COOKIE_TOKEN.fullmatch(token) is None:
            return
        with sqlite3.connect(self.session_db_path, timeout=10) as db:
            db.execute("DELETE FROM web_sessions WHERE cookie_hash=?", (self._cookie_hash(token),))

    def _resolve_token(self, token: str | None) -> WebIdentity | None:
        if not isinstance(token, str) or _COOKIE_TOKEN.fullmatch(token) is None:
            return None
        now = int(time.time())
        with sqlite3.connect(self.session_db_path, timeout=10) as db:
            db.row_factory = sqlite3.Row
            db.execute("DELETE FROM web_sessions WHERE expires_at <= ?", (now,))
            row = db.execute(
                "SELECT user_id,session_id,username FROM web_sessions "
                "WHERE cookie_hash=? AND expires_at>?",
                (self._cookie_hash(token), now),
            ).fetchone()
        if row is None:
            return None
        return WebIdentity(
            user_id=row["user_id"], session_id=row["session_id"], username=row["username"]
        )

    def _set_cookie(self, response: Response, token: str) -> None:
        response.set_cookie(
            key=self.cookie_name,
            value=token,
            max_age=self.session_ttl_seconds,
            path="/",
            httponly=True,
            secure=self.secure_cookie,
            samesite="lax",
        )
        response.headers["Cache-Control"] = "no-store"

    def _clear_cookie(self, response: Response) -> None:
        response.delete_cookie(
            key=self.cookie_name,
            path="/",
            httponly=True,
            secure=self.secure_cookie,
            samesite="lax",
        )
        response.headers["Cache-Control"] = "no-store"

    async def register(
        self, request: Request, response: Response, background_tasks: BackgroundTasks
    ) -> dict[str, object]:
        self._check_origin(request)
        username, password = await self._read_credentials(request)
        try:
            account = await asyncio.to_thread(self.accounts.register, username, password)
            await asyncio.to_thread(self._delete_session, request.cookies.get(self.cookie_name))
            token, session_id = await asyncio.to_thread(self._create_session, account)
        except InvalidAccountInput:
            raise HTTPException(status_code=400, detail="Registration input is invalid") from None
        except UsernameTaken:
            raise HTTPException(status_code=409, detail="Account could not be created") from None
        except (OSError, sqlite3.Error):
            raise HTTPException(status_code=503, detail="Account service is unavailable") from None
        self._set_cookie(response, token)
        if self.warm_memory is not None:
            background_tasks.add_task(self._warm_memory, account.id, session_id)
        return {"authenticated": True, "username": account.username}

    async def login(
        self, request: Request, response: Response, background_tasks: BackgroundTasks
    ) -> dict[str, object]:
        self._check_origin(request)
        username, password = await self._read_credentials(request)
        try:
            account = await asyncio.to_thread(self.accounts.login, username, password)
            old_token = request.cookies.get(self.cookie_name)
            await asyncio.to_thread(self._delete_session, old_token)
            token, session_id = await asyncio.to_thread(self._create_session, account)
        except (InvalidAccountInput, InvalidCredentials):
            raise HTTPException(status_code=401, detail="Invalid username or password") from None
        except (OSError, sqlite3.Error):
            raise HTTPException(status_code=503, detail="Account service is unavailable") from None
        self._set_cookie(response, token)
        if self.warm_memory is not None:
            background_tasks.add_task(self._warm_memory, account.id, session_id)
        return {"authenticated": True, "username": account.username}

    async def logout(self, request: Request, response: Response) -> Response:
        self._check_origin(request)
        try:
            await asyncio.to_thread(self._delete_session, request.cookies.get(self.cookie_name))
        except (OSError, sqlite3.Error):
            raise HTTPException(status_code=503, detail="Account service is unavailable") from None
        self._clear_cookie(response)
        response.status_code = 204
        return response

    async def session(
        self, request: Request, response: Response, background_tasks: BackgroundTasks
    ) -> dict[str, object]:
        response.headers["Cache-Control"] = "no-store"
        try:
            identity = await asyncio.to_thread(
                self._resolve_token, request.cookies.get(self.cookie_name)
            )
        except (OSError, sqlite3.Error):
            raise HTTPException(status_code=503, detail="Account service is unavailable") from None
        if identity is None:
            return {"authenticated": False}
        if self.warm_memory is not None:
            background_tasks.add_task(self._warm_memory, identity.user_id, identity.session_id)
        return {"authenticated": True, "username": identity.username}

    def require_identity(self, request: Request) -> WebIdentity:
        """Resolve the cookie identity; mutating HTTP methods also require same-origin Origin."""
        if request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
            self._check_origin(request)
        try:
            identity = self._resolve_token(request.cookies.get(self.cookie_name))
        except (OSError, sqlite3.Error):
            raise HTTPException(status_code=503, detail="Account service is unavailable") from None
        if identity is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return identity

    async def identity_is_active(self, identity: tuple[str, str]) -> bool:
        """Check whether a server-resolved (user_id, session_id) pair is still active.

        This is intended for periodic realtime-session ownership checks. It does not
        accept or expose the browser cookie or its token and fails closed on DB errors.
        """
        if (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or any(not isinstance(value, str) or not value for value in identity)
        ):
            return False
        try:
            return await asyncio.to_thread(self._identity_is_active, identity)
        except (OSError, sqlite3.Error):
            return False

    def _identity_is_active(self, identity: tuple[str, str]) -> bool:
        now = int(time.time())
        with sqlite3.connect(self.session_db_path, timeout=10) as db:
            row = db.execute(
                "SELECT 1 FROM web_sessions "
                "WHERE user_id=? AND session_id=? AND expires_at>? LIMIT 1",
                (identity[0], identity[1], now),
            ).fetchone()
        return row is not None


def create_web_auth_service(
    *,
    accounts: AccountStore | None = None,
    accounts_db_path: str | Path | None = None,
    session_db_path: str | Path | None = None,
    public_origin: str | None = None,
    session_ttl_seconds: int = SESSION_TTL_SECONDS,
    cookie_name: str = COOKIE_NAME,
    warm_memory: Callable[[str, str], Awaitable[None]] | None = None,
) -> WebAuthService:
    """Create cookie auth for a same-origin browser app; no forwarded headers are trusted."""
    account_path = accounts_db_path or getenv(
        "PMVA_ACCOUNTS_DB",
        str(Path.home() / ".local" / "share" / "conversational-ai-for-astro" / "accounts.sqlite3"),
    )
    session_path = session_db_path or getenv(
        "PMVA_WEB_SESSIONS_DB",
        str(
            Path.home()
            / ".local"
            / "share"
            / "conversational-ai-for-astro"
            / "web-sessions.sqlite3"
        ),
    )
    origin = public_origin or getenv("PMVA_WEB_AUTH_ORIGIN", "")
    if not origin:
        raise ValueError("PMVA_WEB_AUTH_ORIGIN is required")
    return WebAuthService(
        accounts=accounts if accounts is not None else AccountStore(account_path),
        session_db_path=session_path,
        public_origin=origin,
        session_ttl_seconds=session_ttl_seconds,
        cookie_name=cookie_name,
        warm_memory=warm_memory,
    )
