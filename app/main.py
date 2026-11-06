"""Thin same-origin gateway to the existing chat API."""

from personalized_memory.compat import getenv

import os
import ipaddress
import logging
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
from dotenv import dotenv_values
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .local_asr import router as local_asr_router
from .realtime_gateway import build_realtime_router
from .security import load_identity_public_key, require_chat_identity, require_gateway_token
from .web_auth import create_web_auth_service


log = logging.getLogger("uvicorn.error")


def memory_warm_url(chat_url: str) -> str | None:
    """Only send the internal preload secret to a loopback chat service."""
    try:
        parts = urlsplit(chat_url)
        host = parts.hostname or ""
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.casefold() == "localhost"
        if (
            parts.scheme not in {"http", "https"}
            or not loopback
            or parts.username
            or parts.password
        ):
            return None
        return urlunsplit((parts.scheme, parts.netloc, "/internal/memory/warm", "", ""))
    except ValueError:
        return None


def openai_api_key(env_file: Path | None = None) -> str:
    """Prefer the project dotenv credential over an inherited shell value."""
    path = env_file if env_file is not None else Path(__file__).resolve().parent.parent / ".env"
    values = dotenv_values(path, interpolate=False)
    if "OPENAI_API_KEY" in values:
        return values["OPENAI_API_KEY"] or ""
    return os.getenv("OPENAI_API_KEY", "")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("user_id", "session_id", "message")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ChatResponse(BaseModel):
    request_id: str
    message_id: int
    response_id: int
    user_id: str
    session_id: str
    response: str
    context_used: list[str]


def create_app(
    upstream_url: str | None = None,
    *,
    timeout_seconds: float | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    gateway_token: str | None = None,
    identity_public_key=None,
    replay_db_path: str | None = None,
) -> FastAPI:
    """Build the gateway; upstream configuration is kept on the server."""
    url = upstream_url if upstream_url is not None else os.getenv("CHAT_API_URL", "")
    configured_token = (
        gateway_token if gateway_token is not None else getenv("PMVA_GATEWAY_TOKEN", "")
    )
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(os.getenv("CHAT_API_TIMEOUT_SECONDS", "30"))
    )
    if timeout <= 0:
        raise ValueError("CHAT_API_TIMEOUT_SECONDS must be positive")

    app = FastAPI(title="Personalized Memory Chat Gateway")
    app.state.gateway_token = configured_token
    app.state.identity_public_key = (
        identity_public_key if identity_public_key is not None else load_identity_public_key()
    )
    app.state.replay_db_path = replay_db_path or getenv(
        "PMVA_GATEWAY_REPLAY_DB",
        str(
            Path.home()
            / ".local"
            / "share"
            / "conversational-ai-for-astro"
            / "gateway-replays.sqlite3"
        ),
    )
    app.include_router(local_asr_router)

    web_origin = getenv("PMVA_WEB_AUTH_ORIGIN", "").strip()
    if web_origin:
        warm_token = getenv("PMVA_MEMORY_WARM_TOKEN", "")
        warm_url = memory_warm_url(url) if warm_token else None

        async def warm_memory(user_id: str, session_id: str) -> None:
            async with httpx.AsyncClient(timeout=2.0, transport=transport) as client:
                response = await client.post(
                    warm_url,
                    headers={"Authorization": f"Bearer {warm_token}"},
                    json={"user_id": user_id, "session_id": session_id},
                )
                response.raise_for_status()

        if warm_token and not warm_url:
            log.warning("Memory preload disabled: chat service is not on loopback")
        auth = create_web_auth_service(
            public_origin=web_origin, warm_memory=warm_memory if warm_url else None
        )
        app.state.web_auth_service = auth
        app.include_router(auth.router)

        async def authenticate_browser(request: Request) -> tuple[str, str]:
            identity = auth.require_identity(request)
            return identity.user_id, identity.session_id

        async def realtime_chat(user_id: str, session_id: str, message: str) -> str:
            payload = ChatRequest(user_id=user_id, session_id=session_id, message=message)
            response = await forward_chat(payload)
            return response.response

        app.include_router(
            build_realtime_router(
                authenticate_browser,
                realtime_chat,
                api_key=openai_api_key(),
                identity_is_active=auth.identity_is_active,
            )
        )

    async def forward_chat(payload: ChatRequest) -> ChatResponse:
        """Forward one trusted identity-bound request to the memory service."""
        if not url:
            raise HTTPException(status_code=503, detail="Chat service is not configured")

        try:
            async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
                upstream = await client.post(url, json=payload.model_dump())
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="Chat service timed out") from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail="Chat service is unavailable") from exc

        if upstream.status_code == 422:
            raise HTTPException(status_code=422, detail="Chat request was rejected")
        if upstream.status_code == 503:
            raise HTTPException(status_code=503, detail="Chat service is unavailable")
        if not upstream.is_success:
            raise HTTPException(status_code=502, detail="Chat service is unavailable")

        try:
            result = ChatResponse.model_validate(upstream.json())
        except ValueError as exc:
            raise HTTPException(
                status_code=502, detail="Chat service returned an invalid response"
            ) from exc

        # Prevent accidental cross-account/session response mixing. This is not authentication.
        if result.user_id != payload.user_id or result.session_id != payload.session_id:
            raise HTTPException(status_code=502, detail="Chat service returned an invalid response")
        return result

    @app.post(
        "/api/chat",
        response_model=ChatResponse,
        dependencies=[Depends(require_gateway_token)],
    )
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        await require_chat_identity(request, payload)
        return await forward_chat(payload)

    if web_origin:
        web_root = Path(__file__).resolve().parent.parent / "web"
        if (web_root / "index.html").is_file() and (web_root / "app.js").is_file():
            # Mount last so API/auth/realtime routes always take precedence.
            app.mount("/", StaticFiles(directory=web_root, html=True), name="web")

    return app


app = create_app()
