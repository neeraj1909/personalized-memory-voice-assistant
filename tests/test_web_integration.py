from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import threading
import time

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from app.accounts import AccountStore
from app.main import create_app
from app.realtime_gateway import session_config
from app.security import canonical_chat_body


def test_auth_warms_memory_on_loopback_using_server_resolved_identity(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("PMVA_WEB_AUTH_ORIGIN", ORIGIN)
    monkeypatch.setenv("PMVA_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
    monkeypatch.setenv("PMVA_WEB_SESSIONS_DB", str(tmp_path / "sessions.sqlite3"))
    monkeypatch.setenv("PMVA_MEMORY_WARM_TOKEN", "warm-test-secret")
    warm_requests = []

    async def upstream(request):
        warm_requests.append(request)
        return httpx.Response(200, json={"warmed": True})

    app = create_app("http://127.0.0.1:8002/chat", transport=httpx.MockTransport(upstream))
    with TestClient(app, base_url=ORIGIN) as client:
        result = client.post(
            "/api/auth/register",
            json={"username": "warmuser", "password": PASSWORD},
            headers={"Origin": ORIGIN},
        )
        assert result.status_code == 200
        identity = app.state.web_auth_service._resolve_token(client.cookies.get("astro_session"))
        assert identity is not None
        client.get("/api/auth/session")
    assert len(warm_requests) == 2
    for request in warm_requests:
        assert str(request.url) == "http://127.0.0.1:8002/internal/memory/warm"
        assert request.headers["Authorization"] == "Bearer warm-test-secret"
        assert json.loads(request.content) == {
            "user_id": identity.user_id,
            "session_id": identity.session_id,
        }


def test_memory_warm_never_sends_secret_to_remote_chat_origin():
    from app.main import memory_warm_url

    assert memory_warm_url("http://127.0.0.1:8002/chat") == (
        "http://127.0.0.1:8002/internal/memory/warm"
    )
    assert memory_warm_url("https://remote.example/chat") is None
    assert memory_warm_url("http://localhost@remote.example/chat") is None


ORIGIN = "http://localhost"
GATEWAY_TOKEN = "test-gateway-token-" + "x" * 32
PASSWORD = "voice-test-password-123"


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _assertion(payload: dict[str, str], key: Ed25519PrivateKey) -> str:
    issued_at = int(time.time())
    claims = {
        "iss": "astro-voice-streamlit",
        "aud": "astro-local-gateway",
        "sub": payload["user_id"],
        "sid": payload["session_id"],
        "iat": issued_at,
        "exp": issued_at + 30,
        "jti": secrets.token_urlsafe(24),
        "method": "POST",
        "path": "/api/chat",
        "body_sha256": hashlib.sha256(canonical_chat_body(payload)).hexdigest(),
    }
    encoded = _b64(json.dumps(claims, sort_keys=True, separators=(",", ":")).encode())
    signature = key.sign(f"astro-identity-v1.{encoded}".encode())
    return f"{encoded}.{_b64(signature)}"


class FakeSideband:
    def __init__(self):
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[dict] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.response_requested = threading.Event()
        self.closed = threading.Event()

    async def send(self, raw: str) -> None:
        event = json.loads(raw)
        self.sent.append(event)
        if event.get("type") == "response.create":
            self.response_requested.set()
        if event.get("type") == "input_audio_buffer.clear":
            await self.incoming.put(json.dumps({"type": "input_audio_buffer.cleared"}))

    async def recv(self) -> str:
        return await self.incoming.get()

    def feed(self, event: dict) -> None:
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.incoming.put_nowait, json.dumps(event))


class FakeSidebandContext:
    def __init__(self, sideband: FakeSideband):
        self.sideband = sideband

    async def __aenter__(self):
        self.sideband.loop = asyncio.get_running_loop()
        await self.sideband.incoming.put(
            json.dumps({"type": "session.updated", "session": session_config()})
        )
        return self.sideband

    async def __aexit__(self, *_args):
        self.sideband.closed.set()


def _wait(event: threading.Event, timeout: float = 2.0) -> bool:
    return event.wait(timeout)


def test_same_origin_app_auth_realtime_chat_binding_and_signed_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("PMVA_WEB_AUTH_ORIGIN", ORIGIN)
    monkeypatch.setenv("PMVA_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
    monkeypatch.setenv("PMVA_WEB_SESSIONS_DB", str(tmp_path / "sessions.sqlite3"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-provider-key")

    private_key = Ed25519PrivateKey.generate()
    forwarded: list[dict[str, str]] = []
    provider_paths: list[str] = []
    chat_complete = threading.Event()
    sideband = FakeSideband()

    async def chat_upstream(request: httpx.Request) -> httpx.Response:
        payload = json.loads(await request.aread())
        forwarded.append(payload)
        chat_complete.set()
        return httpx.Response(
            200,
            json={
                "request_id": "req-test",
                "message_id": 1,
                "response_id": 2,
                "user_id": payload["user_id"],
                "session_id": payload["session_id"],
                "response": "A personalized answer.",
                "context_used": [],
            },
        )

    async def fake_provider(request: httpx.Request) -> httpx.Response:
        provider_paths.append(request.url.path)
        if request.url.path.endswith("/hangup"):
            return httpx.Response(200)
        return httpx.Response(
            201,
            content=b"v=0\r\nanswer-sdp",
            headers={"Location": "/v1/realtime/calls/integration-call"},
        )

    from app import main as main_module
    from app import realtime_gateway as realtime_module

    monkeypatch.setattr(main_module, "openai_api_key", lambda: "test-only-provider-key")
    original_builder = realtime_module.build_realtime_router

    def builder_with_local_fakes(authenticate, chat, **kwargs):
        return original_builder(
            authenticate,
            chat,
            http_client_factory=lambda: httpx.AsyncClient(
                transport=httpx.MockTransport(fake_provider)
            ),
            ws_connect=lambda *_args, **_kwargs: FakeSidebandContext(sideband),
            identity_check_interval_seconds=0.02,
            drain_seconds=1.0,
            **kwargs,
        )

    monkeypatch.setattr(main_module, "build_realtime_router", builder_with_local_fakes)
    app = create_app(
        "http://mynaksh.test/chat",
        transport=httpx.MockTransport(chat_upstream),
        gateway_token=GATEWAY_TOKEN,
        identity_public_key=private_key.public_key(),
        replay_db_path=str(tmp_path / "replays.sqlite3"),
    )

    with TestClient(app, base_url=ORIGIN) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "Personalized Memory Voice Assistant" in home.text

        registration = client.post(
            "/api/auth/register",
            json={"username": "VoiceUser", "password": PASSWORD},
            headers={"Origin": ORIGIN},
        )
        assert registration.status_code == 200
        assert "HttpOnly" in registration.headers["set-cookie"]

        account = AccountStore(tmp_path / "accounts.sqlite3").login("voiceuser", PASSWORD)
        cookie_token = client.cookies.get("astro_session")
        identity = app.state.web_auth_service._resolve_token(cookie_token)
        assert identity is not None and identity.user_id == account.id

        invalid_origin = client.post(
            "/api/realtime/call",
            content=b"v=0\r\noffer",
            headers={"content-type": "application/sdp", "Origin": "https://attacker.test"},
        )
        assert invalid_origin.status_code == 403

        call = client.post(
            "/api/realtime/call",
            content=b"v=0\r\noffer-sdp",
            headers={"content-type": "application/sdp", "Origin": ORIGIN},
        )
        assert call.status_code == 200
        assert call.headers["x-call-id"] == "integration-call"
        assert call.content == b"v=0\r\nanswer-sdp"
        ready = client.post("/api/realtime/call/integration-call/ready", headers={"Origin": ORIGIN})
        assert ready.status_code == 200
        assert ready.json() == {"ready": True}

        sideband.feed({"type": "response.created", "response": {"id": "turn-1"}})
        sideband.feed(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item-1",
                "transcript": "What should I reflect on?",
            }
        )
        sideband.feed({"type": "input_audio_buffer.committed", "item_id": "item-1"})
        sideband.feed(
            {
                "type": "response.done",
                "response": {
                    "id": "turn-1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "name": "ask_memory",
                            "call_id": "tool-1",
                            "arguments": json.dumps({"message": "What should I reflect on?"}),
                        }
                    ],
                },
            }
        )

        assert _wait(chat_complete)
        assert forwarded[0] == {
            "user_id": account.id,
            "session_id": identity.session_id,
            "message": "What should I reflect on?",
        }
        assert _wait(sideband.response_requested)
        sideband.feed({"type": "response.created", "response": {"id": "speech-1"}})
        sideband.feed(
            {
                "type": "response.done",
                "response": {"id": "speech-1", "status": "completed", "output": []},
            }
        )

        unsigned = client.post(
            "/api/chat",
            json=forwarded[0],
            headers={"Authorization": f"Bearer {GATEWAY_TOKEN}"},
        )
        assert unsigned.status_code == 401
        signed = client.post(
            "/api/chat",
            json=forwarded[0],
            headers={
                "Authorization": f"Bearer {GATEWAY_TOKEN}",
                "X-Astro-Identity": _assertion(forwarded[0], private_key),
            },
        )
        assert signed.status_code == 200
        assert len(forwarded) == 2

        logout = client.post("/api/auth/logout", headers={"Origin": ORIGIN})
        assert logout.status_code == 204
        assert client.get("/api/auth/session").json() == {"authenticated": False}
        assert _wait(sideband.closed)
        assert provider_paths == [
            "/v1/realtime/calls",
            "/v1/realtime/calls/integration-call/hangup",
        ]
