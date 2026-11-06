import json
import base64
import hashlib
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from app.main import create_app, openai_api_key
from app.security import canonical_chat_body

GATEWAY_TOKEN = "test-gateway-token-" + "x" * 32
IDENTITY_PRIVATE_KEY = Ed25519PrivateKey.generate()
IDENTITY_PUBLIC_KEY = IDENTITY_PRIVATE_KEY.public_key()


def test_project_dotenv_key_takes_precedence_over_inherited_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "old-inherited-key")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=new-project-key\n")
    assert openai_api_key(env_file) == "new-project-key"
    env_file.unlink()
    assert openai_api_key(env_file) == "old-inherited-key"


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def identity_assertion(
    payload: dict,
    *,
    jti: str | None = None,
    private_key: Ed25519PrivateKey = IDENTITY_PRIVATE_KEY,
    claim_overrides: dict | None = None,
) -> str:
    issued_at = int(time.time())
    claims = {
        "iss": "astro-voice-streamlit",
        "aud": "astro-local-gateway",
        "sub": payload["user_id"],
        "sid": payload["session_id"],
        "iat": issued_at,
        "exp": issued_at + 30,
        "jti": jti or secrets.token_urlsafe(24),
        "method": "POST",
        "path": "/api/chat",
        "body_sha256": hashlib.sha256(canonical_chat_body(payload)).hexdigest(),
    }
    if claim_overrides:
        claims.update(claim_overrides)
    encoded = _b64(json.dumps(claims, sort_keys=True, separators=(",", ":")).encode())
    signature = private_key.sign(f"astro-identity-v1.{encoded}".encode())
    return f"{encoded}.{_b64(signature)}"


def gateway_app(tmp_path: Path, *args, **kwargs):
    kwargs.setdefault("identity_public_key", IDENTITY_PUBLIC_KEY)
    kwargs.setdefault("replay_db_path", str(tmp_path / "replays.sqlite3"))
    return create_app(*args, **kwargs)


def post_chat(client, payload, token: str = GATEWAY_TOKEN, assertion: str | None = None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if assertion is None:
        assertion = identity_assertion(payload)
    if assertion:
        headers["X-Astro-Identity"] = assertion
    return client.post("/api/chat", json=payload, headers=headers)


def test_gateway_does_not_serve_the_legacy_static_frontend() -> None:
    with TestClient(create_app("http://chat.test/chat")) as client:
        response = client.get("/")
    assert response.status_code == 404


def test_chat_requires_valid_server_bearer_token_before_forwarding(tmp_path: Path) -> None:
    forwarded = False

    def handler(_):
        nonlocal forwarded
        forwarded = True
        return httpx.Response(500)

    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        missing = post_chat(
            client,
            {"user_id": "u", "session_id": "s", "message": "Hi"},
            token="",
        )
        invalid = post_chat(
            client,
            {"user_id": "u", "session_id": "s", "message": "Hi"},
            token="wrong-gateway-token",
        )
    assert missing.status_code == invalid.status_code == 401
    assert not forwarded
    assert GATEWAY_TOKEN not in missing.text + invalid.text


def test_chat_fails_closed_when_gateway_token_is_unconfigured_or_short(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("PMVA_GATEWAY_TOKEN", raising=False)
    payload = {"user_id": "u", "session_id": "s", "message": "Hi"}
    with TestClient(
        create_app("http://chat.test/chat", replay_db_path=str(tmp_path / "missing.sqlite"))
    ) as client:
        missing = post_chat(client, payload)
    with TestClient(
        create_app(
            "http://chat.test/chat",
            gateway_token="short",
            replay_db_path=str(tmp_path / "short.sqlite"),
        )
    ) as client:
        short = post_chat(client, payload)
    assert missing.status_code == short.status_code == 503


def test_chat_forwards_identity_and_returns_contract(tmp_path: Path) -> None:
    seen: list[dict] = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "message_id": 2,
                "response_id": 3,
                "user_id": "user-a",
                "session_id": "session-b",
                "response": "Hello back",
                "context_used": [],
            },
        )

    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = post_chat(
            client,
            {"user_id": "user-a", "session_id": "session-b", "message": "Hello"},
        )

    assert response.status_code == 200
    assert seen == [{"user_id": "user-a", "session_id": "session-b", "message": "Hello"}]
    assert response.json()["response"] == "Hello back"


def test_shared_gateway_token_cannot_assert_an_account_identity(tmp_path: Path) -> None:
    forwarded = False

    def handler(_):
        nonlocal forwarded
        forwarded = True
        return httpx.Response(200, json={})

    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    payload = {"user_id": "victim-account", "session_id": "attacker-session", "message": "Hi"}
    with TestClient(app) as client:
        missing = post_chat(client, payload, assertion="")
        malformed = post_chat(client, payload, assertion="forged.assertion")
        wrong_signer = post_chat(
            client,
            payload,
            assertion=identity_assertion(payload, private_key=Ed25519PrivateKey.generate()),
        )
    assert missing.status_code == malformed.status_code == wrong_signer.status_code == 401
    assert not forwarded


def test_chat_fails_closed_when_identity_verifier_is_unconfigured(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("PMVA_IDENTITY_PUBLIC_KEY_FILE", raising=False)
    forwarded = False

    def handler(_):
        nonlocal forwarded
        forwarded = True
        return httpx.Response(500)

    app = create_app(
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        replay_db_path=str(tmp_path / "replays.sqlite3"),
        transport=httpx.MockTransport(handler),
    )
    payload = {"user_id": "account-a", "session_id": "session-a", "message": "Hi"}
    with TestClient(app) as client:
        response = post_chat(client, payload)
    assert response.status_code == 503
    assert not forwarded


def test_identity_assertion_binds_body_and_is_single_use(tmp_path: Path) -> None:
    seen: list[dict] = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "message_id": 1,
                "response_id": 2,
                "user_id": payload["user_id"],
                "session_id": payload["session_id"],
                "response": "OK",
                "context_used": [],
            },
        )

    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    payload = {"user_id": "account-a", "session_id": "session-a", "message": "Hello"}
    proof = identity_assertion(payload, jti="one-use-replay-test-identifier")
    with TestClient(app) as client:
        altered = post_chat(
            client,
            {**payload, "message": "use another body"},
            assertion=proof,
        )
        accepted = post_chat(client, payload, assertion=proof)
        replayed = post_chat(client, payload, assertion=proof)
    assert altered.status_code == 401
    assert accepted.status_code == 200
    assert replayed.status_code == 409
    assert seen == [payload]


@pytest.mark.parametrize(
    "claim_overrides",
    [
        {"iat": int(time.time()) - 31, "exp": int(time.time()) - 1},
        {"aud": "another-service"},
        {"method": "GET"},
        {"path": "/api/other"},
    ],
    ids=["expired", "wrong-audience", "wrong-method", "wrong-path"],
)
def test_invalid_identity_claims_are_rejected_before_forwarding(
    tmp_path: Path, claim_overrides: dict
) -> None:
    forwarded = False

    def handler(_):
        nonlocal forwarded
        forwarded = True
        return httpx.Response(500)

    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    payload = {"user_id": "account-a", "session_id": "session-a", "message": "Hi"}
    proof = identity_assertion(payload, claim_overrides=claim_overrides)
    with TestClient(app) as client:
        response = post_chat(client, payload, assertion=proof)
    assert response.status_code == 401
    assert not forwarded


def test_replay_store_failure_fails_closed_before_forwarding(tmp_path: Path, monkeypatch) -> None:
    forwarded = False

    def handler(_):
        nonlocal forwarded
        forwarded = True
        return httpx.Response(500)

    def fail_store(*_args, **_kwargs):
        import sqlite3

        raise sqlite3.OperationalError("private database detail")

    monkeypatch.setattr("app.security._consume_assertion_id", fail_store)
    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    payload = {"user_id": "account-a", "session_id": "session-a", "message": "Hi"}
    with TestClient(app) as client:
        response = post_chat(client, payload)
    assert response.status_code == 503
    assert "private database" not in response.text
    assert not forwarded


def test_distinct_concurrent_assertions_pass_but_concurrent_replay_is_single_use(
    tmp_path: Path,
) -> None:
    seen: list[dict] = []
    seen_lock = Lock()

    def handler(request):
        payload = json.loads(request.content)
        with seen_lock:
            seen.append(payload)
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "message_id": 1,
                "response_id": 2,
                "user_id": payload["user_id"],
                "session_id": payload["session_id"],
                "response": "OK",
                "context_used": [],
            },
        )

    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        assert (
            post_chat(
                client,
                {"user_id": "warmup", "session_id": "warmup", "message": "ready"},
            ).status_code
            == 200
        )
        distinct = [
            {"user_id": f"account-{index}", "session_id": f"session-{index}", "message": "Hi"}
            for index in range(2)
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            independent = list(pool.map(lambda body: post_chat(client, body), distinct))

        replay_body = {"user_id": "account-r", "session_id": "session-r", "message": "Once"}
        proof = identity_assertion(replay_body, jti="concurrent-single-use-assertion-id")
        with ThreadPoolExecutor(max_workers=2) as pool:
            replays = list(
                pool.map(
                    lambda _: post_chat(client, replay_body, assertion=proof),
                    range(2),
                )
            )

    assert [response.status_code for response in independent] == [200, 200]
    assert sorted(response.status_code for response in replays) == [200, 409]
    assert len(seen) == 4


def test_chat_rejects_blank_fields_before_upstream(tmp_path: Path) -> None:
    app = gateway_app(tmp_path, "http://chat.test/chat", gateway_token=GATEWAY_TOKEN)
    with TestClient(app) as client:
        response = post_chat(client, {"user_id": " ", "session_id": "s", "message": "Hi"})
    assert response.status_code == 422


def test_chat_accepts_4000_character_message(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request):
        seen.append(json.loads(request.content)["message"])
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "message_id": 2,
                "response_id": 3,
                "user_id": "u",
                "session_id": "s",
                "response": "OK",
                "context_used": [],
            },
        )

    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = post_chat(client, {"user_id": "u", "session_id": "s", "message": "x" * 4000})
    assert response.status_code == 200
    assert seen == ["x" * 4000]


def test_chat_rejects_4001_character_message_before_forwarding(tmp_path: Path) -> None:
    seen = False

    def handler(_):
        nonlocal seen
        seen = True
        return httpx.Response(200, json={})

    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = post_chat(client, {"user_id": "u", "session_id": "s", "message": "x" * 4001})
    assert response.status_code == 422
    assert not seen


def test_upstream_failure_does_not_leak_response_body(tmp_path: Path) -> None:
    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(500, text="private database credential detail")
        ),
    )
    with TestClient(app) as client:
        response = post_chat(client, {"user_id": "u", "session_id": "s", "message": "Hi"})
    assert response.status_code == 502
    assert "private database" not in response.text


def test_upstream_timeout_returns_gateway_timeout(tmp_path: Path) -> None:
    def handler(_):
        raise httpx.ReadTimeout("private upstream diagnostic")

    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(handler),
    )
    with TestClient(app) as client:
        response = post_chat(client, {"user_id": "u", "session_id": "s", "message": "Hi"})
    assert response.status_code == 504
    assert "private upstream diagnostic" not in response.text


def test_upstream_cannot_return_another_users_session(tmp_path: Path) -> None:
    body = {
        "request_id": "req-1",
        "message_id": 2,
        "response_id": 3,
        "user_id": "somebody-else",
        "session_id": "s",
        "response": "private response",
        "context_used": [],
    }
    app = gateway_app(
        tmp_path,
        "http://chat.test/chat",
        gateway_token=GATEWAY_TOKEN,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
    )
    with TestClient(app) as client:
        response = post_chat(client, {"user_id": "u", "session_id": "s", "message": "Hi"})
    assert response.status_code == 502
    assert "private response" not in response.text
