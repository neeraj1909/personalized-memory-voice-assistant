from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.accounts import AccountStore
from app.web_auth import WebAuthService, WebIdentity, create_web_auth_service

LOCAL_ORIGIN = "http://127.0.0.1:8000"
PASSWORD = "local-test-password-123"


def build_app(
    tmp_path: Path,
    *,
    origin: str = LOCAL_ORIGIN,
    ttl: int = 3600,
    warm_memory=None,
) -> tuple[FastAPI, WebAuthService]:
    service = create_web_auth_service(
        accounts=AccountStore(tmp_path / "accounts.sqlite3"),
        session_db_path=tmp_path / "sessions.sqlite3",
        public_origin=origin,
        session_ttl_seconds=ttl,
        warm_memory=warm_memory,
    )
    app = FastAPI()
    app.include_router(service.router)

    @app.get("/protected")
    def protected(identity: WebIdentity = Depends(service.require_identity)):
        return {
            "user_id": identity.user_id,
            "session_id": identity.session_id,
            "username": identity.username,
        }

    @app.post("/protected")
    def protected_mutation(identity: WebIdentity = Depends(service.require_identity)):
        return {"username": identity.username}

    return app, service


def credentials(username: str, password: str = PASSWORD) -> dict[str, str]:
    return {"username": username, "password": password}


def test_memory_warm_uses_authenticated_user_for_login_and_session_restore(tmp_path):
    warmed = []

    async def warm_memory(user_id, session_id):
        warmed.append((user_id, session_id))

    app, _ = build_app(tmp_path, warm_memory=warm_memory)
    with (
        TestClient(app, base_url=LOCAL_ORIGIN) as alice,
        TestClient(app, base_url=LOCAL_ORIGIN) as bob,
    ):
        alice.post(
            "/api/auth/register", json=credentials("alice"), headers={"Origin": LOCAL_ORIGIN}
        )
        alice_identity = alice.get("/protected").json()
        bob.post("/api/auth/register", json=credentials("bob"), headers={"Origin": LOCAL_ORIGIN})
        bob_identity = bob.get("/protected").json()
        alice.get("/api/auth/session")
        alice.post(
            "/api/auth/login",
            json=credentials("alice", "incorrect"),
            headers={"Origin": LOCAL_ORIGIN},
        )
        bob.post("/api/auth/login", json=credentials("bob"), headers={"Origin": LOCAL_ORIGIN})
        bob_new_identity = bob.get("/protected").json()
        alice.post("/api/auth/logout", headers={"Origin": LOCAL_ORIGIN})
        alice.get("/api/auth/session")

    def pair(identity):
        return identity["user_id"], identity["session_id"]

    assert warmed == [
        pair(alice_identity),
        pair(bob_identity),
        pair(alice_identity),
        pair(bob_new_identity),
    ]


def test_memory_warm_failure_does_not_break_authentication(tmp_path):
    async def warm_memory(_user_id, _session_id):
        raise RuntimeError("private backend detail")

    app, _ = build_app(tmp_path, warm_memory=warm_memory)
    with TestClient(app, base_url=LOCAL_ORIGIN) as client:
        registered = client.post(
            "/api/auth/register", json=credentials("alice"), headers={"Origin": LOCAL_ORIGIN}
        )
        restored = client.get("/api/auth/session")
    assert registered.status_code == restored.status_code == 200
    assert restored.json() == {"authenticated": True, "username": "alice"}
    assert "private backend detail" not in registered.text + restored.text


def test_register_sets_opaque_http_only_cookie_and_resolves_server_identity(tmp_path):
    app, _ = build_app(tmp_path)
    with TestClient(app, base_url=LOCAL_ORIGIN) as client:
        response = client.post(
            "/api/auth/register", json=credentials("DemoUser"), headers={"Origin": LOCAL_ORIGIN}
        )
        protected = client.get("/protected")
        session = client.get("/api/auth/session")

    assert response.status_code == 200
    assert response.json() == {"authenticated": True, "username": "demouser"}
    assert "DemoUser" not in response.text
    assert PASSWORD not in response.text
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Path=/" in cookie
    assert "Max-Age=3600" in cookie
    assert "Secure" not in cookie
    assert protected.status_code == 200
    assert protected.json()["username"] == "demouser"
    assert protected.json()["user_id"]
    assert protected.json()["session_id"]
    assert "user_id" not in session.json()
    assert "session_id" not in session.json()
    assert session.json() == {"authenticated": True, "username": "demouser"}


def test_https_deployment_sets_secure_cookie(tmp_path):
    origin = "https://bank.example"
    app, _ = build_app(tmp_path, origin=origin)
    with TestClient(app, base_url=origin) as client:
        response = client.post(
            "/api/auth/register", json=credentials("bank-user"), headers={"Origin": origin}
        )
        protected = client.get("/protected")

    assert response.status_code == protected.status_code == 200
    assert "Secure" in response.headers["set-cookie"]
    assert protected.json()["username"] == "bank-user"


@pytest.mark.parametrize(
    "origin",
    [
        "http://bank.example",
        "http://192.168.1.8:8000",
        "https://user:password@bank.example",
        "https://bank.example/auth",
        "file:///tmp/app",
    ],
)
def test_rejects_unsafe_configured_origin(tmp_path, origin):
    with pytest.raises(ValueError, match="PMVA_WEB_AUTH_ORIGIN"):
        build_app(tmp_path, origin=origin)


def test_auth_factory_requires_explicit_public_origin(tmp_path, monkeypatch):
    monkeypatch.delenv("PMVA_WEB_AUTH_ORIGIN", raising=False)
    with pytest.raises(ValueError, match="PMVA_WEB_AUTH_ORIGIN is required"):
        create_web_auth_service(
            accounts=AccountStore(tmp_path / "accounts.sqlite3"),
            session_db_path=tmp_path / "sessions.sqlite3",
        )


def test_login_uses_generic_error_and_never_echoes_password(tmp_path):
    app, _ = build_app(tmp_path)
    with TestClient(app, base_url=LOCAL_ORIGIN) as client:
        response = client.post(
            "/api/auth/login",
            json=credentials("unknown-user", "wrong-password-secret"),
            headers={"Origin": LOCAL_ORIGIN},
        )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid username or password"}
    assert "unknown-user" not in response.text
    assert "wrong-password-secret" not in response.text
    assert "set-cookie" not in response.headers


def test_invalid_cookie_is_not_authenticated(tmp_path):
    app, _ = build_app(tmp_path)
    with TestClient(
        app, base_url=LOCAL_ORIGIN, cookies={"astro_session": "not-a-valid-token"}
    ) as client:
        protected = client.get("/protected")
        session = client.get("/api/auth/session")
    assert protected.status_code == 401
    assert session.status_code == 200
    assert session.json() == {"authenticated": False}


def test_accounts_are_bound_to_independent_server_sessions_and_logout_revokes(tmp_path):
    app, _ = build_app(tmp_path)
    with (
        TestClient(app, base_url=LOCAL_ORIGIN) as alice,
        TestClient(app, base_url=LOCAL_ORIGIN) as bob,
    ):
        alice_login = alice.post(
            "/api/auth/register", json=credentials("alice"), headers={"Origin": LOCAL_ORIGIN}
        )
        bob_login = bob.post(
            "/api/auth/register", json=credentials("bob"), headers={"Origin": LOCAL_ORIGIN}
        )
        alice_identity = alice.get("/protected").json()
        bob_identity = bob.get("/protected").json()
        logout = alice.post("/api/auth/logout", headers={"Origin": LOCAL_ORIGIN})
        alice_after = alice.get("/protected")
        bob_after = bob.get("/protected")

    assert alice_login.status_code == bob_login.status_code == 200
    assert alice_identity["user_id"] != bob_identity["user_id"]
    assert alice_identity["session_id"] != bob_identity["session_id"]
    assert logout.status_code == 204
    assert "Max-Age=0" in logout.headers["set-cookie"]
    assert alice_after.status_code == 401
    assert bob_after.status_code == 200


@pytest.mark.parametrize("origin", [None, "https://attacker.example"])
def test_csrf_origin_is_required_for_auth_mutations(tmp_path, origin):
    app, _ = build_app(tmp_path)
    with TestClient(app, base_url=LOCAL_ORIGIN) as client:
        headers = {} if origin is None else {"Origin": origin}
        register = client.post("/api/auth/register", json=credentials("csrf-user"), headers=headers)
        login = client.post("/api/auth/login", json=credentials("csrf-user"), headers=headers)
        logout = client.post("/api/auth/logout", headers=headers)

    assert register.status_code == login.status_code == logout.status_code == 403
    assert not register.headers.get("set-cookie")


def test_login_and_logout_require_same_origin_even_with_valid_session(tmp_path):
    app, _ = build_app(tmp_path)
    with TestClient(app, base_url=LOCAL_ORIGIN) as client:
        client.post(
            "/api/auth/register", json=credentials("origin-user"), headers={"Origin": LOCAL_ORIGIN}
        )
        before = client.get("/protected")
        csrf_login = client.post(
            "/api/auth/login",
            json=credentials("origin-user"),
            headers={"Origin": "http://localhost:8000"},
        )
        csrf_logout = client.post("/api/auth/logout", headers={"Origin": "http://localhost:8000"})
        after = client.get("/protected")
    assert before.status_code == after.status_code == 200
    assert csrf_login.status_code == csrf_logout.status_code == 403


def test_protected_mutations_require_same_origin(tmp_path):
    app, _ = build_app(tmp_path)
    with TestClient(app, base_url=LOCAL_ORIGIN) as client:
        client.post(
            "/api/auth/register",
            json=credentials("mutation-user"),
            headers={"Origin": LOCAL_ORIGIN},
        )
        missing_origin = client.post("/protected")
        wrong_origin = client.post("/protected", headers={"Origin": "https://attacker.example"})
        same_origin = client.post("/protected", headers={"Origin": LOCAL_ORIGIN})
    assert missing_origin.status_code == wrong_origin.status_code == 403
    assert same_origin.status_code == 200


def test_bad_registration_body_does_not_echo_password(tmp_path):
    app, _ = build_app(tmp_path)
    with TestClient(app, base_url=LOCAL_ORIGIN) as client:
        response = client.post(
            "/api/auth/register",
            json={"username": "user", "password": "secret-in-response-check", "extra": True},
            headers={"Origin": LOCAL_ORIGIN},
        )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid credentials request"}
    assert "secret-in-response-check" not in response.text


def test_expired_cookie_session_is_rejected(tmp_path, monkeypatch):
    now = 1_700_000_000
    monkeypatch.setattr("app.web_auth.time.time", lambda: now)
    app, _ = build_app(tmp_path, ttl=60)
    with TestClient(app, base_url=LOCAL_ORIGIN) as client:
        response = client.post(
            "/api/auth/register", json=credentials("expiry-user"), headers={"Origin": LOCAL_ORIGIN}
        )
        monkeypatch.setattr("app.web_auth.time.time", lambda: now + 61)
        protected = client.get("/protected")
        session = client.get("/api/auth/session")
    assert response.status_code == 200
    assert protected.status_code == 401
    assert session.json() == {"authenticated": False}


def test_identity_is_active_requires_matching_live_server_pair(tmp_path):
    app, auth = build_app(tmp_path)
    with TestClient(app, base_url=LOCAL_ORIGIN) as client:
        client.post(
            "/api/auth/register", json=credentials("monitor-user"), headers={"Origin": LOCAL_ORIGIN}
        )
        identity = client.get("/protected").json()

    pair = (identity["user_id"], identity["session_id"])
    assert asyncio.run(auth.identity_is_active(pair)) is True
    assert asyncio.run(auth.identity_is_active((pair[0], "another-session"))) is False
    assert asyncio.run(auth.identity_is_active(("another-user", pair[1]))) is False
    assert asyncio.run(auth.identity_is_active((pair[0], ""))) is False


def test_identity_is_active_turns_false_after_logout(tmp_path):
    app, auth = build_app(tmp_path)
    with TestClient(app, base_url=LOCAL_ORIGIN) as client:
        client.post(
            "/api/auth/register",
            json=credentials("logout-monitor"),
            headers={"Origin": LOCAL_ORIGIN},
        )
        identity = client.get("/protected").json()
        pair = (identity["user_id"], identity["session_id"])
        assert asyncio.run(auth.identity_is_active(pair)) is True
        response = client.post("/api/auth/logout", headers={"Origin": LOCAL_ORIGIN})

    assert response.status_code == 204
    assert asyncio.run(auth.identity_is_active(pair)) is False


def test_identity_is_active_turns_false_after_expiry(tmp_path, monkeypatch):
    now = 1_700_000_000
    monkeypatch.setattr("app.web_auth.time.time", lambda: now)
    app, auth = build_app(tmp_path, ttl=60)
    with TestClient(app, base_url=LOCAL_ORIGIN) as client:
        client.post(
            "/api/auth/register",
            json=credentials("expiry-monitor"),
            headers={"Origin": LOCAL_ORIGIN},
        )
        identity = client.get("/protected").json()
        pair = (identity["user_id"], identity["session_id"])
        assert asyncio.run(auth.identity_is_active(pair)) is True
        monkeypatch.setattr("app.web_auth.time.time", lambda: now + 61)
        assert asyncio.run(auth.identity_is_active(pair)) is False
