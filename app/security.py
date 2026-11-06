"""Server-to-server authentication and signed account assertions."""

from personalized_memory.compat import getenv

import base64
import asyncio
import hashlib
import hmac
import json
import re
import sqlite3
import time
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from fastapi import HTTPException, Request

MIN_GATEWAY_TOKEN_LENGTH = 32


def gateway_token_is_configured(token: str | None) -> bool:
    return bool(
        isinstance(token, str)
        and token.isascii()
        and len(token) >= MIN_GATEWAY_TOKEN_LENGTH
        and not any(character.isspace() for character in token)
    )


async def require_gateway_token(request: Request) -> None:
    expected = getattr(request.app.state, "gateway_token", None)
    if not gateway_token_is_configured(expected):
        raise HTTPException(status_code=503, detail="Chat gateway is not configured")

    authorization = request.headers.get("authorization", "")
    scheme, separator, supplied = authorization.partition(" ")
    if (
        not separator
        or scheme.casefold() != "bearer"
        or not supplied
        or any(character.isspace() for character in supplied)
        or not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("ascii"))
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")


def canonical_chat_body(payload: dict) -> bytes:
    """Serialize the fixed gateway chat body identically in signer and verifier."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def load_identity_public_key(path: str | None = None) -> Ed25519PublicKey | None:
    """Load the gateway's public verification key; the private key stays in Streamlit."""
    key_path = path if path is not None else getenv("PMVA_IDENTITY_PUBLIC_KEY_FILE", "")
    if not key_path:
        return None
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_public_key(Path(key_path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("PMVA_IDENTITY_PUBLIC_KEY_FILE must contain an Ed25519 public key")
    return key


def _decode_segment(value: str) -> bytes:
    if not value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("invalid assertion encoding")
    return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)


def _consume_assertion_id(path: str, assertion_id: str, expires_at: int, now: int) -> bool:
    db_path = Path(path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with sqlite3.connect(db_path, timeout=5) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS consumed_identity_assertions ("
            "jti TEXT PRIMARY KEY, expires_at INTEGER NOT NULL)"
        )
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM consumed_identity_assertions WHERE expires_at < ?", (now,))
        try:
            db.execute(
                "INSERT INTO consumed_identity_assertions (jti, expires_at) VALUES (?, ?)",
                (assertion_id, expires_at),
            )
        except sqlite3.IntegrityError:
            return False
    db_path.chmod(0o600)
    return True


async def require_chat_identity(request: Request, payload) -> None:
    """Verify a short-lived, one-use Streamlit assertion bound to the exact chat body."""
    public_key = getattr(request.app.state, "identity_public_key", None)
    if public_key is None:
        raise HTTPException(status_code=503, detail="Chat identity verification is not configured")

    assertion = request.headers.get("x-astro-identity", "")
    if not assertion or len(assertion) > 4096:
        raise HTTPException(status_code=401, detail="Invalid chat identity assertion")
    try:
        encoded_claims, encoded_signature = assertion.split(".", 1)
        signed = f"astro-identity-v1.{encoded_claims}".encode("ascii")
        public_key.verify(_decode_segment(encoded_signature), signed)
        claims = json.loads(_decode_segment(encoded_claims))
    except (ValueError, UnicodeError, json.JSONDecodeError, InvalidSignature):
        raise HTTPException(status_code=401, detail="Invalid chat identity assertion") from None

    now = int(time.time())
    if (
        not isinstance(claims, dict)
        or set(claims)
        != {"iss", "aud", "sub", "sid", "iat", "exp", "jti", "method", "path", "body_sha256"}
        or claims.get("iss") != "astro-voice-streamlit"
        or claims.get("aud") != "astro-local-gateway"
        or claims.get("method") != request.method
        or claims.get("path") != request.url.path
        or claims.get("sub") != payload.user_id
        or claims.get("sid") != payload.session_id
        or type(claims.get("iat")) is not int
        or type(claims.get("exp")) is not int
        or claims["iat"] > now + 5
        or claims["exp"] < now
        or not 0 < claims["exp"] - claims["iat"] <= 60
        or not isinstance(claims.get("jti"), str)
        or not 20 <= len(claims["jti"]) <= 128
        or claims.get("body_sha256")
        != hashlib.sha256(canonical_chat_body(payload.model_dump())).hexdigest()
    ):
        raise HTTPException(status_code=401, detail="Invalid chat identity assertion")

    replay_db = getattr(request.app.state, "replay_db_path", "")
    try:
        unused = await asyncio.to_thread(
            _consume_assertion_id, replay_db, claims["jti"], claims["exp"], now
        )
    except (OSError, sqlite3.Error):
        raise HTTPException(
            status_code=503, detail="Chat identity verification is unavailable"
        ) from None
    if not unused:
        raise HTTPException(status_code=409, detail="Chat identity assertion already used")
