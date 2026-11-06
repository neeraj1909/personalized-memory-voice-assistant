"""Streamlit chat UI for the local personalized memory gateway.

This local demo signs gateway assertions from account-bound Streamlit state.
The assignment API itself does not authenticate user IDs; direct callers who
can reach it can choose IDs. Keep it on a trusted local machine.
"""

from __future__ import annotations

from personalized_memory.compat import getenv

import hashlib
import base64
import io
import ipaddress
import json
import os
import secrets
import time
import uuid
import wave
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import streamlit as st
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from app.accounts import (
    Account,
    AccountStore,
    InvalidAccountInput,
    InvalidCredentials,
    UsernameTaken,
)
from app.security import canonical_chat_body

MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
MAX_MESSAGE_CHARS = 4000
ACCOUNT_DB = getenv(
    "PMVA_ACCOUNTS_DB",
    str(Path.home() / ".local" / "share" / "conversational-ai-for-astro" / "accounts.sqlite3"),
)


class GatewayError(RuntimeError):
    """A safe, user-displayable gateway failure."""


def account_store() -> AccountStore:
    return AccountStore(ACCOUNT_DB)


def _valid_chat_token(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) >= 32
        and all(ord(char) < 128 and not char.isspace() for char in value)
    )


def _chat_gateway_headers() -> dict[str, str]:
    token = getenv("PMVA_GATEWAY_TOKEN", "")
    if not _valid_chat_token(token):
        raise GatewayError(
            "Chat gateway auth is not configured. Set PMVA_GATEWAY_TOKEN to a server-only token with at least 32 ASCII non-whitespace characters."
        )
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _gateway_base_url() -> str:
    """Validate a server-configured gateway origin before attaching its secret token."""
    value = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000").strip()
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        # Accessing .port also validates malformed and out-of-range port syntax.
        _port = parsed.port
        if (
            parsed.scheme not in {"http", "https"}
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "expected a bare HTTP(S) origin without credentials, path, query, or fragment"
            )
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = hostname.casefold() == "localhost"
        if parsed.scheme == "http" and not loopback:
            raise ValueError("plain HTTP is allowed only for localhost or a loopback IP")
    except ValueError as exc:
        raise GatewayError(
            "GATEWAY_URL is invalid. Use HTTPS for remote gateways, or HTTP only for localhost/loopback; do not include credentials or a path."
        ) from exc
    return value.rstrip("/")


def post_chat(user_id: str, session_id: str, message: str) -> str:
    """Send one text turn through the existing same-host gateway contract."""
    payload = {"user_id": user_id, "session_id": session_id, "message": message}
    headers = _chat_gateway_headers()
    headers["X-Astro-Identity"] = _identity_assertion(payload)
    response = httpx.post(
        f"{_gateway_base_url()}/api/chat",
        json=payload,
        headers=headers,
        timeout=float(os.getenv("CHAT_CLIENT_TIMEOUT_SECONDS", "35")),
        follow_redirects=False,
    )
    _raise_gateway_error(response)
    try:
        result = response.json()
    except ValueError as exc:
        raise GatewayError("The chat gateway returned an unreadable response.") from exc
    text = result.get("response") if isinstance(result, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise GatewayError("The chat gateway returned an empty response.")
    return text.strip()


def _identity_assertion(payload: dict[str, str]) -> str:
    """Sign the authenticated Streamlit account/session for this exact chat request."""
    key_path = getenv("PMVA_IDENTITY_PRIVATE_KEY_FILE", "")
    if not key_path:
        raise GatewayError("Chat identity signing is not configured.")
    private_key_path = Path(key_path).expanduser()
    try:
        if os.name == "posix" and private_key_path.stat().st_mode & 0o077:
            raise GatewayError("Chat identity signing key permissions must be 0600.")
        key = serialization.load_pem_private_key(private_key_path.read_bytes(), password=None)
    except GatewayError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise GatewayError("Chat identity signing is not configured.") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise GatewayError("Chat identity signing is not configured.")

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
    encoded = (
        base64.urlsafe_b64encode(
            json.dumps(claims, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        .rstrip(b"=")
        .decode("ascii")
    )
    signature = key.sign(f"astro-identity-v1.{encoded}".encode("ascii"))
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{encoded}.{encoded_signature}"


def validate_recording(wav_bytes: bytes) -> None:
    """Validate the audio widget output against the existing local ASR contract."""
    if not wav_bytes or len(wav_bytes) > MAX_AUDIO_BYTES:
        raise GatewayError("Recording is empty or larger than the 8 MiB limit.")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as recording:
            if (
                recording.getcomptype() != "NONE"
                or recording.getnchannels() != 1
                or recording.getsampwidth() != 2
                or recording.getframerate() != 16000
                or recording.getnframes() <= 0
            ):
                raise GatewayError("Audio must be mono 16 kHz 16-bit PCM WAV.")
            if recording.getnframes() / recording.getframerate() > MAX_AUDIO_SECONDS:
                raise GatewayError("Recording exceeds the 30-second limit.")
            expected_data_bytes = recording.getnframes() * recording.getsampwidth()
            if len(recording.readframes(recording.getnframes())) != expected_data_bytes:
                raise GatewayError("Recording data is incomplete.")
    except (EOFError, OSError, wave.Error) as exc:
        raise GatewayError("The recording is not a valid WAV file.") from exc


def post_transcription(wav_bytes: bytes) -> str:
    """Send one explicitly submitted WAV clip to local Whisper transcription."""
    validate_recording(wav_bytes)
    response = httpx.post(
        f"{_gateway_base_url()}/api/transcribe",
        content=wav_bytes,
        headers={**_chat_gateway_headers(), "Content-Type": "audio/wav"},
        timeout=65,
        follow_redirects=False,
    )
    _raise_gateway_error(response)
    try:
        result = response.json()
    except ValueError as exc:
        raise GatewayError("The transcription gateway returned an unreadable response.") from exc
    text = result.get("text") if isinstance(result, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise GatewayError("No transcript was returned. You can try again or type your message.")
    return text.strip()


def _raise_gateway_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    detail: Any = None
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("error")
    except ValueError:
        pass
    safe_detail = (
        detail
        if isinstance(detail, str) and detail.strip()
        else f"Gateway request failed ({response.status_code})."
    )
    raise GatewayError(safe_detail)


def _new_session_id() -> str:
    return str(uuid.uuid4())


def _release_live_voice_runtime() -> None:
    try:
        from app.live_voice import release_live_voice
    except ImportError:
        runtime = st.session_state.pop("_live_voice_runtime", None)
        if runtime is not None:
            runtime.release()
        return
    release_live_voice()


def _clear_reply_audio() -> None:
    st.session_state.pop("reply_audio", None)
    st.session_state.pop("reply_audio_identity", None)
    st.session_state.pop("reply_audio_message_count", None)
    for key in list(st.session_state):
        if key.startswith("read_aloud_text_"):
            st.session_state.pop(key, None)


def _render_reply_audio() -> None:
    messages = st.session_state.get("messages", [])
    latest = next((item for item in reversed(messages) if item.get("role") == "assistant"), None)
    if latest is None:
        _clear_reply_audio()
        return

    text = latest.get("content", "")
    identity = hashlib.sha256(
        f"{st.session_state['session_id']}:{len(messages)}:{text}".encode("utf-8")
    ).hexdigest()[:16]
    if st.session_state.get("reply_audio_identity") not in {None, identity}:
        _clear_reply_audio()

    speech_key = f"read_aloud_text_{identity}"
    initial_text = text[:500]
    st.text_area(
        "Text to speak",
        value=initial_text,
        key=speech_key,
        max_chars=500,
        height=90,
    )
    if len(text) > 500:
        st.caption(
            "Local speech supports up to 500 characters. The field starts with the first 500; edit it before generating speech if needed."
        )
    st.caption(
        "Optional local speech synthesis. Choose Read aloud to prepare speech, then press play. No audio is generated before that choice."
    )
    if st.button("Read aloud", key=f"read_aloud_{identity}"):
        try:
            from app.local_tts import synthesize_multilingual_wav_sync

            speech_text = str(st.session_state.get(speech_key, "")).strip()
            if not speech_text:
                st.error("Enter text to speak first.")
                return
            st.session_state["reply_audio"] = synthesize_multilingual_wav_sync(speech_text)
            st.session_state["reply_audio_identity"] = identity
            st.session_state["reply_audio_message_count"] = len(messages)
        except Exception:
            st.error("Could not prepare local speech for this reply. Try shorter text.")
    audio = st.session_state.get("reply_audio")
    if audio and st.session_state.get("reply_audio_identity") == identity:
        st.audio(audio, format="audio/wav")


def _authenticate(account: Account) -> None:
    """Bind chat requests to the account returned by the server-side account store."""
    _release_live_voice_runtime()
    st.session_state["account_id"] = account.id
    st.session_state["username"] = account.username
    st.session_state["session_id"] = _new_session_id()
    st.session_state["messages"] = []
    st.session_state["draft_generation"] = 0
    st.session_state["draft_seed"] = ""
    st.session_state["audio_generation"] = 0
    _clear_reply_audio()
    st.session_state["auth_error"] = ""
    st.session_state["auth_notice"] = ""


def _login_callback() -> None:
    username = st.session_state.get("login_username", "")
    password = st.session_state.get("login_password", "")
    try:
        _authenticate(account_store().login(username, password))
        st.session_state["auth_notice"] = "Signed in to this local demo."
    except (InvalidAccountInput, InvalidCredentials) as exc:
        st.session_state["auth_error"] = str(exc)
    finally:
        # Callbacks run before widgets are recreated, so the password field is cleared safely.
        st.session_state["login_password"] = ""


def _register_callback() -> None:
    username = st.session_state.get("register_username", "")
    password = st.session_state.get("register_password", "")
    try:
        _authenticate(account_store().register(username, password))
        st.session_state["auth_notice"] = "Local demo account created."
    except (InvalidAccountInput, UsernameTaken) as exc:
        st.session_state["auth_error"] = str(exc)
    finally:
        st.session_state["register_password"] = ""


def _clear_session_state() -> None:
    _release_live_voice_runtime()
    for key in (
        "account_id",
        "username",
        "user_id",
        "session_id",
        "messages",
        "draft_generation",
        "draft_seed",
        "audio_generation",
        "audio_consent_generation",
        "auth_error",
        "auth_notice",
        "login_username",
        "login_password",
        "register_username",
        "register_password",
        "reply_audio",
        "reply_audio_identity",
        "reply_audio_message_count",
    ):
        st.session_state.pop(key, None)


def _logout_callback() -> None:
    _clear_session_state()


def _new_conversation_callback() -> None:
    _release_live_voice_runtime()
    st.session_state["session_id"] = _new_session_id()
    st.session_state["messages"] = []
    st.session_state["draft_generation"] = st.session_state.get("draft_generation", 0) + 1
    st.session_state["draft_seed"] = ""
    st.session_state["audio_generation"] = st.session_state.get("audio_generation", 0) + 1
    st.session_state["audio_consent_generation"] = (
        st.session_state.get("audio_consent_generation", 0) + 1
    )
    _clear_reply_audio()


def _consent_changed(consent_key: str) -> None:
    if not st.session_state.get(consent_key, False):
        st.session_state["audio_generation"] = st.session_state.get("audio_generation", 0) + 1


def _show_auth_gate() -> None:
    st.title("Personalized Memory Voice Assistant")
    st.caption(
        "A local demo account for this app installation; it is not an online identity provider."
    )
    st.warning(
        "Local account sign-in lets this Streamlit service assert an account to its gateway. The upstream assignment API itself has no authentication and trusts the supplied user ID; anyone who can reach it directly can choose an ID. Keep both services on a trusted local machine.",
        icon="⚠️",
    )
    sign_in, create_account = st.tabs(["Sign in", "Create local account"])
    with sign_in:
        with st.form("login_form"):
            st.text_input("Username", key="login_username", autocomplete="username")
            st.text_input(
                "Password", type="password", key="login_password", autocomplete="current-password"
            )
            st.form_submit_button(
                "Sign in", key="login_submit", on_click=_login_callback, type="primary"
            )
    with create_account:
        with st.form("register_form"):
            st.text_input("Username", key="register_username", autocomplete="username")
            st.text_input(
                "Password",
                type="password",
                key="register_password",
                autocomplete="new-password",
                help="Use at least 12 characters.",
            )
            st.form_submit_button(
                "Create account", key="register_submit", on_click=_register_callback, type="primary"
            )
    if st.session_state.get("auth_error"):
        st.error(st.session_state["auth_error"])
    if st.session_state.get("auth_notice"):
        st.success(st.session_state["auth_notice"])
    st.stop()


def _submit_chat(draft_key: str) -> None:
    message = str(st.session_state.get(draft_key, "")).strip()
    if not message:
        st.session_state["chat_error"] = "Write a message before sending."
        return
    if len(message) > MAX_MESSAGE_CHARS:
        st.session_state["chat_error"] = f"Messages are limited to {MAX_MESSAGE_CHARS} characters."
        return

    _clear_reply_audio()
    try:
        answer = post_chat(st.session_state["account_id"], st.session_state["session_id"], message)
    except (GatewayError, httpx.HTTPError) as exc:
        st.session_state["chat_error"] = (
            str(exc)
            if isinstance(exc, GatewayError)
            else "Could not reach the chat gateway. Check that it is running and try again."
        )
        return
    st.session_state.setdefault("messages", []).append({"role": "user", "content": message})
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.session_state["draft_generation"] = st.session_state.get("draft_generation", 0) + 1
    st.session_state["draft_seed"] = ""
    st.session_state["chat_error"] = ""


def _render_chat() -> None:
    st.title("Personalized Memory Voice Assistant")
    st.caption("Ask a question, or start a voice conversation.")
    token = getenv("PMVA_GATEWAY_TOKEN", "")
    if not _valid_chat_token(token):
        st.warning(
            "Chat is not configured. Set PMVA_GATEWAY_TOKEN on the Streamlit server to a server-only token with at least 32 ASCII non-whitespace characters."
        )
    st.sidebar.markdown("### Local demo account")
    st.sidebar.write(f"Signed in as **{st.session_state['username']}**")
    st.sidebar.caption(
        "This local Streamlit service signs account-bound requests to the gateway. The upstream assignment API has no authentication; direct callers who can reach it can choose user IDs. Keep both services on a trusted local machine."
    )
    st.sidebar.button("Sign out", key="sign_out", on_click=_logout_callback)
    st.sidebar.button(
        "New conversation", key="new_conversation", on_click=_new_conversation_callback
    )
    st.sidebar.caption(
        "New conversation clears this view and starts a new session; it does not delete history held upstream."
    )

    messages = st.session_state.get("messages", [])
    if st.session_state.get("reply_audio") and st.session_state.get(
        "reply_audio_message_count"
    ) != len(messages):
        _clear_reply_audio()

    transcript_rendered = False
    try:
        _gateway_base_url()
        _chat_gateway_headers()
    except GatewayError as exc:
        st.info(f"Live voice is unavailable until gateway configuration is fixed: {exc}")
    else:
        try:
            from app.live_voice import render_live_voice
        except ImportError:
            st.info("Live voice is not available in this installation.")
        else:
            try:
                render_live_voice(
                    st.session_state["account_id"],
                    st.session_state["session_id"],
                    transcribe=post_transcription,
                    chat=post_chat,
                )
                transcript_rendered = True
            except Exception:
                st.error("Live voice could not start. Check audio permissions and retry.")

    if not transcript_rendered:
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    with st.expander("Text and recorded messages"):
        _render_reply_audio()
        st.markdown("#### Record a voice message")
        st.caption(
            "Voice transcription runs locally. Recognition quality depends on the configured model. The browser uploads a recording to this Streamlit server only after consent. "
            "It is sent to the local gateway for transcription only after you choose Transcribe. The gateway limits clips to 30 seconds. "
            "Mixed-language recognition has not been validated."
        )
        consent_generation = st.session_state.get("audio_consent_generation", 0)
        consent_key = f"audio_consent_{consent_generation}"
        st.checkbox(
            "I consent to upload this recording to the Streamlit server for local transcription.",
            key=consent_key,
            on_change=_consent_changed,
            args=(consent_key,),
        )
        if st.session_state.get(consent_key, False):
            st.info(
                "This voice message path is buffered, not streamed. After recording, choose Transcribe to send the WAV to the local gateway. Review the transcript before sending it to chat."
            )
            audio_generation = st.session_state.get("audio_generation", 0)
            audio_key = f"voice_recording_{audio_generation}"
            with st.form(f"transcribe_form_{audio_generation}"):
                audio_value = st.audio_input(
                    "Record a voice clip",
                    sample_rate=16000,
                    key=audio_key,
                    help="Use Stop, then select Transcribe recording. The 30-second maximum is checked when you submit; longer clips are rejected.",
                )
                transcribe_clicked = st.form_submit_button(
                    "Transcribe recording", key=f"transcribe_{audio_generation}"
                )
            if transcribe_clicked:
                if audio_value is None:
                    st.warning("Record a voice clip before choosing Transcribe.")
                else:
                    try:
                        with st.spinner("Transcribing locally…"):
                            transcript = post_transcription(audio_value.getvalue())
                        was_truncated = len(transcript) > MAX_MESSAGE_CHARS
                        st.session_state["draft_seed"] = transcript[:MAX_MESSAGE_CHARS]
                        st.session_state["draft_generation"] = (
                            st.session_state.get("draft_generation", 0) + 1
                        )
                        st.session_state["audio_generation"] = audio_generation + 1
                        st.session_state["audio_consent_generation"] = consent_generation + 1
                        st.session_state["transcription_notice"] = (
                            "Transcript exceeded the message limit and was shortened. Review or edit it before sending."
                            if was_truncated
                            else "Transcript ready. Review or edit it before sending."
                        )
                        st.rerun()
                    except (GatewayError, httpx.HTTPError) as exc:
                        detail = (
                            str(exc)
                            if isinstance(exc, GatewayError)
                            else "Could not reach the local transcription gateway. You can retry or type instead."
                        )
                        st.error(detail)

        notice = st.session_state.pop("transcription_notice", "")
        if notice:
            st.success(notice)
        st.markdown("#### Your message")
        draft_generation = st.session_state.get("draft_generation", 0)
        draft_key = f"draft_{draft_generation}"
        draft_seed = st.session_state.pop("draft_seed", "")
        st.text_area(
            "Write or review your message",
            value=draft_seed,
            key=draft_key,
            max_chars=MAX_MESSAGE_CHARS,
            height=100,
        )
        send_clicked = st.button("Send", key="send_message", type="primary")
        if send_clicked:
            _submit_chat(draft_key)
            if not st.session_state.get("chat_error"):
                st.rerun()
        if st.session_state.get("chat_error"):
            st.error(st.session_state["chat_error"])
        st.caption("Text is sent to chat only when you choose Send.")


def main() -> None:
    st.set_page_config(
        page_title="Personalized Memory Voice Assistant", page_icon="✳", layout="centered"
    )
    for key, default in (
        ("account_id", ""),
        ("username", ""),
        ("session_id", ""),
        ("messages", []),
        ("draft_generation", 0),
        ("draft_seed", ""),
        ("audio_generation", 0),
        ("audio_consent_generation", 0),
        ("auth_error", ""),
        ("auth_notice", ""),
        ("chat_error", ""),
    ):
        st.session_state.setdefault(key, default)
    if not st.session_state.get("account_id"):
        _show_auth_gate()
    # Account IDs are assigned only from AccountStore results on this server.
    st.session_state["user_id"] = st.session_state["account_id"]
    if not st.session_state.get("session_id"):
        st.session_state["session_id"] = _new_session_id()
    _render_chat()


if __name__ == "__main__":
    main()
