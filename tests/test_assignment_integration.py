"""Exercise real local gateway and assignment HTTP services with controlled models."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from app.security import canonical_chat_body


WORKSPACE = Path(__file__).resolve().parents[1]
ASSIGNMENT = Path("/home/neeraj/Code/mynaksh_assignment")
ASSIGNMENT_PYTHON = ASSIGNMENT / ".venv/bin/python"
GATEWAY_TOKEN = "test-gateway-token-" + "x" * 32


ASSIGNMENT_RUNNER = r"""from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from app.config import Settings
from app.main import create_app


def choose_context(messages, info):
    payload = json.loads(messages[-1].parts[-1].content)
    text = payload["current_message"]
    memories = payload["stored_memories"]
    args = {"language_scope": "unchanged"}
    if "Mera goal agle saal शिक्षक बनना है" in text:
        evidence = "Mera goal agle saal शिक्षक बनना है।"
        args["new_memories"] = [{
            "kind": "goal",
            "key": "career_goal",
            "value": "become a teacher next year",
            "evidence": evidence,
        }]
    elif "Aaj se Hindi mein jawab dena." in text:
        goal_ids = [memory["id"] for memory in memories if memory["key"] == "career_goal"]
        args.update({
            "language_scope": "durable",
            "reply_language": "Hindi",
            "language_evidence": text,
            "relevant_memory_ids": goal_ids,
            "new_memories": [{
                "kind": "preference",
                "key": "language_preference",
                "value": "Hindi",
                "evidence": text,
            }],
        })
    elif "goal" in text.casefold():
        by_key = {memory["key"]: memory["id"] for memory in memories}
        args["relevant_memory_ids"] = [
            by_key[key] for key in ("language_preference", "career_goal") if key in by_key
        ]
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])


def create_reply(messages, _info):
    payload = json.loads(messages[-1].parts[-1].content)
    memories = payload["relevant_memories"]
    has_goal = any(memory["key"] == "career_goal" for memory in memories)
    text = "Tumhara goal agle saal teacher banna tha." if has_goal else "Mujhe aapka goal yaad nahi hai."
    return ModelResponse(parts=[TextPart(text)])


settings = Settings(
    _env_file=None,
    database_path=Path(os.environ["E2E_ASSIGNMENT_DB"]),
    llm_base_url=None,
    llm_api_key=None,
    otel_exporter_otlp_endpoint=None,
)
app = create_app(
    settings=settings,
    context_model=FunctionModel(choose_context),
    reply_model=FunctionModel(create_reply),
)
"""


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_http(url: str, process: subprocess.Popen[bytes], log_path: Path) -> None:
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 20
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = log_path.read_text(errors="replace") if log_path.exists() else ""
            raise AssertionError(f"server exited ({process.returncode}): {output[-3000:]}")
        try:
            with opener.open(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (OSError, URLError) as exc:
            last_error = exc
            time.sleep(0.1)
    output = log_path.read_text(errors="replace") if log_path.exists() else ""
    raise AssertionError(f"server readiness timed out ({last_error}): {output[-3000:]}")


def _identity_assertion(payload: dict[str, str], key: Ed25519PrivateKey) -> str:
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
    encoded_claims = (
        base64.urlsafe_b64encode(
            json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        .rstrip(b"=")
        .decode("ascii")
    )
    signature = key.sign(f"astro-identity-v1.{encoded_claims}".encode("ascii"))
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{encoded_claims}.{encoded_signature}"


def test_gateway_to_assignment_chat_keeps_scoped_unicode_memories(tmp_path: Path) -> None:
    """Verify signed identity, real HTTP routing, memory recall and user isolation."""
    if not ASSIGNMENT_PYTHON.is_file():
        raise AssertionError(f"Reference assignment interpreter missing: {ASSIGNMENT_PYTHON}")

    assignment_port, gateway_port = _free_port(), _free_port()
    assignment_url = f"http://127.0.0.1:{assignment_port}"
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    runner = tmp_path / "assignment_e2e_server.py"
    runner.write_text(ASSIGNMENT_RUNNER, encoding="utf-8")
    private_key = Ed25519PrivateKey.generate()
    public_key_path = tmp_path / "identity-public.pem"
    private_key_path = tmp_path / "identity-private.pem"
    private_key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    private_key_path.chmod(0o600)
    assignment_log = tmp_path / "assignment.log"
    gateway_log = tmp_path / "gateway.log"
    assignment_env = os.environ.copy()
    assignment_env.update(
        {
            "E2E_ASSIGNMENT_DB": str(tmp_path / "memory.sqlite3"),
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(ASSIGNMENT))),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LANGUAGE_MODEL"):
        assignment_env.pop(name, None)
    gateway_env = os.environ.copy()
    gateway_env.update(
        {
            "CHAT_API_URL": f"{assignment_url}/chat",
            "PMVA_GATEWAY_TOKEN": GATEWAY_TOKEN,
            "PMVA_IDENTITY_PUBLIC_KEY_FILE": str(public_key_path),
            "PMVA_GATEWAY_REPLAY_DB": str(tmp_path / "replays.sqlite3"),
            "CHAT_API_TIMEOUT_SECONDS": "15",
            "PYTHONPATH": str(WORKSPACE),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    assignment_process: subprocess.Popen[bytes] | None = None
    gateway_process: subprocess.Popen[bytes] | None = None
    assignment_log_handle = assignment_log.open("wb")
    gateway_log_handle = gateway_log.open("wb")
    try:
        assignment_process = subprocess.Popen(
            [
                str(ASSIGNMENT_PYTHON),
                "-m",
                "uvicorn",
                "assignment_e2e_server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(assignment_port),
                "--log-level",
                "warning",
            ],
            cwd=tmp_path,
            env=assignment_env,
            stdout=assignment_log_handle,
            stderr=subprocess.STDOUT,
        )
        _wait_for_http(f"{assignment_url}/openapi.json", assignment_process, assignment_log)
        gateway_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(gateway_port),
                "--log-level",
                "warning",
            ],
            cwd=WORKSPACE,
            env=gateway_env,
            stdout=gateway_log_handle,
            stderr=subprocess.STDOUT,
        )
        _wait_for_http(f"{gateway_url}/openapi.json", gateway_process, gateway_log)

        def send(user_id: str, session_id: str, message: str) -> httpx.Response:
            payload = {"user_id": user_id, "session_id": session_id, "message": message}
            return httpx.post(
                f"{gateway_url}/api/chat",
                json=payload,
                headers={
                    "Authorization": f"Bearer {GATEWAY_TOKEN}",
                    "X-Astro-Identity": _identity_assertion(payload, private_key),
                },
                timeout=20,
                trust_env=False,
            )

        unicode_statement = "Mera goal agle saal शिक्षक बनना है।"
        extracted = send("integration-user", "first-session", unicode_statement)
        assert extracted.status_code == 200, extracted.text
        assert extracted.json()["user_id"] == "integration-user"

        switched = send("integration-user", "second-session", "Aaj se Hindi mein jawab dena.")
        assert switched.status_code == 200, switched.text
        assert switched.json()["context_used"] == ["career_goal"]

        recalled = send("integration-user", "third-session", "Mera goal kya tha?")
        assert recalled.status_code == 200, recalled.text
        assert recalled.json()["context_used"] == ["language_preference", "career_goal"]
        assert "teacher" in recalled.json()["response"]

        isolated = send("different-user", "their-session", "Mera goal kya tha?")
        assert isolated.status_code == 200, isolated.text
        assert isolated.json()["context_used"] == []

        with sqlite3.connect(tmp_path / "memory.sqlite3") as db:
            alice_memory = db.execute(
                "SELECT memory_type,memory_key,value FROM user_memories "
                "WHERE user_id=? ORDER BY id",
                ("integration-user",),
            ).fetchall()
            bob_memory = db.execute(
                "SELECT memory_key FROM user_memories WHERE user_id=?", ("different-user",)
            ).fetchall()
            stored_statement = db.execute(
                "SELECT content FROM conversation_messages WHERE user_id=? "
                "AND session_id=? AND role='user'",
                ("integration-user", "first-session"),
            ).fetchone()
        assert alice_memory == [
            ("goal", "career_goal", "become a teacher next year"),
            ("preference", "language_preference", "Hindi"),
        ]
        assert bob_memory == []
        assert stored_statement == (unicode_statement,)
    finally:
        for process in (gateway_process, assignment_process):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        assignment_log_handle.close()
        gateway_log_handle.close()
        shutil.rmtree(tmp_path, ignore_errors=True)
