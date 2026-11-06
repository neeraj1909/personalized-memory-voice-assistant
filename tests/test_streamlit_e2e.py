"""Streamlit UI through real local gateway and reference-service HTTP boundaries.

Both model agents in the reference service are deterministic in-process PydanticAI
FunctionModels. The runner deliberately supplies Settings(_env_file=None), so this
test never reads the reference repository's private provider configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import pytest
from streamlit.testing.v1 import AppTest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


WORKSPACE = Path(__file__).resolve().parents[1]
ASSIGNMENT = Path("/home/neeraj/Code/mynaksh_assignment")
ASSIGNMENT_PYTHON = ASSIGNMENT / ".venv/bin/python"
GATEWAY_TOKEN = "streamlit-e2e-test-token-" + "x" * 32
PASSWORD = "local-e2e-password-123"
CODE_SWITCHED_GOAL = "Mera goal अगले साल teacher banna hai — please remember this."


ASSIGNMENT_RUNNER = r"""from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from app.config import Settings
from app.main import create_app


def decide(messages, info):
    payload = json.loads(messages[-1].parts[-1].content)
    message = payload["current_message"]
    args = {"language_scope": "unchanged"}
    if "Mera goal अगले साल teacher banna hai" in message:
        evidence = "Mera goal अगले साल teacher banna hai"
        args["new_memories"] = [{
            "kind": "goal",
            "key": "career_goal",
            "value": "become a teacher next year",
            "evidence": evidence,
        }]
    elif "goal" in message.casefold():
        args["relevant_memory_ids"] = [
            item["id"] for item in payload["stored_memories"]
            if item["key"] == "career_goal"
        ]
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])


def reply(messages, _info):
    payload = json.loads(messages[-1].parts[-1].content)
    has_goal = any(
        item["key"] == "career_goal" and item["value"] == "become a teacher next year"
        for item in payload["relevant_memories"]
    )
    text = (
        "Tumhara goal agle saal teacher banna tha."
        if has_goal
        else "Mujhe aapka career goal yaad nahi hai."
    )
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
    context_model=FunctionModel(decide),
    reply_model=FunctionModel(reply),
)
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_http(url: str, process: subprocess.Popen[bytes], log_path: Path) -> None:
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 20
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            details = log_path.read_text(errors="replace") if log_path.exists() else ""
            raise AssertionError(f"server exited with {process.returncode}: {details[-4000:]}")
        try:
            with opener.open(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (OSError, URLError) as exc:
            last_error = exc
            time.sleep(0.1)
    details = log_path.read_text(errors="replace") if log_path.exists() else ""
    raise AssertionError(f"server did not become ready ({last_error}): {details[-4000:]}")


@pytest.fixture
def cleaned_tmp_path(tmp_path: Path):
    """Remove E2E databases, synthetic audio, runner, and logs after each run."""
    try:
        yield tmp_path
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_streamlit_login_memory_cross_session_and_optional_voice_e2e(
    cleaned_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise AppTest against live Uvicorn services without external model calls."""
    tmp_path = cleaned_tmp_path
    if not ASSIGNMENT_PYTHON.is_file():
        pytest.skip(f"reference assignment interpreter is unavailable: {ASSIGNMENT_PYTHON}")
    run_local_asr = os.getenv("RUN_LOCAL_ASR_E2E") == "1"
    whisper_cli = os.getenv("WHISPER_CLI", "").strip() if run_local_asr else ""
    whisper_model = os.getenv("WHISPER_MODEL", "").strip() if run_local_asr else ""
    if run_local_asr:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or not whisper_cli or not whisper_model:
            pytest.fail(
                "RUN_LOCAL_ASR_E2E=1 requires ffmpeg and explicit WHISPER_CLI/WHISPER_MODEL paths"
            )
        if not Path(whisper_cli).is_file() or not Path(whisper_model).is_file():
            pytest.fail("Configured local Whisper CLI/model path does not exist")

    assignment_port = _free_port()
    gateway_port = _free_port()
    assignment_url = f"http://127.0.0.1:{assignment_port}"
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    runner_path = tmp_path / "assignment_e2e_server.py"
    runner_path.write_text(ASSIGNMENT_RUNNER, encoding="utf-8")
    signing_key = Ed25519PrivateKey.generate()
    private_key_path = tmp_path / "identity-private.pem"
    public_key_path = tmp_path / "identity-public.pem"
    private_key_path.write_bytes(
        signing_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_key_path.write_bytes(
        signing_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    private_key_path.chmod(0o600)
    synthetic_wav = b""
    phrase = "Please turn left at the next street."
    if run_local_asr:
        phrase_file = tmp_path / "synthetic-phrase.txt"
        wav_file = tmp_path / "synthetic-phrase.wav"
        phrase_file.write_text(phrase + "\n", encoding="utf-8")
        generated = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"flite=textfile={phrase_file}:voice=kal",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(wav_file),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert generated.returncode == 0, f"synthetic speech generation failed: {generated.stderr}"
        synthetic_wav = wav_file.read_bytes()
    assignment_log = tmp_path / "assignment-server.log"
    gateway_log = tmp_path / "gateway-server.log"

    assignment_env = os.environ.copy()
    assignment_env.update(
        {
            "E2E_ASSIGNMENT_DB": str(tmp_path / "assignment.sqlite3"),
            "PYTHONPATH": os.pathsep.join((str(tmp_path), str(ASSIGNMENT))),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LANGUAGE_MODEL"):
        assignment_env.pop(key, None)

    gateway_env = os.environ.copy()
    gateway_env.update(
        {
            "CHAT_API_URL": f"{assignment_url}/chat",
            "PMVA_GATEWAY_TOKEN": GATEWAY_TOKEN,
            "PMVA_IDENTITY_PUBLIC_KEY_FILE": str(public_key_path),
            "PMVA_GATEWAY_REPLAY_DB": str(tmp_path / "gateway-replays.sqlite3"),
            "CHAT_API_TIMEOUT_SECONDS": "15",
            "PYTHONPATH": str(WORKSPACE),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    if run_local_asr:
        gateway_env.update({"WHISPER_CLI": whisper_cli, "WHISPER_MODEL": whisper_model})
    else:
        gateway_env.pop("WHISPER_CLI", None)
        gateway_env.pop("WHISPER_MODEL", None)

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

        monkeypatch.setenv("GATEWAY_URL", gateway_url)
        monkeypatch.setenv("PMVA_GATEWAY_TOKEN", GATEWAY_TOKEN)
        monkeypatch.setenv("PMVA_IDENTITY_PRIVATE_KEY_FILE", str(private_key_path))
        monkeypatch.setenv("PMVA_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
        app_path = WORKSPACE / "streamlit_app.py"

        # Create Alice, then log in from a fresh AppTest session to prove the
        # account ID is stable while each authenticated UI gets a new chat session.
        registration = AppTest.from_file(app_path, default_timeout=120).run()
        registration.text_input(key="register_username").set_value("alice")
        registration.text_input(key="register_password").set_value(PASSWORD)
        registration.button(key="register_submit").click().run()
        assert not registration.exception
        alice_id = registration.session_state["account_id"]

        alice = AppTest.from_file(app_path, default_timeout=120).run()
        alice.text_input(key="login_username").set_value("alice")
        alice.text_input(key="login_password").set_value(PASSWORD)
        alice.button(key="login_submit").click().run()
        assert not alice.exception
        assert alice.session_state["account_id"] == alice_id
        first_session = alice.session_state["session_id"]

        if run_local_asr:
            # AppTest stages one WAV recording into st.audio_input. The real
            # gateway performs local Whisper transcription; no chat turn occurs.
            alice.checkbox(key="audio_consent_0").set_value(True).run()
            alice.audio_input(key="voice_recording_0").set_value(
                ("synthetic.wav", synthetic_wav, "audio/wav")
            )
            alice.button(key="transcribe_0").click().run(timeout=120)
            assert not alice.exception
            transcript = alice.text_area(key="draft_1").value
            assert transcript and "turn left at the next street" in transcript.casefold()
            with sqlite3.connect(tmp_path / "assignment.sqlite3") as db:
                assert db.execute("SELECT COUNT(*) FROM conversation_messages").fetchone() == (0,)

        # Send a mixed Latin/Devanagari statement through the authenticated UI.
        # In the opt-in audio branch this also proves the transcript is editable.
        draft_key = f"draft_{alice.session_state['draft_generation']}"
        alice.text_area(key=draft_key).set_value(CODE_SWITCHED_GOAL).run()
        alice.button(key="send_message").click().run()
        assert not alice.exception
        assert any(CODE_SWITCHED_GOAL in element.value for element in alice.markdown)

        # The actual Uvicorn reference app handles this follow-up in a new
        # session and should select Alice's stored career-goal memory.
        alice.button(key="new_conversation").click().run()
        second_session = alice.session_state["session_id"]
        assert first_session != second_session
        alice.text_area(key=f"draft_{alice.session_state['draft_generation']}").set_value(
            "Mera goal kya tha?"
        ).run()
        alice.button(key="send_message").click().run()
        assert not alice.exception
        assert any(
            "Tumhara goal agle saal teacher banna tha." in item.value for item in alice.markdown
        )

        # Bob is a distinct local account. The same query must not expose Alice's
        # selected memory, despite using the same assignment service/database.
        bob = AppTest.from_file(app_path, default_timeout=120).run()
        bob.text_input(key="register_username").set_value("bob")
        bob.text_input(key="register_password").set_value(PASSWORD)
        bob.button(key="register_submit").click().run()
        assert not bob.exception
        bob_id = bob.session_state["account_id"]
        assert bob_id != alice_id
        bob.text_area(key="draft_0").set_value("Mera goal kya tha?").run()
        bob.button(key="send_message").click().run()
        assert not bob.exception
        assert any("Mujhe aapka career goal yaad nahi hai." in item.value for item in bob.markdown)

        with sqlite3.connect(tmp_path / "assignment.sqlite3") as db:
            alice_memory = db.execute(
                "SELECT memory_key, value FROM user_memories WHERE user_id=?",
                (alice_id,),
            ).fetchall()
            bob_memory = db.execute(
                "SELECT memory_key, value FROM user_memories WHERE user_id=?",
                (bob_id,),
            ).fetchall()
            stored_message = db.execute(
                "SELECT content FROM conversation_messages WHERE user_id=? AND session_id=? AND role='user'",
                (alice_id, first_session),
            ).fetchone()
        assert alice_memory == [("career_goal", "become a teacher next year")]
        assert bob_memory == []
        assert stored_message == (CODE_SWITCHED_GOAL,)
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
