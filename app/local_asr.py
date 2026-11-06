"""Local Whisper.cpp transcription for short PCM WAV clips."""

import asyncio
import io
import os
import re
import tempfile
import threading
import wave
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from .security import require_gateway_token

MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
TRANSCRIPTION_TIMEOUT_SECONDS = 60
_transcription_lock = threading.Lock()
_WHISPER_LANGUAGE_PATTERN = re.compile(r"(?:auto|[a-z]{2,3})")
# whisper.cpp's English-only ggml model filenames use a `.en` marker.
_ENGLISH_ONLY_MODEL_PATTERN = re.compile(r"\.en(?:[-_.]|$)", re.IGNORECASE)

router = APIRouter()


class TranscriptionResponse(BaseModel):
    text: str


async def _read_audio(request: Request) -> bytes:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "audio/wav":
        raise HTTPException(status_code=415, detail="Send audio as audio/wav")

    audio = bytearray()
    async for chunk in request.stream():
        if len(audio) + len(chunk) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio exceeds the size limit")
        audio.extend(chunk)
    return bytes(audio)


def _configured_language(model: str) -> str:
    # whisper.cpp accepts "auto" for its multilingual models. An explicit
    # deployment setting still takes precedence for predictable workloads.
    language = os.getenv("WHISPER_LANGUAGE", "auto").strip().casefold()
    if not _WHISPER_LANGUAGE_PATTERN.fullmatch(language):
        raise HTTPException(
            status_code=503,
            detail="WHISPER_LANGUAGE must be 'auto' or a 2-3 letter language code",
        )
    if language != "en" and _ENGLISH_ONLY_MODEL_PATTERN.search(Path(model).name):
        raise HTTPException(
            status_code=503,
            detail=(
                "WHISPER_MODEL appears to be English-only; configure a multilingual model "
                "to use WHISPER_LANGUAGE other than 'en'"
            ),
        )
    return language


def _validate_wav(audio: bytes) -> None:
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frames = source.getnframes()
            if (
                source.getcomptype() != "NONE"
                or source.getnchannels() != 1
                or sample_width != 2
                or sample_rate != 16000
                or frames <= 0
            ):
                raise HTTPException(
                    status_code=422, detail="Audio must be mono 16 kHz 16-bit PCM WAV"
                )
            if frames / sample_rate > MAX_AUDIO_SECONDS:
                raise HTTPException(status_code=413, detail="Audio exceeds the duration limit")
            if len(source.readframes(frames)) != frames * sample_width:
                raise HTTPException(status_code=422, detail="WAV audio data is incomplete")
    except (EOFError, OSError, wave.Error) as exc:
        raise HTTPException(status_code=422, detail="Invalid WAV audio") from exc


async def _run_whisper(audio: bytes, cli: str, model: str, language: str) -> str:
    if not _transcription_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Transcription is busy; try again shortly")

    temp_dir: tempfile.TemporaryDirectory | None = None
    process: asyncio.subprocess.Process | None = None
    try:
        try:
            temp_dir = tempfile.TemporaryDirectory(prefix="memory-asr-")
            temp_root = Path(temp_dir.name)
            input_path = temp_root / "input.wav"
            output_prefix = temp_root / "transcription"
            input_path.write_bytes(audio)
            input_path.chmod(0o600)
        except OSError as exc:
            raise HTTPException(
                status_code=503, detail="Local transcription is unavailable"
            ) from exc

        try:
            process = await asyncio.create_subprocess_exec(
                cli,
                "-m",
                model,
                "-t",
                "2",
                "-ng",
                "-l",
                language,
                "-nt",
                "-np",
                "-otxt",
                "-of",
                str(output_prefix),
                "-f",
                str(input_path),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise HTTPException(
                status_code=503, detail="Local transcription is unavailable"
            ) from exc

        try:
            await asyncio.wait_for(process.communicate(), timeout=TRANSCRIPTION_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise HTTPException(status_code=504, detail="Transcription timed out") from exc
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise

        if process.returncode != 0:
            raise HTTPException(status_code=502, detail="Local transcription failed")
        try:
            return output_prefix.with_suffix(".txt").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise HTTPException(status_code=502, detail="Local transcription failed") from exc
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        _transcription_lock.release()


@router.post(
    "/api/transcribe",
    response_model=TranscriptionResponse,
    dependencies=[Depends(require_gateway_token)],
)
async def transcribe(request: Request) -> TranscriptionResponse:
    cli = os.getenv("WHISPER_CLI", "").strip()
    model = os.getenv("WHISPER_MODEL", "").strip()
    if not cli or not model:
        raise HTTPException(status_code=503, detail="Local transcription is not configured")
    language = _configured_language(model)

    audio = await _read_audio(request)
    _validate_wav(audio)
    return TranscriptionResponse(text=await _run_whisper(audio, cli, model, language))
