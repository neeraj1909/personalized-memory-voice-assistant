"""Authenticated server-side signaling and tool sideband for direct Realtime WebRTC.

The browser sends only an SDP offer; the gateway owns model/session policy, provider
credentials, and the owner-bound personalized memory callback. A direct browser data channel
can still send provider events, so policy checks are detection/termination
controls, not a hard boundary against a modified client.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import re
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import quote, urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

log = logging.getLogger("uvicorn.error")

CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
MAX_SDP_BYTES = 200_000
MAX_ACTIVE_SESSIONS = 4
MAX_SESSION_SECONDS = 3600
MAX_CHAT_MESSAGE_CHARS = 4_000
TOOL_NAME = "ask_memory"


class RealtimeBrokerError(Exception):
    """A Realtime call could not be created or safely closed."""


def session_config(model: str = "gpt-realtime-2.1") -> dict[str, Any]:
    """Server-owned policy: one required personalized memory tool, fixed audio and VAD config."""
    return {
        "type": "realtime",
        "model": model,
        "instructions": (
            "You are the voice interface for personalized memory's personalized chat. For every user "
            "turn, call ask_memory with the recognized user message, whatever its topic. "
            "Preserve the user's words and mixed languages in the message; do not translate "
            "or paraphrase it into a single language. Hinglish naturally mixes Hindi and English. "
            "Do not answer using your own knowledge. After the tool succeeds, speak only "
            "its returned answer, preserving its languages and code-switching."
        ),
        "tools": [
            {
                "type": "function",
                "name": TOOL_NAME,
                "description": "Get an answer from the personalized personalized memory chat service.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_CHAT_MESSAGE_CHARS,
                        }
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            }
        ],
        "tool_choice": "required",
        # VAD starts a tool-selection response automatically. Keep that response
        # silent; only the explicit response after a successful tool call speaks.
        "output_modalities": ["text"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "gpt-4o-mini-transcribe"},
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": True,
                    "interrupt_response": True,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            },
            "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "marin"},
        },
        "max_output_tokens": 512,
        "truncation": {"type": "retention_ratio", "retention_ratio": 0.8},
    }


def call_id_from_location(location: str) -> str:
    path = urlsplit(location).path.rstrip("/")
    parts = path.split("/")
    if len(parts) < 4 or parts[-3:-1] != ["realtime", "calls"]:
        raise RealtimeBrokerError("Provider returned an invalid call location")
    call_id = parts[-1]
    if not CALL_ID_PATTERN.fullmatch(call_id):
        raise RealtimeBrokerError("Provider returned an invalid call location")
    return call_id


def _policy_matches(actual: Any, expected: dict[str, Any]) -> bool:
    """Subset comparison: provider defaults may be added, configured values may not drift."""
    if not isinstance(actual, dict):
        return False
    for key in ("model", "instructions", "tool_choice", "output_modalities", "max_output_tokens"):
        if actual.get(key) != expected.get(key):
            return False
    actual_tools = actual.get("tools")
    expected_tools = expected.get("tools")
    if not isinstance(actual_tools, list) or not isinstance(expected_tools, list):
        return False
    if len(actual_tools) != 1 or not _contains_expected(actual_tools[0], expected_tools[0]):
        return False
    return _contains_expected(actual.get("audio"), expected["audio"])


def _contains_expected(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_expected(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_contains_expected(a, e) for a, e in zip(actual, expected))
        )
    return actual == expected


@dataclass
class _Call:
    call_id: str
    owner: tuple[str, str]
    ws: Any
    ws_context: Any
    config: dict[str, Any]
    max_seconds: int
    ready_timeout_seconds: float
    record_usage: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None
    task: asyncio.Task | None = None
    ending: asyncio.Event = field(default_factory=asyncio.Event)
    peer_ready: asyncio.Event = field(default_factory=asyncio.Event)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    finish_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    started_at: float = field(default_factory=lambda: asyncio.get_running_loop().time())
    created: int = 0
    completed: int = 0
    pending_speech_generations: list[int] = field(default_factory=list)
    speech_response_generations: dict[str, int] = field(default_factory=dict)
    pending_tools: int = 0
    generation: int = 0
    tool_tasks: set[asyncio.Task] = field(default_factory=set)
    pending_tool_ids: set[str] = field(default_factory=set)
    tool_outputs: dict[str, str] = field(default_factory=dict)
    active_response_id: str | None = None
    committed: set[str] = field(default_factory=set)
    transcribed: set[str] = field(default_factory=set)
    usage_ids: set[str] = field(default_factory=set)
    clear_ack: bool = False
    finished: bool = False
    sideband_closed: bool = False
    hangup_confirmed: bool = False


AuthCallback = Callable[[Request], tuple[str, str] | Awaitable[tuple[str, str]]]
ChatCallback = Callable[[str, str, str], str | Awaitable[str]]
OwnerActiveCallback = Callable[[tuple[str, str]], bool | Awaitable[bool]]


def build_realtime_router(
    authenticate: AuthCallback,
    chat: ChatCallback,
    *,
    api_key: str,
    model: str = "gpt-realtime-2.1",
    max_sessions: int = MAX_ACTIVE_SESSIONS,
    max_session_seconds: int = MAX_SESSION_SECONDS,
    base_url: str = "https://api.openai.com",
    http_client_factory: Callable[[], Any] | None = None,
    ws_connect: Callable[..., Any] | None = None,
    record_usage: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
    identity_is_active: OwnerActiveCallback | None = None,
    ready_timeout_seconds: float = 30.0,
    identity_check_interval_seconds: float = 10.0,
    drain_seconds: float = 10.0,
) -> APIRouter:
    """Create `/api/realtime/call` and owner-checked `/api/realtime/call/{id}`.

    `authenticate(request)` must perform the application's real browser login,
    CSRF and owner/session checks. `chat` receives the authenticated owner and
    only the function's message text; model arguments never contain identity.
    """
    if not api_key:
        # Route remains constructible for tests, but requests fail closed.
        api_key = ""
    if min(max_sessions, max_session_seconds) <= 0:
        raise ValueError("Realtime limits must be positive")
    if model != "gpt-realtime-2.1":
        raise ValueError("Only the configured Realtime model is supported")
    if not 0 < drain_seconds <= 15:
        raise ValueError("Realtime drain limit must be between 0 and 15 seconds")
    if ready_timeout_seconds <= 0 or identity_check_interval_seconds <= 0:
        raise ValueError("Peer readiness and identity check intervals must be positive")
    max_sessions = min(max_sessions, MAX_ACTIVE_SESSIONS)
    max_session_seconds = min(max_session_seconds, MAX_SESSION_SECONDS)
    router = APIRouter()
    sessions: dict[str, _Call] = {}
    closed_sessions: OrderedDict[str, tuple[tuple[str, str], float]] = OrderedDict()
    sessions_lock = asyncio.Lock()
    pending_creates = 0
    closing = False
    client_factory = http_client_factory or (lambda: httpx.AsyncClient(timeout=20.0))
    provider_base = base_url.rstrip("/")
    parsed_provider = urlsplit(provider_base)
    if (
        parsed_provider.scheme not in {"https", "http"}
        or not parsed_provider.netloc
        or parsed_provider.username
        or parsed_provider.password
        or parsed_provider.path not in {"", "/"}
        or parsed_provider.query
        or parsed_provider.fragment
    ):
        raise ValueError("Realtime provider base must be a bare HTTP(S) origin")

    async def identity_for(request: Request) -> tuple[str, str]:
        try:
            identity = authenticate(request)
            if inspect.isawaitable(identity):
                identity = await identity
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=401, detail="Unauthorized") from exc
        if (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or not all(isinstance(part, str) and part.strip() for part in identity)
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return identity

    async def send(call: _Call, event: dict[str, Any]) -> None:
        async with call.send_lock:
            await call.ws.send(json.dumps(event, ensure_ascii=False))

    async def provider_hangup(call_id: str) -> bool:
        try:
            async with client_factory() as client:
                response = await client.post(
                    f"{provider_base}/v1/realtime/calls/{quote(call_id, safe='')}/hangup",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if response.status_code == 404:
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = None
                    error = payload.get("error") if isinstance(payload, dict) else None
                    if isinstance(error, dict) and error.get("code") == "call_id_not_found":
                        # WebRTC may already have removed the call when its
                        # peer closes before our explicit hangup reaches it.
                        return True
                if not response.is_success:
                    log.warning("Realtime provider hangup returned status=%d", response.status_code)
                return response.is_success
        except Exception as exc:
            log.warning("Realtime provider hangup request failed: %s", type(exc).__name__)
            return False

    async def invoke_chat(owner: tuple[str, str], message: str) -> Any:
        if inspect.iscoroutinefunction(chat):
            return await chat(owner[0], owner[1], message)
        value = await asyncio.to_thread(chat, owner[0], owner[1], message)
        if inspect.isawaitable(value):
            return await value
        return value

    async def finish(call: _Call) -> bool:
        async with call.finish_lock:
            call.finished = True
            if not call.hangup_confirmed:
                call.hangup_confirmed = await provider_hangup(call.call_id)
            if not call.sideband_closed:
                with suppress(Exception):
                    await call.ws_context.__aexit__(None, None, None)
                call.sideband_closed = True
            if call.hangup_confirmed:
                async with sessions_lock:
                    sessions.pop(call.call_id, None)
                    closed_sessions[call.call_id] = (call.owner, time.monotonic() + 3600)
                    closed_sessions.move_to_end(call.call_id)
                    while len(closed_sessions) > 256:
                        closed_sessions.popitem(last=False)
            else:
                # Keep this owner-bound record and its capacity slot so DELETE
                # can retry provider cleanup. The websocket is already closed.
                log.warning("Realtime provider hangup unconfirmed call_id=%s", call.call_id)
            return call.hangup_confirmed

    async def shutdown() -> None:
        nonlocal closing
        closing = True
        async with sessions_lock:
            active = tuple(sessions.values())
        for call in active:
            call.ending.set()
            with suppress(Exception):
                await send(call, {"type": "input_audio_buffer.clear"})
        for call in active:
            if call.task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(call.task), timeout=drain_seconds)
                except asyncio.TimeoutError:
                    call.task.cancel()
                    with suppress(asyncio.CancelledError):
                        await call.task
            # Retry failed monitor cleanup during orderly application shutdown.
            for attempt in range(3):
                if await finish(call):
                    break
                if attempt < 2:
                    await asyncio.sleep(0.1 * (attempt + 1))

    router.add_event_handler("shutdown", shutdown)

    async def tool_output(call: _Call, function: dict[str, Any], generation: int) -> None:
        call_id = function.get("call_id")
        name = function.get("name")
        safe_result = "I couldn't reach the personalized memory chat service. Please try again."
        result_ok = False
        try:
            if (
                name != TOOL_NAME
                or not isinstance(call_id, str)
                or not CALL_ID_PATTERN.fullmatch(call_id)
            ):
                raise ValueError("invalid tool call")
            arguments = json.loads(function.get("arguments", "{}"))
            if not isinstance(arguments, dict) or set(arguments) != {"message"}:
                raise ValueError("invalid tool arguments")
            message = arguments["message"]
            if (
                not isinstance(message, str)
                or not message.strip()
                or len(message) > MAX_CHAT_MESSAGE_CHARS
            ):
                raise ValueError("invalid tool message")
            started = time.monotonic()
            try:
                value = await invoke_chat(call.owner, message.strip())
            finally:
                log.info(
                    "Realtime personalized memory request duration_ms=%d",
                    (time.monotonic() - started) * 1000,
                )
            if not isinstance(value, str) or not value.strip():
                raise ValueError("empty chat answer")
            safe_result = value.strip()
            result_ok = True
        except Exception:
            log.info("Realtime personalized memory tool failed call_id=%s", call.call_id)
        if call.finished:
            return
        if isinstance(call_id, str) and call_id in call.tool_outputs:
            await send(
                call,
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": call.tool_outputs[call_id],
                    },
                },
            )
            return
        output = json.dumps({"answer": safe_result}, ensure_ascii=False)
        if isinstance(call_id, str):
            call.tool_outputs[call_id] = output
        await send(
            call,
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id or "invalid_call",
                    "output": output,
                },
            },
        )
        if not result_ok:
            # Surface a fixed, factual error in the shared conversation. Do not
            # invoke the model to improvise when personalized memory is unavailable.
            await send(
                call,
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": safe_result}],
                    },
                },
            )
            return
        # Successful tool output is spoken once; the second response cannot call
        # the tool recursively.
        if result_ok and not call.ending.is_set() and generation == call.generation:
            call.pending_speech_generations.append(generation)
            await send(
                call,
                {
                    "type": "response.create",
                    "response": {
                        "tool_choice": "none",
                        "instructions": (
                            "Speak only the answer in the preceding function result. "
                            "Preserve its wording and mixed languages; do not translate, "
                            "paraphrase, or add facts."
                        ),
                        "output_modalities": ["audio"],
                        "max_output_tokens": 2048,
                    },
                },
            )

    async def record_usage_event(call: _Call, event: dict[str, Any]) -> None:
        callback = call.record_usage
        if callback is None:
            return
        if inspect.iscoroutinefunction(callback):
            result = await callback(call.call_id, event)
        else:
            result = await asyncio.to_thread(callback, call.call_id, event)
        if inspect.isawaitable(result):
            await result

    async def owner_is_active(owner: tuple[str, str]) -> bool:
        if identity_is_active is None:
            return True
        if inspect.iscoroutinefunction(identity_is_active):
            result = await identity_is_active(owner)
        else:
            result = await asyncio.to_thread(identity_is_active, owner)
        if inspect.isawaitable(result):
            result = await result
        return result is True

    async def monitor(call: _Call) -> None:
        deadline = call.started_at + call.max_seconds
        hard_deadline_drain = False
        next_identity_check = call.started_at + identity_check_interval_seconds
        try:
            while True:
                now = asyncio.get_running_loop().time()
                if (
                    not call.peer_ready.is_set()
                    and now - call.started_at >= call.ready_timeout_seconds
                ):
                    call.ending.set()
                    break
                if identity_is_active is not None and now >= next_identity_check:
                    next_identity_check = now + identity_check_interval_seconds
                    if not await owner_is_active(call.owner):
                        call.ending.set()
                        with suppress(Exception):
                            await send(call, {"type": "input_audio_buffer.clear"})
                        break
                remaining = deadline - now
                if remaining <= 0:
                    if not hard_deadline_drain:
                        hard_deadline_drain = True
                        call.ending.set()
                        deadline = asyncio.get_running_loop().time() + drain_seconds
                        with suppress(Exception):
                            await send(call, {"type": "input_audio_buffer.clear"})
                        continue
                    break
                try:
                    wait_seconds = min(remaining, 0.5)
                    if not call.peer_ready.is_set():
                        wait_seconds = min(
                            wait_seconds,
                            max(0.01, call.ready_timeout_seconds - (now - call.started_at)),
                        )
                    if identity_is_active is not None:
                        wait_seconds = min(
                            wait_seconds,
                            max(0.01, next_identity_check - now),
                        )
                    raw = await asyncio.wait_for(call.ws.recv(), timeout=wait_seconds)
                except asyncio.TimeoutError:
                    if (
                        call.ending.is_set()
                        and call.clear_ack
                        and call.created == call.completed
                        and call.committed <= call.transcribed
                        and call.pending_tools == 0
                    ):
                        break
                    continue
                event = json.loads(raw)
                if not isinstance(event, dict):
                    raise ValueError("invalid provider event")
                kind = event.get("type")
                if kind == "session.updated" and not _policy_matches(
                    event.get("session"), call.config
                ):
                    log.warning("Realtime sideband stopped: session policy mismatch")
                    # Detect provider/client policy mutation; direct browser data
                    # channel control means this cannot prevent all earlier output.
                    call.ending.set()
                    with suppress(Exception):
                        await send(call, {"type": "response.cancel"})
                    break
                if kind == "response.created":
                    response = event.get("response") or {}
                    response_choice = response.get("tool_choice")
                    if response_choice == "required":
                        # A new required-tool response proves any pending
                        # speech request from an older VAD generation was
                        # cancelled before response.created reached sideband.
                        call.pending_speech_generations = [
                            generation
                            for generation in call.pending_speech_generations
                            if generation == call.generation
                        ]
                    is_speech = (
                        bool(call.pending_speech_generations) and response_choice != "required"
                    )
                    response_id = response.get("id")
                    if is_speech:
                        if not isinstance(response_id, str):
                            call.ending.set()
                            break
                        call.speech_response_generations[response_id] = (
                            call.pending_speech_generations.pop(0)
                        )
                    expected_choice = "none" if is_speech else "required"
                    if (
                        "tool_choice" in response and response.get("tool_choice") != expected_choice
                    ) or (
                        "tools" in response
                        and not _contains_expected(response.get("tools"), call.config["tools"])
                    ):
                        call.ending.set()
                        with suppress(Exception):
                            await send(call, {"type": "response.cancel"})
                        break
                    if isinstance(response_id, str):
                        call.active_response_id = response_id
                        generation = call.speech_response_generations.get(response_id)
                        if generation is not None and generation != call.generation:
                            with suppress(Exception):
                                await send(
                                    call,
                                    {"type": "response.cancel", "response_id": response_id},
                                )
                    call.created += 1
                elif kind == "response.done":
                    call.completed += 1
                    response = event.get("response") or {}
                    response_id = response.get("id")
                    is_speech = (
                        isinstance(response_id, str)
                        and response_id in call.speech_response_generations
                    )
                    if is_speech:
                        call.speech_response_generations.pop(response_id, None)
                    if response_id == call.active_response_id:
                        call.active_response_id = None
                    event_id = event.get("event_id")
                    if call.record_usage and (not isinstance(event_id, str) or not event_id):
                        call.ending.set()
                        break
                    if event_id and event_id not in call.usage_ids and call.record_usage:
                        call.usage_ids.add(event_id)
                        await record_usage_event(call, event)
                    if response.get("status") == "completed":
                        if not any(
                            isinstance(item, dict) and item.get("type") == "function_call"
                            for item in response.get("output", [])
                        ):
                            if not is_speech:
                                # A response without a completed personalized memory tool is
                                # outside configured behavior; terminate promptly.
                                call.ending.set()
                                with suppress(Exception):
                                    await send(call, {"type": "response.cancel"})
                                break
                        calls = [
                            item
                            for item in response.get("output", [])
                            if isinstance(item, dict) and item.get("type") == "function_call"
                        ]
                        if len(calls) > 1:
                            call.ending.set()
                            break
                        if calls:
                            tool_id = calls[0].get("call_id")
                            if not isinstance(tool_id, str) or not CALL_ID_PATTERN.fullmatch(
                                tool_id
                            ):
                                call.ending.set()
                                break
                            if tool_id in call.pending_tool_ids:
                                continue
                            if tool_id in call.tool_outputs:
                                await send(
                                    call,
                                    {
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "function_call_output",
                                            "call_id": tool_id,
                                            "output": call.tool_outputs[tool_id],
                                        },
                                    },
                                )
                                continue
                            call.pending_tool_ids.add(tool_id)
                            call.pending_tools += 1
                            generation = call.generation
                            task = asyncio.create_task(tool_output(call, calls[0], generation))
                            call.tool_tasks.add(task)

                            def tool_finished(
                                done: asyncio.Task,
                                active_call=call,
                                active_tool_id=tool_id,
                            ) -> None:
                                active_call.pending_tools = max(0, active_call.pending_tools - 1)
                                active_call.tool_tasks.discard(done)
                                active_call.pending_tool_ids.discard(active_tool_id)

                            task.add_done_callback(tool_finished)
                elif kind == "conversation.item.input_audio_transcription.completed":
                    item_id = event.get("item_id")
                    if isinstance(item_id, str):
                        call.transcribed.add(item_id)
                    event_id = event.get("event_id")
                    if call.record_usage and (not isinstance(event_id, str) or not event_id):
                        call.ending.set()
                        break
                    if event_id and event_id not in call.usage_ids and call.record_usage:
                        call.usage_ids.add(event_id)
                        await record_usage_event(call, event)
                elif kind == "input_audio_buffer.committed":
                    item_id = event.get("item_id")
                    if isinstance(item_id, str):
                        call.committed.add(item_id)
                elif kind == "input_audio_buffer.speech_started":
                    active = call.active_response_id
                    if active:
                        # VAD is configured to interrupt. This sideband cancel is
                        # an additional server-side fence; the browser must also
                        # clear local playback because sideband cannot retract it.
                        with suppress(Exception):
                            await send(call, {"type": "response.cancel", "response_id": active})
                    call.active_response_id = None
                    call.generation += 1
                elif kind == "input_audio_buffer.cleared" and call.ending.is_set():
                    call.clear_ack = True
                elif kind == "session.closed":
                    call.ending.set()
                    break
                elif kind in {"error", "conversation.item.input_audio_transcription.failed"}:
                    error = event.get("error") or {}
                    code = error.get("code") if isinstance(error, dict) else None
                    if kind == "error" and code == "response_cancel_not_active":
                        # VAD may cancel first; our additional interrupt then
                        # races with it. The conversation remains usable.
                        continue
                    log.warning("Realtime sideband stopped: provider event=%s code=%s", kind, code)
                    call.ending.set()
                    break
                if (
                    call.ending.is_set()
                    and call.clear_ack
                    and call.created == call.completed
                    and call.committed <= call.transcribed
                    and call.pending_tools == 0
                ):
                    break
        except asyncio.CancelledError:
            raise
        except Exception:
            log.info("Realtime sideband ended unexpectedly call_id=%s", call.call_id)
        finally:
            for task in tuple(call.tool_tasks):
                task.cancel()
            await finish(call)

    @router.post("/api/realtime/call")
    async def create_call(request: Request) -> Response:
        if closing:
            raise HTTPException(status_code=503, detail="Realtime voice is shutting down")
        if not api_key:
            raise HTTPException(status_code=503, detail="Realtime voice is not configured")
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/sdp"
        ):
            raise HTTPException(status_code=415, detail="Expected application/sdp")
        owner = await identity_for(request)
        offer_buffer = bytearray()
        async for chunk in request.stream():
            offer_buffer.extend(chunk)
            if len(offer_buffer) > MAX_SDP_BYTES:
                raise HTTPException(status_code=422, detail="Invalid SDP offer")
        offer = bytes(offer_buffer)
        if not offer or len(offer) > MAX_SDP_BYTES:
            raise HTTPException(status_code=422, detail="Invalid SDP offer")
        nonlocal pending_creates
        async with sessions_lock:
            if len(sessions) + pending_creates >= max_sessions:
                raise HTTPException(status_code=429, detail="Realtime session limit reached")
            pending_creates += 1
        slot_reserved = True
        config = session_config(model)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Safety-Identifier": hashlib.sha256(owner[0].encode()).hexdigest(),
        }
        call_id: str | None = None
        active_call: _Call | None = None
        ws_context = None
        try:
            async with client_factory() as client:
                response = await client.post(
                    f"{provider_base}/v1/realtime/calls",
                    headers=headers,
                    files={
                        "sdp": (None, offer.decode("utf-8"), "application/sdp"),
                        "session": (None, json.dumps(config), "application/json"),
                    },
                )
            response.raise_for_status()
            call_id = call_id_from_location(response.headers.get("Location", ""))
            answer = response.content
            if not answer or len(answer) > MAX_SDP_BYTES:
                raise RealtimeBrokerError("Provider returned an invalid SDP answer")
            if ws_connect is None:
                from websockets.asyncio.client import connect

                connect_ws = connect
            else:
                connect_ws = ws_connect
            ws_scheme = "wss" if parsed_provider.scheme == "https" else "ws"
            ws_host = parsed_provider.netloc
            ws_url = f"{ws_scheme}://{ws_host}/v1/realtime?call_id={quote(call_id, safe='')}"
            ws_context = connect_ws(
                ws_url,
                additional_headers={
                    **headers,
                    "OpenAI-Safety-Identifier": hashlib.sha256(owner[0].encode()).hexdigest(),
                },
                max_size=2**22,
                open_timeout=10,
            )
            ws = await ws_context.__aenter__()
            call = _Call(
                call_id=call_id,
                owner=owner,
                ws=ws,
                ws_context=ws_context,
                config=config,
                max_seconds=min(max_session_seconds, MAX_SESSION_SECONDS),
                ready_timeout_seconds=ready_timeout_seconds,
                record_usage=record_usage,
            )
            active_call = call
            async with sessions_lock:
                if call_id in sessions:
                    raise RealtimeBrokerError("Realtime session limit reached")
                closed_sessions.pop(call_id, None)
                sessions[call_id] = call
                pending_creates -= 1
                slot_reserved = False
            call.task = asyncio.create_task(monitor(call), name=f"realtime-sideband-{call_id}")
            result = Response(
                content=answer,
                media_type="application/sdp",
                headers={"X-Call-ID": call_id, "Cache-Control": "no-store"},
            )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            if active_call is not None:
                if active_call.task is not None:
                    active_call.task.cancel()
                    with suppress(asyncio.CancelledError):
                        await active_call.task
                await finish(active_call)
            elif ws_context is not None:
                with suppress(Exception):
                    await ws_context.__aexit__(None, None, None)
            if call_id and active_call is None:
                await provider_hangup(call_id)
            log.info("Realtime call setup failed: %s", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Could not start Realtime voice") from None
        finally:
            if slot_reserved:
                async with sessions_lock:
                    pending_creates -= 1

    @router.delete("/api/realtime/call/{call_id}")
    async def end_call(call_id: str, request: Request) -> dict[str, bool]:
        owner = await identity_for(request)
        if not CALL_ID_PATTERN.fullmatch(call_id):
            raise HTTPException(status_code=404, detail="Realtime session not found")
        async with sessions_lock:
            call = sessions.get(call_id)
            closed = closed_sessions.get(call_id)
            if closed and closed[1] <= time.monotonic():
                closed_sessions.pop(call_id, None)
                closed = None
        if call is None and closed and closed[0] == owner:
            return {"ended": True}
        if call is None or call.owner != owner:
            raise HTTPException(status_code=404, detail="Realtime session not found")
        call.ending.set()
        with suppress(Exception):
            await send(call, {"type": "input_audio_buffer.clear"})
        if call.task:
            try:
                await asyncio.wait_for(asyncio.shield(call.task), timeout=drain_seconds)
            except asyncio.TimeoutError:
                call.task.cancel()
                with suppress(asyncio.CancelledError):
                    await call.task
        confirmed = await finish(call)
        if not confirmed:
            raise HTTPException(
                status_code=503, detail="Realtime session closure could not be confirmed"
            )
        return {"ended": True}

    @router.post("/api/realtime/call/{call_id}/ready")
    async def peer_ready(call_id: str, request: Request) -> dict[str, bool]:
        owner = await identity_for(request)
        if not CALL_ID_PATTERN.fullmatch(call_id):
            raise HTTPException(status_code=404, detail="Realtime session not found")
        async with sessions_lock:
            call = sessions.get(call_id)
        if call is None or call.owner != owner or call.ending.is_set():
            raise HTTPException(status_code=404, detail="Realtime session not found")
        call.peer_ready.set()
        return {"ready": True}

    return router
