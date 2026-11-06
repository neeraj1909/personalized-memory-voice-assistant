from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from app.realtime_gateway import (
    TOOL_NAME,
    build_realtime_router,
    call_id_from_location,
    session_config,
)


class FakeSideband:
    def __init__(self):
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, raw: str):
        event = json.loads(raw)
        self.sent.append(event)
        if event.get("type") == "input_audio_buffer.clear":
            await self.incoming.put(json.dumps({"type": "input_audio_buffer.cleared"}))

    async def recv(self):
        return await self.incoming.get()

    async def feed(self, event: dict):
        await self.incoming.put(json.dumps(event))


class FakeSidebandContext:
    def __init__(self, ws: FakeSideband):
        self.ws = ws

    async def __aenter__(self):
        await self.ws.feed({"type": "session.updated", "session": session_config()})
        return self.ws

    async def __aexit__(self, *_args):
        self.ws.closed = True


async def wait_for(predicate, timeout=1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


def build_app(
    *,
    owner=("account-a", "session-a"),
    chat=None,
    max_sessions=2,
    record_usage=None,
    identity_is_active=None,
    ready_timeout_seconds=30,
    identity_check_interval_seconds=10,
    hangup_statuses=None,
):
    ws = FakeSideband()
    provider_requests: list[httpx.Request] = []
    hangup_statuses = list(hangup_statuses or [])

    async def provider(request: httpx.Request) -> httpx.Response:
        provider_requests.append(request)
        if request.url.path.endswith("/hangup"):
            status = hangup_statuses.pop(0) if hangup_statuses else 200
            if isinstance(status, tuple):
                return httpx.Response(status[0], json={"error": {"code": status[1]}})
            return httpx.Response(status)
        return httpx.Response(
            200,
            content=b"v=0\r\nanswer-sdp",
            headers={"Location": "/v1/realtime/calls/rtc-fixture"},
        )

    def authenticate(request):
        if request.headers.get("x-test-auth") != "ok":
            from fastapi import HTTPException

            raise HTTPException(401, "Unauthorized")
        return (
            request.headers.get("x-owner-id", owner[0]),
            request.headers.get("x-session-id", owner[1]),
        )

    router = build_realtime_router(
        authenticate=authenticate,
        chat=chat or (lambda *_args: "A personalized answer."),
        api_key="secret-test-key",
        max_sessions=max_sessions,
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(provider)),
        ws_connect=lambda *_args, **_kwargs: FakeSidebandContext(ws),
        record_usage=record_usage,
        identity_is_active=identity_is_active,
        ready_timeout_seconds=ready_timeout_seconds,
        identity_check_interval_seconds=identity_check_interval_seconds,
        drain_seconds=0.5,
    )
    app = FastAPI()
    app.include_router(router)
    return app, ws, provider_requests


async def open_call(client: httpx.AsyncClient):
    return await client.post(
        "/api/realtime/call",
        content=b"v=0\r\noffer-sdp",
        headers={"content-type": "application/sdp", "x-test-auth": "ok"},
    )


def test_call_uses_server_session_config_and_returns_sdp_and_call_id():
    async def run():
        app, ws, requests = build_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            response = await open_call(client)
            assert response.status_code == 200
            assert response.content == b"v=0\r\nanswer-sdp"
            assert response.headers["content-type"].startswith("application/sdp")
            assert response.headers["x-call-id"] == "rtc-fixture"
            assert response.headers["cache-control"] == "no-store"
            assert "secret-test-key" not in response.text
            request = requests[0]
            assert request.headers["authorization"] == "Bearer secret-test-key"
            assert b"offer-sdp" in request.content
            assert b"ask_memory" in request.content
            assert b"gpt-realtime-2.1" in request.content
            assert b"gpt-4o-mini-transcribe" in request.content
            assert "language" not in session_config()["audio"]["input"]["transcription"]
            assert not ws.closed
            ended = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert ended.status_code == 200
            repeated = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert repeated.status_code == 200
            foreign_after = await client.delete(
                "/api/realtime/call/rtc-fixture",
                headers={"x-test-auth": "ok", "x-owner-id": "other"},
            )
            assert foreign_after.status_code == 404
            assert ended.json() == {"ended": True}
            assert ws.closed
            assert requests[-1].url.path.endswith("/rtc-fixture/hangup")

    asyncio.run(run())


def test_sideband_calls_mynaksh_with_trusted_owner_and_sends_one_followup():
    async def run():
        calls: list[tuple[str, str, str]] = []

        async def chat(user_id, session_id, message):
            calls.append((user_id, session_id, message))
            return "Aap apne career goals par focus करें."

        app, ws, _ = build_app(chat=chat)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            response = await open_call(client)
            assert response.status_code == 200
            await ws.feed({"type": "response.created", "response": {"id": "resp-tool"}})
            await ws.feed(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "event_id": "evt-transcription",
                    "item_id": "item-user-1",
                    "transcript": "Mujhe career change ke लिए क्या करना चाहिए?",
                }
            )
            await ws.feed({"type": "input_audio_buffer.committed", "item_id": "item-user-1"})
            await ws.feed(
                {
                    "type": "response.done",
                    "event_id": "evt-response-tool",
                    "response": {
                        "id": "resp-tool",
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "name": TOOL_NAME,
                                "call_id": "tool-1",
                                "arguments": json.dumps(
                                    {"message": "Mujhe career change ke लिए क्या करना चाहिए?"}
                                ),
                            }
                        ],
                    },
                }
            )
            assert await wait_for(lambda: any(e.get("type") == "response.create" for e in ws.sent))
            assert calls == [
                ("account-a", "session-a", "Mujhe career change ke लिए क्या करना चाहिए?")
            ]
            output = next(e for e in ws.sent if e.get("type") == "conversation.item.create")
            assert output["item"] == {
                "type": "function_call_output",
                "call_id": "tool-1",
                "output": json.dumps(
                    {"answer": "Aap apne career goals par focus करें."}, ensure_ascii=False
                ),
            }
            followups = [e for e in ws.sent if e.get("type") == "response.create"]
            assert len(followups) == 1
            assert followups[0]["response"]["tool_choice"] == "none"
            assert followups[0]["response"]["output_modalities"] == ["audio"]
            assert followups[0]["response"]["max_output_tokens"] == 2048
            assert "mixed languages" in followups[0]["response"]["instructions"]
            await ws.feed(
                {
                    "type": "response.done",
                    "event_id": "evt-response-tool-replay",
                    "response": {
                        "id": "resp-tool",
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "name": TOOL_NAME,
                                "call_id": "tool-1",
                                "arguments": json.dumps(
                                    {"message": "Mujhe career change ke लिए क्या करना चाहिए?"}
                                ),
                            }
                        ],
                    },
                }
            )
            assert await wait_for(
                lambda: (
                    len([e for e in ws.sent if e.get("type") == "conversation.item.create"]) == 2
                )
            )
            assert calls == [
                ("account-a", "session-a", "Mujhe career change ke लिए क्या करना चाहिए?")
            ]
            await ws.feed({"type": "response.created", "response": {"id": "resp-audio"}})
            assert await wait_for(lambda: any(e.get("type") == "response.create" for e in ws.sent))
            await ws.feed({"type": "input_audio_buffer.speech_started"})
            assert await wait_for(lambda: any(e.get("type") == "response.cancel" for e in ws.sent))
            await client.delete("/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"})

    asyncio.run(run())


def test_interrupted_followup_does_not_capture_next_tool_response():
    async def run():
        calls = []

        async def chat(user_id, session_id, message):
            calls.append(message)
            return f"Answer: {message}"

        app, ws, _ = build_app(chat=chat)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            await open_call(client)
            await ws.feed(
                {
                    "type": "response.created",
                    "response": {"id": "tool-response-1", "tool_choice": "required"},
                }
            )
            await ws.feed(
                {
                    "type": "response.done",
                    "response": {
                        "id": "tool-response-1",
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "name": TOOL_NAME,
                                "call_id": "tool-1",
                                "arguments": '{"message":"first"}',
                            }
                        ],
                    },
                }
            )
            assert await wait_for(lambda: any(e.get("type") == "response.create" for e in ws.sent))
            # VAD interrupts the requested speech before its response.created event.
            await ws.feed({"type": "input_audio_buffer.speech_started"})
            await ws.feed(
                {
                    "type": "response.created",
                    "response": {"id": "tool-response-2", "tool_choice": "required"},
                }
            )
            await ws.feed(
                {
                    "type": "response.done",
                    "response": {
                        "id": "tool-response-2",
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "name": TOOL_NAME,
                                "call_id": "tool-2",
                                "arguments": '{"message":"second"}',
                            }
                        ],
                    },
                }
            )
            assert await wait_for(lambda: calls == ["first", "second"])
            # This is the current legitimate speech response. The stale first
            # reservation must not make its response.created look obsolete.
            await ws.feed(
                {
                    "type": "response.created",
                    "response": {"id": "speech-response-2", "tool_choice": "none"},
                }
            )
            await asyncio.sleep(0.03)
            assert not any(
                e.get("type") == "response.cancel" and e.get("response_id") == "speech-response-2"
                for e in ws.sent
            )
            await client.delete("/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"})

    asyncio.run(run())


def test_tool_failure_returns_safe_fallback_without_asking_model_to_answer():
    async def run():
        def fail_chat(*_args):
            raise RuntimeError("private failure text")

        app, ws, _ = build_app(chat=fail_chat)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            await open_call(client)
            await ws.feed({"type": "response.created", "response": {"id": "resp-tool"}})
            await ws.feed(
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp-tool",
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "name": TOOL_NAME,
                                "call_id": "tool-1",
                                "arguments": '{"message":"hello"}',
                            }
                        ],
                    },
                }
            )
            assert await wait_for(
                lambda: any(e.get("type") == "conversation.item.create" for e in ws.sent)
            )
            outputs = [e for e in ws.sent if e.get("type") == "conversation.item.create"]
            output = next(e for e in outputs if e["item"].get("type") == "function_call_output")
            assert "couldn't reach" in output["item"]["output"]
            assert any(
                e["item"].get("role") == "assistant"
                and "couldn't reach" in e["item"]["content"][0]["text"]
                for e in outputs
            )
            assert not any(e.get("type") == "response.create" for e in ws.sent)
            await client.delete("/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"})

    asyncio.run(run())


def test_delete_is_owner_checked_and_session_limit_is_bounded():
    async def run():
        app, _ws, requests = build_app(max_sessions=1)
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://astro") as client:
            assert (await open_call(client)).status_code == 200
            second = await open_call(client)
            assert second.status_code == 429
            foreign = await client.delete(
                "/api/realtime/call/rtc-fixture",
                headers={"x-test-auth": "ok", "x-owner-id": "other"},
            )
            assert foreign.status_code == 404
            assert len(requests) == 1
            ended = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert ended.status_code == 200

    asyncio.run(run())


def test_cancel_already_completed_response_does_not_end_conversation():
    async def run():
        app, ws, requests = build_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            await ws.feed({"type": "response.created", "response": {"id": "old-response"}})
            await ws.feed({"type": "input_audio_buffer.speech_started"})
            assert await wait_for(lambda: any(e.get("type") == "response.cancel" for e in ws.sent))
            # Server VAD can already have interrupted the response before the
            # server's additional cancellation reaches the provider.
            await ws.feed({"type": "error", "error": {"code": "response_cancel_not_active"}})
            await ws.feed({"type": "input_audio_buffer.speech_started"})
            await ws.feed(
                {
                    "type": "response.done",
                    "response": {
                        "id": "new-response",
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "name": TOOL_NAME,
                                "call_id": "next-turn",
                                "arguments": json.dumps({"message": "What does the moon mean?"}),
                            }
                        ],
                    },
                }
            )
            assert await wait_for(lambda: any(e.get("type") == "response.create" for e in ws.sent))
            assert not ws.closed
            assert not any(r.url.path.endswith("/hangup") for r in requests)
            assert (
                await client.delete("/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"})
            ).status_code == 200

    asyncio.run(run())


def test_unknown_provider_error_still_terminates_conversation():
    async def run():
        app, ws, requests = build_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            await ws.feed({"type": "error", "error": {"code": "unexpected_error"}})
            assert await wait_for(lambda: ws.closed)
            assert any(r.url.path.endswith("/hangup") for r in requests)

    asyncio.run(run())


def test_provider_confirmed_missing_call_is_an_idempotent_success():
    async def run():
        app, _ws, requests = build_app(hangup_statuses=[(404, "call_id_not_found")])
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            response = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert response.status_code == 200
            assert response.json() == {"ended": True}
            assert len([r for r in requests if r.url.path.endswith("/hangup")]) == 1

    asyncio.run(run())


def test_unknown_provider_not_found_remains_retryable():
    async def run():
        app, _ws, requests = build_app(
            hangup_statuses=[(404, "unknown_code"), (404, "unknown_code"), 200]
        )
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            response = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert response.status_code == 503
            retry = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert retry.status_code == 200
            assert len([r for r in requests if r.url.path.endswith("/hangup")]) == 3

    asyncio.run(run())


def test_failed_monitor_hangup_remains_owner_retryable_until_provider_confirms():
    async def run():
        app, ws, requests = build_app(hangup_statuses=[503, 503, 204])
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            changed = session_config()
            changed["tool_choice"] = "none"
            await ws.feed({"type": "session.updated", "session": changed})
            assert await wait_for(lambda: ws.closed)
            # Monitor cleanup's failed DELETE remains addressable to its owner
            # and reserves capacity until a confirmed hangup.
            failed = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert failed.status_code == 503
            assert len([r for r in requests if r.url.path.endswith("/hangup")]) == 2
            retry = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert retry.status_code == 200
            assert retry.json() == {"ended": True}
            assert len([r for r in requests if r.url.path.endswith("/hangup")]) == 3
            assert all(
                request.headers.get("authorization") == "Bearer secret-test-key"
                for request in requests
                if request.url.path.endswith("/hangup")
            )
            gone = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert gone.status_code == 200

    asyncio.run(run())


def test_session_tool_is_single_required_tool_and_location_parser_is_strict():
    config = session_config()
    assert config["tool_choice"] == "required"
    assert [tool["name"] for tool in config["tools"]] == [TOOL_NAME]
    assert config["output_modalities"] == ["text"]
    assert config["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert config["audio"]["input"]["format"]["rate"] == 24000
    assert call_id_from_location("https://provider.test/v1/realtime/calls/abc-123") == "abc-123"
    with pytest.raises(Exception, match="invalid call location"):
        call_id_from_location("https://attacker.test/redirect")


def test_session_routes_every_topic_to_mynaksh_without_astrology_restriction():
    config = session_config()
    instructions = config["instructions"].lower()
    assert "every user turn" in instructions
    assert "whatever its topic" in instructions
    assert "ask_memory" in instructions
    assert "do not translate" in instructions
    assert "hinglish" in instructions
    assert "astrolog" not in instructions
    assert "astrolog" not in config["tools"][0]["description"].lower()
    assert config["tool_choice"] == "required"


def test_more_than_twenty_provider_responses_do_not_force_end_before_user_does():
    async def run():
        completed = []
        app, ws, _ = build_app(
            record_usage=lambda _call_id, event: completed.append(event["response"]["id"])
        )
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            for index in range(21):
                response_id = f"resp-{index}"
                await ws.feed({"type": "response.created", "response": {"id": response_id}})
                await ws.feed(
                    {
                        "type": "response.done",
                        "event_id": f"done-{index}",
                        "response": {"id": response_id, "status": "cancelled", "output": []},
                    }
                )
            assert await wait_for(lambda: len(completed) == 21)
            assert completed == [f"resp-{index}" for index in range(21)]
            assert not ws.closed
            ended = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert ended.status_code == 200

    asyncio.run(run())


def test_usage_events_are_forwarded_once_by_provider_event_id():
    async def run():
        usage = []
        app, ws, _ = build_app(
            record_usage=lambda call_id, event: usage.append((call_id, event["event_id"]))
        )
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            event = {
                "type": "conversation.item.input_audio_transcription.completed",
                "event_id": "usage-1",
                "item_id": "input-1",
                "transcript": "hello",
            }
            await ws.feed(event)
            await ws.feed(event)
            assert await wait_for(lambda: usage == [("rtc-fixture", "usage-1")])
            await client.delete("/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"})

    asyncio.run(run())


def test_policy_drift_terminates_call_and_hangs_up():
    async def run():
        app, ws, requests = build_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            changed = session_config()
            changed["tool_choice"] = "none"
            await ws.feed({"type": "session.updated", "session": changed})
            assert await wait_for(lambda: len(requests) >= 2)
            assert ws.closed
            gone = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert gone.status_code == 200

    asyncio.run(run())


def test_datachannel_response_override_is_detected_and_terminated():
    async def run():
        app, ws, requests = build_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            await ws.feed(
                {
                    "type": "response.created",
                    "response": {"id": "untrusted-response", "tool_choice": "none"},
                }
            )
            assert await wait_for(lambda: len(requests) >= 2)
            assert ws.closed

    asyncio.run(run())


def test_missing_usage_event_id_fails_closed_when_usage_accounting_is_enabled():
    async def run():
        app, ws, requests = build_app(record_usage=lambda *_: None)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            await ws.feed(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "input-1",
                    "transcript": "hello",
                }
            )
            assert await wait_for(lambda: len(requests) >= 2)
            assert ws.closed

    asyncio.run(run())


def test_signaling_without_peer_ready_is_hung_up_and_ready_extends_lifecycle():
    async def run():
        app, _ws, requests = build_app(ready_timeout_seconds=0.1)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            assert await wait_for(lambda: len(requests) >= 2)
            assert requests[-1].url.path.endswith("/rtc-fixture/hangup")

        app, ws, requests = build_app(ready_timeout_seconds=0.1)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            ready = await client.post(
                "/api/realtime/call/rtc-fixture/ready", headers={"x-test-auth": "ok"}
            )
            assert ready.status_code == 200
            await asyncio.sleep(0.15)
            assert len(requests) == 1
            assert not ws.closed
            ended = await client.delete(
                "/api/realtime/call/rtc-fixture", headers={"x-test-auth": "ok"}
            )
            assert ended.status_code == 200

    asyncio.run(run())


def test_auth_revocation_closes_established_call():
    async def run():
        active = True
        app, ws, requests = build_app(
            identity_is_active=lambda _owner: active,
            identity_check_interval_seconds=0.05,
            ready_timeout_seconds=1,
        )
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://astro"
        ) as client:
            assert (await open_call(client)).status_code == 200
            ready = await client.post(
                "/api/realtime/call/rtc-fixture/ready", headers={"x-test-auth": "ok"}
            )
            assert ready.status_code == 200
            active = False
            assert await wait_for(lambda: len(requests) >= 2)
            assert ws.closed

    asyncio.run(run())
