"""Optional, content-free OpenTelemetry tracing for the chat service."""

from __future__ import annotations

from contextlib import AbstractContextManager
from urllib.parse import urlsplit

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NoOpTracerProvider, Span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

PROJECT_NAME = "personalized-memory-chat"
_SPAN_NAMES = frozenset(
    {
        "chat.request",
        "chat.service",
        "chat.history",
        "chat.language",
        "chat.memory.extract",
        "chat.memory.persist",
        "chat.memory.retrieve",
        "chat.main_agent",
    }
)
_noop_tracer = NoOpTracerProvider().get_tracer("personalized_memory_chat.chat")
_provider: TracerProvider | None = None


def configure_tracing(endpoint: str | None) -> TracerProvider | None:
    """Enable direct OTLP/HTTP export to a local Phoenix endpoint when configured."""

    global _provider
    if not endpoint:
        return None
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.path != "/v1/traces"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("OTLP endpoint must be an HTTP(S) /v1/traces URL without credentials")
    if _provider is not None:
        raise RuntimeError("tracing is already configured")

    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": PROJECT_NAME, "openinference.project.name": PROJECT_NAME}
        )
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint, headers={"x-project-name": PROJECT_NAME})
        )
    )
    Agent.instrument_all(
        InstrumentationSettings(
            tracer_provider=provider,
            include_content=False,
            include_binary_content=False,
        )
    )
    _provider = provider
    return provider


def trace_step(name: str, *, traceparent: str | None = None) -> AbstractContextManager[Span]:
    """Start one named span without attaching messages, identifiers, or exceptions."""

    if name not in _SPAN_NAMES:
        raise ValueError(f"unsupported trace step: {name}")
    if traceparent is not None and name != "chat.request":
        raise ValueError("traceparent is only accepted on the request span")
    tracer = (
        _provider.get_tracer("personalized_memory_chat.chat")
        if _provider is not None
        else _noop_tracer
    )
    parent_context = (
        TraceContextTextMapPropagator().extract({"traceparent": traceparent})
        if traceparent is not None
        else None
    )
    return tracer.start_as_current_span(
        name,
        context=parent_context,
        record_exception=False,
        set_status_on_exception=False,
    )


def shutdown_tracing(provider: TracerProvider | None) -> None:
    """Flush queued spans before the application exits."""

    global _provider
    if provider is None:
        return
    if provider is not _provider:
        raise ValueError("tracing provider does not belong to this application")
    Agent.instrument_all(False)
    _provider = None
    provider.shutdown()
