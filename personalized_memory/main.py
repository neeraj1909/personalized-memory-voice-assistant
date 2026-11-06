"""FastAPI boundary for personalized chat."""

from personalized_memory.compat import getenv

import logging
import secrets
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from .chat import (
    CONTEXT_INSTRUCTIONS,
    REPLY_INSTRUCTIONS,
    ChatEngine,
    ContextDecision,
    EmptyReplyError,
    PromptBudgetError,
)
from .cache import MemoryCacheError
from .config import Settings, get_settings
from .fast_chat import FAST_INSTRUCTIONS, FastChatEngine, FastDecision
from .observability import configure_tracing, shutdown_tracing, trace_step
from .storage import ChatStore

logger = logging.getLogger("personalized_memory_chat.api")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1)

    @field_validator("user_id", "session_id", "message")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ChatResponse(BaseModel):
    request_id: str
    message_id: int
    response_id: int
    user_id: str
    session_id: str
    response: str
    context_used: list[str]


class WarmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)

    @field_validator("user_id", "session_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


def create_app(
    *,
    settings: Settings | None = None,
    context_model: Model | None = None,
    reply_model: Model | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        store = ChatStore(runtime_settings.database_path)
        store.initialize()
        provider = configure_tracing(runtime_settings.otel_exporter_otlp_endpoint)
        sdk_client: AsyncOpenAI | None = None
        application.state.chat_engine = None
        try:
            if context_model is None or reply_model is None:
                if (
                    runtime_settings.llm_provider.strip().casefold() == "openai-compatible"
                    and runtime_settings.llm_base_url
                    and runtime_settings.llm_api_key
                ):
                    sdk_client = AsyncOpenAI(
                        base_url=runtime_settings.llm_base_url.strip().rstrip("/"),
                        api_key=runtime_settings.llm_api_key.get_secret_value(),
                        max_retries=0,
                    )
                    openai_provider = OpenAIProvider(openai_client=sdk_client)
                    configured_context = OpenAIChatModel(
                        runtime_settings.language_model or runtime_settings.llm_model,
                        provider=openai_provider,
                    )
                    configured_reply = OpenAIChatModel(
                        runtime_settings.llm_model, provider=openai_provider
                    )
                else:
                    configured_context = configured_reply = None
            else:
                configured_context = configured_reply = None

            decision_model = context_model or configured_context
            answer_model = reply_model or configured_reply
            if decision_model is not None and answer_model is not None:
                model_settings = ModelSettings(timeout=runtime_settings.llm_timeout_seconds)
                context_agent = Agent(
                    decision_model,
                    name="context_decision",
                    output_type=ContextDecision,
                    instructions=CONTEXT_INSTRUCTIONS,
                    model_settings=model_settings,
                    retries=0,
                )
                reply_agent = Agent(
                    answer_model,
                    name="chat_reply",
                    instructions=REPLY_INSTRUCTIONS,
                    model_settings=model_settings,
                    retries=0,
                )
                application.state.context_agent = context_agent
                application.state.reply_agent = reply_agent
                if getenv("PMVA_MEMORY_FAST") == "1":
                    fast_agent = Agent(
                        answer_model,
                        name="fast_memory_chat",
                        output_type=FastDecision,
                        instructions=FAST_INSTRUCTIONS,
                        model_settings=model_settings,
                        retries=0,
                    )
                    application.state.chat_engine = FastChatEngine(
                        store, fast_agent, runtime_settings
                    )
                else:
                    application.state.chat_engine = ChatEngine(
                        store, context_agent, reply_agent, runtime_settings
                    )
            yield
        finally:
            try:
                if sdk_client is not None:
                    await sdk_client.close()
            finally:
                shutdown_tracing(provider)

    application = FastAPI(title=runtime_settings.app_name, lifespan=lifespan)

    @application.post("/internal/memory/warm")
    async def warm_memory(payload: WarmRequest, request: Request) -> dict[str, bool]:
        token = getenv("PMVA_MEMORY_WARM_TOKEN", "")
        supplied = request.headers.get("authorization", "")
        if not token or not secrets.compare_digest(supplied, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="Unauthorized")
        engine = request.app.state.chat_engine
        if not isinstance(engine, FastChatEngine):
            raise HTTPException(status_code=503, detail="Memory preload unavailable")
        try:
            async with engine.cache.acquire(payload.user_id, payload.session_id):
                pass
        except MemoryCacheError as exc:
            raise HTTPException(status_code=503, detail="Memory preload unavailable") from exc
        return {"warmed": True}

    @application.middleware("http")
    async def request_trace(request: Request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        with trace_step("chat.request", traceparent=request.headers.get("traceparent")):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.post("/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        request: Request,
    ) -> ChatResponse:
        if len(payload.message) > runtime_settings.max_message_chars:
            raise HTTPException(status_code=422, detail="message exceeds the configured limit")
        engine: ChatEngine | None = request.app.state.chat_engine
        if engine is None:
            raise HTTPException(status_code=503, detail="LLM service is not configured")
        try:
            with trace_step("chat.service"):
                result = await engine.chat(payload.user_id, payload.session_id, payload.message)
        except PromptBudgetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MemoryCacheError as exc:
            raise HTTPException(status_code=503, detail="Memory cache unavailable") from exc
        except (
            ModelAPIError,
            UnexpectedModelBehavior,
            UsageLimitExceeded,
            APIError,
            ValidationError,
            EmptyReplyError,
        ) as exc:
            logger.warning(
                "model_request_failed request_id=%s error_type=%s",
                request.state.request_id,
                type(exc).__name__,
            )
            raise HTTPException(status_code=502, detail="LLM provider unavailable") from exc
        return ChatResponse(
            request_id=request.state.request_id,
            message_id=result.message_id,
            response_id=result.response_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
            response=result.response,
            context_used=result.context_used,
        )

    return application


app = create_app()
