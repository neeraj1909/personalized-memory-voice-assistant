"""Runtime settings; credentials stay in the private .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration with safe defaults for local development and tests."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Personalized Memory Chat"
    database_path: Path = Path("memory.db")
    max_message_chars: int = Field(default=128_000, ge=1)
    prompt_max_chars: int = Field(default=256_000, ge=1000)
    memory_limit: int = Field(default=5, ge=1, le=20)
    memory_candidates: int = Field(default=40, ge=1, le=200)
    language_history_limit: int = Field(default=3, ge=0, le=20)
    language_model: str | None = None
    eval_judge_model: str | None = None
    otel_exporter_otlp_endpoint: str | None = None

    llm_provider: str = "openai-compatible"
    llm_model: str = "chatgpt/gpt-6.1-sol"
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=120)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
