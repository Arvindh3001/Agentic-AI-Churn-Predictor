"""
Pydantic Settings — centralised configuration for the Churn Intelligence Platform.
All values are loaded from environment variables / .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    database_url: str = Field(
        default="postgresql://churn_user:churn_pass@localhost:5432/churn_db",
        description="PostgreSQL connection string",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )


class ChromaDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    chromadb_host: str = Field(default="localhost")
    chromadb_port: int = Field(default=8001)

    @property
    def chromadb_url(self) -> str:
        return f"http://{self.chromadb_host}:{self.chromadb_port}"


class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    mlflow_tracking_uri: str = Field(default="http://localhost:5001")
    mlflow_experiment_name: str = Field(default="churn-prediction")
    mlflow_model_name: str = Field(default="churn-ensemble")


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    openai_api_key: str = Field(default="")
    google_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    default_llm_provider: Literal["openai", "anthropic", "google"] = Field(default="openai")
    default_llm_model: str = Field(default="gpt-4o")

    # LangSmith
    langchain_tracing_v2: bool = Field(default=False)
    langchain_api_key: str = Field(default="")
    langchain_project: str = Field(default="churn-platform-dev")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")


class JWTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    jwt_secret_key: str = Field(default="change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=60)


class SlackSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    slack_webhook_url: str = Field(default="")
    slack_channel: str = Field(default="#churn-alerts")


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    raw_data_path: Path = Field(default=BASE_DIR / "data" / "raw")
    processed_data_path: Path = Field(default=BASE_DIR / "data" / "processed")
    synthetic_data_path: Path = Field(default=BASE_DIR / "data" / "synthetic")
    feast_repo_path: Path = Field(default=BASE_DIR / "data" / "feature_store")

    @field_validator("raw_data_path", "processed_data_path", "synthetic_data_path", mode="before")
    @classmethod
    def make_path(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    churn_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_interval_alpha: float = Field(default=0.1, ge=0.0, le=1.0)


class AppSettings(BaseSettings):
    """Root settings object — compose all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
        env_nested_delimiter="__",
    )

    app_env: Literal["development", "staging", "production"] = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    workers: int = Field(default=4, ge=1)

    # Nested
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    chromadb: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    slack: SlackSettings = Field(default_factory=SlackSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


# Singleton — import and use this everywhere
settings = AppSettings()

if __name__ == "__main__":
    import json

    print(json.dumps(settings.model_dump(mode="json"), indent=2))
    logger.info("Settings loaded successfully", env=settings.app_env)
