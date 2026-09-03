"""
Centralized application configuration, loaded from environment variables
(with sane defaults for local dev). Using pydantic-settings means config is
validated at startup rather than failing deep inside some service later.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_ENGINE_", extra="ignore")

    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expiry_seconds: int = 3600

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    rate_limit_requests_per_minute: int = 60

    conversation_ttl_seconds: int = 3600
    conversation_max_messages: int = 50

    low_confidence_threshold: float = 0.35        # below this -> needs_clarification
    high_risk_confidence_threshold: float = 0.75   # below this -> require confirmation for write tools

    log_level: str = "INFO"


settings = Settings()
