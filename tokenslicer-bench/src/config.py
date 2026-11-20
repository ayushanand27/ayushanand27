"""Typed configuration loaded from environment variables / `.env`."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Built-in defaults. Prices are USD per 1M tokens and WILL drift: verify on the provider's pricing page
# and override with WORKER_PRICE_* in `.env` for any serious run.
PROVIDER_DEFAULTS: dict[str, dict[str, object]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "price_in_miss": 0.28,
        "price_in_hit": 0.028,
        "price_out": 0.42,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-chat",
        "price_in_miss": 0.30,
        "price_in_hit": None,
        "price_out": 1.20,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "price_in_miss": 0.59,
        "price_in_hit": None,
        "price_out": 0.79,
    },
    "custom": {
        "base_url": None,
        "model": None,
        "price_in_miss": None,
        "price_in_hit": None,
        "price_out": None,
    },
}


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or inconsistent."""


class ProviderConfig(BaseModel):
    """Everything needed to talk to one OpenAI-compatible endpoint and price its usage."""

    model_config = ConfigDict(frozen=True)

    name: str
    base_url: str
    api_key: str
    model: str
    price_in_miss: float  # USD / 1M uncached input tokens
    price_in_hit: float  # USD / 1M cached input tokens (== miss price when no cache discount)
    price_out: float  # USD / 1M output tokens


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Worker
    worker_provider: Literal["deepseek", "openrouter", "groq", "custom"] = "deepseek"
    deepseek_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    worker_api_key: SecretStr | None = None
    worker_base_url: str | None = None
    worker_model: str | None = None
    worker_price_input_miss: float | None = None
    worker_price_input_hit: float | None = None
    worker_price_output: float | None = None

    # Pipeline A
    pipeline_a_mode: Literal["simulate", "live"] = "simulate"
    frontier_price_input: float = 3.00
    frontier_price_output: float = 15.00
    frontier_base_url: str | None = None
    frontier_api_key: SecretStr | None = None
    frontier_model: str | None = None

    # Currency / limits
    usd_to_inr: float = 88.0
    max_output_tokens: int = 4096
    request_timeout_sec: float = 180.0
    max_retries: int = 6
    raw_context_token_cap: int = 110_000
    git_apply_recount: bool = False

    def worker_config(self) -> ProviderConfig:
        provider = self.worker_provider
        defaults = PROVIDER_DEFAULTS[provider]
        key_by_provider = {
            "deepseek": self.deepseek_api_key,
            "openrouter": self.openrouter_api_key,
            "groq": self.groq_api_key,
            "custom": self.worker_api_key,
        }
        secret = key_by_provider[provider]
        if secret is None or not secret.get_secret_value().strip():
            raise ConfigError(
                f"No API key for WORKER_PROVIDER={provider!r}. Set "
                f"{provider.upper() if provider != 'custom' else 'WORKER'}_API_KEY in .env."
            )

        base_url = self.worker_base_url or defaults["base_url"]
        model = self.worker_model or defaults["model"]
        price_miss = (
            self.worker_price_input_miss
            if self.worker_price_input_miss is not None
            else defaults["price_in_miss"]
        )
        price_out = (
            self.worker_price_output if self.worker_price_output is not None else defaults["price_out"]
        )
        price_hit = (
            self.worker_price_input_hit
            if self.worker_price_input_hit is not None
            else defaults["price_in_hit"]
        )
        missing = [
            label
            for label, value in (
                ("WORKER_BASE_URL", base_url),
                ("WORKER_MODEL", model),
                ("WORKER_PRICE_INPUT_MISS", price_miss),
                ("WORKER_PRICE_OUTPUT", price_out),
            )
            if value is None
        ]
        if missing:
            raise ConfigError(f"WORKER_PROVIDER={provider!r} requires: {', '.join(missing)}")

        return ProviderConfig(
            name=provider,
            base_url=str(base_url),
            api_key=secret.get_secret_value().strip(),
            model=str(model),
            price_in_miss=float(price_miss),  # type: ignore[arg-type]
            price_in_hit=float(price_hit if price_hit is not None else price_miss),  # type: ignore[arg-type]
            price_out=float(price_out),  # type: ignore[arg-type]
        )

    def frontier_config(self) -> ProviderConfig:
        """Only needed for PIPELINE_A_MODE=live."""
        missing = [
            label
            for label, value in (
                ("FRONTIER_BASE_URL", self.frontier_base_url),
                ("FRONTIER_API_KEY", self.frontier_api_key),
                ("FRONTIER_MODEL", self.frontier_model),
            )
            if not value
        ]
        if missing:
            raise ConfigError(f"PIPELINE_A_MODE=live requires: {', '.join(missing)}")
        assert self.frontier_api_key is not None and self.frontier_base_url and self.frontier_model
        return ProviderConfig(
            name="frontier",
            base_url=self.frontier_base_url,
            api_key=self.frontier_api_key.get_secret_value().strip(),
            model=self.frontier_model,
            price_in_miss=self.frontier_price_input,
            price_in_hit=self.frontier_price_input,
            price_out=self.frontier_price_output,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
