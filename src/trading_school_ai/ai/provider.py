"""OpenAI-compatible provider abstraction. API keys come from env vars only."""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from ..config.settings import AIConfig


class ProviderError(Exception):
    """Transport / provider failure. Never fatal for Python-only learning."""


def mask(secret: Optional[str]) -> str:
    if not secret:
        return "<unset>"
    return f"{secret[:3]}***{secret[-2:]}" if len(secret) > 6 else "***"


class Provider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> tuple[str, int]:
        """Return (raw_text, tokens_used)."""


class NullProvider(Provider):
    """AI_OFF. Calling it is always an explicit, loud failure - never a fake answer."""

    def complete(self, system: str, user: str) -> tuple[str, int]:
        raise ProviderError(
            "AI is disabled (AI_OFF). No LLM call was made and no response is fabricated."
        )


class OpenAICompatibleProvider(Provider):
    def __init__(self, cfg: AIConfig):
        self.cfg = cfg
        self.api_key = os.environ.get(cfg.api_key_env)
        if not self.api_key:
            raise ProviderError(
                f"Missing API key: set the {cfg.api_key_env} environment variable."
            )

    def complete(self, system: str, user: str) -> tuple[str, int]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderError(
                "httpx is required for the LLM adapter: pip install 'trading-school-ai[ai]'"
            ) from exc
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = httpx.post(
                    f"{self.cfg.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload, timeout=self.cfg.timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                tokens = int(data.get("usage", {}).get("total_tokens", 0))
                return text, tokens
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        raise ProviderError(f"LLM request failed after retries: {last}")


def build_provider(cfg: AIConfig) -> Provider:
    if not cfg.enabled or cfg.provider == "null":
        return NullProvider()
    return OpenAICompatibleProvider(cfg)
