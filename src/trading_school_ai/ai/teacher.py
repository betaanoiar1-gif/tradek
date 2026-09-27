"""The LLM Teacher: proposes research hypotheses, never trading decisions."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config.settings import AIConfig
from .provider import NullProvider, Provider, ProviderError, build_provider, mask
from .schema import Hypothesis, HypothesisRejected, validate_response

SYSTEM_PROMPT = (
    "You are a quantitative research teacher for an automated trading research "
    "system. You receive only compact statistical summaries. You must reply with "
    "a single JSON object containing: concept, hypothesis, expected_effect, "
    "proposed_experiment, genome_template, confidence, rationale. "
    "You must NEVER issue buy/sell decisions, order sizes, order prices, leverage "
    "or any executable instruction. Every suggestion is a research hypothesis that "
    "will be tested by a Python experiment engine before it means anything."
)


@dataclass
class TokenGovernor:
    budget: int
    used: int = 0

    def allows(self, estimate: int = 0) -> bool:
        return (self.used + estimate) <= self.budget

    def charge(self, tokens: int) -> None:
        self.used += max(0, int(tokens))

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.used)


@dataclass
class TeacherStatus:
    enabled: bool
    provider: str
    key_present: bool
    masked_key: str
    tokens_used: int
    tokens_remaining: int
    calls: int
    cache_hits: int
    last_error: str = ""


class Teacher:
    """Event-driven only. Failure here never stops Python-only learning."""

    def __init__(self, cfg: AIConfig, cache_dir: Optional[str | Path] = None,
                 provider: Optional[Provider] = None):
        self.cfg = cfg
        self._init_error = ""
        if provider is not None:
            self.provider = provider
        else:
            try:
                self.provider = build_provider(cfg)
            except ProviderError as exc:
                # Credentials/transport problems must never crash the run:
                # Python-only learning continues unaffected.
                self._init_error = f"PROVIDER_UNAVAILABLE: {exc}"
                self.provider = NullProvider()
        self.governor = TokenGovernor(cfg.token_budget)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        self._last_call_ts: float = 0.0
        self.calls = 0
        self.cache_hits = 0
        self.last_error = ""

    # ------------------------------------------------------------------ util
    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def status(self) -> TeacherStatus:
        import os

        key = os.environ.get(self.cfg.api_key_env)
        return TeacherStatus(
            enabled=self.enabled, provider=self.cfg.provider,
            key_present=bool(key), masked_key=mask(key),
            tokens_used=self.governor.used, tokens_remaining=self.governor.remaining,
            calls=self.calls, cache_hits=self.cache_hits, last_error=self.last_error,
        )

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha256((self.cfg.model + "|" + prompt).encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        return p.read_text() if p.exists() else None

    def _cache_put(self, key: str, text: str) -> None:
        if self.cache_dir:
            (self.cache_dir / f"{key}.json").write_text(text)

    def _cooldown_ok(self) -> bool:
        return (time.monotonic() - self._last_call_ts) >= self.cfg.cooldown_seconds

    # ------------------------------------------------------------- main call
    def propose(self, summary: dict, force: bool = False) -> Optional[Hypothesis]:
        """Request one hypothesis. Returns None on any non-fatal failure."""
        if not self.enabled:
            self.last_error = "AI_OFF"
            return None
        if self._init_error:
            self.last_error = self._init_error
            return None
        prompt = json.dumps(summary, sort_keys=True, default=str)
        key = self._cache_key(prompt)
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            try:
                hyp = validate_response(json.loads(cached))
            except (HypothesisRejected, json.JSONDecodeError) as exc:
                self.last_error = f"CACHED_INVALID: {exc}"
                return None
        else:
            if not force and not self._cooldown_ok():
                self.last_error = "COOLDOWN"
                return None
            if not self.governor.allows(1000):
                self.last_error = "TOKEN_BUDGET_EXHAUSTED"
                return None
            try:
                text, tokens = self.provider.complete(SYSTEM_PROMPT, prompt)
            except ProviderError as exc:
                self.last_error = f"PROVIDER_ERROR: {exc}"
                return None
            self.calls += 1
            self._last_call_ts = time.monotonic()
            self.governor.charge(tokens)
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                self.last_error = f"INVALID_JSON: {exc}"
                return None
            try:
                hyp = validate_response(payload)
            except HypothesisRejected as exc:
                self.last_error = f"SCHEMA_REJECTED: {exc}"
                return None
            self._cache_put(key, text)
        hid = hyp.hypothesis_id()
        if hid in self._seen:
            self.last_error = "DUPLICATE"
            return None
        self._seen.add(hid)
        self.last_error = ""
        return hyp
