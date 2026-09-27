"""Validated hypothesis schema. Research suggestions only - never orders."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from ..strategies.genome import Genome

# Any attempt to issue an executable instruction is rejected outright.
FORBIDDEN_PATTERNS = [
    r"\bbuy\s+now\b", r"\bsell\s+now\b", r"\bplace\s+(an?\s+)?order\b",
    r"\bmarket\s+order\b", r"\blimit\s+order\b", r"\bexecute\s+(a\s+)?trade\b",
    r"\border_size\b", r"\bposition_size\s*[:=]", r"\bleverage\s*[:=]\s*\d",
    r"\bsend\s+to\s+exchange\b", r"\bskip\s+backtest\b", r"\bbypass\b",
]
FORBIDDEN_KEYS = {
    "order", "orders", "order_size", "order_price", "quantity", "qty", "size",
    "price", "side_to_execute", "execute", "live", "api_key", "leverage",
}


class HypothesisRejected(Exception):
    """The teacher response violated the research-only contract."""


class Hypothesis(BaseModel):
    concept: str = Field(min_length=3, max_length=200)
    hypothesis: str = Field(min_length=10, max_length=2000)
    expected_effect: str = Field(min_length=3, max_length=1000)
    proposed_experiment: str = Field(min_length=3, max_length=2000)
    genome_template: dict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("concept", "hypothesis", "expected_effect", "proposed_experiment")
    @classmethod
    def _no_commands(cls, v: str) -> str:
        low = v.lower()
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat, low):
                raise ValueError(f"Trading-instruction pattern rejected: {pat}")
        return v

    @field_validator("genome_template")
    @classmethod
    def _no_execution_keys(cls, v: dict) -> dict:
        def walk(node):
            if isinstance(node, dict):
                for k, sub in node.items():
                    if str(k).lower() in FORBIDDEN_KEYS:
                        raise ValueError(f"Forbidden execution key in template: {k}")
                    walk(sub)
            elif isinstance(node, list):
                for sub in node:
                    walk(sub)

        walk(v)
        return v

    def hypothesis_id(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:20]

    def to_genome(self) -> Genome:
        """Convert to a candidate Genome. Invalid templates raise, never fall back."""
        try:
            return Genome(**{**self.genome_template, "label": f"ai-{self.hypothesis_id()[:8]}"})
        except Exception as exc:  # noqa: BLE001
            raise HypothesisRejected(f"genome_template is not a valid Genome: {exc}") from exc


def validate_response(payload: dict) -> Hypothesis:
    try:
        return Hypothesis(**payload)
    except Exception as exc:  # noqa: BLE001
        raise HypothesisRejected(str(exc)) from exc
