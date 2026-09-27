"""Typed, validated, hashable strategy Genome."""
from __future__ import annotations

import hashlib
import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from ..features.registry import (
    FeatureSpec,
    available_features,
    feature_defaults,
    feature_lookback,
)

Operator = Literal["<", "<=", ">", ">=", "cross_above", "cross_below"]
Operand = Literal["value", "feature", "close"]

# Bounded parameter ranges enforced at validation time.
PARAM_BOUNDS: dict[str, tuple[int, int]] = {"window": (2, 2000)}


class FeatureRef(BaseModel):
    model_config = {"frozen": True}
    name: str
    params: dict[str, float] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in available_features():
            raise ValueError(f"Unknown feature {v!r}; available: {available_features()}")
        return v

    @model_validator(mode="after")
    def _bounds(self):
        for k, v in self.params.items():
            if k in PARAM_BOUNDS:
                lo, hi = PARAM_BOUNDS[k]
                if not (lo <= v <= hi):
                    raise ValueError(f"{self.name}.{k}={v} outside bounds [{lo},{hi}]")
        return self

    def to_spec(self) -> FeatureSpec:
        merged = {**feature_defaults(self.name), **self.params}
        return FeatureSpec(self.name, merged)

    @property
    def key(self) -> str:
        return self.to_spec().key()

    @property
    def lookback(self) -> int:
        return feature_lookback(self.to_spec())


class Condition(BaseModel):
    model_config = {"frozen": True}
    left: FeatureRef
    op: Operator
    right_kind: Operand = "value"
    right_value: float = 0.0
    right_feature: Optional[FeatureRef] = None

    @model_validator(mode="after")
    def _consistent(self):
        if self.right_kind == "feature" and self.right_feature is None:
            raise ValueError("right_kind='feature' requires right_feature")
        if self.right_kind != "feature" and self.right_feature is not None:
            raise ValueError("right_feature only valid with right_kind='feature'")
        return self

    def features(self) -> list[FeatureRef]:
        out = [self.left]
        if self.right_feature is not None:
            out.append(self.right_feature)
        return out


class RiskParams(BaseModel):
    model_config = {"frozen": True}
    stop_atr_mult: float = Field(2.0, ge=0.1, le=20.0)
    take_profit_atr_mult: float = Field(3.0, ge=0.1, le=50.0)
    atr_window: int = Field(14, ge=2, le=500)
    position_fraction: float = Field(1.0, gt=0.0, le=1.0)


class Genome(BaseModel):
    """A complete, bounded, serializable strategy definition."""

    model_config = {"frozen": True}

    side: Literal["long", "short", "both"] = "long"
    entry: list[Condition] = Field(min_length=1)
    exit: list[Condition] = Field(default_factory=list)
    filters: list[Condition] = Field(default_factory=list)
    risk: RiskParams = Field(default_factory=RiskParams)
    max_holding_bars: Optional[int] = Field(default=None, ge=1, le=100_000)
    label: str = "unnamed"

    def features(self) -> list[FeatureRef]:
        seen: dict[str, FeatureRef] = {}
        for cond in [*self.entry, *self.exit, *self.filters]:
            for f in cond.features():
                seen.setdefault(f.key, f)
        atr = FeatureRef(name="atr", params={"window": self.risk.atr_window})
        seen.setdefault(atr.key, atr)
        return [seen[k] for k in sorted(seen)]

    def signal_features(self) -> list[FeatureRef]:
        """Features that participate in signal logic (ATR excluded unless used)."""
        seen: dict[str, FeatureRef] = {}
        for cond in [*self.entry, *self.exit, *self.filters]:
            for f in cond.features():
                seen.setdefault(f.key, f)
        return [seen[k] for k in sorted(seen)]

    def max_lookback(self) -> int:
        """L, in canonical 1m bars.

        Deliberately computed from *signal* features so that changing the ATR
        multiplier (or ATR window used only for stop distance) cannot alter L
        for signal purposes. ATR warmup is handled by the backtester itself.
        """
        return max((f.lookback for f in self.signal_features()), default=0)

    def holding_horizon(self, default_max_holding_bars: int) -> int:
        """H from the Genome when explicit, otherwise from config."""
        if self.max_holding_bars is not None:
            return int(self.max_holding_bars)
        return int(default_max_holding_bars)

    def canonical_payload(self) -> dict:
        return json.loads(self.model_dump_json(exclude={"label"}))

    def genome_hash(self) -> str:
        blob = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def genome_id(self) -> str:
        return self.genome_hash()[:16]

    def feature_hash(self) -> str:
        blob = json.dumps(sorted(f.key for f in self.features()), separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_json(self) -> str:
        return self.model_dump_json()

    @staticmethod
    def from_json(text: str) -> "Genome":
        return Genome.model_validate_json(text)
