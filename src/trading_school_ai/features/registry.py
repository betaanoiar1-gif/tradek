"""Causal feature engine.

Contracts:
  * Every feature is computed from information available up to and including bar i.
  * No feature applies an execution shift. The execution layer owns timing.
  * Feature windows crossing a declared gap are NaN until enough valid
    observations have accumulated inside the new contiguous segment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd

from ..data.dataset import CanonicalDataset


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    params: dict

    def key(self) -> str:
        ps = ",".join(f"{k}={self.params[k]}" for k in sorted(self.params))
        return f"{self.name}({ps})"


FeatureFn = Callable[[pd.DataFrame, dict], pd.Series]
_REGISTRY: Dict[str, dict] = {}


def register(name: str, lookback: Callable[[dict], int], defaults: dict):
    def deco(fn: FeatureFn):
        _REGISTRY[name] = {"fn": fn, "lookback": lookback, "defaults": defaults}
        return fn

    return deco


def available_features() -> list[str]:
    return sorted(_REGISTRY)


def feature_defaults(name: str) -> dict:
    return dict(_REGISTRY[_check(name)]["defaults"])


def _check(name: str) -> str:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown feature {name!r}; available: {available_features()}")
    return name


def feature_lookback(spec: FeatureSpec) -> int:
    meta = _REGISTRY[_check(spec.name)]
    params = {**meta["defaults"], **spec.params}
    return int(meta["lookback"](params))


# --------------------------------------------------------------------------
# Feature implementations (all causal, all in canonical 1m bars)
# --------------------------------------------------------------------------
@register("sma", lambda p: int(p["window"]), {"window": 20})
def _sma(df: pd.DataFrame, p: dict) -> pd.Series:
    w = int(p["window"])
    return df["close"].rolling(w, min_periods=w).mean()


@register("ema", lambda p: int(p["window"]) * 3, {"window": 20})
def _ema(df: pd.DataFrame, p: dict) -> pd.Series:
    w = int(p["window"])
    out = df["close"].ewm(span=w, adjust=False, min_periods=w).mean()
    return out


@register("rsi", lambda p: int(p["window"]) + 1, {"window": 14})
def _rsi(df: pd.DataFrame, p: dict) -> pd.Series:
    w = int(p["window"])
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.rolling(w, min_periods=w).mean()
    al = loss.rolling(w, min_periods=w).mean()
    rs = ag / al.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[(al == 0) & (ag > 0)] = 100.0
    rsi[(al == 0) & (ag == 0)] = 50.0
    return rsi


@register("atr", lambda p: int(p["window"]) + 1, {"window": 14})
def _atr(df: pd.DataFrame, p: dict) -> pd.Series:
    w = int(p["window"])
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(w, min_periods=w).mean()


@register("roc", lambda p: int(p["window"]) + 1, {"window": 30})
def _roc(df: pd.DataFrame, p: dict) -> pd.Series:
    w = int(p["window"])
    return df["close"].pct_change(w)


@register("zscore", lambda p: int(p["window"]), {"window": 60})
def _zscore(df: pd.DataFrame, p: dict) -> pd.Series:
    w = int(p["window"])
    m = df["close"].rolling(w, min_periods=w).mean()
    s = df["close"].rolling(w, min_periods=w).std(ddof=0)
    return (df["close"] - m) / s.replace(0.0, np.nan)


@register("vol", lambda p: int(p["window"]) + 1, {"window": 60})
def _vol(df: pd.DataFrame, p: dict) -> pd.Series:
    w = int(p["window"])
    return df["close"].pct_change().rolling(w, min_periods=w).std(ddof=0)


def compute_feature(dataset: CanonicalDataset, spec: FeatureSpec) -> pd.Series:
    """Compute a single feature with gap-aware invalidation."""
    meta = _REGISTRY[_check(spec.name)]
    params = {**meta["defaults"], **spec.params}
    lookback = int(meta["lookback"](params))
    df = dataset.frame
    segments = dataset.gaps.segment_ids(df.index)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for seg in np.unique(segments):
        mask = segments == seg
        sub = df.loc[mask]
        if len(sub) == 0:
            continue
        values = meta["fn"](sub, params)
        # enforce the lookback warmup inside the segment (gap = hard boundary)
        values = values.copy()
        values.iloc[: min(lookback - 1, len(values))] = np.nan
        out.loc[mask] = values.to_numpy()
    out.name = spec.key()
    return out


def compute_features(
    dataset: CanonicalDataset, specs: list[FeatureSpec]
) -> pd.DataFrame:
    if not specs:
        return pd.DataFrame(index=dataset.frame.index)
    cols = {}
    for spec in specs:
        cols[spec.key()] = compute_feature(dataset, spec)
    return pd.DataFrame(cols, index=dataset.frame.index)


def max_lookback(specs: list[FeatureSpec]) -> int:
    """L = max(feature.lookback) in canonical 1m bars (0 when no features)."""
    return max((feature_lookback(s) for s in specs), default=0)
