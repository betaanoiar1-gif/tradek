"""Multi-timeframe construction from canonical 1m data only.

Rules (v1.0):
  * right-labeled, right-closed resampling
  * incomplete candles dropped (bar count must equal expected count)
  * candles overlapping a declared gap are invalidated (dropped)
  * every candle carries close_time and available_from = close_time + 1ms
  * at decision timestamp t, only candles with close_time < t may be used
"""
from __future__ import annotations

import pandas as pd

from ..data.dataset import CanonicalDataset

ONE_MS = pd.Timedelta("1ms")
SUPPORTED = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
}


def timeframe_minutes(tf: str) -> int:
    if tf not in SUPPORTED:
        raise ValueError(f"Unsupported timeframe {tf!r}; supported: {sorted(SUPPORTED)}")
    return SUPPORTED[tf]


def build_mtf(dataset: CanonicalDataset, timeframe: str) -> pd.DataFrame:
    """Return MTF OHLCV with close_time / available_from, gap-invalidated."""
    minutes = timeframe_minutes(timeframe)
    src = dataset.frame
    if minutes == 1:
        out = src.copy()
        out["bar_count"] = 1
        out["close_time"] = out.index + pd.Timedelta(minutes=1) - ONE_MS
    else:
        rule = f"{minutes}min"
        agg = src.resample(rule, label="right", closed="right", origin="epoch").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            bar_count=("close", "count"),
        )
        agg = agg.dropna(subset=["open", "close"])
        # right-labeled + right-closed: label == close_time of the bucket
        agg["close_time"] = agg.index - ONE_MS
        # incomplete candles are dropped
        agg = agg[agg["bar_count"] == minutes]
        out = agg
    out = out.copy()
    out["available_from"] = out["close_time"] + ONE_MS
    out = _invalidate_gap_candles(out, dataset, minutes)
    out.index.name = "bucket_end" if minutes > 1 else "open_time"
    return out


def _invalidate_gap_candles(
    frame: pd.DataFrame, dataset: CanonicalDataset, minutes: int
) -> pd.DataFrame:
    gaps = dataset.gaps.frame
    if gaps.empty or frame.empty:
        return frame
    candle_start = frame["close_time"] + ONE_MS - pd.Timedelta(minutes=minutes)
    candle_end = frame["close_time"] + ONE_MS
    keep = pd.Series(True, index=frame.index)
    for gs, ge in zip(gaps["gap_start"], gaps["gap_end"]):
        overlap = (candle_start < ge) & (candle_end > gs)
        keep &= ~overlap
    return frame[keep]


def visible_candles(mtf: pd.DataFrame, decision_ts: pd.Timestamp) -> pd.DataFrame:
    """Only candles whose close_time is strictly before the decision timestamp."""
    return mtf[mtf["close_time"] < decision_ts]


def align_to_base(mtf: pd.DataFrame, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill MTF values onto the 1m grid respecting available_from.

    A 1m bar at open_time o has decision timestamp o + 1min - 1ms + 1ms = o + 1min,
    but the value usable *for a decision made at o* is the last MTF candle with
    close_time < o. This function returns exactly that (no lookahead).
    """
    if mtf.empty:
        return pd.DataFrame(index=base_index, columns=mtf.columns, dtype="float64")
    cols = [c for c in mtf.columns if c not in ("close_time", "available_from")]
    pos = mtf["close_time"].to_numpy().searchsorted(base_index.to_numpy(), side="left") - 1
    out = pd.DataFrame(index=base_index, columns=cols, dtype="float64")
    valid = pos >= 0
    if valid.any():
        out.loc[valid, cols] = mtf[cols].to_numpy()[pos[valid]]
    return out
