"""Compact statistical summaries for the teacher. Never raw OHLCV dumps."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data.dataset import CanonicalDataset

MAX_SUMMARY_FIELDS = 40


def dataset_summary(dataset: CanonicalDataset) -> dict[str, Any]:
    df = dataset.frame
    ret = df["close"].pct_change().dropna()
    return {
        "bars": int(len(df)),
        "start": str(df.index[0]) if len(df) else None,
        "end": str(df.index[-1]) if len(df) else None,
        "gap_count": len(dataset.gaps),
        "missing_bars": dataset.gaps.total_missing_bars(),
        "return_mean_bp": round(float(ret.mean()) * 1e4, 4) if len(ret) else 0.0,
        "return_std_bp": round(float(ret.std()) * 1e4, 4) if len(ret) else 0.0,
        "skew": round(float(ret.skew()), 4) if len(ret) > 2 else 0.0,
        "kurtosis": round(float(ret.kurt()), 4) if len(ret) > 3 else 0.0,
        "abs_return_autocorr_1": round(float(ret.abs().autocorr(1)), 4) if len(ret) > 2 else 0.0,
        "trend_total_pct": round(float(df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 3)
        if len(df) > 1 else 0.0,
    }


def experiment_summary(rows: list[dict], limit: int = 10) -> list[dict]:
    out = []
    for r in rows[:limit]:
        out.append(
            {
                "genome_id": r.get("genome_id"),
                "fitness": round(float(r.get("fitness", 0.0)), 4),
                "status": r.get("status"),
                "rejection_reason": r.get("rejection_reason"),
            }
        )
    return out


def failure_summary(rows: list[dict]) -> dict:
    return {str(r.get("rejection_reason")): int(r.get("n", 0)) for r in rows}
