"""MTF and causality tests (SYNTHETIC data)."""
import numpy as np
import pandas as pd
import pytest

from trading_school_ai.data.dataset import build_dataset
from trading_school_ai.features.registry import FeatureSpec, compute_feature
from trading_school_ai.mtf.resample import ONE_MS, align_to_base, build_mtf, visible_candles
from .conftest import make_frame


def test_right_labeled_right_closed(synthetic_dataset):
    m5 = build_mtf(synthetic_dataset, "5m")
    first = m5.iloc[0]
    # label is the bucket end; close_time = label - 1ms
    assert m5.index[0] - ONE_MS == first["close_time"]
    assert first["available_from"] == first["close_time"] + ONE_MS
    assert (m5["bar_count"] == 5).all()


def test_incomplete_candles_dropped():
    df = make_frame(103)  # 103 minutes -> last 5m bucket incomplete
    ds = build_dataset(df)
    m5 = build_mtf(ds, "5m")
    assert (m5["bar_count"] == 5).all()
    assert m5["close_time"].iloc[-1] < ds.frame.index[-1]


def test_gap_candles_invalidated(gapped_dataset):
    m15 = build_mtf(gapped_dataset, "15m")
    g0, g1 = gapped_dataset.gaps.frame.iloc[0]
    starts = m15["close_time"] + ONE_MS - pd.Timedelta(minutes=15)
    ends = m15["close_time"] + ONE_MS
    assert not ((starts < g1) & (ends > g0)).any()


def test_visibility_rule_strict(synthetic_dataset):
    m5 = build_mtf(synthetic_dataset, "5m")
    t = m5["close_time"].iloc[10]
    vis = visible_candles(m5, t)
    assert (vis["close_time"] < t).all()
    assert m5["close_time"].iloc[10] not in set(vis["close_time"])


def test_align_to_base_has_no_lookahead(synthetic_dataset):
    m5 = build_mtf(synthetic_dataset, "5m")
    aligned = align_to_base(m5, synthetic_dataset.frame.index)
    for i in (100, 500, 1200):
        ts = synthetic_dataset.frame.index[i]
        val = aligned["close"].iloc[i]
        if np.isnan(val):
            continue
        usable = m5[m5["close_time"] < ts]
        assert val == usable["close"].iloc[-1]


def test_future_mutation_does_not_change_past_features(synthetic_dataset):
    spec = FeatureSpec("sma", {"window": 20})
    base = compute_feature(synthetic_dataset, spec)
    df = synthetic_dataset.frame.reset_index()
    df.loc[2500:, "close"] *= 1.5
    df.loc[2500:, "high"] = df.loc[2500:, ["high", "close"]].max(axis=1) * 1.5
    mutated = compute_feature(build_dataset(df), spec)
    pd.testing.assert_series_equal(base.iloc[:2400], mutated.iloc[:2400])


def test_feature_window_invalid_after_gap(gapped_dataset):
    spec = FeatureSpec("sma", {"window": 20})
    s = compute_feature(gapped_dataset, spec)
    gap_start = gapped_dataset.gaps.frame["gap_start"].iloc[0]
    after = s[s.index >= gap_start]
    assert after.iloc[:19].isna().all()
    assert not np.isnan(after.iloc[19])
