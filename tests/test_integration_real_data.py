"""REAL DATA integration tests.

These tests load the actual canonical BTCUSDT 1m dataset. They are SKIPPED with
an explicit reason when the dataset is not present. A skipped test is never
reported as a pass.

Point them at the dataset with:
    TSA_CANONICAL_PATH=/path/BTCUSDT-1m-canonical.parquet \
    TSA_GAPS_PATH=/path/BTCUSDT-1m-missing-gaps.parquet pytest -q
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from trading_school_ai.config.settings import CANONICAL_DATASET_SHA256, load_settings
from trading_school_ai.data.dataset import load_canonical, validate_dataset

SETTINGS = load_settings(None)
CANONICAL = Path(SETTINGS.data.canonical_path)
REASON = (
    f"Canonical dataset not available at {CANONICAL}. "
    "Set TSA_CANONICAL_PATH (and TSA_GAPS_PATH) to run real-data integration "
    "tests. No substitute or synthetic data is used here."
)
real_data = pytest.mark.skipif(not CANONICAL.exists(), reason=REASON)


@real_data
def test_real_dataset_contract():
    report = validate_dataset(SETTINGS.data)
    assert report.ok, report.errors
    assert report.rows > 0
    assert report.sha256 is not None


@real_data
def test_real_dataset_hash_matches_contract():
    report = validate_dataset(SETTINGS.data)
    assert report.sha256 == CANONICAL_DATASET_SHA256, (
        "Dataset hash does not match the documented contract hash; "
        f"got {report.sha256}"
    )


@real_data
def test_real_dataset_backtest_and_metrics():
    from trading_school_ai.backtest.engine import run_backtest
    from trading_school_ai.metrics.core import compute_metrics
    from trading_school_ai.strategies.genome import Condition, FeatureRef, Genome

    ds = load_canonical(SETTINGS.data)
    g = Genome(entry=[Condition(left=FeatureRef(name="rsi", params={"window": 14}),
                                op="<", right_value=30.0)],
               exit=[Condition(left=FeatureRef(name="rsi", params={"window": 14}),
                               op=">", right_value=70.0)],
               max_holding_bars=1440, label="real-rsi")
    res = run_backtest(ds, g, SETTINGS.backtest, SETTINGS.strategy)
    m = compute_metrics(res)
    assert m.trade_count >= 0
    gaps = ds.gaps.frame
    for _, row in gaps.iterrows():
        for t in res.trades:
            assert not (row["gap_start"] <= t.entry_time < row["gap_end"])


@real_data
def test_real_dataset_walkforward():
    from trading_school_ai.strategies.genome import Condition, FeatureRef, Genome
    from trading_school_ai.walkforward.engine import walk_forward

    ds = load_canonical(SETTINGS.data)
    g = Genome(entry=[Condition(left=FeatureRef(name="rsi", params={"window": 14}),
                                op="<", right_value=30.0)],
               max_holding_bars=1440, label="real-wf")
    folds = walk_forward(ds, g, SETTINGS, folds=3)
    assert len(folds) >= 1
    for f in folds:
        assert f.train_end < f.test_start
