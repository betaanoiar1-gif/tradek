"""Shared fixtures.

SYNTHETIC DATA NOTICE: every dataset built here is synthetic and is used only
for unit / engine tests. It is never presented as real market data, and real
data integration is covered separately in tests/test_integration_real_data.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_school_ai.config.settings import Settings
from trading_school_ai.data.dataset import build_dataset
from trading_school_ai.strategies.genome import Condition, FeatureRef, Genome, RiskParams


def make_frame(n=3000, seed=11, start="2024-01-01"):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.0006, n) + 0.0004 * np.sin(np.arange(n) / 150)
    px = 30000 * np.exp(np.cumsum(r))
    hi = px * (1 + np.abs(rng.normal(0, 0.0004, n)))
    lo = px * (1 - np.abs(rng.normal(0, 0.0004, n)))
    return pd.DataFrame({
        "open_time": idx, "open": px, "high": np.maximum(hi, px),
        "low": np.minimum(lo, px), "close": px, "volume": rng.uniform(1, 5, n),
    })


@pytest.fixture
def synthetic_dataset():
    """SYNTHETIC unit-test dataset (no gaps)."""
    return build_dataset(make_frame())


@pytest.fixture
def gapped_dataset():
    """SYNTHETIC unit-test dataset with one removed block => one declared gap."""
    df = make_frame()
    df = pd.concat([df.iloc[:1500], df.iloc[1560:]]).reset_index(drop=True)
    return build_dataset(df)


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.storage.runtime_dir = str(tmp_path / "runtime")
    # Permissive gates for unit tests so that an ACCEPTED candidate exists.
    # Gate behaviour itself is tested explicitly in test_metrics_fitness.py.
    s.fitness.min_trades = 1
    s.fitness.min_profit_factor = 0.0
    s.evolution.population_size = 6
    s.evolution.generations = 2
    s.montecarlo.runs = 8
    return s


@pytest.fixture
def simple_genome():
    return Genome(
        side="long",
        entry=[Condition(left=FeatureRef(name="rsi", params={"window": 14}),
                         op="<", right_value=35.0)],
        exit=[Condition(left=FeatureRef(name="rsi", params={"window": 14}),
                        op=">", right_value=65.0)],
        risk=RiskParams(stop_atr_mult=2.0, take_profit_atr_mult=3.0, atr_window=14),
        max_holding_bars=240,
        label="test-simple",
    )
