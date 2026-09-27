import numpy as np
import pytest

from trading_school_ai.config.settings import Settings
from trading_school_ai.strategies.genome import Condition, FeatureRef, Genome, RiskParams
from trading_school_ai.walkforward.purge import (
    build_split, chronological_split, embargo_mask, purge_count, purge_mask,
)


def _genome(atr_mult=2.0, atr_window=14, holding=None, window=50):
    return Genome(
        entry=[Condition(left=FeatureRef(name="sma", params={"window": window}),
                         op=">", right_value=0.0)],
        risk=RiskParams(stop_atr_mult=atr_mult, atr_window=atr_window),
        max_holding_bars=holding, label="p",
    )


def test_L_is_max_signal_feature_lookback():
    g = Genome(entry=[
        Condition(left=FeatureRef(name="sma", params={"window": 50}), op=">", right_value=0),
        Condition(left=FeatureRef(name="rsi", params={"window": 14}), op=">", right_value=0),
    ], label="L")
    assert g.max_lookback() == 50


def test_H_from_genome():
    assert _genome(holding=333).holding_horizon(1440) == 333


def test_H_fallback_from_config():
    s = Settings()
    assert s.strategy.strategy_max_holding_bars == 1440
    assert _genome(holding=None).holding_horizon(s.strategy.strategy_max_holding_bars) == 1440


def test_atr_multiplier_does_not_change_H_or_L():
    a, b = _genome(atr_mult=1.0), _genome(atr_mult=9.5)
    assert a.holding_horizon(1440) == b.holding_horizon(1440)
    assert a.max_lookback() == b.max_lookback()


def test_atr_multiplier_does_not_change_purge_count():
    a, b = _genome(atr_mult=1.0), _genome(atr_mult=9.5)
    pa = purge_count(10000, a.max_lookback(), a.holding_horizon(1440), 0.6, 0.2)
    pb = purge_count(10000, b.max_lookback(), b.holding_horizon(1440), 0.6, 0.2)
    assert pa == pb and pa > 0


def test_atr_window_does_not_change_purge_count():
    a, b = _genome(atr_window=7), _genome(atr_window=100)
    pa = purge_count(10000, a.max_lookback(), a.holding_horizon(1440), 0.6, 0.2)
    pb = purge_count(10000, b.max_lookback(), b.holding_horizon(1440), 0.6, 0.2)
    assert pa == pb


def test_purge_only_on_label_overlap():
    idx = np.arange(0, 100)
    # test interval [100, 199], horizon 10 -> only i >= 90 have labels reaching 100
    mask = purge_mask(idx, 100, 199, 10)
    assert mask[90:].all()
    assert not mask[:90].any()


def test_feature_overlap_alone_does_not_purge():
    # huge lookback, zero horizon -> nothing is purged
    mask = purge_mask(np.arange(0, 100), 100, 199, 0)
    assert not mask.any()


def test_embargo_boundaries():
    idx = np.arange(195, 215)
    mask = embargo_mask(idx, test_end=199, embargo_bars=5)
    assert list(idx[mask]) == [200, 201, 202, 203, 204]


def test_build_split_is_chronological_and_disjoint():
    sp = build_split(10000, lookback=50, horizon=100, train_frac=0.6,
                     validation_frac=0.2, embargo_bars=10)
    assert sp.train.max() < sp.validation.min() < sp.test.min()
    assert len(set(sp.train) & set(sp.test)) == 0
    assert sp.train.min() >= 49


def test_chronological_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        chronological_split(100, 0.9, 0.2)
