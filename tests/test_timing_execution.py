"""Execution timing contract: single shift, next-valid-open, no gap execution."""
import numpy as np
import pandas as pd

from trading_school_ai.backtest.engine import run_backtest
from trading_school_ai.backtest.types import ExitReason
from trading_school_ai.data.dataset import build_dataset
from trading_school_ai.strategies.genome import Condition, FeatureRef, Genome, RiskParams
from trading_school_ai.strategies.signals import generate_signals
from .conftest import make_frame


def _always_long():
    return Genome(
        entry=[Condition(left=FeatureRef(name="sma", params={"window": 2}),
                         op=">", right_value=0.0)],
        max_holding_bars=5, label="always",
    )


def test_signal_layer_applies_no_shift(synthetic_dataset, simple_genome):
    sig = generate_signals(synthetic_dataset, simple_genome).frame
    idx = synthetic_dataset.frame.index
    assert (sig["close_time"] == idx + pd.Timedelta(minutes=1) - pd.Timedelta("1ms")).all()
    assert (sig["decision_timestamp"] == sig["close_time"] + pd.Timedelta("1ms")).all()


def test_execution_happens_at_next_bar_open(synthetic_dataset):
    g = _always_long()
    res = run_backtest(synthetic_dataset, g, _cfg(), _scfg())
    sig = generate_signals(synthetic_dataset, g).frame
    first_sig = _first_actionable(sig)
    t = res.trades[0]
    assert t.entry_time == synthetic_dataset.frame.index[first_sig + 1]
    open_px = synthetic_dataset.frame["open"].iloc[first_sig + 1]
    assert t.entry_price >= open_px  # adverse slippage on a buy


def test_no_double_shift(synthetic_dataset):
    g = _always_long()
    res = run_backtest(synthetic_dataset, g, _cfg(), _scfg())
    sig = generate_signals(synthetic_dataset, g).frame
    first_sig = _first_actionable(sig)
    delta = res.trades[0].entry_time - synthetic_dataset.frame.index[first_sig]
    assert delta == pd.Timedelta(minutes=1)


def test_no_execution_across_gap(gapped_dataset):
    g = _always_long()
    res = run_backtest(gapped_dataset, g, _cfg(), _scfg())
    gap_start = gapped_dataset.gaps.frame["gap_start"].iloc[0]
    gap_end = gapped_dataset.gaps.frame["gap_end"].iloc[0]
    for t in res.trades:
        assert not (gap_start <= t.entry_time < gap_end)
        assert not (gap_start <= t.exit_time < gap_end)
        # the bar right before the gap never opens a position
        assert t.entry_time != gap_start - pd.Timedelta(minutes=1)


def test_gap_boundary_exit_recorded(gapped_dataset):
    g = Genome(entry=[Condition(left=FeatureRef(name="sma", params={"window": 2}),
                                op=">", right_value=0.0)],
               risk=RiskParams(stop_atr_mult=20.0, take_profit_atr_mult=50.0,
                               atr_window=200),
               max_holding_bars=5000, label="hold")
    res = run_backtest(gapped_dataset, g, _cfg(), _scfg())
    reasons = {t.exit_reason for t in res.trades}
    assert ExitReason.GAP_BOUNDARY.value in reasons


def _first_actionable(sig):
    """First signal bar whose ATR (stop distance) is available at decision time."""
    ok = sig["entry_long"].to_numpy() & ~np.isnan(sig["atr"].to_numpy())
    return int(np.flatnonzero(ok)[0])


def _cfg():
    from trading_school_ai.config.settings import BacktestConfig
    return BacktestConfig()


def _scfg():
    from trading_school_ai.config.settings import StrategyConfig
    return StrategyConfig()
