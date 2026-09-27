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


def test_stale_signal_is_not_executed_after_a_gap(gapped_dataset):
    """A signal produced on the last bar before a gap must be DROPPED.

    The rule is: execution happens at the open of the first 1m bar strictly
    after decision_timestamp. If that exact bar is missing, there is no valid
    execution candle, so the intent expires. It is never carried across the gap
    and filled at the first bar after the gap (that would be a stale fill).
    """
    g = _always_long()
    idx = gapped_dataset.frame.index
    gap_start = gapped_dataset.gaps.frame["gap_start"].iloc[0]
    gap_end = gapped_dataset.gaps.frame["gap_end"].iloc[0]
    last_before = idx[idx < gap_start][-1]
    first_after = idx[idx >= gap_end][0]

    sig = generate_signals(gapped_dataset, g).frame
    assert bool(sig.loc[last_before, "entry_long"])  # a signal really exists there

    res = run_backtest(gapped_dataset, g, _cfg(), _scfg())
    entries = {t.entry_time for t in res.trades}
    # the first bar after the gap must not be a fill of the pre-gap signal
    for t in res.trades:
        if t.entry_time == first_after:
            # only admissible if its own signal bar is the bar right before it,
            # which cannot exist across a gap -> so this must never happen
            raise AssertionError("stale pre-gap signal was executed after the gap")
    assert last_before not in entries


def test_execution_index_requires_the_true_next_minute(gapped_dataset):
    """Engine-level check of the contiguity requirement."""
    from trading_school_ai.backtest.engine import _minute_step

    idx = gapped_dataset.frame.index
    step = _minute_step(idx)
    ts = idx.asi8
    gap_start = gapped_dataset.gaps.frame["gap_start"].iloc[0]
    i = int(np.flatnonzero(idx == idx[idx < gap_start][-1])[0])
    assert ts[i + 1] - ts[i] > step  # the next stored bar is NOT the next minute
