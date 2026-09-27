"""Backtest engine correctness (SYNTHETIC deterministic fixtures)."""
import numpy as np
import pandas as pd
import pytest

from trading_school_ai.backtest.engine import run_backtest
from trading_school_ai.backtest.types import ExitReason
from trading_school_ai.config.settings import BacktestConfig, StrategyConfig
from trading_school_ai.data.dataset import build_dataset
from trading_school_ai.strategies.genome import Condition, FeatureRef, Genome, RiskParams


def ramp(n=200, slope=1.0, start=100.0, spread=0.0):
    idx = pd.date_range("2024-03-01", periods=n, freq="1min", tz="UTC")
    px = start + slope * np.arange(n, dtype=float)
    return build_dataset(pd.DataFrame({
        "open_time": idx, "open": px, "high": px + spread, "low": px - spread,
        "close": px, "volume": np.ones(n)}))


def always(side="long", holding=10, stop=20.0, tp=50.0, atr_w=2):
    return Genome(side=side,
                  entry=[Condition(left=FeatureRef(name="sma", params={"window": 2}),
                                   op=">", right_value=0.0)],
                  risk=RiskParams(stop_atr_mult=stop, take_profit_atr_mult=tp,
                                  atr_window=atr_w),
                  max_holding_bars=holding, label="always")


def test_long_trade_pnl_no_costs():
    ds = ramp(60, slope=1.0)
    cfg = BacktestConfig(fee_rate=0.0, slippage_bps=0.0, initial_equity=1000.0)
    res = run_backtest(ds, always(holding=10), cfg, StrategyConfig())
    t = res.trades[0]
    assert t.side == "long"
    assert t.bars_held == 10
    assert t.exit_price - t.entry_price == pytest.approx(10.0)
    assert t.gross_pnl == pytest.approx(t.quantity * 10.0)
    assert t.net_pnl == pytest.approx(t.gross_pnl)


def test_short_trade_loses_on_uptrend():
    ds = ramp(60, slope=1.0)
    cfg = BacktestConfig(fee_rate=0.0, slippage_bps=0.0, initial_equity=1000.0)
    res = run_backtest(ds, always(side="short", holding=10), cfg, StrategyConfig())
    t = res.trades[0]
    assert t.side == "short"
    assert t.net_pnl < 0


def test_shorts_disabled_by_config():
    ds = ramp(60)
    cfg = BacktestConfig(allow_short=False, fee_rate=0.0, slippage_bps=0.0)
    res = run_backtest(ds, always(side="short"), cfg, StrategyConfig())
    assert res.trades == []


def test_fees_reduce_pnl():
    ds = ramp(60, slope=1.0)
    free = run_backtest(ds, always(holding=10), BacktestConfig(fee_rate=0.0, slippage_bps=0.0), StrategyConfig())
    paid = run_backtest(ds, always(holding=10), BacktestConfig(fee_rate=0.002, slippage_bps=0.0), StrategyConfig())
    assert paid.trades[0].fees > 0
    assert paid.trades[0].net_pnl < free.trades[0].net_pnl


def test_slippage_is_adverse():
    ds = ramp(60, slope=1.0)
    clean = run_backtest(ds, always(holding=10), BacktestConfig(fee_rate=0.0, slippage_bps=0.0), StrategyConfig())
    slipped = run_backtest(ds, always(holding=10), BacktestConfig(fee_rate=0.0, slippage_bps=50.0), StrategyConfig())
    assert slipped.trades[0].entry_price > clean.trades[0].entry_price
    assert slipped.trades[0].exit_price < clean.trades[0].exit_price
    assert slipped.trades[0].slippage_cost > 0


def test_stop_loss_triggers():
    n = 60
    idx = pd.date_range("2024-03-01", periods=n, freq="1min", tz="UTC")
    px = np.concatenate([np.full(10, 100.0), np.linspace(100.0, 80.0, n - 10)])
    ds = build_dataset(pd.DataFrame({"open_time": idx, "open": px, "high": px + 0.5,
                                     "low": px - 0.5, "close": px, "volume": 1.0}))
    g = always(holding=500, stop=1.0, tp=50.0, atr_w=2)
    res = run_backtest(ds, g, BacktestConfig(fee_rate=0.0, slippage_bps=0.0), StrategyConfig())
    assert res.trades[0].exit_reason == ExitReason.STOP_LOSS.value


def test_take_profit_triggers():
    ds = ramp(120, slope=1.0, spread=0.5)
    g = always(holding=500, stop=20.0, tp=1.0, atr_w=2)
    res = run_backtest(ds, g, BacktestConfig(fee_rate=0.0, slippage_bps=0.0), StrategyConfig())
    assert res.trades[0].exit_reason == ExitReason.TAKE_PROFIT.value


def test_time_exit_uses_H():
    ds = ramp(120, slope=0.01)
    res = run_backtest(ds, always(holding=7, stop=20.0, tp=50.0), BacktestConfig(), StrategyConfig())
    assert res.trades[0].bars_held == 7
    assert res.trades[0].exit_reason == ExitReason.TIME_EXIT.value


def test_no_overlapping_positions_by_default():
    ds = ramp(200, slope=0.01)
    res = run_backtest(ds, always(holding=10, stop=20.0, tp=50.0), BacktestConfig(), StrategyConfig())
    times = sorted([(t.entry_time, t.exit_time) for t in res.trades])
    for (a_in, a_out), (b_in, b_out) in zip(times, times[1:]):
        assert b_in >= a_out


def test_deterministic_ledger(synthetic_dataset, simple_genome):
    a = run_backtest(synthetic_dataset, simple_genome, BacktestConfig(), StrategyConfig())
    b = run_backtest(synthetic_dataset, simple_genome, BacktestConfig(), StrategyConfig())
    pd.testing.assert_frame_equal(a.ledger(), b.ledger())
    pd.testing.assert_series_equal(a.equity, b.equity)


def test_no_trades_from_random_numbers(synthetic_dataset):
    """A genome that can never trigger must produce zero trades."""
    g = Genome(entry=[Condition(left=FeatureRef(name="rsi", params={"window": 14}),
                                op="<", right_value=-1.0)], label="never")
    res = run_backtest(synthetic_dataset, g, BacktestConfig(), StrategyConfig())
    assert res.trades == []
    assert (res.equity == res.initial_equity).all()
