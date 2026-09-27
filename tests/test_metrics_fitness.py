import numpy as np
import pandas as pd
import pytest

from trading_school_ai.backtest.engine import run_backtest
from trading_school_ai.backtest.types import BacktestResult
from trading_school_ai.config.settings import FitnessConfig
from trading_school_ai.fitness.score import REJECT_SCORE, compute_fitness, rejection_gate
from trading_school_ai.metrics.core import (PF_CAP, compute_metrics, daily_returns,
                                            max_drawdown, sharpe_ratio, sortino_ratio)


def test_metrics_come_from_real_ledger(synthetic_dataset, simple_genome, settings):
    res = run_backtest(synthetic_dataset, simple_genome, settings.backtest, settings.strategy)
    m = compute_metrics(res)
    ledger = res.ledger()
    assert m.trade_count == len(ledger)
    assert m.net_pnl == pytest.approx(ledger["net_pnl"].sum())
    assert m.total_fees == pytest.approx(ledger["fees"].sum())


def test_no_trades_degenerate_case():
    eq = pd.Series([100.0] * 10, index=pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC"))
    m = compute_metrics(BacktestResult([], eq, 100.0, 10, 0))
    assert m.trade_count == 0 and m.profit_factor == 0.0 and not m.valid


def test_zero_volatility_sharpe_is_zero():
    dr = pd.Series([0.0, 0.0, 0.0])
    assert sharpe_ratio(dr) == 0.0
    assert sortino_ratio(dr) == 0.0


def test_no_downside_sortino_capped():
    dr = pd.Series([0.01, 0.02, 0.03])
    assert sortino_ratio(dr) == PF_CAP


def test_insufficient_observations():
    assert sharpe_ratio(pd.Series([0.01])) == 0.0
    assert max_drawdown(pd.Series(dtype="float64")) == 0.0


def test_fitness_is_deterministic_and_absolute(synthetic_dataset, simple_genome, settings):
    res = run_backtest(synthetic_dataset, simple_genome, settings.backtest, settings.strategy)
    m = compute_metrics(res)
    a = compute_fitness(m, settings.fitness)
    b = compute_fitness(m, settings.fitness)
    assert a.score == b.score
    assert a.formula_version == settings.fitness.formula_version


def test_fitness_independent_of_population(synthetic_dataset, simple_genome, settings):
    """Fitness must not depend on any other candidate's results."""
    res = run_backtest(synthetic_dataset, simple_genome, settings.backtest, settings.strategy)
    m = compute_metrics(res)
    lone = compute_fitness(m, settings.fitness).score
    # evaluating many other genomes changes nothing
    from trading_school_ai.learning.population import seed_population
    for g in seed_population(5, 1):
        run_backtest(synthetic_dataset, g, settings.backtest, settings.strategy)
    assert compute_fitness(m, settings.fitness).score == lone


def test_rejection_gates():
    from trading_school_ai.metrics.core import Metrics
    cfg = FitnessConfig(min_trades=10, min_profit_factor=1.0, max_drawdown_limit=0.2)
    few = Metrics(0.1, 2.0, 0.6, 5, 0.05, 1.0, 1.0, 0.1, 0, 0, 0, 0, 10, True)
    assert rejection_gate(few, cfg).startswith("MIN_TRADES")
    assert compute_fitness(few, cfg).score == REJECT_SCORE
    bad_pf = Metrics(0.1, 0.5, 0.6, 50, 0.05, 1.0, 1.0, 0.1, 0, 0, 0, 0, 10, True)
    assert rejection_gate(bad_pf, cfg).startswith("PROFIT_FACTOR")
    deep = Metrics(0.1, 2.0, 0.6, 50, 0.9, 1.0, 1.0, 0.1, 0, 0, 0, 0, 10, True)
    assert rejection_gate(deep, cfg).startswith("MAX_DRAWDOWN")
    good = Metrics(0.1, 2.0, 0.6, 50, 0.05, 1.0, 1.0, 0.1, 0, 0, 0, 0, 10, True)
    assert rejection_gate(good, cfg) == ""
    assert compute_fitness(good, cfg).accepted
