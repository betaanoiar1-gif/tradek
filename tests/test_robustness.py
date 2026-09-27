import numpy as np
import pytest

from trading_school_ai.backtest.engine import run_backtest
from trading_school_ai.metrics.core import compute_metrics
from trading_school_ai.montecarlo.runner import run_monte_carlo
from trading_school_ai.stress.overlays import parameter_perturbation, run_stress


def test_monte_carlo_deterministic(synthetic_dataset, simple_genome, settings):
    a = run_monte_carlo(synthetic_dataset, simple_genome, settings, base_seed=42)
    b = run_monte_carlo(synthetic_dataset, simple_genome, settings, base_seed=42)
    assert a.returns == b.returns
    c = run_monte_carlo(synthetic_dataset, simple_genome, settings, base_seed=43)
    assert c.returns != a.returns


def test_monte_carlo_perturbations_are_adverse(synthetic_dataset, simple_genome, settings):
    """Costs can only get worse, so MC returns must not beat the clean baseline."""
    settings.montecarlo.dropout_prob = 0.0
    mc = run_monte_carlo(synthetic_dataset, simple_genome, settings, base_seed=7)
    base_res = run_backtest(synthetic_dataset, simple_genome, settings.backtest, settings.strategy)
    base = compute_metrics(base_res)
    assert base.trade_count > 0
    assert max(mc.returns) <= base.total_return + 1e-12


def test_monte_carlo_does_not_mutate_canonical_data(synthetic_dataset, simple_genome, settings):
    before = synthetic_dataset.frame.copy(deep=True)
    hash_before = synthetic_dataset.sha256
    run_monte_carlo(synthetic_dataset, simple_genome, settings, base_seed=1)
    assert synthetic_dataset.sha256 == hash_before
    assert synthetic_dataset.frame.equals(before)


def test_stress_deterministic_and_pure(synthetic_dataset, simple_genome, settings):
    before = synthetic_dataset.frame.copy(deep=True)
    a = run_stress(synthetic_dataset, simple_genome, settings, base_seed=5)
    b = run_stress(synthetic_dataset, simple_genome, settings, base_seed=5)
    assert [s.metrics for s in a.scenarios] == [s.metrics for s in b.scenarios]
    assert synthetic_dataset.frame.equals(before)
    assert a.kind == "PASS_REPORT"


def test_stress_is_not_a_fitness_input():
    """The fitness module must not reference stress or Monte Carlo at all."""
    import inspect

    from trading_school_ai.fitness import score

    import ast

    tree = ast.parse(inspect.getsource(score))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    assert not any("stress" in m or "montecarlo" in m for m in imported)
    sig = inspect.signature(score.compute_fitness)
    assert list(sig.parameters) == ["metrics", "cfg"]


def test_higher_fees_reduce_stress_returns(synthetic_dataset, simple_genome, settings):
    settings.stress.fee_multipliers = [1.0, 10.0]
    rep = run_stress(synthetic_dataset, simple_genome, settings, base_seed=3)
    fees = [s for s in rep.scenarios if s.name.startswith("fee")]
    assert fees[-1].metrics["total_return"] <= fees[0].metrics["total_return"] + 1e-12


def test_parameter_perturbation_reports_sensitivity(synthetic_dataset, simple_genome, settings):
    out = parameter_perturbation(synthetic_dataset, simple_genome, settings, (0.9, 1.1))
    assert out["base"]["trade_count"] >= 0
    assert len(out["perturbed"]) == 2
    assert all("metrics" in p for p in out["perturbed"])


def test_walkforward_folds(synthetic_dataset, simple_genome, settings):
    from trading_school_ai.walkforward.engine import walk_forward
    folds = walk_forward(synthetic_dataset, simple_genome, settings, folds=2)
    assert len(folds) == 2
    for f in folds:
        assert f.train_end < f.test_start
