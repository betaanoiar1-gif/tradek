"""Purged walk-forward evaluation over contiguous chronological folds."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..backtest.engine import run_backtest
from ..config.settings import Settings
from ..data.dataset import CanonicalDataset
from ..fitness.score import compute_fitness
from ..metrics.core import compute_metrics
from ..strategies.genome import Genome


@dataclass
class Fold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_fitness: float
    test_metrics: dict


def walk_forward(
    dataset: CanonicalDataset,
    genome: Genome,
    settings: Settings,
    folds: int = 4,
) -> list[Fold]:
    """Anchored-window walk-forward with embargo between train and test."""
    if folds < 1:
        raise ValueError("folds must be >= 1")
    n = len(dataset.frame)
    H = genome.holding_horizon(settings.strategy.strategy_max_holding_bars)
    embargo = settings.split.embargo_bars
    block = n // (folds + 1)
    if block <= (H + embargo + genome.max_lookback() + 2):
        raise ValueError(
            f"Dataset too short for {folds} walk-forward folds "
            f"(block={block} bars, needs > H+embargo+L)"
        )
    out: list[Fold] = []
    idx = dataset.frame.index
    for k in range(folds):
        tr0, tr1 = 0, block * (k + 1) - 1
        # purge+embargo gap between train end and test start
        te0 = tr1 + 1 + H + embargo
        te1 = min(te0 + block - 1, n - 1)
        if te0 >= te1:
            break
        train_ds = CanonicalDataset(
            dataset.frame.iloc[tr0 : tr1 + 1], dataset.gaps, dataset.sha256, dataset.source_path
        )
        test_ds = CanonicalDataset(
            dataset.frame.iloc[te0 : te1 + 1], dataset.gaps, dataset.sha256, dataset.source_path
        )
        tr_res = run_backtest(train_ds, genome, settings.backtest, settings.strategy)
        te_res = run_backtest(test_ds, genome, settings.backtest, settings.strategy)
        tr_fit = compute_fitness(compute_metrics(tr_res), settings.fitness)
        out.append(
            Fold(
                fold=k,
                train_start=idx[tr0],
                train_end=idx[tr1],
                test_start=idx[te0],
                test_end=idx[te1],
                train_fitness=tr_fit.score,
                test_metrics=compute_metrics(te_res).to_dict(),
            )
        )
    return out
