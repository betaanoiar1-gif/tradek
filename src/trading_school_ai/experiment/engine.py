"""The Experiment Engine: the single gate every hypothesis must pass.

Python-generated and LLM-proposed genomes take exactly the same path.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd

from .. import CODE_VERSION
from ..backtest.engine import run_backtest
from ..backtest.types import BacktestResult
from ..config.settings import Settings
from ..data.dataset import CanonicalDataset
from ..fitness.score import FitnessResult, compute_fitness
from ..metrics.core import Metrics, compute_metrics
from ..strategies.genome import Genome
from .seeds import derive_seed, experiment_id

SplitName = Literal["train", "validation", "test", "full"]


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class ExperimentResult:
    experiment_id: str
    counter: int
    seed: int
    split: str
    genome_hash: str
    genome_id: str
    feature_hash: str
    dataset_hash: Optional[str]
    dataset_source: Optional[str]
    config_hash: str
    code_version: str
    git_commit: Optional[str]
    metrics: dict
    fitness: float
    fitness_accepted: bool
    rejection_reason: str
    formula_version: str
    status: str
    ledger_path: Optional[str] = None
    genome_json: str = ""
    error: str = ""
    _result: Optional[BacktestResult] = field(default=None, repr=False, compare=False)

    def to_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return d


def split_dataset(
    dataset: CanonicalDataset, settings: Settings, split: SplitName
) -> CanonicalDataset:
    """Chronological slicing. Test is never touched during learning."""
    if split == "full":
        return dataset
    n = len(dataset.frame)
    tr = int(n * settings.split.train_frac)
    va = int(n * (settings.split.train_frac + settings.split.validation_frac))
    if split == "train":
        sub = dataset.frame.iloc[:tr]
    elif split == "validation":
        sub = dataset.frame.iloc[tr:va]
    elif split == "test":
        sub = dataset.frame.iloc[va:]
    else:  # pragma: no cover
        raise ValueError(split)
    return CanonicalDataset(sub, dataset.gaps, dataset.sha256, dataset.source_path)


def run_experiment(
    dataset: CanonicalDataset,
    genome: Genome,
    settings: Settings,
    counter: int,
    split: SplitName = "train",
    ledger_dir: Optional[str] = None,
    commit: Optional[str] = None,
) -> ExperimentResult:
    """Evaluate one genome on one split through the real backtester."""
    ghash = genome.genome_hash()
    dhash = dataset.sha256 or "NO_HASH"
    chash = settings.config_hash()
    eid = experiment_id(ghash, dhash, chash, counter, split)
    seed = derive_seed(ghash, dhash, chash, counter, split)
    sub = split_dataset(dataset, settings, split)
    try:
        result = run_backtest(
            sub, genome, settings.backtest, settings.strategy,
            genome_id=genome.genome_id, experiment_id=eid,
        )
        metrics: Metrics = compute_metrics(result)
        fitness: FitnessResult = compute_fitness(metrics, settings.fitness)
        status = "OK" if fitness.accepted else "REJECTED"
        err = ""
    except Exception as exc:  # noqa: BLE001 - invalid genomes must not kill the run
        return ExperimentResult(
            experiment_id=eid, counter=counter, seed=seed, split=split,
            genome_hash=ghash, genome_id=genome.genome_id,
            feature_hash=genome.feature_hash(), dataset_hash=dataset.sha256,
            dataset_source=dataset.source_path, config_hash=chash,
            code_version=CODE_VERSION,
            git_commit=commit if commit is not None else git_commit(),
            metrics={}, fitness=float("-inf"), fitness_accepted=False,
            rejection_reason="EXPERIMENT_ERROR", formula_version=settings.fitness.formula_version,
            status="ERROR", genome_json=genome.to_json(), error=f"{type(exc).__name__}: {exc}",
        )
    ledger_path = None
    if ledger_dir:
        from pathlib import Path

        p = Path(ledger_dir)
        p.mkdir(parents=True, exist_ok=True)
        ledger_path = str(p / f"{eid}.parquet")
        result.ledger().to_parquet(ledger_path, index=False)
    return ExperimentResult(
        experiment_id=eid, counter=counter, seed=seed, split=split,
        genome_hash=ghash, genome_id=genome.genome_id,
        feature_hash=genome.feature_hash(), dataset_hash=dataset.sha256,
        dataset_source=dataset.source_path, config_hash=chash,
        code_version=CODE_VERSION,
        git_commit=commit if commit is not None else git_commit(),
        metrics=metrics.to_dict(), fitness=fitness.score,
        fitness_accepted=fitness.accepted, rejection_reason=fitness.rejection_reason,
        formula_version=fitness.formula_version, status=status,
        ledger_path=ledger_path, genome_json=genome.to_json(), _result=result,
    )
