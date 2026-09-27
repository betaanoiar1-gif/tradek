"""Rerun a stored experiment and compare outputs within a documented tolerance."""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..config.settings import Settings
from ..data.dataset import CanonicalDataset
from ..memory.db import Memory
from ..strategies.genome import Genome
from .engine import run_experiment

# Documented tolerance: metrics are floating point aggregates of identical
# deterministic simulations, so the only admissible difference is FP noise.
DEFAULT_TOLERANCE = 1e-9


@dataclass
class ReproduceReport:
    experiment_id: str
    reproduced_experiment_id: str
    identical_id: bool
    tolerance: float
    mismatches: dict
    ok: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def reproduce(
    memory: Memory, experiment_id: str, dataset: CanonicalDataset,
    settings: Settings, tolerance: float = DEFAULT_TOLERANCE,
) -> ReproduceReport:
    row = memory.get_experiment(experiment_id)
    if row is None:
        raise KeyError(f"Unknown experiment_id {experiment_id}")
    genome = Genome.from_json(row["genome_json"])
    stored_metrics = json.loads(row["metrics_json"] or "{}")
    rerun = run_experiment(
        dataset, genome, settings, counter=int(row["counter"]), split=row["split"],
        commit=row["git_commit"],
    )
    mismatches: dict = {}
    for key, old in stored_metrics.items():
        new = rerun.metrics.get(key)
        if isinstance(old, bool) or isinstance(new, bool):
            if bool(old) != bool(new):
                mismatches[key] = {"stored": old, "rerun": new}
        elif isinstance(old, (int, float)) and isinstance(new, (int, float)):
            if abs(float(old) - float(new)) > tolerance:
                mismatches[key] = {"stored": old, "rerun": new}
        elif old != new:
            mismatches[key] = {"stored": old, "rerun": new}
    if abs(float(row["fitness"]) - rerun.fitness) > tolerance:
        mismatches["fitness"] = {"stored": row["fitness"], "rerun": rerun.fitness}
    if row["seed"] != rerun.seed:
        mismatches["seed"] = {"stored": row["seed"], "rerun": rerun.seed}
    identical = rerun.experiment_id == experiment_id
    return ReproduceReport(
        experiment_id=experiment_id, reproduced_experiment_id=rerun.experiment_id,
        identical_id=identical, tolerance=tolerance, mismatches=mismatches,
        ok=identical and not mismatches,
    )
