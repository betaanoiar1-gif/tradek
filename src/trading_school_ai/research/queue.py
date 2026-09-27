"""Research queue: routes hypotheses (AI or Python) into the Experiment Engine.

There is exactly one evaluation path. A hypothesis is a claim until Python
statistics judge it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..ai.schema import Hypothesis, HypothesisRejected
from ..config.settings import Settings
from ..data.dataset import CanonicalDataset
from ..experiment.engine import ExperimentResult, run_experiment
from ..memory.db import Memory
from ..strategies.genome import Genome


@dataclass
class ResearchOutcome:
    hypothesis_id: str
    source: str
    accepted_by_schema: bool
    genome_id: Optional[str]
    experiment: Optional[ExperimentResult]
    verdict: str
    detail: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["experiment"] = self.experiment.to_row() if self.experiment else None
        return d


def evaluate_hypothesis(
    hypothesis: Hypothesis,
    dataset: CanonicalDataset,
    settings: Settings,
    memory: Optional[Memory] = None,
    source: str = "ai",
) -> ResearchOutcome:
    hid = hypothesis.hypothesis_id()
    try:
        genome: Genome = hypothesis.to_genome()
    except HypothesisRejected as exc:
        if memory:
            memory.record_hypothesis(hid, source, hypothesis.model_dump(), "INVALID_GENOME")
        return ResearchOutcome(hid, source, True, None, None, "REJECTED_INVALID_GENOME", str(exc))
    counter = memory.next_counter() if memory else 0
    result = run_experiment(dataset, genome, settings, counter=counter, split="train")
    if memory:
        memory.record_experiment(result)
        memory.record_hypothesis(
            hid, source, hypothesis.model_dump(),
            "VERIFIED" if result.fitness_accepted else "FALSIFIED", result.experiment_id,
        )
    verdict = "VERIFIED" if result.fitness_accepted else "FALSIFIED"
    return ResearchOutcome(hid, source, True, genome.genome_id, result, verdict,
                           result.rejection_reason)
