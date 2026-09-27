"""Evolutionary search. Every candidate is evaluated by the real Experiment Engine.

Selection uses TRAIN only. Validation is observer-only. Test is never consulted.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config.settings import Settings
from ..data.dataset import CanonicalDataset
from ..experiment.engine import ExperimentResult, run_experiment
from ..experiment.seeds import derive_seed
from ..memory.db import Memory
from ..strategies.genome import Genome
from .population import crossover, mutate, seed_population


@dataclass
class Candidate:
    genome: Genome
    train: ExperimentResult
    validation: Optional[ExperimentResult] = None

    @property
    def fitness(self) -> float:
        return self.train.fitness


@dataclass
class EvolutionReport:
    generations: int
    evaluated: int
    best: Optional[Candidate]
    archive: list[dict] = field(default_factory=list)
    diversity: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generations": self.generations,
            "evaluated": self.evaluated,
            "best_genome_id": self.best.genome.genome_id if self.best else None,
            "best_fitness": self.best.fitness if self.best else None,
            "best_experiment_id": self.best.train.experiment_id if self.best else None,
            "diversity": self.diversity,
            "archive_size": len(self.archive),
        }


_WORKER_STATE: dict = {}


def _worker_init(dataset: CanonicalDataset, settings: Settings) -> None:
    # read-only, per-process state; never mutated by workers
    _WORKER_STATE["dataset"] = dataset
    _WORKER_STATE["settings"] = settings


def _worker_eval(args: tuple[str, int, str, Optional[str]]) -> ExperimentResult:
    genome_json, counter, split, commit = args
    genome = Genome.from_json(genome_json)
    return run_experiment(
        _WORKER_STATE["dataset"], genome, _WORKER_STATE["settings"],
        counter=counter, split=split, commit=commit,
    )


def _evaluate(
    genomes: list[Genome], dataset: CanonicalDataset, settings: Settings,
    counters: list[int], split: str, workers: int, commit: Optional[str],
) -> list[ExperimentResult]:
    jobs = [(g.to_json(), c, split, commit) for g, c in zip(genomes, counters)]
    if workers <= 1:
        _worker_init(dataset, settings)
        return [_worker_eval(j) for j in jobs]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(dataset, settings)
    ) as pool:
        # map preserves input order -> deterministic aggregation
        return list(pool.map(_worker_eval, jobs))


def _diversity(genomes: list[Genome]) -> float:
    ids = {g.genome_hash() for g in genomes}
    return len(ids) / max(len(genomes), 1)


def evolve(
    dataset: CanonicalDataset,
    settings: Settings,
    memory: Optional[Memory] = None,
    generations: Optional[int] = None,
    population_size: Optional[int] = None,
    observer_validation: bool = True,
    commit: Optional[str] = None,
) -> EvolutionReport:
    cfg = settings.evolution
    gens = generations if generations is not None else cfg.generations
    psize = population_size if population_size is not None else cfg.population_size
    population = seed_population(psize, cfg.master_seed)
    counter = memory.next_counter() if memory else 0
    archive: dict[str, Candidate] = {}
    diversity: list[float] = []
    evaluated = 0

    for gen in range(gens):
        todo = [g for g in population if g.genome_hash() not in archive]
        counters = list(range(counter, counter + len(todo)))
        counter += len(todo)
        results = _evaluate(
            todo, dataset, settings, counters, "train", cfg.workers, commit
        )
        evaluated += len(results)
        for g, r in zip(todo, results):
            cand = Candidate(genome=g, train=r)
            archive[g.genome_hash()] = cand
            if memory:
                memory.record_experiment(r)
        diversity.append(_diversity(population))

        scored = sorted(
            (archive[g.genome_hash()] for g in population),
            key=lambda c: (-c.fitness, c.genome.genome_hash()),
        )
        if gen == gens - 1:
            break
        rng = np.random.default_rng(derive_seed(cfg.master_seed, "gen", gen))
        elites = [c.genome for c in scored[: cfg.elitism]]
        children: list[Genome] = list(elites)
        pool_list = [c for c in scored]
        while len(children) < psize:
            p1 = _tournament(pool_list, rng, cfg.tournament_size)
            p2 = _tournament(pool_list, rng, cfg.tournament_size)
            child = crossover(p1, p2, rng) if rng.random() < cfg.crossover_rate else p1
            if rng.random() < cfg.mutation_rate:
                child = mutate(child, rng)
            try:
                Genome.model_validate(child.model_dump())
            except Exception:  # noqa: BLE001 - invalid genomes are skipped, never faked
                continue
            children.append(child)
        population = children[:psize]

    ranked = sorted(archive.values(), key=lambda c: (-c.fitness, c.genome.genome_hash()))
    best = ranked[0] if ranked and ranked[0].train.fitness_accepted else (
        ranked[0] if ranked else None
    )
    if best is not None and observer_validation:
        # Validation is observer-only: it can never change selection above.
        val = run_experiment(
            dataset, best.genome, settings, counter=counter, split="validation",
            commit=commit,
        )
        best.validation = val
        if memory:
            memory.record_experiment(val)
    return EvolutionReport(
        generations=gens, evaluated=evaluated, best=best,
        archive=[{"genome_id": c.genome.genome_id, "fitness": c.fitness,
                  "experiment_id": c.train.experiment_id,
                  "status": c.train.status} for c in ranked],
        diversity=diversity,
    )


def _tournament(pool: list[Candidate], rng: np.random.Generator, k: int) -> Genome:
    k = max(1, min(k, len(pool)))
    picks = rng.choice(len(pool), size=k, replace=False)
    best = min(picks, key=lambda i: (-pool[i].fitness, pool[i].genome.genome_hash()))
    return pool[best].genome
