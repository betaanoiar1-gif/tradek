"""Stress overlays. PASS_REPORT only: never a Fitness input, never a rejection gate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backtest.engine import run_backtest
from ..config.settings import Settings
from ..data.dataset import CanonicalDataset
from ..experiment.seeds import derive_seed
from ..metrics.core import compute_metrics
from ..strategies.genome import Genome


@dataclass
class StressScenario:
    name: str
    fee_multiplier: float
    slippage_multiplier: float
    execution_delay_bars: int
    missing_signal_prob: float
    metrics: dict


@dataclass
class StressReport:
    scenarios: list[StressScenario]
    kind: str = "PASS_REPORT"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "note": "Stress is reported, not scored. It never enters fitness.",
            "scenarios": [s.__dict__ for s in self.scenarios],
        }

    def survived(self, min_return: float = 0.0) -> list[str]:
        return [s.name for s in self.scenarios if s.metrics.get("total_return", 0.0) >= min_return]


def run_stress(
    dataset: CanonicalDataset, genome: Genome, settings: Settings, base_seed: int
) -> StressReport:
    cfg = settings.stress
    n = len(dataset.frame)
    checksum = float(np.nansum(dataset.frame.to_numpy()))
    scenarios: list[StressScenario] = []
    combos = []
    for fm in cfg.fee_multipliers:
        combos.append(("fee", fm, 1.0, 0, 0.0))
    for sm in cfg.slippage_multipliers:
        combos.append(("slippage", 1.0, sm, 0, 0.0))
    for d in cfg.execution_delay_bars:
        combos.append(("delay", 1.0, 1.0, d, 0.0))
    for p in cfg.missing_signal_probs:
        combos.append(("missing_signal", 1.0, 1.0, 0, p))
    # volatility-related execution stress: slippage scales with realized volatility
    combos.append(("volatility_execution", 1.0, 5.0, 1, 0.0))
    for kind, fm, sm, delay, prob in combos:
        mask = None
        if prob > 0:
            rng = np.random.default_rng(derive_seed(base_seed, "stress", kind, prob))
            mask = rng.random(n) < prob
        res = run_backtest(
            dataset, genome, settings.backtest, settings.strategy,
            fee_multiplier=fm, slippage_multiplier=sm,
            execution_delay_bars=delay, signal_dropout_mask=mask,
        )
        scenarios.append(
            StressScenario(
                name=f"{kind}:fee={fm},slip={sm},delay={delay},miss={prob}",
                fee_multiplier=fm, slippage_multiplier=sm,
                execution_delay_bars=delay, missing_signal_prob=prob,
                metrics=compute_metrics(res).to_dict(),
            )
        )
    if float(np.nansum(dataset.frame.to_numpy())) != checksum:  # pragma: no cover
        raise RuntimeError("Stress overlays mutated canonical data")
    return StressReport(scenarios)


def parameter_perturbation(
    dataset: CanonicalDataset, genome: Genome, settings: Settings,
    multipliers: tuple[float, ...] = (0.8, 0.9, 1.1, 1.2),
) -> dict:
    """Perturb genome parameters and report metric sensitivity."""
    out = {"base": None, "perturbed": []}
    base = compute_metrics(run_backtest(dataset, genome, settings.backtest, settings.strategy))
    out["base"] = base.to_dict()
    for m in multipliers:
        data = genome.model_dump()
        for cond in data["entry"]:
            if cond["right_kind"] == "value":
                cond["right_value"] = float(cond["right_value"]) * m
            else:
                cond["left"]["params"] = {
                    k: max(2.0, round(v * m)) for k, v in cond["left"]["params"].items()
                }
        g2 = type(genome)(**data)
        met = compute_metrics(run_backtest(dataset, g2, settings.backtest, settings.strategy))
        out["perturbed"].append({"multiplier": m, "genome_id": g2.genome_id,
                                 "metrics": met.to_dict()})
    return out
