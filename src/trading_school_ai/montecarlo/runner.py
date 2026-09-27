"""Monte Carlo robustness.

What is perturbed (documented, deterministic per seed):
  * extra slippage in basis points, ALWAYS adverse (fills move against us)
  * fee multiplier >= 1.0 (costs can only get worse)
  * random signal dropout (entries silently missed)

What is NOT touched:
  * canonical OHLCV values and the dataset hash (read-only input)
  * gap boundaries and the execution timing rule
Trades are never reshuffled: each run is a full re-simulation, so trade-level
costs, ordering and gap boundaries stay internally consistent.
"""
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
class MonteCarloReport:
    runs: int
    base_return: float
    returns: list[float]
    profit_factors: list[float]
    max_drawdowns: list[float]
    percentile_gate: float
    gate_value: float
    gate_threshold: float
    passed: bool
    dataset_hash_before: str | None
    dataset_hash_after: str | None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["mean_return"] = float(np.mean(self.returns)) if self.returns else 0.0
        d["median_return"] = float(np.median(self.returns)) if self.returns else 0.0
        d["worst_return"] = float(np.min(self.returns)) if self.returns else 0.0
        return d


def run_monte_carlo(
    dataset: CanonicalDataset,
    genome: Genome,
    settings: Settings,
    base_seed: int,
) -> MonteCarloReport:
    cfg = settings.montecarlo
    before = dataset.sha256
    frame_checksum = float(np.nansum(dataset.frame.to_numpy()))
    base = compute_metrics(
        run_backtest(dataset, genome, settings.backtest, settings.strategy)
    )
    rets, pfs, dds = [], [], []
    n = len(dataset.frame)
    for k in range(cfg.runs):
        rng = np.random.default_rng(derive_seed(base_seed, "mc", k))
        extra_slip = float(rng.uniform(0.0, cfg.slippage_extra_bps_max))
        fee_mult = float(rng.uniform(1.0, cfg.fee_multiplier_max))
        dropout = rng.random(n) < cfg.dropout_prob
        res = run_backtest(
            dataset, genome, settings.backtest, settings.strategy,
            fee_multiplier=fee_mult, extra_slippage_bps=extra_slip,
            signal_dropout_mask=dropout,
        )
        m = compute_metrics(res)
        rets.append(m.total_return)
        pfs.append(m.profit_factor)
        dds.append(m.max_drawdown)
    after_checksum = float(np.nansum(dataset.frame.to_numpy()))
    if after_checksum != frame_checksum:  # pragma: no cover - defensive invariant
        raise RuntimeError("Monte Carlo mutated canonical data")
    gate_value = float(np.percentile(rets, cfg.percentile_gate)) if rets else 0.0
    return MonteCarloReport(
        runs=cfg.runs, base_return=base.total_return, returns=rets,
        profit_factors=pfs, max_drawdowns=dds,
        percentile_gate=cfg.percentile_gate, gate_value=gate_value,
        gate_threshold=cfg.min_gate_return,
        passed=bool(gate_value >= cfg.min_gate_return),
        dataset_hash_before=before, dataset_hash_after=dataset.sha256,
    )
