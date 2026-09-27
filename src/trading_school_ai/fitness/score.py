"""Deterministic, absolute fitness. Train metrics only. No population scaling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.settings import FitnessConfig
from ..metrics.core import Metrics

REJECT_SCORE = -1e9


@dataclass
class FitnessResult:
    score: float
    accepted: bool
    rejection_reason: str
    formula_version: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def rejection_gate(metrics: Metrics, cfg: FitnessConfig) -> str:
    """Return the first failing gate, or '' when all gates pass."""
    if not metrics.valid or metrics.trade_count == 0:
        return "NO_TRADES"
    if metrics.trade_count < cfg.min_trades:
        return f"MIN_TRADES<{cfg.min_trades}"
    if metrics.profit_factor < cfg.min_profit_factor:
        return f"PROFIT_FACTOR<{cfg.min_profit_factor}"
    if metrics.max_drawdown > cfg.max_drawdown_limit:
        return f"MAX_DRAWDOWN>{cfg.max_drawdown_limit}"
    if not np.isfinite(metrics.total_return):
        return "NON_FINITE_RETURN"
    return ""


def compute_fitness(metrics: Metrics, cfg: FitnessConfig) -> FitnessResult:
    """Absolute score. Monte Carlo and Stress results never enter this formula."""
    reason = rejection_gate(metrics, cfg)
    if reason:
        return FitnessResult(REJECT_SCORE, False, reason, cfg.formula_version)
    pf_term = cfg.profit_factor_weight * float(np.log1p(max(metrics.profit_factor - 1.0, 0.0)))
    sharpe_term = cfg.sharpe_weight * float(metrics.sharpe)
    dd_term = cfg.drawdown_penalty * float(metrics.max_drawdown)
    score = sharpe_term + pf_term - dd_term
    return FitnessResult(float(score), True, "", cfg.formula_version)
