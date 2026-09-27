"""Metrics computed strictly from the real backtest ledger and equity curve.

Degenerate-case policy (documented, deterministic):
  no trades            -> all trade metrics 0, profit_factor = 0.0, valid=False
  no losses            -> profit_factor = float('inf') reported as PF_CAP (1e9)
  zero volatility      -> Sharpe/Sortino = 0.0
  no downside returns  -> Sortino = PF_CAP when mean > 0 else 0.0
  <2 daily observations-> Sharpe/Sortino = 0.0
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..backtest.types import BacktestResult

PF_CAP = 1e9
MINUTES_PER_YEAR = 365 * 24 * 60


@dataclass
class Metrics:
    total_return: float
    profit_factor: float
    win_rate: float
    trade_count: int
    max_drawdown: float
    sharpe: float
    sortino: float
    exposure: float
    total_fees: float
    total_slippage: float
    gross_pnl: float
    net_pnl: float
    avg_bars_held: float
    valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _safe(x: float) -> float:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return 0.0
    return float(x)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    arr = equity.to_numpy(dtype="float64")
    peak = np.maximum.accumulate(arr)
    dd = np.where(peak > 0, (peak - arr) / peak, 0.0)
    return float(np.max(dd))


def daily_returns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype="float64")
    daily = equity.resample("1D").last().dropna()
    return daily.pct_change().dropna()


def sharpe_ratio(dret: pd.Series) -> float:
    if len(dret) < 2:
        return 0.0
    sd = float(dret.std(ddof=1))
    if sd == 0.0 or np.isnan(sd):
        return 0.0
    return float(dret.mean() / sd * np.sqrt(365.0))


def sortino_ratio(dret: pd.Series) -> float:
    if len(dret) < 2:
        return 0.0
    downside = dret[dret < 0]
    mean = float(dret.mean())
    if len(downside) == 0:
        return PF_CAP if mean > 0 else 0.0
    dsd = float(np.sqrt(np.mean(np.square(downside))))
    if dsd == 0.0:
        return PF_CAP if mean > 0 else 0.0
    return float(mean / dsd * np.sqrt(365.0))


def compute_metrics(result: BacktestResult) -> Metrics:
    ledger = result.ledger()
    eq = result.equity
    n_trades = len(ledger)
    initial = result.initial_equity
    total_return = (
        float(eq.iloc[-1] / initial - 1.0) if len(eq) and initial > 0 else 0.0
    )
    if n_trades == 0:
        return Metrics(
            total_return=total_return, profit_factor=0.0, win_rate=0.0, trade_count=0,
            max_drawdown=max_drawdown(eq), sharpe=0.0, sortino=0.0,
            exposure=0.0, total_fees=0.0, total_slippage=0.0, gross_pnl=0.0,
            net_pnl=0.0, avg_bars_held=0.0, valid=False,
        )
    net = ledger["net_pnl"].to_numpy(dtype="float64")
    wins = net[net > 0].sum()
    losses = -net[net < 0].sum()
    pf = PF_CAP if losses == 0 else float(wins / losses)
    dret = daily_returns(eq)
    return Metrics(
        total_return=_safe(total_return),
        profit_factor=_safe(min(pf, PF_CAP)),
        win_rate=float((net > 0).mean()),
        trade_count=int(n_trades),
        max_drawdown=_safe(max_drawdown(eq)),
        sharpe=_safe(sharpe_ratio(dret)),
        sortino=_safe(sortino_ratio(dret)),
        exposure=float(result.exposure_bars / result.bars) if result.bars else 0.0,
        total_fees=float(ledger["fees"].sum()),
        total_slippage=float(ledger["slippage_cost"].sum()),
        gross_pnl=float(ledger["gross_pnl"].sum()),
        net_pnl=float(ledger["net_pnl"].sum()),
        avg_bars_held=float(ledger["bars_held"].mean()),
        valid=True,
    )
