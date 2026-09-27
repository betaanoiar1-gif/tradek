from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


class ExitReason(str, Enum):
    SIGNAL = "SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TIME_EXIT = "TIME_EXIT"
    GAP_BOUNDARY = "GAP_BOUNDARY"
    END_OF_DATA = "END_OF_DATA"


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    fees: float
    slippage_cost: float
    exit_reason: str
    gross_pnl: float
    net_pnl: float
    bars_held: int
    genome_id: str = ""
    experiment_id: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["entry_time"] = pd.Timestamp(self.entry_time)
        d["exit_time"] = pd.Timestamp(self.exit_time)
        return d


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity: pd.Series
    initial_equity: float
    bars: int
    exposure_bars: int
    config_snapshot: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def ledger(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "entry_time", "exit_time", "side", "entry_price", "exit_price",
                    "quantity", "fees", "slippage_cost", "exit_reason", "gross_pnl",
                    "net_pnl", "bars_held", "genome_id", "experiment_id",
                ]
            )
        return pd.DataFrame([t.to_dict() for t in self.trades])
