"""Execution interfaces. Only risk-approved orders may enter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..risk.engine import ApprovedOrder


class ExecutionError(Exception):
    pass


class LiveLockError(ExecutionError):
    """Live trading requires config + LIVE_CONFIRM=1 + --live. All three."""


@dataclass
class InstrumentMeta:
    inst_id: str
    tick_size: float
    lot_size: float
    min_size: float
    source: str  # 'exchange' or 'unavailable' - never silently assumed


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: str
    raw: dict = field(default_factory=dict)


class ExecutionVenue(ABC):
    mode: str = "paper"

    @abstractmethod
    def instrument(self, symbol: str) -> InstrumentMeta: ...

    @abstractmethod
    def submit(self, order: ApprovedOrder) -> Fill: ...

    @abstractmethod
    def position(self, symbol: str) -> float: ...


def round_to_step(value: float, step: float) -> float:
    """Round DOWN to the venue step; never round up past available size."""
    if step <= 0:
        raise ValueError("step must be positive")
    import math

    return math.floor(value / step + 1e-12) * step
