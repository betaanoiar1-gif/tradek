"""Paper execution venue - the default everywhere, including all tests."""
from __future__ import annotations

from ..risk.engine import ApprovedOrder
from .base import ExecutionVenue, Fill, InstrumentMeta, round_to_step


class PaperVenue(ExecutionVenue):
    mode = "paper"

    def __init__(self, meta: dict[str, InstrumentMeta] | None = None,
                 fee_rate: float = 0.001, slippage_bps: float = 1.0):
        self._meta = meta or {}
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self._positions: dict[str, float] = {}
        self.fills: list[Fill] = []
        self._seq = 0

    def instrument(self, symbol: str) -> InstrumentMeta:
        if symbol in self._meta:
            return self._meta[symbol]
        # Paper venue simulates a conservative instrument; explicitly labelled.
        return InstrumentMeta(symbol, 0.1, 1e-8, 1e-5, source="paper_simulated")

    def submit(self, order: ApprovedOrder) -> Fill:
        meta = self.instrument(order.intent.symbol)
        qty = round_to_step(order.approved_quantity, meta.lot_size)
        if qty < meta.min_size:
            return Fill(order.approval_id, order.intent.symbol, order.intent.side,
                        0.0, 0.0, "REJECTED_MIN_SIZE")
        direction = 1 if order.intent.side == "buy" else -1
        px = order.intent.price * (1 + direction * self.slippage_bps / 1e4)
        px = round(px / meta.tick_size) * meta.tick_size
        self._seq += 1
        self._positions[order.intent.symbol] = (
            self._positions.get(order.intent.symbol, 0.0) + direction * qty
        )
        fill = Fill(f"paper-{self._seq}", order.intent.symbol, order.intent.side,
                    qty, px, "FILLED")
        self.fills.append(fill)
        return fill

    def position(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)
