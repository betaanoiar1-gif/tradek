"""Risk Engine: independent of strategy logic and of the LLM.

Nothing reaches execution without an explicit approval object from here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from ..config.settings import RiskConfig

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    quantity: float
    price: float
    reason: str = ""

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.price)


@dataclass(frozen=True)
class ApprovedOrder:
    """The only object the execution layer accepts."""

    intent: OrderIntent
    approved_quantity: float
    approval_id: str
    risk_snapshot: dict


@dataclass
class RiskState:
    equity: float
    open_exposure: float = 0.0
    realized_pnl_today: float = 0.0
    peak_equity: float = 0.0
    last_order_ts: float = -1e18
    halted: bool = False
    halt_reason: str = ""

    def drawdown(self) -> float:
        peak = max(self.peak_equity, self.equity)
        return 0.0 if peak <= 0 else (peak - self.equity) / peak


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    order: Optional[ApprovedOrder] = None


class RiskEngine:
    def __init__(self, cfg: RiskConfig, state: RiskState):
        self.cfg = cfg
        self.state = state
        self.state.peak_equity = max(state.peak_equity, state.equity)
        self.alerts: list[str] = []

    # ------------------------------------------------------------- controls
    def trip_kill_switch(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        self.alerts.append(f"KILL_SWITCH: {reason}")

    def update_equity(self, equity: float) -> None:
        self.state.equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)

    def register_fill(self, notional: float, realized_pnl: float = 0.0) -> None:
        self.state.open_exposure += notional
        self.state.realized_pnl_today += realized_pnl

    def report_position_mismatch(self, local: float, exchange: float) -> str:
        """Any mismatch is a hard STOP + ALERT; never a silent guess."""
        if abs(local - exchange) > 1e-12:
            msg = f"POSITION_MISMATCH local={local} exchange={exchange}"
            self.trip_kill_switch(msg)
            return "STOP_AND_ALERT"
        return "OK"

    # -------------------------------------------------------------- gateway
    def approve(self, intent: OrderIntent, now: Optional[float] = None) -> RiskDecision:
        now = time.monotonic() if now is None else now
        if self.cfg.kill_switch or self.state.halted:
            return RiskDecision(False, f"HALTED: {self.state.halt_reason or 'kill_switch'}")
        if intent.quantity <= 0 or intent.price <= 0:
            return RiskDecision(False, "INVALID_ORDER")
        if now - self.state.last_order_ts < self.cfg.cooldown_seconds:
            return RiskDecision(False, "COOLDOWN")
        if self.state.realized_pnl_today <= -abs(self.cfg.daily_loss_limit):
            self.trip_kill_switch("DAILY_LOSS_LIMIT")
            return RiskDecision(False, "DAILY_LOSS_LIMIT")
        if self.state.drawdown() > self.cfg.max_drawdown_limit:
            self.trip_kill_switch("MAX_DRAWDOWN")
            return RiskDecision(False, "MAX_DRAWDOWN")
        notional = intent.notional
        if notional > self.cfg.max_position_quote:
            return RiskDecision(False, "MAX_POSITION_EXCEEDED")
        if self.state.open_exposure + notional > self.cfg.max_exposure_quote:
            return RiskDecision(False, "MAX_EXPOSURE_EXCEEDED")
        if self.state.equity > 0:
            leverage = (self.state.open_exposure + notional) / self.state.equity
            if leverage > self.cfg.max_leverage + 1e-12:
                return RiskDecision(False, "LEVERAGE_EXCEEDED")
        self.state.last_order_ts = now
        snapshot = {
            "equity": self.state.equity,
            "open_exposure": self.state.open_exposure,
            "drawdown": self.state.drawdown(),
            "realized_pnl_today": self.state.realized_pnl_today,
        }
        approval_id = f"{intent.symbol}:{intent.side}:{intent.quantity:.8f}:{intent.price:.8f}"
        return RiskDecision(
            True, "APPROVED",
            ApprovedOrder(intent, intent.quantity, approval_id, snapshot),
        )
