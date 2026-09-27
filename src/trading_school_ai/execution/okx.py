"""OKX Spot adapter behind the three live locks.

Exchange metadata (tick size, lot size, minimum size) is ALWAYS retrieved from
the exchange. Nothing about the instrument is hardcoded as authoritative.
"""
from __future__ import annotations

import os
import time
from typing import Optional

from ..risk.engine import ApprovedOrder
from .base import ExecutionError, ExecutionVenue, Fill, InstrumentMeta, LiveLockError, round_to_step

SYMBOL_MAP = {"BTCUSDT": "BTC-USDT"}


def to_okx_symbol(canonical: str) -> str:
    if canonical in SYMBOL_MAP:
        return SYMBOL_MAP[canonical]
    if "-" in canonical:
        return canonical
    for quote in ("USDT", "USDC", "USD"):
        if canonical.endswith(quote):
            return f"{canonical[: -len(quote)]}-{quote}"
    raise ValueError(f"Cannot map {canonical!r} to an OKX SPOT instrument id")


def check_live_locks(live_enabled_in_config: bool, cli_live_flag: bool) -> None:
    """All three locks must be open, otherwise live execution is refused."""
    if not live_enabled_in_config:
        raise LiveLockError("LOCK_1_FAILED: execution.live_enabled is false in config")
    if os.environ.get("LIVE_CONFIRM") != "1":
        raise LiveLockError("LOCK_2_FAILED: environment variable LIVE_CONFIRM=1 is not set")
    if not cli_live_flag:
        raise LiveLockError("LOCK_3_FAILED: the --live CLI flag was not provided")


class OKXSpotVenue(ExecutionVenue):
    mode = "live"

    def __init__(self, live_enabled_in_config: bool, cli_live_flag: bool,
                 base_url: str = "https://www.okx.com", client=None):
        check_live_locks(live_enabled_in_config, cli_live_flag)
        self.base_url = base_url
        self._client = client
        self._meta_cache: dict[str, InstrumentMeta] = {}

    def _get(self, path: str, params: dict) -> dict:
        if self._client is not None:
            return self._client.get(path, params)
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ExecutionError("httpx required for OKX access") from exc
        r = httpx.get(f"{self.base_url}{path}", params=params, timeout=15.0)
        r.raise_for_status()
        return r.json()

    def instrument(self, symbol: str) -> InstrumentMeta:
        inst_id = to_okx_symbol(symbol)
        if inst_id in self._meta_cache:
            return self._meta_cache[inst_id]
        data = self._get("/api/v5/public/instruments",
                         {"instType": "SPOT", "instId": inst_id})
        items = data.get("data") or []
        if not items:
            raise ExecutionError(
                f"OKX returned no SPOT instrument metadata for {inst_id}; refusing to "
                "guess tick/lot/min size."
            )
        d = items[0]
        meta = InstrumentMeta(
            inst_id=inst_id, tick_size=float(d["tickSz"]),
            lot_size=float(d["lotSz"]), min_size=float(d["minSz"]), source="exchange",
        )
        self._meta_cache[inst_id] = meta
        return meta

    def available_balance(self, ccy: str) -> float:
        data = self._get("/api/v5/account/balance", {"ccy": ccy})
        items = data.get("data") or []
        if not items:
            raise ExecutionError(f"Could not read OKX balance for {ccy}")
        for det in items[0].get("details", []):
            if det.get("ccy") == ccy:
                return float(det.get("availBal", 0.0))
        raise ExecutionError(f"No balance entry for {ccy}")

    def submit(self, order: ApprovedOrder) -> Fill:
        meta = self.instrument(order.intent.symbol)
        qty = round_to_step(order.approved_quantity, meta.lot_size)
        if qty < meta.min_size:
            return Fill(order.approval_id, meta.inst_id, order.intent.side, 0.0, 0.0,
                        "REJECTED_MIN_SIZE")
        px = round_to_step(order.intent.price, meta.tick_size)
        body = {"instId": meta.inst_id, "tdMode": "cash", "side": order.intent.side,
                "ordType": "limit", "px": f"{px}", "sz": f"{qty}"}
        if self._client is None:
            raise ExecutionError(
                "Live OKX order submission requires an authenticated signed client; "
                "none configured. v1.0 ships the adapter unsigned by design."
            )
        resp = self._client.post("/api/v5/trade/order", body)
        items = resp.get("data") or []
        if not items or items[0].get("sCode") not in ("0", 0):
            raise ExecutionError(f"OKX order rejected: {resp}")
        ord_id = items[0]["ordId"]
        return self.poll_order(meta.inst_id, ord_id, order.approval_id)

    def poll_order(self, inst_id: str, ord_id: str, approval_id: str,
                   attempts: int = 10, delay: float = 0.5) -> Fill:
        last: dict = {}
        for _ in range(attempts):
            data = self._get("/api/v5/trade/order", {"instId": inst_id, "ordId": ord_id})
            items = data.get("data") or []
            if items:
                last = items[0]
                if last.get("state") in ("filled", "canceled"):
                    return Fill(ord_id, inst_id, last.get("side", ""),
                                float(last.get("accFillSz", 0) or 0),
                                float(last.get("avgPx", 0) or 0),
                                str(last.get("state")).upper(), raw=last)
            time.sleep(delay)
        return Fill(ord_id, inst_id, last.get("side", ""),
                    float(last.get("accFillSz", 0) or 0),
                    float(last.get("avgPx", 0) or 0), "PENDING", raw=last)

    def position(self, symbol: str) -> float:
        base = to_okx_symbol(symbol).split("-")[0]
        return self.available_balance(base)

    def reconcile(self, symbol: str, local_position: float, risk_engine) -> str:
        """Mismatch => STOP and ALERT via the Risk Engine. Never a silent guess."""
        exchange_pos = self.position(symbol)
        return risk_engine.report_position_mismatch(local_position, exchange_pos)
