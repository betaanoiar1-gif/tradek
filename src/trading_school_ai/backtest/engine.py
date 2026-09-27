"""Event-driven backtester over canonical 1m OHLCV.

Timing contract (applied here exactly once):
  signal on bar i  ->  decision_timestamp = close_time(i) + 1ms
                   ->  execution at the OPEN of the first 1m bar whose
                      open timestamp is strictly greater than decision_timestamp.
The next bar in the array is used only if it is the true next minute; otherwise
the first available later bar is used and, when a declared gap intervenes, the
entry is suppressed (no execution during / across missing intervals).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.settings import BacktestConfig, StrategyConfig
from ..data.dataset import CanonicalDataset
from ..strategies.genome import Genome
from ..strategies.signals import generate_signals
from .types import BacktestResult, ExitReason, Trade


def _minute_step(index: pd.DatetimeIndex) -> int:
    """Integer value of one minute in the index's own datetime resolution."""
    factors = {"s": 1, "ms": 10**3, "us": 10**6, "ns": 10**9}
    unit = getattr(index, "unit", "ns")
    if unit not in factors:
        raise ValueError(f"Unsupported datetime resolution {unit!r}")
    return 60 * factors[unit]


def _apply_slippage(price: float, side: int, bps: float, adverse: bool = True) -> float:
    """Slippage always moves the fill against the trader."""
    delta = price * bps / 10_000.0
    if adverse:
        return price + delta if side > 0 else price - delta
    return price


def run_backtest(
    dataset: CanonicalDataset,
    genome: Genome,
    bt_cfg: BacktestConfig,
    strat_cfg: StrategyConfig,
    fee_multiplier: float = 1.0,
    slippage_multiplier: float = 1.0,
    extra_slippage_bps: float = 0.0,
    execution_delay_bars: int = 0,
    signal_dropout_mask: np.ndarray | None = None,
    genome_id: str = "",
    experiment_id: str = "",
) -> BacktestResult:
    """Run a deterministic event-driven backtest. Returns real trades only."""
    frame = dataset.frame
    n = len(frame)
    if n == 0:
        return BacktestResult([], pd.Series(dtype="float64"), bt_cfg.initial_equity, 0, 0)

    sig = generate_signals(dataset, genome).frame
    entry_long = sig["entry_long"].to_numpy()
    entry_short = sig["entry_short"].to_numpy()
    exit_sig = sig["exit_signal"].to_numpy()
    atr = sig["atr"].to_numpy()
    if signal_dropout_mask is not None:
        keep = ~signal_dropout_mask
        entry_long = entry_long & keep
        entry_short = entry_short & keep

    if not bt_cfg.allow_short:
        entry_short = np.zeros(n, dtype=bool)

    o = frame["open"].to_numpy()
    h = frame["high"].to_numpy()
    l = frame["low"].to_numpy()
    c = frame["close"].to_numpy()
    ts = frame.index
    ts_ns = ts.asi8
    minute_step = _minute_step(ts)

    gap_flags = dataset.gaps.boundary_flags(ts)
    H = genome.holding_horizon(strat_cfg.strategy_max_holding_bars)
    fee = bt_cfg.fee_rate * fee_multiplier
    slip_bps = bt_cfg.slippage_bps * slippage_multiplier + extra_slippage_bps
    frac = min(bt_cfg.position_fraction, genome.risk.position_fraction)

    equity = np.empty(n, dtype="float64")
    cash = float(bt_cfg.initial_equity)
    trades: list[Trade] = []
    warnings: list[str] = []
    exposure_bars = 0

    in_pos = False
    side = 0
    qty = 0.0
    entry_price = 0.0
    entry_idx = -1
    entry_fee = 0.0
    entry_slip = 0.0
    stop_px = np.nan
    tp_px = np.nan
    pending: tuple[int, int] | None = None  # (execute_at_index, side)

    def next_exec_index(i: int) -> int | None:
        """First bar strictly after decision_timestamp of bar i, gap-aware."""
        j = i + 1 + execution_delay_bars
        if j >= n:
            return None
        expected = ts_ns[i] + minute_step * (1 + execution_delay_bars)
        if ts_ns[j] != expected:
            # a declared missing interval sits between: do not invent a candle
            return None
        return j

    for i in range(n):
        price = c[i]
        if in_pos:
            exposure_bars += 1

        # ---- exits are evaluated intrabar on the current bar -------------
        if in_pos:
            exit_price = None
            reason = None
            if side > 0:
                if not np.isnan(stop_px) and l[i] <= stop_px:
                    exit_price, reason = stop_px, ExitReason.STOP_LOSS
                elif not np.isnan(tp_px) and h[i] >= tp_px:
                    exit_price, reason = tp_px, ExitReason.TAKE_PROFIT
            else:
                if not np.isnan(stop_px) and h[i] >= stop_px:
                    exit_price, reason = stop_px, ExitReason.STOP_LOSS
                elif not np.isnan(tp_px) and l[i] <= tp_px:
                    exit_price, reason = tp_px, ExitReason.TAKE_PROFIT
            if reason is None and (i - entry_idx) >= H:
                exit_price, reason = c[i], ExitReason.TIME_EXIT
            if reason is None and exit_sig[i] and i > entry_idx:
                exit_price, reason = c[i], ExitReason.SIGNAL
            if reason is None and gap_flags[i]:
                if bt_cfg.gap_exit == "last_valid_close":
                    exit_price, reason = c[i], ExitReason.GAP_BOUNDARY
            if reason is None and i == n - 1:
                exit_price, reason = c[i], ExitReason.END_OF_DATA
            if reason is not None:
                fill = _apply_slippage(exit_price, -side, slip_bps)
                notional = fill * qty
                exit_fee = notional * fee
                gross = (fill - entry_price) * qty * side
                total_fees = entry_fee + exit_fee
                slip_cost = entry_slip + abs(fill - exit_price) * qty
                net = gross - total_fees
                cash += net
                trades.append(
                    Trade(
                        entry_time=ts[entry_idx],
                        exit_time=ts[i],
                        side="long" if side > 0 else "short",
                        entry_price=float(entry_price),
                        exit_price=float(fill),
                        quantity=float(qty),
                        fees=float(total_fees),
                        slippage_cost=float(slip_cost),
                        exit_reason=reason.value,
                        gross_pnl=float(gross),
                        net_pnl=float(net),
                        bars_held=int(i - entry_idx),
                        genome_id=genome_id or genome.genome_id,
                        experiment_id=experiment_id,
                    )
                )
                in_pos = False
                side = 0
                qty = 0.0

        # ---- pending entry executes at this bar's OPEN -------------------
        if pending is not None and pending[0] == i:
            _, want_side = pending
            pending = None
            blocked = in_pos and not bt_cfg.allow_overlapping_positions
            if not blocked and not gap_flags[i]:
                fill = _apply_slippage(o[i], want_side, slip_bps)
                equity_now = cash
                notional = equity_now * frac
                q = notional / fill if fill > 0 else 0.0
                a = atr[i - 1] if i > 0 else np.nan
                # A position is never opened without a valid stop distance:
                # ATR must be available at the decision bar.
                if q > 0 and not np.isnan(a):
                    in_pos = True
                    side = want_side
                    qty = q
                    entry_price = fill
                    entry_idx = i
                    entry_fee = notional * fee
                    entry_slip = abs(fill - o[i]) * q
                    cash -= entry_fee
                    stop_px = fill - a * genome.risk.stop_atr_mult * side
                    tp_px = fill + a * genome.risk.take_profit_atr_mult * side

        # ---- signal -> schedule execution --------------------------------
        if pending is None and (entry_long[i] or entry_short[i]):
            if not (in_pos and not bt_cfg.allow_overlapping_positions):
                want = 1 if entry_long[i] else -1
                j = next_exec_index(i)
                if j is not None:
                    pending = (j, want)

        mtm = cash
        if in_pos:
            mtm = cash + (price - entry_price) * qty * side
        equity[i] = mtm

    eq = pd.Series(equity, index=ts, name="equity")
    return BacktestResult(
        trades=trades,
        equity=eq,
        initial_equity=float(bt_cfg.initial_equity),
        bars=n,
        exposure_bars=exposure_bars,
        config_snapshot={
            "fee_rate": fee,
            "slippage_bps": slip_bps,
            "H": H,
            "L": genome.max_lookback(),
            "allow_short": bt_cfg.allow_short,
            "gap_exit": bt_cfg.gap_exit,
            "execution_delay_bars": execution_delay_bars,
        },
        warnings=warnings,
    )
