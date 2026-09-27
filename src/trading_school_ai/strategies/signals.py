"""Signal generation. Produces entry/exit intents WITHOUT any execution shift."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.dataset import CanonicalDataset
from ..features.registry import compute_features
from .genome import Condition, Genome

LONG, FLAT, SHORT = 1, 0, -1


def _series_for(cond_side: str, values: pd.DataFrame, ref_key: str, close: pd.Series):
    if ref_key == "__close__":
        return close
    return values[ref_key]


def evaluate_condition(
    cond: Condition, values: pd.DataFrame, close: pd.Series
) -> pd.Series:
    left = values[cond.left.key]
    if cond.right_kind == "feature":
        right = values[cond.right_feature.key]
    elif cond.right_kind == "close":
        right = close
    else:
        right = pd.Series(cond.right_value, index=left.index)
    if cond.op == "<":
        out = left < right
    elif cond.op == "<=":
        out = left <= right
    elif cond.op == ">":
        out = left > right
    elif cond.op == ">=":
        out = left >= right
    elif cond.op == "cross_above":
        out = (left > right) & (left.shift(1) <= right.shift(1))
    elif cond.op == "cross_below":
        out = (left < right) & (left.shift(1) >= right.shift(1))
    else:  # pragma: no cover - Operator literal prevents this
        raise ValueError(f"Unsupported operator {cond.op}")
    invalid = left.isna() | right.isna()
    if cond.op in ("cross_above", "cross_below"):
        invalid = invalid | left.shift(1).isna() | right.shift(1).isna()
    return out.where(~invalid, False) & ~invalid


def _all_conditions(conds: list[Condition], values, close) -> pd.Series:
    if not conds:
        return pd.Series(True, index=close.index)
    acc = None
    for c in conds:
        s = evaluate_condition(c, values, close)
        acc = s if acc is None else (acc & s)
    return acc


class SignalFrame:
    """Signals indexed by the 1m bar that produced them.

    `decision_timestamp` = bar close_time + 1ms. The execution layer converts
    this into the next valid 1m open. Nothing here is shifted.
    """

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    @property
    def entry_long(self) -> np.ndarray:
        return self.frame["entry_long"].to_numpy()

    @property
    def entry_short(self) -> np.ndarray:
        return self.frame["entry_short"].to_numpy()

    @property
    def exit_signal(self) -> np.ndarray:
        return self.frame["exit_signal"].to_numpy()


def generate_signals(dataset: CanonicalDataset, genome: Genome) -> SignalFrame:
    specs = [f.to_spec() for f in genome.features()]
    values = compute_features(dataset, specs)
    close = dataset.frame["close"]
    entry = _all_conditions(genome.entry, values, close)
    filt = _all_conditions(genome.filters, values, close)
    raw_entry = entry & filt
    exit_sig = (
        _all_conditions(genome.exit, values, close)
        if genome.exit
        else pd.Series(False, index=close.index)
    )
    idx = dataset.frame.index
    one_min = pd.Timedelta(minutes=1)
    close_time = idx + one_min - pd.Timedelta("1ms")
    decision_ts = close_time + pd.Timedelta("1ms")
    # Side mapping (v1.0): 'long' -> long entries only, 'short' -> short entries
    # only, 'both' -> the trigger opens long and the exit condition opens short.
    if genome.side == "long":
        e_long, e_short = raw_entry, pd.Series(False, index=close.index)
    elif genome.side == "short":
        e_long, e_short = pd.Series(False, index=close.index), raw_entry
    else:
        e_long = raw_entry
        e_short = exit_sig & filt
    idx = dataset.frame.index
    one_min = pd.Timedelta(minutes=1)
    close_time = idx + one_min - pd.Timedelta("1ms")
    out = pd.DataFrame(
        {
            "entry_long": e_long.to_numpy(),
            "entry_short": e_short.to_numpy(),
            "exit_signal": exit_sig.to_numpy(),
            "close_time": close_time,
            "decision_timestamp": close_time + pd.Timedelta("1ms"),
        },
        index=idx,
    )
    atr_key = f"atr(window={float(genome.risk.atr_window)})"
    if atr_key not in values.columns:
        atr_key = [c for c in values.columns if c.startswith("atr(")][0]
    out["atr"] = values[atr_key].to_numpy()
    return SignalFrame(out)
