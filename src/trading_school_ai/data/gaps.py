"""Gap registry: declared missing intervals in canonical 1m data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import SchemaError

GAP_COLUMNS = ("gap_start", "gap_end")


@dataclass(frozen=True)
class GapRegistry:
    """Half-open missing intervals [gap_start, gap_end) in UTC."""

    frame: pd.DataFrame

    @staticmethod
    def empty() -> "GapRegistry":
        return GapRegistry(
            pd.DataFrame(
                {
                    "gap_start": pd.Series([], dtype="datetime64[ns, UTC]"),
                    "gap_end": pd.Series([], dtype="datetime64[ns, UTC]"),
                }
            )
        )

    @staticmethod
    def from_frame(df: pd.DataFrame) -> "GapRegistry":
        cols = {c.lower(): c for c in df.columns}
        start = cols.get("gap_start") or cols.get("start") or cols.get("missing_start")
        end = cols.get("gap_end") or cols.get("end") or cols.get("missing_end")
        if start is None or end is None:
            raise SchemaError(
                f"Gap registry must expose start/end columns, got {list(df.columns)}"
            )
        out = pd.DataFrame(
            {
                "gap_start": pd.to_datetime(df[start], utc=True),
                "gap_end": pd.to_datetime(df[end], utc=True),
            }
        )
        if (out["gap_end"] <= out["gap_start"]).any():
            raise SchemaError("Gap registry contains non-positive intervals")
        out = out.sort_values("gap_start").reset_index(drop=True)
        return GapRegistry(out)

    @staticmethod
    def read_parquet(path: str | Path) -> "GapRegistry":
        return GapRegistry.from_frame(pd.read_parquet(path))

    @staticmethod
    def derive_from_index(index: pd.DatetimeIndex, freq: str = "1min") -> "GapRegistry":
        """Derive gaps from holes in an otherwise regular timestamp index."""
        if len(index) < 2:
            return GapRegistry.empty()
        step = pd.Timedelta(freq)
        diffs = index[1:] - index[:-1]
        hole = diffs > step
        starts = index[:-1][hole] + step
        ends = index[1:][hole]
        return GapRegistry.from_frame(
            pd.DataFrame({"gap_start": starts, "gap_end": ends})
        )

    def __len__(self) -> int:
        return len(self.frame)

    def total_missing_bars(self, freq: str = "1min") -> int:
        if self.frame.empty:
            return 0
        step = pd.Timedelta(freq)
        return int(((self.frame["gap_end"] - self.frame["gap_start"]) // step).sum())

    def covers(self, ts: pd.Timestamp) -> bool:
        if self.frame.empty:
            return False
        s = self.frame["gap_start"]
        e = self.frame["gap_end"]
        return bool(((s <= ts) & (ts < e)).any())

    def boundary_flags(self, index: pd.DatetimeIndex) -> np.ndarray:
        """True at each bar that is the last valid bar immediately before a gap."""
        flags = np.zeros(len(index), dtype=bool)
        if self.frame.empty or len(index) == 0:
            return flags
        pos = index.searchsorted(self.frame["gap_start"].to_numpy(), side="left") - 1
        for p in pos:
            if 0 <= p < len(index):
                flags[p] = True
        return flags

    def segment_ids(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Contiguous-data segment id per bar; increments after every gap."""
        ids = np.zeros(len(index), dtype=np.int64)
        if self.frame.empty or len(index) == 0:
            return ids
        starts = self.frame["gap_start"].to_numpy()
        ids = index.searchsorted(starts, side="left")
        counts = np.zeros(len(index), dtype=np.int64)
        for p in ids:
            if 0 <= p < len(index):
                counts[p:] += 1
        return counts
