"""Canonical dataset loading and validation. No silent repair, ever."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config.settings import DataConfig
from .errors import (
    MissingDatasetError,
    OHLCValidityError,
    SchemaError,
    TemporalContractError,
)
from .gaps import GapRegistry
from .hashing import file_sha256

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
TIMESTAMP_CANDIDATES = ("open_time", "timestamp", "time", "date", "datetime")


@dataclass
class ValidationReport:
    path: str
    rows: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    sha256: Optional[str]
    expected_sha256: Optional[str]
    hash_match: Optional[bool]
    gap_count: int
    missing_bars: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "rows": self.rows,
            "start": str(self.start),
            "end": str(self.end),
            "sha256": self.sha256,
            "expected_sha256": self.expected_sha256,
            "hash_match": self.hash_match,
            "gap_count": self.gap_count,
            "missing_bars": self.missing_bars,
            "errors": self.errors,
            "warnings": self.warnings,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class CanonicalDataset:
    """Immutable view of validated canonical 1m OHLCV data."""

    frame: pd.DataFrame
    gaps: GapRegistry
    sha256: Optional[str]
    source_path: Optional[str]

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index

    def __len__(self) -> int:
        return len(self.frame)

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> "CanonicalDataset":
        sub = self.frame.loc[start:end]
        return CanonicalDataset(sub, self.gaps, self.sha256, self.source_path)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    ts_col = None
    for cand in TIMESTAMP_CANDIDATES:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "open_time"})
            ts_col = "open_time"
        else:
            raise SchemaError(
                f"No timestamp column found; expected one of {TIMESTAMP_CANDIDATES}"
            )
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required OHLCV columns: {missing}")
    ts = df[ts_col]
    if pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
        unit = "ms" if float(ts.iloc[0]) > 1e11 else "s"
        idx = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(ts, utc=True)
    out = df[list(REQUIRED_COLUMNS)].astype("float64")
    out.index = pd.DatetimeIndex(idx, name="open_time")
    return out


def validate_frame(df: pd.DataFrame) -> list[str]:
    """Return a list of contract violations (empty list == valid)."""
    errors: list[str] = []
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return ["Index is not a DatetimeIndex"]
    if idx.tz is None or str(idx.tz) not in ("UTC", "utc"):
        errors.append(f"Index timezone must be UTC, got {idx.tz}")
    if idx.has_duplicates:
        errors.append(f"Duplicate timestamps: {int(idx.duplicated().sum())}")
    if not idx.is_monotonic_increasing:
        errors.append("Timestamps are not monotonically increasing")
    o, h, l, c, v = (df[k].to_numpy() for k in REQUIRED_COLUMNS)
    if np.isnan(df[list(REQUIRED_COLUMNS)].to_numpy()).any():
        errors.append("NaN values present in OHLCV columns")
    bad_hl = int(np.sum(h < l))
    if bad_hl:
        errors.append(f"high < low in {bad_hl} rows")
    bad_range = int(np.sum((o > h) | (o < l) | (c > h) | (c < l)))
    if bad_range:
        errors.append(f"open/close outside [low, high] in {bad_range} rows")
    if int(np.sum(v < 0)):
        errors.append(f"negative volume in {int(np.sum(v < 0))} rows")
    if int(np.sum((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0))):
        errors.append("non-positive prices present")
    return errors


def build_dataset(
    df: pd.DataFrame,
    gaps: GapRegistry | None = None,
    sha256: str | None = None,
    source_path: str | None = None,
    strict: bool = True,
) -> CanonicalDataset:
    frame = _normalize(df)
    errors = validate_frame(frame)
    if errors and strict:
        raise OHLCValidityError("; ".join(errors))
    registry = gaps if gaps is not None else GapRegistry.derive_from_index(frame.index)
    return CanonicalDataset(frame, registry, sha256, source_path)


def load_canonical(cfg: DataConfig, strict: bool = True) -> CanonicalDataset:
    """Load the canonical dataset. Raises rather than substituting data."""
    path = Path(cfg.canonical_path).expanduser()
    if not path.exists():
        raise MissingDatasetError(
            "Canonical dataset not found at "
            f"{path}. Set data.canonical_path in config or TSA_CANONICAL_PATH. "
            "No substitute data will be downloaded or synthesized."
        )
    df = pd.read_parquet(path)
    digest = file_sha256(path)
    if cfg.enforce_hash and cfg.expected_sha256 and digest != cfg.expected_sha256:
        raise SchemaError(
            f"Dataset hash mismatch: expected {cfg.expected_sha256}, got {digest}"
        )
    gaps_path = Path(cfg.gaps_path).expanduser()
    gaps = GapRegistry.read_parquet(gaps_path) if gaps_path.exists() else None
    frame = _normalize(df)
    errors = validate_frame(frame)
    if errors and strict:
        raise TemporalContractError("; ".join(errors))
    if gaps is None:
        gaps = GapRegistry.derive_from_index(frame.index)
    return CanonicalDataset(frame, gaps, digest, str(path))


def validate_dataset(cfg: DataConfig) -> ValidationReport:
    """Non-raising validation used by `tsa data validate` and `tsa doctor`."""
    path = Path(cfg.canonical_path).expanduser()
    report = ValidationReport(
        path=str(path),
        rows=0,
        start=None,
        end=None,
        sha256=None,
        expected_sha256=cfg.expected_sha256,
        hash_match=None,
        gap_count=0,
        missing_bars=0,
    )
    if not path.exists():
        report.errors.append(f"MISSING_DEPENDENCY: canonical dataset not found at {path}")
        return report
    try:
        raw = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        report.errors.append(f"READ_FAILURE: {exc}")
        return report
    report.sha256 = file_sha256(path)
    if cfg.expected_sha256:
        report.hash_match = report.sha256 == cfg.expected_sha256
        if not report.hash_match:
            msg = (
                f"HASH_MISMATCH: expected {cfg.expected_sha256} got {report.sha256}"
            )
            (report.errors if cfg.enforce_hash else report.warnings).append(msg)
    try:
        frame = _normalize(raw)
    except SchemaError as exc:
        report.errors.append(f"SCHEMA: {exc}")
        return report
    report.rows = len(frame)
    if report.rows:
        report.start, report.end = frame.index[0], frame.index[-1]
    report.errors.extend(validate_frame(frame))
    gaps_path = Path(cfg.gaps_path).expanduser()
    if gaps_path.exists():
        try:
            gaps = GapRegistry.read_parquet(gaps_path)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"GAP_REGISTRY: {exc}")
            return report
        derived = GapRegistry.derive_from_index(frame.index)
        if len(derived) != len(gaps):
            report.warnings.append(
                f"GAP_REGISTRY_MISMATCH: declared {len(gaps)} gaps, "
                f"index implies {len(derived)}"
            )
    else:
        report.warnings.append(f"GAP_REGISTRY_ABSENT: {gaps_path}; gaps derived from index")
        gaps = GapRegistry.derive_from_index(frame.index)
    report.gap_count = len(gaps)
    report.missing_bars = gaps.total_missing_bars()
    return report
