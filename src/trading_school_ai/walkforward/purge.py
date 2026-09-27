"""Chronological splitting with label-overlap purging and embargo.

Indexing conventions (canonical 1m bar indices, inclusive endpoints):
  feature_interval(i) = [i - L + 1, i]
  label_interval(i)   = (i, i + H]        (i.e. [i+1, i+H])

A training sample i is purged if and only if its LABEL interval intersects the
test interval [test_start, test_end]. Feature-window overlap alone never purges.
Embargo removes an additional `embargo_bars` samples AFTER the test interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_range: tuple[int, int]
    validation_range: tuple[int, int]
    test_range: tuple[int, int]


def chronological_split(
    n: int, train_frac: float, validation_frac: float
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 < train_frac < 1) or not (0 <= validation_frac < 1):
        raise ValueError("invalid fractions")
    if train_frac + validation_frac >= 1.0:
        raise ValueError("train_frac + validation_frac must be < 1")
    tr_end = int(n * train_frac) - 1
    va_end = int(n * (train_frac + validation_frac)) - 1
    return (0, tr_end), (tr_end + 1, va_end), (va_end + 1, n - 1)


def purge_mask(
    candidate_indices: np.ndarray,
    test_start: int,
    test_end: int,
    horizon: int,
) -> np.ndarray:
    """True where the sample must be PURGED (label interval hits the test set)."""
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    label_start = candidate_indices + 1
    label_end = candidate_indices + horizon
    return (label_start <= test_end) & (label_end >= test_start)


def embargo_mask(
    candidate_indices: np.ndarray, test_end: int, embargo_bars: int
) -> np.ndarray:
    """True where the sample falls inside the post-test embargo window."""
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0")
    if embargo_bars == 0:
        return np.zeros(len(candidate_indices), dtype=bool)
    return (candidate_indices > test_end) & (
        candidate_indices <= test_end + embargo_bars
    )


def build_split(
    n: int,
    lookback: int,
    horizon: int,
    train_frac: float,
    validation_frac: float,
    embargo_bars: int,
) -> Split:
    (tr0, tr1), (va0, va1), (te0, te1) = chronological_split(
        n, train_frac, validation_frac
    )
    all_idx = np.arange(n)
    # warmup: samples without a complete feature window are not decision samples
    warm = all_idx >= max(lookback - 1, 0)
    train = all_idx[(all_idx >= tr0) & (all_idx <= tr1) & warm]
    purged = purge_mask(train, te0, te1, horizon) | purge_mask(train, va0, va1, horizon)
    embargoed = embargo_mask(train, te1, embargo_bars) | embargo_mask(
        train, va1, embargo_bars
    )
    train = train[~(purged | embargoed)]
    validation = all_idx[(all_idx >= va0) & (all_idx <= va1) & warm]
    test = all_idx[(all_idx >= te0) & (all_idx <= te1) & warm]
    return Split(train, validation, test, (tr0, tr1), (va0, va1), (te0, te1))


def purge_count(
    n: int, lookback: int, horizon: int, train_frac: float, validation_frac: float
) -> int:
    """Number of train samples removed purely by label-overlap purging."""
    (tr0, tr1), (va0, va1), (te0, te1) = chronological_split(
        n, train_frac, validation_frac
    )
    idx = np.arange(tr0, tr1 + 1)
    idx = idx[idx >= max(lookback - 1, 0)]
    mask = purge_mask(idx, te0, te1, horizon) | purge_mask(idx, va0, va1, horizon)
    return int(mask.sum())
