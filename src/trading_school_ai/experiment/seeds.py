"""Deterministic seed derivation. No UUIDs, no wall-clock time in the seed path."""
from __future__ import annotations

import hashlib

MASK64 = (1 << 64) - 1


def derive_seed(*parts: object) -> int:
    """Stable 63-bit seed from arbitrary structured parts."""
    blob = "|".join(str(p) for p in parts).encode()
    digest = hashlib.sha256(blob).digest()
    return int.from_bytes(digest[:8], "big") & (MASK64 >> 1)


def experiment_id(
    genome_hash: str, dataset_hash: str, config_hash: str, counter: int, split: str
) -> str:
    blob = f"{genome_hash}|{dataset_hash}|{config_hash}|{counter}|{split}".encode()
    return hashlib.sha256(blob).hexdigest()[:24]
