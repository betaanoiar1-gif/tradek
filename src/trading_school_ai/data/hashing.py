"""Deterministic hashing for datasets and arbitrary payloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def payload_sha256(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()
