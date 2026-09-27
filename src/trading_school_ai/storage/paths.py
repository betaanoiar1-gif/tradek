"""Runtime directory layout (local filesystem only)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config.settings import Settings


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    ledgers: Path
    reports: Path
    cache: Path
    logs: Path

    @staticmethod
    def build(settings: Settings, create: bool = True) -> "RuntimePaths":
        root = settings.runtime_path
        paths = RuntimePaths(
            root=root, ledgers=root / "ledgers", reports=root / "reports",
            cache=root / "cache", logs=root / "logs",
        )
        if create:
            for p in (paths.root, paths.ledgers, paths.reports, paths.cache, paths.logs):
                p.mkdir(parents=True, exist_ok=True)
        return paths
