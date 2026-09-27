"""Colab bootstrap: mount Drive, validate paths, initialize LOCAL runtime dirs.

Drive is used for the dataset and for backups only. SQLite and all active
runtime writes stay on the local Colab filesystem.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from ..config.settings import Settings, load_settings
from ..observability.doctor import run_doctor
from ..storage.paths import RuntimePaths

DEFAULT_LOCAL_RUNTIME = "/content/tsa_runtime"


def in_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").exists()


def mount_drive(mountpoint: str = "/content/drive") -> bool:
    """Mount Google Drive when running in Colab. Returns True on success."""
    if Path(mountpoint, "MyDrive").exists():
        return True
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return False
    drive.mount(mountpoint)
    return Path(mountpoint, "MyDrive").exists()


def bootstrap(config: str | None = None, runtime_dir: str | None = None) -> dict:
    """One-call setup: mount, configure local runtime, run doctor."""
    mounted = mount_drive() if in_colab() else False
    settings: Settings = load_settings(config)
    if runtime_dir:
        settings.storage.runtime_dir = runtime_dir
    elif in_colab():
        settings.storage.runtime_dir = DEFAULT_LOCAL_RUNTIME
    paths = RuntimePaths.build(settings, create=True)
    report = run_doctor(settings)
    return {
        "in_colab": in_colab(),
        "drive_mounted": mounted,
        "runtime_dir": str(paths.root),
        "db_path": str(settings.db_path),
        "canonical_path": settings.data.canonical_path,
        "gaps_path": settings.data.gaps_path,
        "doctor_ok": report["ok"],
        "doctor": report,
    }
