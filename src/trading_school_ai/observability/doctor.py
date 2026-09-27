"""Environment and contract diagnostics used by `tsa doctor`."""
from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

from ..config.settings import Settings
from ..data.dataset import validate_dataset
from ..drive.sync import drive_available
from ..storage.paths import RuntimePaths


def run_doctor(settings: Settings) -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, severity: str = "error"):
        checks.append({"check": name, "ok": bool(ok), "detail": detail,
                       "severity": "info" if ok else severity})

    add("python", sys.version_info >= (3, 10),
        f"{platform.python_version()} on {platform.system()}")
    for mod in ("numpy", "pandas", "pyarrow", "typer", "pydantic"):
        try:
            m = importlib.import_module(mod)
            add(f"dep:{mod}", True, getattr(m, "__version__", "unknown"))
        except ImportError as exc:
            add(f"dep:{mod}", False, str(exc))
    try:
        importlib.import_module("httpx")
        add("dep:httpx(optional)", True, "available (AI adapter usable)", "warning")
    except ImportError:
        add("dep:httpx(optional)", False,
            "not installed - AI teacher unavailable, AI_OFF unaffected", "warning")

    paths = RuntimePaths.build(settings, create=True)
    add("runtime_dir", paths.root.exists(), str(paths.root))
    db = settings.db_path
    add("sqlite_local", "/drive/" not in str(db).lower(),
        f"{db} (runtime DB must not live on Drive)")

    report = validate_dataset(settings.data)
    add("dataset", report.ok,
        f"rows={report.rows} gaps={report.gap_count} "
        f"hash_match={report.hash_match} errors={report.errors}",
        "error")
    for w in report.warnings:
        add("dataset_warning", False, w, "warning")

    drive_dir = settings.storage.drive_backup_dir
    add("drive_backup", drive_available(drive_dir),
        f"{drive_dir} (optional; configure storage.drive_backup_dir)", "warning")

    add("ai_mode", True,
        "AI_OFF (default)" if not settings.ai.enabled else f"AI_ON provider={settings.ai.provider}")
    key_present = bool(os.environ.get(settings.ai.api_key_env))
    add("ai_key", True,
        f"{settings.ai.api_key_env} {'present' if key_present else 'absent'} "
        "(not required for AI_OFF)")
    live_ok = (not settings.execution.live_enabled) and os.environ.get("LIVE_CONFIRM") != "1"
    add("live_locks", True,
        f"execution.mode={settings.execution.mode} live_enabled="
        f"{settings.execution.live_enabled} LIVE_CONFIRM="
        f"{os.environ.get('LIVE_CONFIRM', 'unset')} -> "
        f"{'LIVE DISABLED' if live_ok else 'LIVE PARTIALLY UNLOCKED'}")

    errors = [c for c in checks if not c["ok"] and c["severity"] == "error"]
    return {
        "ok": not errors,
        "config_hash": settings.config_hash(),
        "checks": checks,
        "dataset_report": report.to_dict(),
    }
