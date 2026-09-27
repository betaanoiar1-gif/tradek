"""Drive sync: copy + checksum verify + lock. Drive is a backup target only."""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..data.hashing import file_sha256


class SyncError(Exception):
    pass


@dataclass
class SyncResult:
    copied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    verified: int = 0

    def to_dict(self) -> dict:
        return {"copied": self.copied, "skipped": self.skipped,
                "failed": self.failed, "verified": self.verified,
                "ok": not self.failed}


class DirLock:
    """Cooperative lock file so two runtimes cannot sync the same dir at once."""

    def __init__(self, path: Path, stale_seconds: int = 3600):
        self.path = Path(path) / ".tsa-sync.lock"
        self.stale_seconds = stale_seconds

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age < self.stale_seconds:
                raise SyncError(f"Sync lock held by another process: {self.path}")
            self.path.unlink()
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        if self.path.exists():
            self.path.unlink()


def sync_to_drive(local_dir: str | Path, drive_dir: str | Path,
                  overwrite: bool = False) -> SyncResult:
    """Copy local artifacts to Drive and verify checksums.

    Existing valid backups are never overwritten unless overwrite=True.
    """
    src = Path(local_dir)
    dst = Path(drive_dir)
    if not src.exists():
        raise SyncError(f"Local source directory does not exist: {src}")
    try:
        dst.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SyncError(
            f"Drive destination unavailable: {dst} ({exc}). "
            "Configure storage.drive_backup_dir to a writable path."
        ) from exc
    result = SyncResult()
    with DirLock(dst):
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            if path.name == ".tsa-sync.lock":
                continue
            rel = path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = file_sha256(path)
            if target.exists():
                if file_sha256(target) == digest:
                    result.skipped.append(str(rel))
                    result.verified += 1
                    continue
                if not overwrite:
                    result.skipped.append(f"{rel} (differs; overwrite=False)")
                    continue
            try:
                shutil.copy2(path, target)
                if file_sha256(target) != digest:
                    result.failed.append(f"{rel} (checksum mismatch after copy)")
                else:
                    result.copied.append(str(rel))
                    result.verified += 1
            except OSError as exc:
                result.failed.append(f"{rel} ({exc})")
    return result


def restore_from_drive(drive_dir: str | Path, local_dir: str | Path,
                       overwrite: bool = False) -> SyncResult:
    """Explicit restore in the opposite direction."""
    return sync_to_drive(drive_dir, local_dir, overwrite=overwrite)


def drive_available(drive_dir: str | Path) -> bool:
    p = Path(drive_dir)
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".tsa-probe"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False
