import json
import subprocess
import sys

import pandas as pd
import pytest
from typer.testing import CliRunner

from trading_school_ai.cli.main import app
from trading_school_ai.drive.sync import DirLock, SyncError, restore_from_drive, sync_to_drive
from .conftest import make_frame

runner = CliRunner()


def _write_dataset(tmp_path):
    p = tmp_path / "canon.parquet"
    df = make_frame(4000)
    df.to_parquet(p)
    return p


def _config(tmp_path, data_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"data:\n  canonical_path: {data_path}\n  gaps_path: {tmp_path/'gaps.parquet'}\n"
        f"  expected_sha256: null\n  enforce_hash: false\n"
        f"fitness:\n  min_trades: 1\n"
        f"evolution:\n  population_size: 4\n  generations: 1\n"
        f"storage:\n  runtime_dir: {tmp_path/'runtime'}\n"
        f"  drive_backup_dir: {tmp_path/'backup'}\n"
    )
    return cfg


def test_version_command():
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0 and json.loads(r.stdout)["code_version"]


def test_doctor_and_data_validate(tmp_path):
    cfg = _config(tmp_path, _write_dataset(tmp_path))
    r = runner.invoke(app, ["doctor", "-c", str(cfg)])
    payload = json.loads(r.stdout)
    assert payload["ok"] is True
    r2 = runner.invoke(app, ["data", "validate", "-c", str(cfg)])
    assert r2.exit_code == 0 and json.loads(r2.stdout)["rows"] == 4000


def test_doctor_reports_missing_dataset(tmp_path):
    cfg = _config(tmp_path, tmp_path / "absent.parquet")
    r = runner.invoke(app, ["doctor", "-c", str(cfg)])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["ok"] is False


def test_learn_backtest_report_reproduce_cycle(tmp_path):
    cfg = _config(tmp_path, _write_dataset(tmp_path))
    r = runner.invoke(app, ["learn", "--ai-off", "-c", str(cfg)])
    assert r.exit_code == 0, r.stdout
    out = json.loads(r.stdout)
    assert out["ai_mode"] == "AI_OFF" and out["evaluated"] > 0
    eid = out["best_experiment_id"]

    rep = runner.invoke(app, ["report", "-e", eid, "-c", str(cfg)])
    assert rep.exit_code == 0 and json.loads(rep.stdout)["experiment_id"] == eid

    rp = runner.invoke(app, ["reproduce", eid, "-c", str(cfg)])
    assert rp.exit_code == 0, rp.stdout
    assert json.loads(rp.stdout)["ok"] is True

    mem = runner.invoke(app, ["memory", "-c", str(cfg)])
    assert json.loads(mem.stdout)["experiments"] > 0

    rb = runner.invoke(app, ["robustness", "-e", eid, "-c", str(cfg)])
    assert rb.exit_code == 0, rb.stdout


def test_report_refuses_unknown_experiment(tmp_path):
    cfg = _config(tmp_path, _write_dataset(tmp_path))
    r = runner.invoke(app, ["report", "-e", "does-not-exist", "-c", str(cfg)])
    assert r.exit_code == 2


def test_sync_copies_and_verifies(tmp_path):
    src, dst = tmp_path / "local", tmp_path / "drive"
    src.mkdir()
    (src / "a.txt").write_text("hello")
    res = sync_to_drive(src, dst)
    assert res.copied == ["a.txt"] and res.verified == 1
    again = sync_to_drive(src, dst)
    assert again.skipped and not again.copied  # valid backup not overwritten
    (src / "a.txt").write_text("changed")
    no_over = sync_to_drive(src, dst)
    assert any("overwrite=False" in s for s in no_over.skipped)
    forced = sync_to_drive(src, dst, overwrite=True)
    assert forced.copied == ["a.txt"]
    back = restore_from_drive(dst, tmp_path / "restored")
    assert back.copied == ["a.txt"]


def test_sync_lock(tmp_path):
    d = tmp_path / "drive"
    d.mkdir()
    with DirLock(d):
        with pytest.raises(SyncError):
            with DirLock(d):
                pass


def test_sync_reports_missing_source(tmp_path):
    with pytest.raises(SyncError):
        sync_to_drive(tmp_path / "nope", tmp_path / "drive")
