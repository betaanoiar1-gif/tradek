"""AI transport failure must never break Python-only learning."""
import json

from typer.testing import CliRunner

from trading_school_ai.ai.teacher import Teacher
from trading_school_ai.cli.main import app
from trading_school_ai.config.settings import AIConfig
from .conftest import make_frame

runner = CliRunner()


def test_teacher_without_api_key_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("TSA_LLM_API_KEY", raising=False)
    t = Teacher(AIConfig(enabled=True, provider="openai_compatible"))
    assert t.propose({"a": 1}) is None
    assert "PROVIDER_UNAVAILABLE" in t.status().last_error


def test_learn_with_ai_still_completes_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("TSA_LLM_API_KEY", raising=False)
    data = tmp_path / "c.parquet"
    make_frame(4000).to_parquet(data)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"data:\n  canonical_path: {data}\n  gaps_path: {tmp_path/'g.parquet'}\n"
        f"  expected_sha256: null\nfitness:\n  min_trades: 1\n"
        f"evolution:\n  population_size: 4\n  generations: 1\n"
        f"storage:\n  runtime_dir: {tmp_path/'rt'}\n  drive_backup_dir: {tmp_path/'bk'}\n")
    r = runner.invoke(app, ["learn", "--with-ai", "-c", str(cfg)])
    assert r.exit_code == 0, r.stdout
    out = json.loads(r.stdout)
    assert out["evaluated"] > 0
    assert out["teacher"]["proposed"] is False
    assert "PROVIDER_UNAVAILABLE" in out["teacher"]["reason"]
