"""Reporting reads PERSISTED results. It never silently recomputes experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ..memory.db import Memory


class ReportError(Exception):
    pass


def experiment_report(memory: Memory, experiment_id: str) -> dict:
    row = memory.get_experiment(experiment_id)
    if row is None:
        raise ReportError(
            f"Experiment {experiment_id} is not present in memory. "
            "Reporting does not recompute experiments; run it first."
        )
    metrics = json.loads(row["metrics_json"] or "{}")
    ledger_rows = None
    if row.get("ledger_path") and Path(row["ledger_path"]).exists():
        ledger_rows = int(len(pd.read_parquet(row["ledger_path"])))
    return {
        "experiment_id": row["experiment_id"],
        "genome_id": row["genome_id"],
        "split": row["split"],
        "status": row["status"],
        "fitness": row["fitness"],
        "fitness_accepted": bool(row["fitness_accepted"]),
        "rejection_reason": row["rejection_reason"],
        "formula_version": row["formula_version"],
        "dataset_hash": row["dataset_hash"],
        "dataset_source": row["dataset_source"],
        "config_hash": row["config_hash"],
        "code_version": row["code_version"],
        "git_commit": row["git_commit"],
        "seed": row["seed"],
        "metrics": metrics,
        "ledger_path": row["ledger_path"],
        "ledger_rows": ledger_rows,
    }


def leaderboard(memory: Memory, limit: int = 10, split: str = "train") -> list[dict]:
    rows = memory.top_experiments(limit=limit, split=split)
    out = []
    for r in rows:
        m = json.loads(r["metrics_json"] or "{}")
        out.append({
            "experiment_id": r["experiment_id"], "genome_id": r["genome_id"],
            "fitness": round(float(r["fitness"]), 6),
            "total_return": round(float(m.get("total_return", 0.0)), 6),
            "profit_factor": round(float(m.get("profit_factor", 0.0)), 4),
            "trades": m.get("trade_count", 0),
            "max_drawdown": round(float(m.get("max_drawdown", 0.0)), 4),
        })
    return out


def render_markdown(memory: Memory, limit: int = 10) -> str:
    rows = leaderboard(memory, limit=limit)
    summary = memory.summary()
    lines = ["# Trading School AI - Experiment Report", "",
             f"- Database: `{summary['db_path']}`",
             f"- Experiments recorded: {summary['experiments']}",
             f"- Accepted by fitness gates: {summary['accepted']}",
             f"- Hypotheses recorded: {summary['hypotheses']}", "",
             "## Train leaderboard (selection metric)", ""]
    if not rows:
        lines.append("_No accepted experiments recorded yet._")
    else:
        lines.append("| experiment_id | genome_id | fitness | return | PF | trades | maxDD |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['experiment_id']} | {r['genome_id']} | {r['fitness']} | "
                f"{r['total_return']} | {r['profit_factor']} | {r['trades']} | "
                f"{r['max_drawdown']} |")
    lines += ["", "## Failure modes", ""]
    for f in summary["failure_modes"]:
        lines.append(f"- `{f['rejection_reason']}`: {f['n']}")
    return "\n".join(lines) + "\n"
