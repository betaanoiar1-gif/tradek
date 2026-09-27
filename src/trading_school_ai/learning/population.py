"""Seeded population, mutation and crossover over bounded Genomes."""
from __future__ import annotations

import numpy as np

from ..strategies.genome import Condition, FeatureRef, Genome, RiskParams

WINDOWS = [5, 10, 14, 20, 30, 50, 60, 100, 200]
THRESHOLD_FEATURES = {
    "rsi": [20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 65.0, 70.0, 75.0, 80.0],
    "zscore": [-2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5],
    "roc": [-0.01, -0.005, -0.002, 0.002, 0.005, 0.01],
    "vol": [0.0005, 0.001, 0.002, 0.005],
}
TREND_FEATURES = ["sma", "ema"]
HOLDING_CHOICES = [60, 120, 240, 480, 720, 1440]


def _threshold_condition(rng: np.random.Generator, entry: bool) -> Condition:
    name = str(rng.choice(list(THRESHOLD_FEATURES)))
    window = int(rng.choice(WINDOWS))
    thr = float(rng.choice(THRESHOLD_FEATURES[name]))
    op = "<" if (entry ^ (rng.random() < 0.5)) else ">"
    return Condition(
        left=FeatureRef(name=name, params={"window": window}),
        op=op, right_kind="value", right_value=thr,
    )


def _cross_condition(rng: np.random.Generator) -> Condition:
    fast = int(rng.choice(WINDOWS[:5]))
    slow = int(rng.choice(WINDOWS[4:]))
    name = str(rng.choice(TREND_FEATURES))
    op = "cross_above" if rng.random() < 0.5 else "cross_below"
    return Condition(
        left=FeatureRef(name=name, params={"window": fast}),
        op=op, right_kind="feature",
        right_feature=FeatureRef(name=name, params={"window": slow}),
    )


def random_genome(rng: np.random.Generator, index: int = 0) -> Genome:
    entry = [_cross_condition(rng) if rng.random() < 0.5 else _threshold_condition(rng, True)]
    exit_ = [_threshold_condition(rng, False)]
    risk = RiskParams(
        stop_atr_mult=float(np.round(rng.uniform(0.5, 5.0), 2)),
        take_profit_atr_mult=float(np.round(rng.uniform(0.5, 8.0), 2)),
        atr_window=int(rng.choice([7, 14, 21, 30])),
        position_fraction=float(np.round(rng.uniform(0.2, 1.0), 2)),
    )
    return Genome(
        side="long" if rng.random() < 0.7 else "short",
        entry=entry, exit=exit_, risk=risk,
        max_holding_bars=int(rng.choice(HOLDING_CHOICES)),
        label=f"seed-{index}",
    )


def seed_population(size: int, seed: int) -> list[Genome]:
    rng = np.random.default_rng(seed)
    return [random_genome(rng, i) for i in range(size)]


def mutate(genome: Genome, rng: np.random.Generator) -> Genome:
    data = genome.model_dump()
    choice = rng.integers(0, 4)
    if choice == 0 and data["entry"]:
        data["entry"] = [_cross_condition(rng).model_dump()
                         if rng.random() < 0.5 else
                         _threshold_condition(rng, True).model_dump()]
    elif choice == 1:
        data["exit"] = [_threshold_condition(rng, False).model_dump()]
    elif choice == 2:
        r = dict(data["risk"])
        r["stop_atr_mult"] = float(np.clip(r["stop_atr_mult"] * rng.uniform(0.6, 1.6), 0.1, 20.0))
        r["take_profit_atr_mult"] = float(
            np.clip(r["take_profit_atr_mult"] * rng.uniform(0.6, 1.6), 0.1, 50.0))
        data["risk"] = r
    else:
        data["max_holding_bars"] = int(rng.choice(HOLDING_CHOICES))
    data["label"] = f"mut-{genome.genome_id[:6]}"
    return Genome(**data)


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    da, db = a.model_dump(), b.model_dump()
    child = {
        "side": da["side"] if rng.random() < 0.5 else db["side"],
        "entry": da["entry"] if rng.random() < 0.5 else db["entry"],
        "exit": db["exit"] if rng.random() < 0.5 else da["exit"],
        "filters": da["filters"] if rng.random() < 0.5 else db["filters"],
        "risk": da["risk"] if rng.random() < 0.5 else db["risk"],
        "max_holding_bars": da["max_holding_bars"] if rng.random() < 0.5
        else db["max_holding_bars"],
        "label": f"x-{a.genome_id[:4]}{b.genome_id[:4]}",
    }
    return Genome(**child)
