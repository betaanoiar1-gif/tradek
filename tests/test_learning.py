import json

import pytest

from trading_school_ai.experiment.engine import run_experiment, split_dataset
from trading_school_ai.experiment.seeds import derive_seed, experiment_id
from trading_school_ai.learning.evolution import evolve
from trading_school_ai.memory.db import Memory
from trading_school_ai.strategies.genome import Genome


def test_experiment_is_deterministic(synthetic_dataset, simple_genome, settings):
    a = run_experiment(synthetic_dataset, simple_genome, settings, counter=0)
    b = run_experiment(synthetic_dataset, simple_genome, settings, counter=0)
    assert a.experiment_id == b.experiment_id
    assert a.seed == b.seed
    assert a.metrics == b.metrics
    assert a.fitness == b.fitness


def test_seeds_do_not_use_uuid_or_clock():
    s1 = derive_seed("g", "d", "c", 3, "train")
    s2 = derive_seed("g", "d", "c", 3, "train")
    assert s1 == s2 and s1 >= 0
    assert experiment_id("g", "d", "c", 3, "train") == experiment_id("g", "d", "c", 3, "train")


def test_genome_ids_are_deterministic_and_content_addressed(simple_genome):
    clone = Genome.from_json(simple_genome.to_json())
    assert clone.genome_hash() == simple_genome.genome_hash()
    relabelled = Genome(**{**simple_genome.model_dump(), "label": "other"})
    assert relabelled.genome_hash() == simple_genome.genome_hash()


def test_splits_are_chronological_and_disjoint(synthetic_dataset, settings):
    tr = split_dataset(synthetic_dataset, settings, "train").frame
    va = split_dataset(synthetic_dataset, settings, "validation").frame
    te = split_dataset(synthetic_dataset, settings, "test").frame
    assert tr.index[-1] < va.index[0] < te.index[0]
    assert len(tr) + len(va) + len(te) == len(synthetic_dataset.frame)


def test_evolution_uses_train_only_for_selection(synthetic_dataset, settings, tmp_path):
    mem = Memory(tmp_path / "m.sqlite")
    rep = evolve(synthetic_dataset, settings, mem)
    assert rep.best is not None
    rows = [mem.get_experiment(a["experiment_id"]) for a in rep.archive]
    assert {r["split"] for r in rows} == {"train"}


def test_validation_is_observer_only(synthetic_dataset, settings, tmp_path):
    mem = Memory(tmp_path / "m.sqlite")
    rep = evolve(synthetic_dataset, settings, mem, observer_validation=True)
    best_by_train = max(rep.archive, key=lambda a: (a["fitness"], ))
    assert rep.best.train.experiment_id == best_by_train["experiment_id"]
    assert rep.best.validation is not None
    assert rep.best.validation.split == "validation"


def test_test_split_never_touched_during_learning(synthetic_dataset, settings, tmp_path):
    mem = Memory(tmp_path / "m.sqlite")
    evolve(synthetic_dataset, settings, mem)
    splits = {r["split"] for r in
              mem.conn.execute("SELECT DISTINCT split FROM experiments").fetchall()
              for r in [dict(r)]}
    assert "test" not in splits


def test_evolution_reproducible(synthetic_dataset, settings, tmp_path):
    r1 = evolve(synthetic_dataset, settings, Memory(tmp_path / "a.sqlite"))
    r2 = evolve(synthetic_dataset, settings, Memory(tmp_path / "b.sqlite"))
    assert [a["genome_id"] for a in r1.archive] == [a["genome_id"] for a in r2.archive]
    assert [a["fitness"] for a in r1.archive] == [a["fitness"] for a in r2.archive]


def test_serial_and_parallel_agree(synthetic_dataset, settings, tmp_path):
    settings.evolution.workers = 1
    serial = evolve(synthetic_dataset, settings, Memory(tmp_path / "s.sqlite"))
    settings.evolution.workers = 2
    parallel = evolve(synthetic_dataset, settings, Memory(tmp_path / "p.sqlite"))
    assert [a["experiment_id"] for a in serial.archive] == \
           [a["experiment_id"] for a in parallel.archive]
    assert [a["fitness"] for a in serial.archive] == [a["fitness"] for a in parallel.archive]


def test_invalid_genome_is_not_faked(synthetic_dataset, settings):
    with pytest.raises(Exception):
        Genome(entry=[])  # empty entry conditions are invalid


def test_memory_records_provenance(synthetic_dataset, simple_genome, settings, tmp_path):
    mem = Memory(tmp_path / "m.sqlite")
    res = run_experiment(synthetic_dataset, simple_genome, settings, counter=0)
    mem.record_experiment(res)
    row = mem.get_experiment(res.experiment_id)
    for key in ("seed", "dataset_hash", "feature_hash", "genome_hash", "config_hash",
                "code_version", "formula_version", "status", "genome_json"):
        assert key in row
    assert json.loads(row["metrics_json"])["trade_count"] == res.metrics["trade_count"]


def test_memory_refuses_drive_path(tmp_path):
    with pytest.raises(ValueError):
        Memory("/content/drive/MyDrive/x/memory.sqlite")


def test_reproduce_command_matches(synthetic_dataset, simple_genome, settings, tmp_path):
    from trading_school_ai.experiment.reproduce import reproduce
    mem = Memory(tmp_path / "m.sqlite")
    res = run_experiment(synthetic_dataset, simple_genome, settings, counter=0)
    mem.record_experiment(res)
    rep = reproduce(mem, res.experiment_id, synthetic_dataset, settings)
    assert rep.ok and rep.identical_id and rep.mismatches == {}


def test_experiment_records_dataset_provenance(synthetic_dataset, simple_genome,
                                               settings, tmp_path):
    """Every experiment records WHICH dataset produced it, so synthetic and real
    results can never be silently mixed in the same memory database."""
    from trading_school_ai.data.dataset import CanonicalDataset
    tagged = CanonicalDataset(synthetic_dataset.frame, synthetic_dataset.gaps,
                              "deadbeef", "/tmp/labelled-synthetic.parquet")
    mem = Memory(tmp_path / "m.sqlite")
    res = run_experiment(tagged, simple_genome, settings, counter=0)
    mem.record_experiment(res)
    row = mem.get_experiment(res.experiment_id)
    assert row["dataset_hash"] == "deadbeef"
    assert row["dataset_source"] == "/tmp/labelled-synthetic.parquet"


def test_memory_migrates_old_schema_without_data_loss(tmp_path):
    """A v1 database must keep working (additive migration only)."""
    import sqlite3
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """CREATE TABLE experiments (
             experiment_id TEXT PRIMARY KEY, counter INTEGER NOT NULL,
             seed INTEGER NOT NULL, split TEXT NOT NULL, genome_hash TEXT NOT NULL,
             genome_id TEXT NOT NULL, feature_hash TEXT NOT NULL, dataset_hash TEXT,
             config_hash TEXT NOT NULL, code_version TEXT NOT NULL, git_commit TEXT,
             fitness REAL NOT NULL, fitness_accepted INTEGER NOT NULL,
             rejection_reason TEXT, formula_version TEXT, status TEXT NOT NULL,
             ledger_path TEXT, genome_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
             error TEXT);""")
    conn.execute("INSERT INTO experiments VALUES ('old1',0,1,'train','g','gi','fh',"
                 "'dh','ch','1.0.0',NULL,1.0,1,'','fitness-v1','OK',NULL,'{}','{}','')")
    conn.commit()
    conn.close()
    mem = Memory(path)
    assert mem.get_experiment("old1")["fitness"] == 1.0
    assert "dataset_source" in mem.get_experiment("old1")


def test_no_winner_is_claimed_when_all_candidates_are_rejected(
        synthetic_dataset, settings, tmp_path):
    """If every candidate fails the fitness gates, `best` must be None."""
    settings.fitness.min_trades = 10 ** 9  # nothing can pass
    rep = evolve(synthetic_dataset, settings, Memory(tmp_path / "m.sqlite"))
    assert rep.best is None
    d = rep.to_dict()
    assert d["all_rejected"] is True
    assert d["best_genome_id"] is None and d["best_experiment_id"] is None
    assert sum(d["rejection_summary"].values()) == len(rep.archive)


def test_best_is_always_an_accepted_candidate(synthetic_dataset, settings, tmp_path):
    settings.fitness.min_trades = 1
    rep = evolve(synthetic_dataset, settings, Memory(tmp_path / "m.sqlite"))
    if rep.best is not None:
        assert rep.best.train.fitness_accepted is True
        assert rep.best.fitness == max(
            a["fitness"] for a in rep.archive
            if a["experiment_id"] == rep.best.train.experiment_id)
        assert rep.to_dict()["all_rejected"] is False
