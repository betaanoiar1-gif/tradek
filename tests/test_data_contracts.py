import numpy as np
import pandas as pd
import pytest

from trading_school_ai.config.settings import DataConfig
from trading_school_ai.data.dataset import build_dataset, validate_dataset, validate_frame, load_canonical
from trading_school_ai.data.errors import MissingDatasetError, OHLCValidityError
from trading_school_ai.data.gaps import GapRegistry
from trading_school_ai.data.hashing import file_sha256, payload_sha256
from .conftest import make_frame


def test_schema_requires_ohlcv():
    df = make_frame(100).drop(columns=["volume"])
    with pytest.raises(Exception):
        build_dataset(df)


def test_utc_enforced(synthetic_dataset):
    assert str(synthetic_dataset.frame.index.tz) == "UTC"


def test_monotonic_detection():
    df = make_frame(100)
    df = pd.concat([df.iloc[50:], df.iloc[:50]])
    with pytest.raises(OHLCValidityError):
        build_dataset(df)


def test_duplicate_timestamps_detected():
    df = make_frame(100)
    df = pd.concat([df, df.iloc[[50]]]).sort_values("open_time")
    errs = validate_frame(build_dataset(df, strict=False).frame)
    assert any("Duplicate" in e for e in errs)


def test_invalid_ohlc_detected():
    df = make_frame(100)
    df.loc[10, "high"] = df.loc[10, "low"] - 1.0
    with pytest.raises(OHLCValidityError):
        build_dataset(df)


def test_negative_volume_detected():
    df = make_frame(100)
    df.loc[5, "volume"] = -1.0
    with pytest.raises(OHLCValidityError):
        build_dataset(df)


def test_hash_verification(tmp_path):
    p = tmp_path / "x.parquet"
    make_frame(50).to_parquet(p)
    assert file_sha256(p) == file_sha256(p)
    assert payload_sha256({"a": 1}) == payload_sha256({"a": 1})


def test_gap_detection(gapped_dataset):
    assert len(gapped_dataset.gaps) == 1
    assert gapped_dataset.gaps.total_missing_bars() == 60


def test_missing_dataset_raises_and_does_not_substitute(tmp_path):
    cfg = DataConfig(canonical_path=str(tmp_path / "nope.parquet"))
    with pytest.raises(MissingDatasetError) as exc:
        load_canonical(cfg)
    assert "No substitute data" in str(exc.value)
    report = validate_dataset(cfg)
    assert not report.ok and "MISSING_DEPENDENCY" in report.errors[0]


def test_hash_mismatch_reported(tmp_path):
    p = tmp_path / "d.parquet"
    make_frame(200).to_parquet(p)
    cfg = DataConfig(canonical_path=str(p), gaps_path=str(tmp_path / "g.parquet"),
                     expected_sha256="0" * 64, enforce_hash=False)
    rep = validate_dataset(cfg)
    assert rep.hash_match is False
    assert any("HASH_MISMATCH" in w for w in rep.warnings)
    cfg.enforce_hash = True
    assert not validate_dataset(cfg).ok


def test_gap_registry_boundaries(gapped_dataset):
    flags = gapped_dataset.gaps.boundary_flags(gapped_dataset.frame.index)
    assert flags.sum() == 1
    assert gapped_dataset.frame.index[flags][0] == gapped_dataset.gaps.frame["gap_start"][0] - pd.Timedelta(minutes=1)
