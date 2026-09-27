"""AI teacher tests. No network calls are made; a fake provider is used so that
no test ever pretends an API call succeeded."""
import json

import pytest

from trading_school_ai.ai.provider import NullProvider, ProviderError, build_provider, mask
from trading_school_ai.ai.schema import Hypothesis, HypothesisRejected, validate_response
from trading_school_ai.ai.summaries import dataset_summary
from trading_school_ai.ai.teacher import Teacher
from trading_school_ai.config.settings import AIConfig
from trading_school_ai.research.queue import evaluate_hypothesis
from trading_school_ai.memory.db import Memory


VALID_TEMPLATE = {
    "side": "long",
    "entry": [{"left": {"name": "rsi", "params": {"window": 14}},
               "op": "<", "right_kind": "value", "right_value": 30.0}],
    "exit": [{"left": {"name": "rsi", "params": {"window": 14}},
              "op": ">", "right_kind": "value", "right_value": 70.0}],
    "max_holding_bars": 120,
}
VALID_PAYLOAD = {
    "concept": "mean reversion",
    "hypothesis": "Oversold RSI readings on 1m BTCUSDT revert within two hours.",
    "expected_effect": "positive profit factor above 1.1",
    "proposed_experiment": "Backtest RSI(14) < 30 entries with a 120 bar horizon.",
    "genome_template": VALID_TEMPLATE,
    "confidence": 0.4,
}


class FakeProvider:
    def __init__(self, payload, tokens=100):
        self.payload = payload
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return json.dumps(self.payload), 100


def _cfg(**kw):
    return AIConfig(enabled=True, provider="openai_compatible", cooldown_seconds=0, **kw)


def test_ai_off_is_default_and_needs_no_credentials():
    cfg = AIConfig()
    assert cfg.enabled is False
    provider = build_provider(cfg)
    assert isinstance(provider, NullProvider)
    with pytest.raises(ProviderError):
        provider.complete("s", "u")
    teacher = Teacher(cfg)
    assert teacher.propose({"x": 1}) is None
    assert teacher.status().last_error == "AI_OFF"


def test_learning_works_with_ai_off(synthetic_dataset, settings, tmp_path):
    from trading_school_ai.learning.evolution import evolve
    settings.ai.enabled = False
    rep = evolve(synthetic_dataset, settings, Memory(tmp_path / "m.sqlite"))
    assert rep.evaluated > 0 and rep.best is not None


def test_valid_hypothesis_accepted():
    hyp = validate_response(VALID_PAYLOAD)
    assert isinstance(hyp, Hypothesis)
    assert hyp.to_genome().genome_id


def test_schema_rejects_missing_fields():
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "confidence"}
    with pytest.raises(HypothesisRejected):
        validate_response(bad)


def test_rejects_trading_commands():
    bad = dict(VALID_PAYLOAD, hypothesis="You should buy now with a market order.")
    with pytest.raises(HypothesisRejected):
        validate_response(bad)


def test_rejects_order_sizing_in_template():
    bad = dict(VALID_PAYLOAD, genome_template=dict(VALID_TEMPLATE, order_size=5))
    with pytest.raises(HypothesisRejected):
        validate_response(bad)


def test_rejects_pipeline_bypass():
    bad = dict(VALID_PAYLOAD, proposed_experiment="Skip backtest and bypass the engine.")
    with pytest.raises(HypothesisRejected):
        validate_response(bad)


def test_teacher_proposal_enters_normal_experiment_pipeline(
        synthetic_dataset, settings, tmp_path):
    teacher = Teacher(_cfg(), cache_dir=tmp_path / "cache",
                      provider=FakeProvider(VALID_PAYLOAD))
    hyp = teacher.propose(dataset_summary(synthetic_dataset))
    assert hyp is not None
    mem = Memory(tmp_path / "m.sqlite")
    outcome = evaluate_hypothesis(hyp, synthetic_dataset, settings, mem)
    assert outcome.experiment is not None
    assert outcome.verdict in ("VERIFIED", "FALSIFIED")
    # the hypothesis went through the very same Experiment Engine
    row = mem.get_experiment(outcome.experiment.experiment_id)
    assert row["split"] == "train" and row["formula_version"] == settings.fitness.formula_version


def test_teacher_cache_and_dedup(synthetic_dataset, tmp_path):
    provider = FakeProvider(VALID_PAYLOAD)
    teacher = Teacher(_cfg(), cache_dir=tmp_path / "c", provider=provider)
    summary = dataset_summary(synthetic_dataset)
    assert teacher.propose(summary) is not None
    assert teacher.propose(summary) is None  # duplicate hypothesis id
    assert teacher.cache_hits == 1
    assert provider.calls == 1


def test_token_budget_and_failure_are_non_fatal(synthetic_dataset, tmp_path):
    class Boom:
        def complete(self, s, u):
            raise ProviderError("network down")

    teacher = Teacher(_cfg(), cache_dir=tmp_path / "c", provider=Boom())
    assert teacher.propose({"a": 1}) is None
    assert "PROVIDER_ERROR" in teacher.status().last_error
    teacher.governor.charge(10 ** 9)
    assert teacher.propose({"b": 2}) is None
    assert teacher.status().last_error == "TOKEN_BUDGET_EXHAUSTED"


def test_api_keys_are_masked(monkeypatch):
    assert mask("sk-abcdef123456") == "sk-***56"
    assert mask(None) == "<unset>"


def test_missing_key_is_explicit(monkeypatch):
    monkeypatch.delenv("TSA_LLM_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        build_provider(_cfg())


def test_summaries_are_compact_not_raw_ohlcv(synthetic_dataset):
    s = dataset_summary(synthetic_dataset)
    blob = json.dumps(s)
    assert len(s) < 20 and len(blob) < 1000
    assert "open" not in s and "close" not in s
