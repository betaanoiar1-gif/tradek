# Validation Report — Trading School AI v1.0

**Status: `CORE TESTS PASS`**

Not `INTEGRATION VERIFIED`, not `PAPER VALIDATION COMPLETE`, not `PRODUCTION CANDIDATE`.
Reason: the canonical BTCUSDT dataset was not present in this environment, so the four
real-data integration tests were skipped, and no paper-operation session against a live
venue was run.

Environment: Python 3.11.2, numpy/pandas/pyarrow/typer/pydantic, Linux sandbox.

## 1. Test command and actual results

```
$ pytest -q
125 passed, 4 skipped in 7.37s      (129 collected)
```

```
$ lint-imports
AI must not import risk/execution        KEPT
Risk must not depend on LLM              KEPT
Backtest must not use live execution     KEPT
Learning must not import execution or risk KEPT
Contracts: 4 kept, 0 broken.
```

Coverage by area (all executed):

| area | file | tests |
|---|---|---|
| data contracts, hash, gaps | `test_data_contracts.py` | 11 |
| MTF, causality, no-lookahead | `test_mtf_causality.py` | 7 |
| timing, single shift, gap execution | `test_timing_execution.py` | 5 |
| purge/embargo/L/H/ATR independence | `test_purge_embargo.py` | 11 |
| backtest engine | `test_backtest_engine.py` | 11 |
| metrics + fitness | `test_metrics_fitness.py` | 7 |
| experiment/learning/reproducibility | `test_learning.py` | 12 |
| Monte Carlo / stress / walk-forward | `test_robustness.py` | 8 |
| AI teacher isolation and schema | `test_ai_teacher.py` | 13 |
| AI failure non-fatal | `test_ai_failure_nonfatal.py` | 2 |
| risk and execution safety | `test_risk_execution.py` | 18 |
| architecture boundaries | `test_architecture.py` | 10 |
| CLI + Drive sync | `test_cli_and_sync.py` | 8 |
| real-data integration | `test_integration_real_data.py` | 4 (skipped) |

## 2. Skipped tests and exact reasons

All four skips are in `tests/test_integration_real_data.py`:

```
SKIPPED: Canonical dataset not available at
/content/drive/MyDrive/TradingSchoolAI/datasets/canonical/spot/BTCUSDT/1m/BTCUSDT-1m-canonical.parquet.
Set TSA_CANONICAL_PATH (and TSA_GAPS_PATH) to run real-data integration tests.
No substitute or synthetic data is used here.
```

* `test_real_dataset_contract` — schema/UTC/monotonic/OHLC validation on real data.
* `test_real_dataset_hash_matches_contract` — sha256 vs `cb8cbee…c8c1f`.
* `test_real_dataset_backtest_and_metrics` — real backtest, asserts no trade opens inside a declared gap.
* `test_real_dataset_walkforward` — purged walk-forward on real data.

These are **not** reported as passing. Running them requires the actual dataset.

## 3. End-to-end execution evidence (synthetic dataset, explicitly labelled)

A 20,000-bar synthetic parquet was written to `/tmp/e2e` (clearly synthetic; it carries
its own hash and is never presented as the canonical dataset) and the full CLI cycle was
executed:

```
$ tsa doctor -c cfg.yaml                       -> exit 0, ok: true
$ tsa data validate -c cfg.yaml                -> rows=20000, gap_count=0, errors=[]
$ tsa learn --ai-off -c cfg.yaml               -> evaluated=19, generations=3,
                                                  diversity=[1.0, 0.7, 0.8],
                                                  best=367b2cfbb300cfcf, ai_mode=AI_OFF
$ tsa report -e eb1a7029d410314b1a646c13       -> 18 trades, fees 80.06, slippage 8.01,
                                                  maxDD 0.00115, seed 7717604236406567243,
                                                  git_commit ca4062c…, formula fitness-v1
$ tsa walkforward -g best.json --folds 2       -> 2 folds, train_end < test_start each
$ tsa robustness -e <eid>                      -> MC runs=30, gate(p5)=0.01293 >= 0.0 PASS,
                                                  dataset hash before == after,
                                                  12 stress scenarios
$ tsa sync -c cfg.yaml                         -> copied memory.sqlite, verified=1, ok
$ tsa memory -c cfg.yaml                       -> 20 experiments, 2 accepted,
                                                  failures: PROFIT_FACTOR<1.0 x16, NO_TRADES x2
$ tsa learn --with-ai -c cfg.yaml (no key)     -> exit 0, learning completed,
                                                  teacher.proposed=false,
                                                  reason=PROVIDER_UNAVAILABLE
```

Note: metrics on synthetic data are meaningless as trading results (the generator has an
embedded sinusoidal drift). They are shown only as evidence that the pipeline computes
real numbers from a real ledger.

## 4. Reproducibility verification

```
$ tsa reproduce eb1a7029d410314b1a646c13 -c cfg.yaml
{"experiment_id": "eb1a7029d410314b1a646c13",
 "reproduced_experiment_id": "eb1a7029d410314b1a646c13",
 "identical_id": true, "tolerance": 1e-09, "mismatches": {}, "ok": true}
```

Additional automated evidence:

* `test_experiment_is_deterministic` — identical experiment_id, seed, metrics, fitness.
* `test_seeds_do_not_use_uuid_or_clock` — seeds derive only from hashes + counter + split.
* `test_evolution_reproducible` — two independent runs produce identical archives.
* `test_serial_and_parallel_agree` — `workers=1` and `workers=2` produce identical
  experiment ids and fitness values in identical order (ProcessPool determinism verified,
  not assumed).
* `test_monte_carlo_deterministic` — same seed ⇒ identical distributions; different seed ⇒ different.

## 5. AI_OFF confirmation

* `AIConfig.enabled` defaults to `False`; `build_provider` returns `NullProvider`.
* `NullProvider.complete` raises rather than fabricating a response — no hardcoded
  teacher answers exist anywhere in the codebase.
* `test_learning_works_with_ai_off` and the `tsa learn --ai-off` run above both completed
  with **no API key set in the environment**.
* `tsa learn --with-ai` without credentials completes the Python-only cycle and reports
  `PROVIDER_UNAVAILABLE` instead of crashing (`test_learn_with_ai_still_completes_without_credentials`).

## 6. Live execution disabled by default

* `Settings().execution.mode == "paper"`, `live_enabled == False` (`test_paper_is_default`).
* `check_live_locks` enforces three independent locks; each one missing raises
  `LOCK_1_FAILED` / `LOCK_2_FAILED` / `LOCK_3_FAILED` (`test_live_three_locks`).
* `OKXSpotVenue.__init__` runs the lock check, so the live venue cannot even be
  constructed by default (`test_okx_venue_cannot_be_built_without_locks`).
* `tsa doctor` prints the live-lock state: in this environment `LIVE DISABLED`.
* No test or learning workflow touches a live venue.

## 7. Implemented vs explicitly unsupported

**Implemented and tested:** dataset validation + gap registry + hashing; MTF resampling
with availability timestamps; causal feature engine with gap-aware warmup; typed bounded
Genome with deterministic hashing; event-driven backtester (long/short, fees, slippage,
SL/TP, time exit, gap exit, next-valid-open); metrics with defined degenerate cases;
absolute fitness with rejection gates; purge/embargo; purged walk-forward; experiment
engine with full provenance; SQLite WAL memory; deterministic evolution (serial and
parallel); Monte Carlo; stress overlays; parameter perturbation; reproduce; AI teacher
with schema validation, cache, dedup, token governor, cooldown; research queue; risk
engine; paper venue; OKX metadata/reconciliation behind live locks; Drive sync with
checksums and lock; reporting from persisted results; doctor; full CLI; Colab bootstrap.

**Explicitly unsupported in v1.0 (fails clearly, never pretends):** signed OKX order
submission; futures/margin/leverage trading; multi-instrument portfolios; tick-level
intrabar sequencing; Numba acceleration; MTF features inside Genome conditions;
automatic dataset download.

## 8. Handoff

The system is a working research pipeline with enforced temporal and architectural
contracts. Before it could be called a production candidate it needs: (a) the four
real-data integration tests executed green against the canonical dataset with a matching
hash, (b) a walk-forward + Monte Carlo campaign on that real data, (c) a sustained paper
session with reconciliation, and (d) a security review of the signed-order path, which
is deliberately absent today.
