# Trading School AI

**Python trading robot = STUDENT. LLM = TEACHER. Python statistics = JUDGE.**

The Python system learns strategies through evolutionary search, real backtests and
deterministic statistics. The LLM can only look at compact statistical summaries and
propose *research hypotheses*. Every hypothesis — Python-generated or LLM-proposed —
must pass through the same Experiment Engine before it means anything.

**The whole system runs with `AI_OFF` and requires no API credentials.**

---

## Repository tree

```
trading-school-ai/
├── pyproject.toml            # packaging + pytest config
├── setup.cfg                 # import-linter architecture contracts
├── configs/
│   ├── default.yaml          # Drive/Colab defaults (AI_OFF, paper execution)
│   └── local-synthetic.yaml  # local runs; dataset path must be supplied
├── notebooks/
│   └── colab_quickstart.md   # the two Colab cells
├── src/trading_school_ai/
│   ├── config/               # typed settings, config hash
│   ├── data/                 # canonical loading, validation, gaps, hashing
│   ├── mtf/                  # right-labeled/right-closed resampling, availability
│   ├── features/             # causal feature registry with explicit lookbacks
│   ├── strategies/           # Genome (typed, bounded, hashable) + signals
│   ├── backtest/             # event-driven engine, trade ledger, equity curve
│   ├── metrics/              # PF, Sharpe, Sortino, drawdown, exposure, costs
│   ├── fitness/              # absolute deterministic fitness + rejection gates
│   ├── walkforward/          # purge/embargo + purged walk-forward
│   ├── experiment/           # experiment engine, seeds, reproduce
│   ├── learning/             # population, mutation/crossover, evolution
│   ├── montecarlo/           # adverse perturbation robustness
│   ├── stress/               # stress overlays (PASS_REPORT) + perturbation
│   ├── memory/               # SQLite (WAL) experience memory
│   ├── storage/              # local runtime layout
│   ├── drive/                # checksum-verified Drive backup/restore with lock
│   ├── ai/                   # teacher, schema, provider, token governor
│   ├── research/             # hypothesis -> experiment routing
│   ├── risk/                 # independent risk engine + approval object
│   ├── execution/            # paper venue (default), OKX spot behind live locks
│   ├── reporting/            # reads persisted results only
│   ├── observability/        # masked logging, doctor
│   ├── colab/                # bootstrap
│   └── cli/                  # tsa command line
└── tests/                    # 129 tests (4 real-data tests skip without the dataset)
```

## Setup

```bash
git clone <this repo> && cd trading-school-ai
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ",ai" for the optional LLM adapter
pytest -q                        # run the suite
lint-imports                     # verify architecture contracts
```

## Dataset contract

| item | value |
|---|---|
| canonical | `/content/drive/MyDrive/TradingSchoolAI/datasets/canonical/spot/BTCUSDT/1m/BTCUSDT-1m-canonical.parquet` |
| gap registry | `.../BTCUSDT-1m-missing-gaps.parquet` |
| contract sha256 | `cb8cbeedc1ae34b68078e5fb02126f92aab8d064d33fa19b0159acb666ed8c1f` |

Paths are configurable (`data.canonical_path`, or `TSA_CANONICAL_PATH` /
`TSA_GAPS_PATH`). If the dataset is missing, the system reports the exact missing
dependency and stops. **It never downloads or synthesizes substitute data.** Hash
mismatch is reported as a warning by default and becomes fatal with
`data.enforce_hash: true`.

## Colab workflow

**Cell 1 — bootstrap**

```python
!pip install -q "git+https://github.com/<you>/trading-school-ai.git"
from trading_school_ai.colab import bootstrap
info = bootstrap(runtime_dir="/content/tsa_runtime")   # mounts Drive, creates local dirs, runs doctor
info["doctor_ok"], info["db_path"], info["canonical_path"]
```

**Cell 2 — run**

```python
!tsa doctor -c /content/configs/default.yaml
!tsa learn --ai-off -c /content/configs/default.yaml
```

SQLite and all active runtime writes stay on the local Colab filesystem
(`/content/tsa_runtime`). Drive is used for the dataset and for `tsa sync` backups only.

## CLI

```bash
tsa version
tsa doctor -c configs/default.yaml
tsa data validate -c configs/default.yaml
tsa backtest -g genome.json -c configs/default.yaml
tsa learn --ai-off -c configs/default.yaml
tsa learn --with-ai -c configs/default.yaml      # degrades gracefully with no key
tsa walkforward -g genome.json --folds 4 -c configs/default.yaml
tsa robustness -e <experiment_id> -c configs/default.yaml
tsa report --markdown -c configs/default.yaml
tsa report -e <experiment_id> -c configs/default.yaml
tsa memory -c configs/default.yaml
tsa sync -c configs/default.yaml            # add --restore to pull back from Drive
tsa reproduce <experiment_id> -c configs/default.yaml
```

## Temporal contracts

* Canonical data is 1m UTC OHLCV; higher timeframes are built **only** from it with
  right-labeled, right-closed resampling; incomplete candles are dropped.
* Every MTF candle carries `close_time` and `available_from = close_time + 1ms`;
  at decision timestamp `t` only candles with `close_time < t` are usable.
* A signal on bar `i` has `decision_timestamp = close_time(i) + 1ms`. Execution happens
  at the **open of the first valid 1m bar strictly after** that timestamp. If that bar is
  missing (declared gap), no candle is invented and no execution occurs.
* Feature/signal layers apply **no** shift. The execution layer applies the rule once.
* A gap is a hard boundary: feature windows restart, MTF candles overlapping a gap are
  invalidated, no position is opened inside/across a gap, and an open position exits at
  the last valid close with reason `GAP_BOUNDARY` (a **modeling assumption**, not a
  guaranteed real fill).

## Purge / embargo / horizons

```
feature_interval(i) = [i-L+1, i]        L = max(feature.lookback), in 1m bars
label_interval(i)   = (i, i+H]          H = genome.max_holding_bars
                                          or config.strategy_max_holding_bars (default 1440)
```

A training sample is purged **only** when its label interval intersects the held-out
interval; feature-window overlap alone never purges. Embargo is a separate exclusion
applied after the held-out interval. ATR is a price-distance mechanism only: it sets
stop/target distance and can never change `H`, `L`, or the purge count (tested).

## Fitness

Absolute, deterministic, computed from **train** metrics only, never normalized against
the current population. Gates (`NO_TRADES`, `MIN_TRADES`, `PROFIT_FACTOR`,
`MAX_DRAWDOWN`, `NON_FINITE_RETURN`) run before scoring:

```
score = sharpe_weight * sharpe + profit_factor_weight * log1p(max(PF-1, 0)) - drawdown_penalty * maxDD
```

Monte Carlo and Stress results **never** enter the formula. The formula version and
config hash are persisted with every experiment.

## Safety model

* Paper execution is the default everywhere, including all tests.
* Live requires **all three**: `execution.live_enabled: true` **and** `LIVE_CONFIRM=1`
  **and** the `--live` flag. Each missing lock fails with `LOCK_1/2/3_FAILED`.
* The risk engine is independent of strategy and LLM; execution accepts only an
  `ApprovedOrder` produced by it. Position mismatch triggers STOP + ALERT.
* OKX SPOT metadata (tick/lot/min size) is always retrieved from the exchange;
  fee/slippage defaults in config are explicitly labelled fallback assumptions.
* The LLM cannot emit orders: responses containing trading instructions, order sizing
  keys, or pipeline-bypass language are rejected by schema validation.

## Known limitations (v1.0)

* `side: "both"` uses the exit condition as the short trigger; genuinely symmetric
  dual-side genomes are not generated by the evolutionary seeder.
* Single-instrument (BTCUSDT spot) and single-position-at-a-time by default.
* Intrabar SL/TP resolution uses bar high/low; when both are touched inside one bar the
  stop is assumed first (conservative), not tick-level sequencing.
* Signed OKX order submission is intentionally not shipped: `OKXSpotVenue.submit`
  requires an authenticated client and otherwise refuses with a clear error.
* Numba acceleration is not used; the pure-Python/NumPy engine is preferred for exact
  semantics.
* MTF features are available via `mtf.align_to_base` but the v1.0 Genome expresses
  conditions on 1m features only.
* Fees/slippage defaults (0.1% / 1bp) are fallback assumptions, not venue-verified.

See `VALIDATION_REPORT.md` for the executed test results and reproducibility evidence.
