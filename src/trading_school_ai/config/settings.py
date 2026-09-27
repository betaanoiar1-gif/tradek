"""Typed configuration for Trading School AI (no hidden defaults in core logic)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

CANONICAL_DATASET_SHA256 = (
    "cb8cbeedc1ae34b68078e5fb02126f92aab8d064d33fa19b0159acb666ed8c1f"
)


class DataConfig(BaseModel):
    symbol: str = "BTCUSDT"
    market: Literal["spot"] = "spot"
    base_timeframe: str = "1m"
    canonical_path: str = (
        "/content/drive/MyDrive/TradingSchoolAI/datasets/canonical/spot/"
        "BTCUSDT/1m/BTCUSDT-1m-canonical.parquet"
    )
    gaps_path: str = (
        "/content/drive/MyDrive/TradingSchoolAI/datasets/canonical/spot/"
        "BTCUSDT/1m/BTCUSDT-1m-missing-gaps.parquet"
    )
    expected_sha256: Optional[str] = CANONICAL_DATASET_SHA256
    enforce_hash: bool = False  # hash mismatch is reported; enforcement is opt-in


class BacktestConfig(BaseModel):
    fee_rate: float = 0.001           # fallback assumption, documented in README
    slippage_bps: float = 1.0         # fallback assumption, documented in README
    initial_equity: float = 10_000.0
    position_fraction: float = 1.0
    allow_short: bool = True
    allow_overlapping_positions: bool = False
    gap_exit: Literal["last_valid_close", "forbid"] = "last_valid_close"


class StrategyConfig(BaseModel):
    strategy_max_holding_bars: int = 1440  # documented v1.0 default (1 day of 1m bars)


class SplitConfig(BaseModel):
    train_frac: float = 0.6
    validation_frac: float = 0.2
    embargo_bars: int = 60


class FitnessConfig(BaseModel):
    formula_version: str = "fitness-v1"
    min_trades: int = 20
    min_profit_factor: float = 1.0
    max_drawdown_limit: float = 0.5
    sharpe_weight: float = 1.0
    profit_factor_weight: float = 0.5
    drawdown_penalty: float = 1.0


class EvolutionConfig(BaseModel):
    population_size: int = 16
    generations: int = 3
    elitism: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    master_seed: int = 20240101
    workers: int = 1


class MonteCarloConfig(BaseModel):
    runs: int = 100
    slippage_extra_bps_max: float = 3.0
    fee_multiplier_max: float = 1.5
    dropout_prob: float = 0.05
    percentile_gate: float = 5.0
    min_gate_return: float = 0.0


class StressConfig(BaseModel):
    fee_multipliers: list[float] = Field(default_factory=lambda: [1.0, 2.0, 5.0])
    slippage_multipliers: list[float] = Field(default_factory=lambda: [1.0, 3.0, 10.0])
    execution_delay_bars: list[int] = Field(default_factory=lambda: [0, 1, 5])
    missing_signal_probs: list[float] = Field(default_factory=lambda: [0.0, 0.1])


class RiskConfig(BaseModel):
    max_position_quote: float = 5_000.0
    max_exposure_quote: float = 10_000.0
    daily_loss_limit: float = 500.0
    max_drawdown_limit: float = 0.25
    max_leverage: float = 1.0
    cooldown_seconds: int = 60
    kill_switch: bool = False


class AIConfig(BaseModel):
    enabled: bool = False  # AI_OFF is the default
    provider: Literal["openai_compatible", "null"] = "null"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "TSA_LLM_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 2
    token_budget: int = 100_000
    cooldown_seconds: int = 30


class ExecutionConfig(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    live_enabled: bool = False
    exchange: Literal["okx", "paper"] = "paper"


class StorageConfig(BaseModel):
    runtime_dir: str = "./runtime"
    db_filename: str = "memory.sqlite"
    drive_backup_dir: str = "/content/drive/MyDrive/TradingSchoolAI/backups"


class Settings(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    fitness: FitnessConfig = Field(default_factory=FitnessConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    montecarlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    stress: StressConfig = Field(default_factory=StressConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    def config_hash(self) -> str:
        """Hash of the settings that can affect experiment OUTPUTS.

        Runtime-only knobs (worker count, storage paths, AI transport) are
        excluded on purpose: they must never change a result, and including
        them would make identical experiments look different.
        """
        payload = self.model_dump(exclude={"storage", "ai"})
        payload["evolution"].pop("workers", None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @property
    def runtime_path(self) -> Path:
        return Path(self.storage.runtime_dir).expanduser().resolve()

    @property
    def db_path(self) -> Path:
        return self.runtime_path / self.storage.db_filename


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    """Load settings from YAML, overlaying environment overrides."""
    raw: dict = {}
    if path is not None:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        raw = yaml.safe_load(p.read_text()) or {}
    settings = Settings(**raw)
    env_canonical = os.environ.get("TSA_CANONICAL_PATH")
    if env_canonical:
        settings.data.canonical_path = env_canonical
    env_gaps = os.environ.get("TSA_GAPS_PATH")
    if env_gaps:
        settings.data.gaps_path = env_gaps
    env_runtime = os.environ.get("TSA_RUNTIME_DIR")
    if env_runtime:
        settings.storage.runtime_dir = env_runtime
    return settings
