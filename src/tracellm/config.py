"""Configuration management — loads default.yml + CLI/env overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ── Sub-models ─────────────────────────────────────────────────────────────────

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8400
    workers: int = 1
    cors_origins: list[str] = ["*"]
    max_concurrent_requests: int = 64
    request_timeout_seconds: int = 300


class KVCacheConfig(BaseModel):
    enabled: bool = True
    max_cache_size_gb: float = 4.0
    eviction_policy: str = "lru"


class InferenceConfig(BaseModel):
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: list[str] = []
    stream: bool = True
    batch_size: int = 1
    kv_cache: KVCacheConfig = KVCacheConfig()


class ModelsConfig(BaseModel):
    cache_dir: str = "~/.tracellm/models"
    default_dtype: str = "auto"
    default_device: str = "auto"
    max_loaded_models: int = 2
    trust_remote_code: bool = False


class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "auto"


class QLoraConfig(BaseModel):
    bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True


class FinetuneDefaults(BaseModel):
    method: str = "lora"
    lora: LoraConfig = LoraConfig()
    qlora: QLoraConfig = QLoraConfig()


class DeepSpeedConfig(BaseModel):
    enabled: bool = False
    config: str = "config/training_profiles/deepspeed_z2.json"


class TrainingConfig(BaseModel):
    output_dir: str = "~/.tracellm/checkpoints"
    logging_dir: str = "~/.tracellm/logs"
    default_optimizer: str = "adamw"
    default_scheduler: str = "cosine"
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    seed: int = 42
    dataloader_workers: int = 4
    finetune_defaults: FinetuneDefaults = FinetuneDefaults()
    deepspeed: DeepSpeedConfig = DeepSpeedConfig()


class RecursiveConfig(BaseModel):
    max_iterations: int = 10
    convergence_threshold: float = 0.95
    skip_adversarial: bool = False


class ScaffoldConfig(BaseModel):
    default_profile: str = "standard"     # full | standard | quick | code | creative | research
    refine_iterations: int = 5
    refine_convergence: float = 0.90
    verify_retries: int = 2
    show_reasoning: bool = False
    skip_adversarial: bool = False


class BatchConfig(BaseModel):
    max_concurrency: int = 4
    default_max_tokens: int = 4096


class BenchmarkConfig(BaseModel):
    default_max_tokens: int = 128
    default_num_runs: int = 3
    warmup_runs: int = 1


class RLHFConfig(BaseModel):
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"       # sigmoid | hinge | ipo
    ppo_kl_penalty: float = 0.2
    default_learning_rate: float = 5e-7
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32


class LoggingConfig(BaseModel):
    level: str = "info"
    file: str = "~/.tracellm/tracellm.log"
    rich_console: bool = True


# ── Root config ────────────────────────────────────────────────────────────────

class TraceConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    models: ModelsConfig = ModelsConfig()
    inference: InferenceConfig = InferenceConfig()
    recursive: RecursiveConfig = RecursiveConfig()
    scaffold: ScaffoldConfig = ScaffoldConfig()
    training: TrainingConfig = TrainingConfig()
    batch: BatchConfig = BatchConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    rlhf: RLHFConfig = RLHFConfig()
    logging: LoggingConfig = LoggingConfig()


_CONFIG_SEARCH_PATHS = [
    Path("tracellm.yml"),
    Path("config/default.yml"),
    Path.home() / ".tracellm" / "config.yml",
]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str | Path | None = None) -> TraceConfig:
    """Load config from YAML file with fallback search paths."""
    raw: dict[str, Any] = {}

    paths = [Path(config_path)] if config_path else _CONFIG_SEARCH_PATHS
    for p in paths:
        p = p.expanduser()
        if p.exists():
            with open(p) as f:
                raw = yaml.safe_load(f) or {}
            break

    # Environment variable overrides (TRACELLM_SERVER__PORT=9000, etc.)
    env_prefix = "TRACELLM_"
    for key, val in os.environ.items():
        if key.startswith(env_prefix):
            parts = key[len(env_prefix) :].lower().split("__")
            d = raw
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = val

    return TraceConfig(**raw)
