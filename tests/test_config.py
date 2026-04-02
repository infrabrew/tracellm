"""Tests for configuration loading and merging."""

import os
import tempfile
from pathlib import Path

import yaml

from tracellm.config import TraceConfig, load_config


def test_default_config_values():
    """TraceConfig should have sane defaults without any file."""
    config = TraceConfig()
    assert config.server.port == 8400
    assert config.server.host == "0.0.0.0"
    assert config.inference.max_tokens == 4096
    assert config.inference.temperature == 0.7
    assert config.training.default_optimizer == "adamw"
    assert config.training.finetune_defaults.method == "lora"
    assert config.training.finetune_defaults.lora.r == 16


def test_load_config_from_yaml():
    """Config should load and override values from a YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump({
            "server": {"port": 9999, "workers": 4},
            "inference": {"temperature": 0.3},
        }, f)
        f.flush()

        config = load_config(f.name)
        assert config.server.port == 9999
        assert config.server.workers == 4
        assert config.inference.temperature == 0.3
        # Non-overridden values should keep defaults
        assert config.server.host == "0.0.0.0"
        assert config.inference.max_tokens == 4096

    os.unlink(f.name)


def test_kv_cache_defaults():
    config = TraceConfig()
    assert config.inference.kv_cache.enabled is True
    assert config.inference.kv_cache.max_cache_size_gb == 4.0
    assert config.inference.kv_cache.eviction_policy == "lru"


def test_training_defaults():
    config = TraceConfig()
    assert config.training.gradient_checkpointing is True
    assert config.training.mixed_precision == "bf16"
    assert config.training.seed == 42
    assert config.training.deepspeed.enabled is False


def test_qlora_defaults():
    config = TraceConfig()
    qlora = config.training.finetune_defaults.qlora
    assert qlora.bits == 4
    assert qlora.quant_type == "nf4"
    assert qlora.double_quant is True


def test_recursive_defaults():
    config = TraceConfig()
    assert config.recursive.max_iterations == 10
    assert config.recursive.convergence_threshold == 0.95
    assert config.recursive.skip_adversarial is False


def test_scaffold_defaults():
    config = TraceConfig()
    assert config.scaffold.default_profile == "standard"
    assert config.scaffold.refine_iterations == 5
    assert config.scaffold.refine_convergence == 0.90
    assert config.scaffold.verify_retries == 2
    assert config.scaffold.show_reasoning is False
    assert config.scaffold.skip_adversarial is False


def test_recursive_config_override():
    """Recursive config should be overridable from YAML."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump({
            "recursive": {"max_iterations": 20, "convergence_threshold": 0.99},
        }, f)
        f.flush()
        config = load_config(f.name)
        assert config.recursive.max_iterations == 20
        assert config.recursive.convergence_threshold == 0.99
        assert config.recursive.skip_adversarial is False  # default kept
    os.unlink(f.name)


def test_scaffold_config_override():
    """Scaffold config should be overridable from YAML."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump({
            "scaffold": {"default_profile": "full", "verify_retries": 5},
        }, f)
        f.flush()
        config = load_config(f.name)
        assert config.scaffold.default_profile == "full"
        assert config.scaffold.verify_retries == 5
    os.unlink(f.name)
