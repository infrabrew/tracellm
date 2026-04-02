"""Tests for model format detection."""

import json
import tempfile
from pathlib import Path

from tracellm.models.formats import (
    ModelFormat,
    ModelArch,
    detect_format,
    detect_architecture,
    get_model_size_estimate,
)


def test_detect_transformers_format(tmp_path):
    """Directory with config.json should be detected as transformers."""
    config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 1024)

    assert detect_format(tmp_path) == ModelFormat.TRANSFORMERS


def test_detect_mamba_format(tmp_path):
    """Directory with config.json containing mamba should detect Mamba."""
    config = {"model_type": "mamba", "architectures": ["MambaForCausalLM"]}
    (tmp_path / "config.json").write_text(json.dumps(config))

    assert detect_format(tmp_path) == ModelFormat.MAMBA


def test_detect_gguf_format(tmp_path):
    """File with .gguf extension should be detected as GGUF."""
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"\x00" * 1024)

    assert detect_format(gguf_file) == ModelFormat.GGUF


def test_detect_gguf_in_directory(tmp_path):
    """Directory containing .gguf files should detect as GGUF."""
    (tmp_path / "model-q4.gguf").write_bytes(b"\x00" * 1024)

    assert detect_format(tmp_path) == ModelFormat.GGUF


def test_detect_unknown_format(tmp_path):
    """Empty directory should be unknown."""
    (tmp_path / "readme.txt").write_text("hello")
    assert detect_format(tmp_path) == ModelFormat.UNKNOWN


def test_detect_architecture_llama(tmp_path):
    config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_architecture(tmp_path) == ModelArch.LLAMA


def test_detect_architecture_mistral(tmp_path):
    config = {"model_type": "mistral"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_architecture(tmp_path) == ModelArch.MISTRAL


def test_detect_architecture_from_architectures_list(tmp_path):
    config = {"model_type": "custom", "architectures": ["Qwen2ForCausalLM"]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_architecture(tmp_path) == ModelArch.QWEN


def test_model_size_estimate(tmp_path):
    """Size estimate should sum weight files."""
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * (1024 * 1024))  # 1 MB
    (tmp_path / "model-00002.safetensors").write_bytes(b"\x00" * (1024 * 1024))  # 1 MB
    (tmp_path / "config.json").write_text("{}")  # should not be counted

    size = get_model_size_estimate(tmp_path)
    assert 0.001 < size < 0.003  # ~2 MB = ~0.002 GB


def test_nonexistent_path():
    assert detect_format(Path("/nonexistent/path")) == ModelFormat.UNKNOWN
    assert detect_architecture(Path("/nonexistent/path")) == ModelArch.UNKNOWN
