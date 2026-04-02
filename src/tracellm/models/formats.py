"""Model format detection and metadata extraction."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from tracellm.utils.logging import get_logger

log = get_logger("tracellm.models.formats")


class ModelFormat(str, Enum):
    TRANSFORMERS = "transformers"  # HuggingFace safetensors / pytorch_model.bin
    GGUF = "gguf"                 # llama.cpp quantized format
    MAMBA = "mamba"               # mamba-ssm native checkpoints
    SAFETENSORS = "safetensors"   # raw safetensors (non-HF layout)
    UNKNOWN = "unknown"


class ModelArch(str, Enum):
    LLAMA = "llama"
    MISTRAL = "mistral"
    QWEN = "qwen"
    PHI = "phi"
    GEMMA = "gemma"
    MAMBA = "mamba"
    MAMBA2 = "mamba2"
    GPT_NEOX = "gpt_neox"
    FALCON = "falcon"
    DEEPSEEK = "deepseek"
    UNKNOWN = "unknown"


def detect_format(model_path: Path) -> ModelFormat:
    """Detect the model format from files on disk."""
    model_path = Path(model_path)

    if not model_path.exists():
        return ModelFormat.UNKNOWN

    if model_path.is_file():
        if model_path.suffix == ".gguf":
            return ModelFormat.GGUF
        if model_path.suffix == ".safetensors":
            return ModelFormat.SAFETENSORS
        return ModelFormat.UNKNOWN

    # Directory-based detection
    files = {f.name for f in model_path.iterdir()}

    if "config.json" in files:
        config_data = json.loads((model_path / "config.json").read_text())
        arch = config_data.get("model_type", "").lower()
        if "mamba" in arch:
            return ModelFormat.MAMBA
        return ModelFormat.TRANSFORMERS

    if any(f.endswith(".gguf") for f in files):
        return ModelFormat.GGUF

    if any(f.endswith(".safetensors") for f in files):
        return ModelFormat.SAFETENSORS

    return ModelFormat.UNKNOWN


def detect_architecture(model_path: Path) -> ModelArch:
    """Detect the model architecture from config.json."""
    config_file = Path(model_path) / "config.json"
    if not config_file.exists():
        return ModelArch.UNKNOWN

    config = json.loads(config_file.read_text())
    model_type = config.get("model_type", "").lower()

    arch_map = {
        "llama": ModelArch.LLAMA,
        "mistral": ModelArch.MISTRAL,
        "qwen2": ModelArch.QWEN,
        "qwen": ModelArch.QWEN,
        "phi3": ModelArch.PHI,
        "phi": ModelArch.PHI,
        "gemma": ModelArch.GEMMA,
        "gemma2": ModelArch.GEMMA,
        "mamba": ModelArch.MAMBA,
        "mamba2": ModelArch.MAMBA2,
        "gpt_neox": ModelArch.GPT_NEOX,
        "falcon": ModelArch.FALCON,
        "deepseek": ModelArch.DEEPSEEK,
    }

    for key, arch in arch_map.items():
        if key in model_type:
            return arch

    # Check architectures list
    architectures = config.get("architectures", [])
    for a in architectures:
        a_lower = a.lower()
        for key, arch in arch_map.items():
            if key in a_lower:
                return arch

    return ModelArch.UNKNOWN


def get_model_size_estimate(model_path: Path) -> float:
    """Estimate model size in GB from files on disk."""
    model_path = Path(model_path)
    total_bytes = 0

    if model_path.is_file():
        return model_path.stat().st_size / (1024**3)

    weight_extensions = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
    for f in model_path.rglob("*"):
        if f.suffix in weight_extensions:
            total_bytes += f.stat().st_size

    return round(total_bytes / (1024**3), 2)
