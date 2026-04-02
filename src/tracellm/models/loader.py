"""Model loader — loads transformer, Mamba, and GGUF models into memory."""

from __future__ import annotations

import gc
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import torch

from tracellm.config import TraceConfig
from tracellm.models.formats import ModelFormat, detect_format
from tracellm.models.registry import ModelRegistry, ModelCard
from tracellm.utils.hardware import resolve_device, resolve_dtype
from tracellm.utils.logging import get_logger

log = get_logger("tracellm.models.loader")


class LoadedModel:
    """Container for a loaded model + tokenizer + metadata."""

    def __init__(
        self,
        name: str,
        model: Any,
        tokenizer: Any,
        device: str,
        dtype: torch.dtype,
        format: ModelFormat,
    ):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.format = format
        self.loaded_at = time.time()

    def unload(self) -> None:
        """Free model from memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info(f"Unloaded model '{self.name}'")


class ModelLoader:
    """Manages loading/unloading models with LRU eviction."""

    def __init__(self, config: TraceConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self.max_loaded = config.models.max_loaded_models
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()

    @property
    def loaded_models(self) -> list[str]:
        return list(self._loaded.keys())

    def get(self, name: str) -> LoadedModel | None:
        """Get a loaded model, moving it to MRU position."""
        if name in self._loaded:
            self._loaded.move_to_end(name)
            return self._loaded[name]
        return None

    def load(self, name: str, device: str = "auto", dtype: str = "auto") -> LoadedModel:
        """Load a model by registry name. Evicts LRU if at capacity."""
        # Return cached if already loaded
        existing = self.get(name)
        if existing:
            log.info(f"Model '{name}' already loaded")
            return existing

        card = self.registry.get(name)
        if not card:
            raise ValueError(f"Model '{name}' not found in registry. Run `tracellm pull` first.")

        # Evict LRU if at capacity
        while len(self._loaded) >= self.max_loaded:
            evict_name, evict_model = self._loaded.popitem(last=False)
            evict_model.unload()
            log.info(f"Evicted LRU model '{evict_name}'")

        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(dtype)
        model_format = detect_format(Path(card.path))

        log.info(f"Loading '{name}' ({model_format.value}) on {resolved_device} as {resolved_dtype}")

        if model_format == ModelFormat.GGUF:
            loaded = self._load_gguf(card, resolved_device)
        elif model_format == ModelFormat.MAMBA:
            loaded = self._load_mamba(card, resolved_device, resolved_dtype)
        else:
            loaded = self._load_transformers(card, resolved_device, resolved_dtype)

        self._loaded[name] = loaded
        self.registry.touch(name)
        log.info(f"Model '{name}' loaded successfully")
        return loaded

    def unload(self, name: str) -> bool:
        """Unload a model from memory."""
        loaded = self._loaded.pop(name, None)
        if loaded:
            loaded.unload()
            return True
        return False

    def unload_all(self) -> None:
        """Unload all models."""
        for name in list(self._loaded.keys()):
            self.unload(name)

    def _load_transformers(
        self, card: ModelCard, device: str, dtype: torch.dtype
    ) -> LoadedModel:
        """Load a HuggingFace Transformers model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            card.path,
            trust_remote_code=self.config.models.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            card.path,
            torch_dtype=dtype,
            device_map=device if device == "auto" else {"": device},
            trust_remote_code=self.config.models.trust_remote_code,
        )
        model.eval()

        return LoadedModel(
            name=card.name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            format=ModelFormat.TRANSFORMERS,
        )

    def _load_mamba(
        self, card: ModelCard, device: str, dtype: torch.dtype
    ) -> LoadedModel:
        """Load a Mamba / Mamba-2 / Mamba-3 model."""
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        except ImportError:
            # Fall back to transformers if mamba_ssm not installed
            log.warning("mamba-ssm not installed, falling back to transformers loader")
            return self._load_transformers(card, device, dtype)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            card.path,
            trust_remote_code=self.config.models.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = MambaLMHeadModel.from_pretrained(
            card.path,
            dtype=dtype,
            device=device,
        )
        model.eval()

        return LoadedModel(
            name=card.name,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            format=ModelFormat.MAMBA,
        )

    def _load_gguf(self, card: ModelCard, device: str) -> LoadedModel:
        """Load a GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install 'tracellm[gguf]'"
            )

        # Find the .gguf file
        path = Path(card.path)
        if path.is_file():
            gguf_path = path
        else:
            gguf_files = list(path.glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No .gguf files found in {path}")
            gguf_path = gguf_files[0]

        n_gpu_layers = -1 if device in ("cuda", "mps") else 0

        model = Llama(
            model_path=str(gguf_path),
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        return LoadedModel(
            name=card.name,
            model=model,
            tokenizer=None,  # llama.cpp handles tokenization internally
            device=device,
            dtype=torch.float16,
            format=ModelFormat.GGUF,
        )
