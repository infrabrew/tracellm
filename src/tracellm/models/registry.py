"""Model registry — tracks downloaded/local models and their metadata."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, HfApi

from tracellm.models.formats import ModelFormat, ModelArch, detect_format, detect_architecture, get_model_size_estimate
from tracellm.utils.logging import get_logger

log = get_logger("tracellm.models.registry")


@dataclass
class ModelCard:
    """Metadata for a registered model."""
    name: str                         # short name: "llama3-8b", "mamba-2.8b"
    path: str                         # absolute path on disk
    source: str = ""                  # "huggingface:meta-llama/Meta-Llama-3-8B" or "local"
    format: str = "unknown"
    architecture: str = "unknown"
    size_gb: float = 0.0
    parameters: str = ""              # "8B", "70B", etc.
    quantization: str = ""            # "q4_k_m", "awq", "gptq", ""
    added_at: float = 0.0
    last_used_at: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelCard:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """Manages the local model catalog backed by a JSON file."""

    def __init__(self, cache_dir: str = "~/.tracellm/models"):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._catalog_path = self.cache_dir / "catalog.json"
        self._catalog: dict[str, ModelCard] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        if self._catalog_path.exists():
            data = json.loads(self._catalog_path.read_text())
            self._catalog = {k: ModelCard.from_dict(v) for k, v in data.items()}

    def _save_catalog(self) -> None:
        data = {k: v.to_dict() for k, v in self._catalog.items()}
        self._catalog_path.write_text(json.dumps(data, indent=2))

    def list_models(self) -> list[ModelCard]:
        """Return all registered models."""
        return list(self._catalog.values())

    def get(self, name: str) -> ModelCard | None:
        """Retrieve a model card by name."""
        return self._catalog.get(name)

    def register(self, name: str, path: str, **kwargs) -> ModelCard:
        """Register a local model path."""
        p = Path(path).expanduser().resolve()
        card = ModelCard(
            name=name,
            path=str(p),
            source=kwargs.get("source", "local"),
            format=detect_format(p).value,
            architecture=detect_architecture(p).value,
            size_gb=get_model_size_estimate(p),
            parameters=kwargs.get("parameters", ""),
            quantization=kwargs.get("quantization", ""),
            added_at=time.time(),
            last_used_at=time.time(),
            tags=kwargs.get("tags", []),
        )
        self._catalog[name] = card
        self._save_catalog()
        log.info(f"Registered model '{name}' ({card.format}, {card.size_gb:.1f} GB)")
        return card

    def pull(
        self,
        repo_id: str,
        name: str | None = None,
        revision: str = "main",
        quantization: str = "",
    ) -> ModelCard:
        """Download a model from HuggingFace Hub and register it."""
        name = name or repo_id.split("/")[-1].lower()
        dest = self.cache_dir / name

        log.info(f"Pulling '{repo_id}' → {dest}")
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )

        return self.register(
            name=name,
            path=str(dest),
            source=f"huggingface:{repo_id}",
            quantization=quantization,
        )

    def remove(self, name: str, delete_files: bool = False) -> bool:
        """Unregister a model. Optionally delete files from disk."""
        card = self._catalog.pop(name, None)
        if not card:
            return False
        if delete_files:
            p = Path(card.path)
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
                log.info(f"Deleted model files at {p}")
        self._save_catalog()
        log.info(f"Removed model '{name}' from registry")
        return True

    def touch(self, name: str) -> None:
        """Update last_used_at timestamp."""
        card = self._catalog.get(name)
        if card:
            card.last_used_at = time.time()
            self._save_catalog()

    def search_hub(self, query: str, limit: int = 10) -> list[dict]:
        """Search HuggingFace Hub for models."""
        api = HfApi()
        results = api.list_models(search=query, limit=limit, sort="downloads")
        return [
            {
                "id": m.id,
                "downloads": m.downloads,
                "likes": m.likes,
                "tags": m.tags[:5] if m.tags else [],
            }
            for m in results
        ]
