"""Tests for model registry."""

import json
from pathlib import Path

from tracellm.models.registry import ModelRegistry, ModelCard


def test_register_and_get(tmp_path):
    """Should register a model and retrieve it by name."""
    registry = ModelRegistry(cache_dir=str(tmp_path / "models"))

    # Create a fake model directory
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

    card = registry.register("test-model", str(model_dir), tags=["test"])

    assert card.name == "test-model"
    assert card.format == "transformers"
    assert card.architecture == "llama"
    assert "test" in card.tags

    # Retrieve
    retrieved = registry.get("test-model")
    assert retrieved is not None
    assert retrieved.name == "test-model"


def test_list_models(tmp_path):
    registry = ModelRegistry(cache_dir=str(tmp_path / "models"))

    model_dir = tmp_path / "m1"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "mistral"}))

    registry.register("model-a", str(model_dir))
    registry.register("model-b", str(model_dir))

    models = registry.list_models()
    assert len(models) == 2
    names = {m.name for m in models}
    assert "model-a" in names
    assert "model-b" in names


def test_remove_model(tmp_path):
    registry = ModelRegistry(cache_dir=str(tmp_path / "models"))

    model_dir = tmp_path / "m1"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    registry.register("removable", str(model_dir))
    assert registry.get("removable") is not None

    registry.remove("removable")
    assert registry.get("removable") is None


def test_remove_nonexistent(tmp_path):
    registry = ModelRegistry(cache_dir=str(tmp_path / "models"))
    assert registry.remove("nonexistent") is False


def test_catalog_persistence(tmp_path):
    """Registry should persist across instances."""
    cache_dir = str(tmp_path / "models")

    model_dir = tmp_path / "m1"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "phi"}))

    registry1 = ModelRegistry(cache_dir=cache_dir)
    registry1.register("persistent", str(model_dir))

    # New instance should see the model
    registry2 = ModelRegistry(cache_dir=cache_dir)
    assert registry2.get("persistent") is not None
    assert registry2.get("persistent").architecture == "phi"
