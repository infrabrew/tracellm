"""Tests for RLHF training module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tracellm.config import TraceConfig
from tracellm.training.rlhf import (
    RLHFJob,
    RLHFManager,
    load_preference_dataset,
    load_reward_dataset,
)


class TestRLHFJob:
    def test_creation(self):
        job = RLHFJob(job_id="dpo-abc123", model_name="llama3", method="dpo")
        assert job.job_id == "dpo-abc123"
        assert job.model_name == "llama3"
        assert job.method == "dpo"
        assert job.status == "initializing"
        assert job.epoch == 0.0
        assert job.loss is None
        assert job.reward_accuracy is None
        assert job._stop_flag is False

    def test_all_methods(self):
        for method in ["dpo", "reward", "ppo"]:
            job = RLHFJob(job_id=f"{method}-test", model_name="model", method=method)
            assert job.method == method


class TestLoadPreferenceDataset:
    def test_load_jsonl(self, tmp_path):
        data_file = tmp_path / "prefs.jsonl"
        lines = [
            json.dumps({
                "prompt": "What is 2+2?",
                "chosen": "The answer is 4.",
                "rejected": "I think it's 5.",
            }),
            json.dumps({
                "prompt": "What color is the sky?",
                "chosen": "Blue.",
                "rejected": "Green.",
            }),
        ]
        data_file.write_text("\n".join(lines))

        ds = load_preference_dataset(str(data_file))
        assert len(ds) == 2
        assert "prompt" in ds.column_names
        assert "chosen" in ds.column_names
        assert "rejected" in ds.column_names

    def test_alternative_column_names(self, tmp_path):
        data_file = tmp_path / "prefs.jsonl"
        lines = [
            json.dumps({
                "question": "What is AI?",
                "preferred": "AI is artificial intelligence.",
                "dispreferred": "AI is magic.",
            }),
        ]
        data_file.write_text("\n".join(lines))

        ds = load_preference_dataset(str(data_file))
        assert "prompt" in ds.column_names
        assert "chosen" in ds.column_names
        assert "rejected" in ds.column_names

    def test_missing_column_raises(self, tmp_path):
        data_file = tmp_path / "bad.jsonl"
        data_file.write_text(json.dumps({"text": "hello", "label": 1}))

        with pytest.raises(ValueError, match="missing"):
            load_preference_dataset(str(data_file))

    def test_csv_format(self, tmp_path):
        data_file = tmp_path / "prefs.csv"
        data_file.write_text(
            "prompt,chosen,rejected\n"
            "What is 2+2?,4,5\n"
            "Color of sky?,Blue,Green\n"
        )

        ds = load_preference_dataset(str(data_file))
        assert len(ds) == 2


class TestLoadRewardDataset:
    def test_load_text_labels(self, tmp_path):
        data_file = tmp_path / "rewards.jsonl"
        lines = [
            json.dumps({"text": "Good answer", "label": 1.0}),
            json.dumps({"text": "Bad answer", "label": 0.0}),
        ]
        data_file.write_text("\n".join(lines))

        ds, text_field, label_field = load_reward_dataset(str(data_file))
        assert len(ds) == 2
        assert text_field == "text"
        assert label_field == "label"

    def test_convert_preference_to_reward(self, tmp_path):
        data_file = tmp_path / "prefs.jsonl"
        lines = [
            json.dumps({
                "prompt": "What is 2+2?",
                "chosen": "4",
                "rejected": "5",
            }),
        ]
        data_file.write_text("\n".join(lines))

        ds, text_field, label_field = load_reward_dataset(str(data_file))
        # Each preference pair becomes 2 reward entries
        assert len(ds) == 2
        assert text_field == "text"
        assert label_field == "label"


class TestRLHFManager:
    def _make_manager(self):
        config = TraceConfig()
        registry = MagicMock()
        return RLHFManager(config, registry)

    def test_get_status_nonexistent(self):
        manager = self._make_manager()
        assert manager.get_status("nonexistent") is None

    def test_list_jobs_empty(self):
        manager = self._make_manager()
        assert manager.list_jobs() == []

    def test_stop_nonexistent(self):
        # Should not raise
        manager = self._make_manager()
        manager.stop("nonexistent")

    @patch("tracellm.training.rlhf.AutoModelForCausalLM")
    @patch("tracellm.training.rlhf.AutoTokenizer")
    @patch("tracellm.training.rlhf.load_preference_dataset")
    @patch("tracellm.training.rlhf.resolve_device", return_value="cpu")
    @patch("tracellm.training.rlhf.resolve_dtype")
    def test_start_dpo_creates_job(self, mock_dtype, mock_device, mock_ds,
                                    mock_tok, mock_model):
        import torch
        mock_dtype.return_value = torch.float32

        manager = self._make_manager()
        card = MagicMock()
        card.path = "/tmp/model"
        manager.registry.get.return_value = card

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"
        mock_tok.from_pretrained.return_value = tokenizer

        job_id = manager.start_dpo(
            model="test-model",
            dataset="test-dataset",
            epochs=1,
            batch_size=1,
        )

        assert job_id.startswith("dpo-")
        status = manager.get_status(job_id)
        assert status is not None
        assert status["method"] == "dpo"
        assert status["model"] == "test-model"

    @patch("tracellm.training.rlhf.AutoModelForSequenceClassification")
    @patch("tracellm.training.rlhf.AutoTokenizer")
    @patch("tracellm.training.rlhf.load_reward_dataset")
    @patch("tracellm.training.rlhf.resolve_device", return_value="cpu")
    @patch("tracellm.training.rlhf.resolve_dtype")
    def test_start_reward_model_creates_job(self, mock_dtype, mock_device,
                                             mock_ds, mock_tok, mock_model):
        import torch
        mock_dtype.return_value = torch.float32

        manager = self._make_manager()
        card = MagicMock()
        card.path = "/tmp/model"
        manager.registry.get.return_value = card

        job_id = manager.start_reward_model(
            model="test-model",
            dataset="test-dataset",
        )

        assert job_id.startswith("reward-")
        status = manager.get_status(job_id)
        assert status["method"] == "reward"

    def test_start_ppo_creates_job(self):
        manager = self._make_manager()
        card = MagicMock()
        card.path = "/tmp/model"
        manager.registry.get.return_value = card

        job_id = manager.start_ppo(
            model="test-model",
            reward_model="test-reward",
            dataset="test-dataset",
        )

        assert job_id.startswith("ppo-")
        status = manager.get_status(job_id)
        assert status["method"] == "ppo"

    def test_stop_sets_flag(self):
        manager = self._make_manager()
        # Manually inject a job
        job = RLHFJob(job_id="dpo-test", model_name="m", method="dpo")
        manager._jobs["dpo-test"] = job

        manager.stop("dpo-test")
        assert job._stop_flag is True
        assert job.status == "stopping"

    def test_list_jobs_returns_all(self):
        manager = self._make_manager()
        for i in range(3):
            job = RLHFJob(job_id=f"job-{i}", model_name="m", method="dpo")
            manager._jobs[f"job-{i}"] = job

        jobs = manager.list_jobs()
        assert len(jobs) == 3

    def test_status_includes_elapsed(self):
        import time
        manager = self._make_manager()
        job = RLHFJob(job_id="dpo-t", model_name="m", method="dpo")
        job.started_at = time.time() - 10  # 10 seconds ago
        manager._jobs["dpo-t"] = job

        status = manager.get_status("dpo-t")
        assert status["elapsed_seconds"] >= 10

    def test_dpo_model_not_found(self):
        manager = self._make_manager()
        manager.registry.get.return_value = None

        job_id = manager.start_dpo(model="missing", dataset="d")

        # Wait for thread to finish
        import time
        time.sleep(0.5)

        status = manager.get_status(job_id)
        assert status["status"] == "failed"
        assert "not found" in status["error"]
