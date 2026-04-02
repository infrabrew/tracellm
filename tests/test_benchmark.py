"""Tests for model benchmarking engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from tracellm.inference.benchmark import (
    BENCHMARK_PROMPTS,
    BenchmarkEngine,
    BenchmarkResult,
    BenchmarkRun,
    MemorySnapshot,
    PromptBenchmark,
)
from tracellm.inference.sampling import SamplingParams


class TestBenchmarkPrompts:
    def test_all_categories_present(self):
        assert "short" in BENCHMARK_PROMPTS
        assert "medium" in BENCHMARK_PROMPTS
        assert "long" in BENCHMARK_PROMPTS
        assert "code" in BENCHMARK_PROMPTS

    def test_prompts_non_empty(self):
        for name, prompt in BENCHMARK_PROMPTS.items():
            assert len(prompt) > 0, f"Prompt '{name}' is empty"

    def test_length_ordering(self):
        assert len(BENCHMARK_PROMPTS["short"]) < len(BENCHMARK_PROMPTS["medium"])
        assert len(BENCHMARK_PROMPTS["medium"]) < len(BENCHMARK_PROMPTS["long"])


class TestMemorySnapshot:
    def test_defaults(self):
        snap = MemorySnapshot()
        assert snap.gpu_allocated_gb == 0.0
        assert snap.gpu_reserved_gb == 0.0
        assert snap.gpu_peak_gb == 0.0
        assert snap.ram_used_gb == 0.0


class TestBenchmarkRun:
    def test_creation(self):
        run = BenchmarkRun(
            prompt_tokens=50,
            completion_tokens=100,
            time_to_first_token_s=0.05,
            total_time_s=1.0,
            tokens_per_second=100.0,
        )
        assert run.prompt_tokens == 50
        assert run.completion_tokens == 100
        assert run.tokens_per_second == 100.0


class TestBenchmarkEngine:
    def _mock_engine(self):
        engine = MagicMock()

        # Mock loader
        loaded_model = MagicMock()
        loaded_model.device = "cpu"
        loaded_model.dtype = "float32"
        loaded_model.tokenizer.encode.return_value = list(range(20))
        engine.loader.load.return_value = loaded_model

        # Mock stream to yield tokens
        def mock_stream(model, prompt, params):
            for i in range(10):
                yield f"token{i}"

        engine.stream.side_effect = mock_stream

        # Mock generate for warmup
        gen_result = MagicMock()
        gen_result.text = "warmup"
        gen_result.tokens_generated = 10
        gen_result.prompt_tokens = 20
        gen_result.finish_reason = "stop"
        gen_result.generation_time_s = 0.5
        gen_result.tokens_per_second = 20.0
        engine.generate.return_value = gen_result

        return engine

    @patch("tracellm.inference.benchmark.detect_hardware")
    @patch("tracellm.inference.benchmark.torch")
    @patch("tracellm.inference.benchmark.psutil")
    def test_run_single_category(self, mock_psutil, mock_torch, mock_hw):
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 4 * (1024**3)
        mock_psutil.Process.return_value = mock_process

        hw = MagicMock()
        hw.platform = "darwin"
        hw.cpu_count = 8
        hw.ram_total_gb = 32.0
        hw.gpu_count = 0
        hw.best_device = "cpu"
        hw.best_dtype = "float32"
        hw.gpus = []
        mock_hw.return_value = hw

        engine = self._mock_engine()
        bench = BenchmarkEngine(engine=engine)

        result = bench.run(
            model_name="test-model",
            max_tokens=64,
            num_runs=2,
            warmup_runs=1,
            prompt_categories=["short"],
        )

        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "test-model"
        assert "short" in result.prompt_results
        assert result.prompt_results["short"].avg_tokens_per_second > 0
        assert result.total_tokens_generated > 0

    @patch("tracellm.inference.benchmark.detect_hardware")
    @patch("tracellm.inference.benchmark.torch")
    @patch("tracellm.inference.benchmark.psutil")
    def test_run_custom_prompt(self, mock_psutil, mock_torch, mock_hw):
        mock_torch.cuda.is_available.return_value = False
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 2 * (1024**3)
        mock_psutil.Process.return_value = mock_process

        hw = MagicMock()
        hw.platform = "linux"
        hw.cpu_count = 4
        hw.ram_total_gb = 16.0
        hw.gpu_count = 0
        hw.best_device = "cpu"
        hw.best_dtype = "float32"
        hw.gpus = []
        mock_hw.return_value = hw

        engine = self._mock_engine()
        bench = BenchmarkEngine(engine=engine)

        result = bench.run(
            model_name="test-model",
            max_tokens=32,
            num_runs=1,
            warmup_runs=0,
            prompt_categories=[],
            custom_prompt="What is the meaning of life?",
        )

        assert "custom" in result.prompt_results

    @patch("tracellm.inference.benchmark.detect_hardware")
    @patch("tracellm.inference.benchmark.torch")
    @patch("tracellm.inference.benchmark.psutil")
    def test_progress_callback(self, mock_psutil, mock_torch, mock_hw):
        mock_torch.cuda.is_available.return_value = False
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 2 * (1024**3)
        mock_psutil.Process.return_value = mock_process

        hw = MagicMock()
        hw.platform = "linux"
        hw.cpu_count = 4
        hw.ram_total_gb = 16.0
        hw.gpu_count = 0
        hw.best_device = "cpu"
        hw.best_dtype = "float32"
        hw.gpus = []
        mock_hw.return_value = hw

        engine = self._mock_engine()
        bench = BenchmarkEngine(engine=engine)

        progress_calls = []

        def on_progress(category, run_idx, total):
            progress_calls.append((category, run_idx, total))

        result = bench.run(
            model_name="test-model",
            max_tokens=32,
            num_runs=3,
            warmup_runs=0,
            prompt_categories=["short"],
            on_progress=on_progress,
        )

        assert len(progress_calls) == 3
        assert all(cat == "short" for cat, _, _ in progress_calls)

    def test_percentile_calculation(self):
        assert BenchmarkEngine._percentile([1, 2, 3, 4, 5], 50) == 3.0
        assert BenchmarkEngine._percentile([1, 2, 3, 4, 5], 0) == 1.0
        assert BenchmarkEngine._percentile([1, 2, 3, 4, 5], 100) == 5.0
        assert BenchmarkEngine._percentile([], 50) == 0.0
        assert BenchmarkEngine._percentile([42], 50) == 42.0

    def test_percentile_interpolation(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        p25 = BenchmarkEngine._percentile(data, 25)
        assert 10.0 <= p25 <= 30.0

        p75 = BenchmarkEngine._percentile(data, 75)
        assert 30.0 <= p75 <= 50.0
