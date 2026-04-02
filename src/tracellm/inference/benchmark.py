"""Model benchmarking — measure latency, throughput, and memory usage."""

from __future__ import annotations

import gc
import statistics
import time
import uuid
from dataclasses import dataclass, field

import torch

from tracellm.inference.engine import InferenceEngine, GenerationResult
from tracellm.inference.sampling import SamplingParams
from tracellm.utils.hardware import detect_hardware
from tracellm.utils.logging import get_logger

log = get_logger("tracellm.inference.benchmark")

# Standard benchmark prompts at varying lengths
BENCHMARK_PROMPTS = {
    "short": "What is the capital of France?",
    "medium": (
        "Explain the key differences between supervised and unsupervised machine learning. "
        "Include examples of algorithms for each category and discuss when you would choose "
        "one approach over the other in a real-world application."
    ),
    "long": (
        "Write a comprehensive technical guide on building a production-ready REST API. "
        "Cover the following topics in detail: authentication and authorization using JWT tokens, "
        "rate limiting strategies for different tiers of users, input validation and sanitization, "
        "error handling patterns and HTTP status codes, database connection pooling and query optimization, "
        "caching strategies using Redis, logging and monitoring best practices, "
        "deployment considerations for containerized environments, "
        "API versioning approaches, and documentation with OpenAPI/Swagger. "
        "For each topic, provide code examples in Python using FastAPI."
    ),
    "code": (
        "Implement a thread-safe LRU cache in Python with the following features: "
        "configurable max size, TTL expiration per entry, hit/miss statistics tracking, "
        "and a method to bulk-load entries. Include type hints and docstrings."
    ),
}


@dataclass
class BenchmarkRun:
    """Result of a single generation run within a benchmark."""
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token_s: float
    total_time_s: float
    tokens_per_second: float


@dataclass
class PromptBenchmark:
    """Benchmark results for a single prompt category."""
    category: str
    prompt_tokens: int
    runs: list[BenchmarkRun]
    avg_completion_tokens: float
    avg_time_to_first_token_s: float
    avg_total_time_s: float
    avg_tokens_per_second: float
    p50_tokens_per_second: float
    p90_tokens_per_second: float
    p99_tokens_per_second: float
    min_tokens_per_second: float
    max_tokens_per_second: float


@dataclass
class MemorySnapshot:
    """GPU/system memory usage at a point in time."""
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_peak_gb: float = 0.0
    ram_used_gb: float = 0.0


@dataclass
class BenchmarkResult:
    """Full benchmark report for a model."""
    benchmark_id: str
    model_name: str
    device: str
    dtype: str
    prompt_results: dict[str, PromptBenchmark]
    memory_baseline: MemorySnapshot
    memory_loaded: MemorySnapshot
    memory_peak: MemorySnapshot
    total_time_s: float
    total_tokens_generated: int
    overall_tokens_per_second: float
    hardware_summary: dict


class BenchmarkEngine:
    """Run standardized benchmarks against loaded models.

    Measures:
      - Time to first token (TTFT)
      - Tokens per second (throughput)
      - Latency percentiles (p50, p90, p99)
      - GPU/RAM memory usage (baseline, loaded, peak)
      - Results across multiple prompt lengths
    """

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def run(
        self,
        model_name: str,
        max_tokens: int = 128,
        num_runs: int = 3,
        warmup_runs: int = 1,
        prompt_categories: list[str] | None = None,
        custom_prompt: str | None = None,
        on_progress: callable | None = None,
    ) -> BenchmarkResult:
        """Run a full benchmark suite against a model.

        Args:
            model_name: Model to benchmark.
            max_tokens: Max tokens to generate per run.
            num_runs: Number of timed runs per prompt category.
            warmup_runs: Untimed warmup runs (for JIT, cache warming).
            prompt_categories: Which prompt categories to test (default: all).
            custom_prompt: Additional custom prompt to benchmark.
            on_progress: Callback(category, run_idx, total_runs) for UI.
        """
        benchmark_id = f"bench-{uuid.uuid4().hex[:8]}"
        categories = prompt_categories or list(BENCHMARK_PROMPTS.keys())

        prompts = {k: v for k, v in BENCHMARK_PROMPTS.items() if k in categories}
        if custom_prompt:
            prompts["custom"] = custom_prompt

        params = SamplingParams(
            temperature=0.0,  # Greedy for reproducibility
            max_tokens=max_tokens,
            top_k=1,
        )

        log.info(f"[{benchmark_id}] Benchmarking '{model_name}' — "
                 f"{len(prompts)} categories, {num_runs} runs each")

        # Memory baseline (before model load)
        mem_baseline = self._snapshot_memory()

        # Load model
        start_total = time.time()
        loaded = self.engine.loader.load(model_name)

        mem_loaded = self._snapshot_memory()

        # Reset peak tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prompt_results: dict[str, PromptBenchmark] = {}
        total_tokens = 0

        for cat_name, prompt_text in prompts.items():
            # Warmup (untimed)
            for _ in range(warmup_runs):
                self.engine.generate(model_name, prompt_text, params)

            runs: list[BenchmarkRun] = []
            for run_idx in range(num_runs):
                if on_progress:
                    on_progress(cat_name, run_idx + 1, num_runs)

                run = self._timed_run(model_name, prompt_text, params)
                runs.append(run)
                total_tokens += run.completion_tokens

            tps_values = [r.tokens_per_second for r in runs]
            sorted_tps = sorted(tps_values)

            prompt_results[cat_name] = PromptBenchmark(
                category=cat_name,
                prompt_tokens=runs[0].prompt_tokens,
                runs=runs,
                avg_completion_tokens=statistics.mean(r.completion_tokens for r in runs),
                avg_time_to_first_token_s=round(
                    statistics.mean(r.time_to_first_token_s for r in runs), 4
                ),
                avg_total_time_s=round(statistics.mean(r.total_time_s for r in runs), 4),
                avg_tokens_per_second=round(statistics.mean(tps_values), 1),
                p50_tokens_per_second=round(self._percentile(sorted_tps, 50), 1),
                p90_tokens_per_second=round(self._percentile(sorted_tps, 90), 1),
                p99_tokens_per_second=round(self._percentile(sorted_tps, 99), 1),
                min_tokens_per_second=round(min(tps_values), 1),
                max_tokens_per_second=round(max(tps_values), 1),
            )

        mem_peak = self._snapshot_memory()
        elapsed_total = time.time() - start_total

        hw = detect_hardware()
        hw_summary = {
            "platform": hw.platform,
            "cpu_count": hw.cpu_count,
            "ram_total_gb": hw.ram_total_gb,
            "gpu_count": hw.gpu_count,
            "best_device": hw.best_device,
            "best_dtype": hw.best_dtype,
        }
        if hw.gpus:
            hw_summary["gpus"] = [
                {"name": g.name, "vram_gb": g.memory_total_gb}
                for g in hw.gpus
            ]

        return BenchmarkResult(
            benchmark_id=benchmark_id,
            model_name=model_name,
            device=str(loaded.device),
            dtype=str(loaded.dtype),
            prompt_results=prompt_results,
            memory_baseline=mem_baseline,
            memory_loaded=mem_loaded,
            memory_peak=mem_peak,
            total_time_s=round(elapsed_total, 3),
            total_tokens_generated=total_tokens,
            overall_tokens_per_second=round(
                total_tokens / max(elapsed_total, 0.001), 1
            ),
            hardware_summary=hw_summary,
        )

    def _timed_run(
        self,
        model_name: str,
        prompt: str,
        params: SamplingParams,
    ) -> BenchmarkRun:
        """Execute a single timed generation, measuring TTFT and throughput."""
        start = time.time()
        first_token_time = None
        token_count = 0
        prompt_tokens = 0

        # Use streaming to measure time-to-first-token
        for token in self.engine.stream(model_name, prompt, params):
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1

        elapsed = time.time() - start
        ttft = (first_token_time - start) if first_token_time else elapsed

        # Get prompt token count from a quick tokenization
        loaded = self.engine.loader.load(model_name)
        if hasattr(loaded, "tokenizer") and loaded.tokenizer:
            prompt_tokens = len(loaded.tokenizer.encode(prompt))

        return BenchmarkRun(
            prompt_tokens=prompt_tokens,
            completion_tokens=token_count,
            time_to_first_token_s=round(ttft, 4),
            total_time_s=round(elapsed, 4),
            tokens_per_second=round(token_count / max(elapsed, 0.001), 1),
        )

    def _snapshot_memory(self) -> MemorySnapshot:
        """Capture current memory usage."""
        import psutil

        process = psutil.Process()
        ram_used = process.memory_info().rss / (1024**3)

        gpu_alloc = 0.0
        gpu_reserved = 0.0
        gpu_peak = 0.0

        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)

        return MemorySnapshot(
            gpu_allocated_gb=round(gpu_alloc, 3),
            gpu_reserved_gb=round(gpu_reserved, 3),
            gpu_peak_gb=round(gpu_peak, 3),
            ram_used_gb=round(ram_used, 3),
        )

    @staticmethod
    def _percentile(sorted_data: list[float], pct: float) -> float:
        """Compute percentile from pre-sorted data."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (pct / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        d = k - f
        return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])
