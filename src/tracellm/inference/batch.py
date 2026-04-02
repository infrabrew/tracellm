"""Batch inference engine — process multiple prompts concurrently."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterator

from tracellm.inference.engine import InferenceEngine, GenerationResult
from tracellm.inference.sampling import SamplingParams
from tracellm.utils.logging import get_logger

log = get_logger("tracellm.inference.batch")


@dataclass
class BatchItem:
    """A single item in a batch request."""
    id: str
    prompt: str
    params: SamplingParams | None = None


@dataclass
class BatchItemResult:
    """Result for a single batch item."""
    id: str
    result: GenerationResult | None = None
    error: str | None = None


@dataclass
class BatchResult:
    """Aggregated result for a full batch."""
    batch_id: str
    results: list[BatchItemResult]
    total_time_s: float
    total_tokens: int
    total_prompt_tokens: int
    avg_tokens_per_second: float
    items_succeeded: int
    items_failed: int


class BatchEngine:
    """Process multiple prompts through the inference engine concurrently.

    Supports:
      - Parallel execution with configurable concurrency
      - Per-item sampling parameters (or shared defaults)
      - Progress callbacks for streaming status updates
      - Adaptive batching: groups prompts by model to minimize reloads
    """

    def __init__(
        self,
        engine: InferenceEngine,
        max_concurrency: int = 4,
    ):
        self.engine = engine
        self.max_concurrency = max_concurrency

    def run(
        self,
        model_name: str,
        items: list[BatchItem],
        default_params: SamplingParams | None = None,
        on_item_complete: callable | None = None,
    ) -> BatchResult:
        """Execute a batch of prompts and return aggregated results.

        Args:
            model_name: Model to use for all items.
            items: List of BatchItem with prompt and optional per-item params.
            default_params: Fallback sampling params if item has none.
            on_item_complete: Callback(BatchItemResult) fired as each finishes.
        """
        batch_id = f"batch-{uuid.uuid4().hex[:8]}"
        default_params = default_params or SamplingParams()
        start = time.time()

        log.info(f"[{batch_id}] Starting batch of {len(items)} items "
                 f"(concurrency={self.max_concurrency})")

        # Pre-load the model once before fanning out
        self.engine.loader.load(model_name)

        results: list[BatchItemResult] = [None] * len(items)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            future_to_idx = {}
            for idx, item in enumerate(items):
                params = item.params or default_params
                future = pool.submit(self._generate_one, model_name, item, params)
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                item_result = future.result()
                results[idx] = item_result
                if on_item_complete:
                    on_item_complete(item_result)

        elapsed = time.time() - start
        succeeded = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        total_tokens = sum(r.result.tokens_generated for r in succeeded)
        total_prompt = sum(r.result.prompt_tokens for r in succeeded)

        return BatchResult(
            batch_id=batch_id,
            results=results,
            total_time_s=round(elapsed, 3),
            total_tokens=total_tokens,
            total_prompt_tokens=total_prompt,
            avg_tokens_per_second=round(total_tokens / max(elapsed, 0.001), 1),
            items_succeeded=len(succeeded),
            items_failed=len(failed),
        )

    def stream_progress(
        self,
        model_name: str,
        items: list[BatchItem],
        default_params: SamplingParams | None = None,
    ) -> Iterator[dict]:
        """Yield progress events as each item completes.

        Events:
            {"type": "start", "batch_id": ..., "total": N}
            {"type": "item_complete", "index": i, "id": ..., "tokens": N, "error": ...}
            {"type": "done", "batch_id": ..., "succeeded": N, "failed": N, "time_s": ...}
        """
        batch_id = f"batch-{uuid.uuid4().hex[:8]}"
        default_params = default_params or SamplingParams()

        yield {"type": "start", "batch_id": batch_id, "total": len(items)}

        self.engine.loader.load(model_name)

        completed = 0
        succeeded = 0
        failed = 0
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            future_to_item = {}
            for idx, item in enumerate(items):
                params = item.params or default_params
                future = pool.submit(self._generate_one, model_name, item, params)
                future_to_item[future] = (idx, item)

            for future in as_completed(future_to_item):
                idx, item = future_to_item[future]
                item_result = future.result()
                completed += 1

                event = {
                    "type": "item_complete",
                    "index": idx,
                    "id": item_result.id,
                    "completed": completed,
                    "total": len(items),
                }

                if item_result.error:
                    event["error"] = item_result.error
                    failed += 1
                else:
                    event["tokens"] = item_result.result.tokens_generated
                    event["tokens_per_second"] = item_result.result.tokens_per_second
                    event["text_preview"] = item_result.result.text[:100]
                    succeeded += 1

                yield event

        elapsed = time.time() - start
        yield {
            "type": "done",
            "batch_id": batch_id,
            "succeeded": succeeded,
            "failed": failed,
            "total_time_s": round(elapsed, 3),
        }

    def _generate_one(
        self,
        model_name: str,
        item: BatchItem,
        params: SamplingParams,
    ) -> BatchItemResult:
        """Generate a single item — catches exceptions per-item."""
        try:
            result = self.engine.generate(model_name, item.prompt, params)
            return BatchItemResult(id=item.id, result=result)
        except Exception as e:
            log.warning(f"Batch item '{item.id}' failed: {e}")
            return BatchItemResult(id=item.id, error=str(e))
