"""Tests for batch inference engine."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from tracellm.inference.batch import BatchEngine, BatchItem, BatchResult, BatchItemResult
from tracellm.inference.engine import GenerationResult
from tracellm.inference.sampling import SamplingParams


def _make_gen_result(text: str = "Hello", tokens: int = 5) -> GenerationResult:
    return GenerationResult(
        text=text,
        tokens_generated=tokens,
        prompt_tokens=10,
        finish_reason="stop",
        generation_time_s=0.5,
        tokens_per_second=10.0,
    )


def _mock_engine():
    engine = MagicMock()
    engine.generate.return_value = _make_gen_result()
    engine.loader.load.return_value = MagicMock()
    return engine


class TestBatchItem:
    def test_creation(self):
        item = BatchItem(id="test-1", prompt="Hello world")
        assert item.id == "test-1"
        assert item.prompt == "Hello world"
        assert item.params is None

    def test_with_custom_params(self):
        params = SamplingParams(temperature=0.5)
        item = BatchItem(id="test-2", prompt="Hi", params=params)
        assert item.params.temperature == 0.5


class TestBatchEngine:
    def test_run_single_item(self):
        engine = _mock_engine()
        batch = BatchEngine(engine=engine, max_concurrency=1)

        items = [BatchItem(id="item-0", prompt="What is 2+2?")]
        result = batch.run("test-model", items)

        assert isinstance(result, BatchResult)
        assert len(result.results) == 1
        assert result.items_succeeded == 1
        assert result.items_failed == 0
        assert result.total_tokens == 5

    def test_run_multiple_items(self):
        engine = _mock_engine()
        batch = BatchEngine(engine=engine, max_concurrency=2)

        items = [
            BatchItem(id=f"item-{i}", prompt=f"Prompt {i}")
            for i in range(5)
        ]
        result = batch.run("test-model", items)

        assert len(result.results) == 5
        assert result.items_succeeded == 5
        assert result.items_failed == 0
        assert result.total_tokens == 25  # 5 items * 5 tokens each
        assert result.batch_id.startswith("batch-")

    def test_run_with_failures(self):
        engine = _mock_engine()
        engine.generate.side_effect = [
            _make_gen_result(),
            RuntimeError("Model OOM"),
            _make_gen_result(),
        ]
        batch = BatchEngine(engine=engine, max_concurrency=1)

        items = [BatchItem(id=f"item-{i}", prompt=f"P{i}") for i in range(3)]
        result = batch.run("test-model", items)

        assert result.items_succeeded == 2
        assert result.items_failed == 1
        assert any(r.error is not None for r in result.results)

    def test_run_with_callback(self):
        engine = _mock_engine()
        batch = BatchEngine(engine=engine, max_concurrency=1)

        items = [BatchItem(id="item-0", prompt="Hello")]
        completed = []

        def on_complete(item_result):
            completed.append(item_result)

        batch.run("test-model", items, on_item_complete=on_complete)
        assert len(completed) == 1

    def test_run_with_per_item_params(self):
        engine = _mock_engine()
        batch = BatchEngine(engine=engine, max_concurrency=1)

        custom_params = SamplingParams(temperature=0.1, max_tokens=64)
        items = [
            BatchItem(id="item-0", prompt="Hello", params=custom_params),
            BatchItem(id="item-1", prompt="World"),
        ]

        default_params = SamplingParams(temperature=0.7, max_tokens=512)
        result = batch.run("test-model", items, default_params)

        assert result.items_succeeded == 2
        # Verify the custom params were used for item-0
        calls = engine.generate.call_args_list
        assert calls[0][0][2].temperature == 0.1  # item-0 used custom
        assert calls[1][0][2].temperature == 0.7  # item-1 used default

    def test_stream_progress(self):
        engine = _mock_engine()
        batch = BatchEngine(engine=engine, max_concurrency=1)

        items = [BatchItem(id=f"item-{i}", prompt=f"P{i}") for i in range(3)]
        events = list(batch.stream_progress("test-model", items))

        # Should have: 1 start + 3 item_complete + 1 done
        assert events[0]["type"] == "start"
        assert events[0]["total"] == 3
        assert sum(1 for e in events if e["type"] == "item_complete") == 3
        assert events[-1]["type"] == "done"
        assert events[-1]["succeeded"] == 3

    def test_concurrency_respected(self):
        engine = _mock_engine()
        batch = BatchEngine(engine=engine, max_concurrency=2)

        items = [BatchItem(id=f"item-{i}", prompt=f"P{i}") for i in range(10)]
        result = batch.run("test-model", items)

        assert result.items_succeeded == 10

    def test_empty_batch(self):
        engine = _mock_engine()
        batch = BatchEngine(engine=engine)

        result = batch.run("test-model", [])
        assert len(result.results) == 0
        assert result.items_succeeded == 0
        assert result.items_failed == 0

    def test_result_ordering_preserved(self):
        """Results should be in the same order as inputs, regardless of completion order."""
        call_count = 0

        def slow_then_fast(model, prompt, params):
            nonlocal call_count
            call_count += 1
            return _make_gen_result(text=prompt)

        engine = _mock_engine()
        engine.generate.side_effect = slow_then_fast
        batch = BatchEngine(engine=engine, max_concurrency=4)

        items = [BatchItem(id=f"item-{i}", prompt=f"Prompt-{i}") for i in range(5)]
        result = batch.run("test-model", items)

        for i, r in enumerate(result.results):
            assert r.id == f"item-{i}"
