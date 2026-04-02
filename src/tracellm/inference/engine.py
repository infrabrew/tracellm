"""Core inference engine — ties together model loading, sampling, and caching."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

import torch

from tracellm.config import TraceConfig
from tracellm.inference.cache import KVCacheManager
from tracellm.inference.sampling import SamplingParams, sample_token
from tracellm.models.formats import ModelFormat
from tracellm.models.loader import LoadedModel, ModelLoader
from tracellm.models.registry import ModelRegistry
from tracellm.utils.logging import get_logger

log = get_logger("tracellm.inference.engine")


@dataclass
class GenerationResult:
    """Result of a completed generation."""
    text: str
    tokens_generated: int
    prompt_tokens: int
    finish_reason: str           # "stop" | "length" | "stop_sequence"
    generation_time_s: float
    tokens_per_second: float


class InferenceEngine:
    """High-level inference engine supporting streaming and batch generation."""

    def __init__(self, config: TraceConfig):
        self.config = config
        self.registry = ModelRegistry(cache_dir=config.models.cache_dir)
        self.loader = ModelLoader(config, self.registry)
        self.kv_cache = KVCacheManager(
            max_size_gb=config.inference.kv_cache.max_cache_size_gb,
            enabled=config.inference.kv_cache.enabled,
        )

    def load_model(self, name: str, device: str = "auto", dtype: str = "auto") -> LoadedModel:
        """Load a model by name."""
        return self.loader.load(name, device=device, dtype=dtype)

    def generate(
        self,
        model_name: str,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> GenerationResult:
        """Synchronous generation — returns the full result."""
        params = params or SamplingParams()
        loaded = self.loader.load(model_name)
        start = time.time()

        if loaded.format == ModelFormat.GGUF:
            return self._generate_gguf(loaded, prompt, params, start)

        return self._generate_transformers(loaded, prompt, params, start)

    def stream(
        self,
        model_name: str,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> Iterator[str]:
        """Streaming generation — yields tokens as they're produced."""
        params = params or SamplingParams()
        loaded = self.loader.load(model_name)

        if loaded.format == ModelFormat.GGUF:
            yield from self._stream_gguf(loaded, prompt, params)
        else:
            yield from self._stream_transformers(loaded, prompt, params)

    def _generate_transformers(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: SamplingParams,
        start: float,
    ) -> GenerationResult:
        """Generate with a HuggingFace Transformers or Mamba model."""
        tokenizer = loaded.tokenizer
        model = loaded.model

        inputs = tokenizer(prompt, return_tensors="pt").to(loaded.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        generated_tokens = []
        past_key_values = None
        current_ids = input_ids
        finish_reason = "length"

        with torch.inference_mode():
            for step in range(params.max_tokens):
                outputs = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                # Build full sequence for repetition penalty
                all_ids = torch.cat(
                    [input_ids, torch.tensor([generated_tokens], device=input_ids.device)],
                    dim=1,
                ) if generated_tokens else input_ids

                next_token = sample_token(logits, params, all_ids)
                token_id = next_token.item()

                # EOS check
                if token_id == tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break

                generated_tokens.append(token_id)
                current_ids = next_token

                # Stop sequence check
                if params.stop_sequences:
                    text_so_far = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    for seq in params.stop_sequences:
                        if seq in text_so_far:
                            text_so_far = text_so_far[: text_so_far.index(seq)]
                            finish_reason = "stop_sequence"
                            break
                    if finish_reason == "stop_sequence":
                        break

        elapsed = time.time() - start
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        n_gen = len(generated_tokens)

        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            prompt_tokens=prompt_len,
            finish_reason=finish_reason,
            generation_time_s=round(elapsed, 3),
            tokens_per_second=round(n_gen / max(elapsed, 0.001), 1),
        )

    def _stream_transformers(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: SamplingParams,
    ) -> Iterator[str]:
        """Streaming generation for Transformers/Mamba models."""
        tokenizer = loaded.tokenizer
        model = loaded.model

        inputs = tokenizer(prompt, return_tensors="pt").to(loaded.device)
        input_ids = inputs["input_ids"]

        generated_tokens = []
        past_key_values = None
        current_ids = input_ids
        text_buffer = ""

        with torch.inference_mode():
            for step in range(params.max_tokens):
                outputs = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                all_ids = torch.cat(
                    [input_ids, torch.tensor([generated_tokens], device=input_ids.device)],
                    dim=1,
                ) if generated_tokens else input_ids

                next_token = sample_token(logits, params, all_ids)
                token_id = next_token.item()

                if token_id == tokenizer.eos_token_id:
                    break

                generated_tokens.append(token_id)
                current_ids = next_token

                # Decode incrementally
                new_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                delta = new_text[len(text_buffer) :]
                text_buffer = new_text

                if delta:
                    # Stop sequence check
                    if params.stop_sequences:
                        for seq in params.stop_sequences:
                            if seq in text_buffer:
                                # Yield everything up to the stop sequence
                                final = text_buffer[: text_buffer.index(seq)]
                                final_delta = final[len(text_buffer) - len(delta) :]
                                if final_delta:
                                    yield final_delta
                                return
                    yield delta

    def _generate_gguf(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: SamplingParams,
        start: float,
    ) -> GenerationResult:
        """Generate with a llama.cpp GGUF model."""
        model = loaded.model  # Llama instance
        result = model(
            prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repetition_penalty,
            stop=params.stop_sequences or [],
        )

        elapsed = time.time() - start
        text = result["choices"][0]["text"]
        n_gen = result["usage"]["completion_tokens"]
        prompt_tokens = result["usage"]["prompt_tokens"]

        finish_reason = result["choices"][0].get("finish_reason", "stop")

        return GenerationResult(
            text=text,
            tokens_generated=n_gen,
            prompt_tokens=prompt_tokens,
            finish_reason=finish_reason,
            generation_time_s=round(elapsed, 3),
            tokens_per_second=round(n_gen / max(elapsed, 0.001), 1),
        )

    def _stream_gguf(
        self,
        loaded: LoadedModel,
        prompt: str,
        params: SamplingParams,
    ) -> Iterator[str]:
        """Streaming generation for GGUF models."""
        model = loaded.model
        for chunk in model(
            prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repetition_penalty,
            stop=params.stop_sequences or [],
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            if text:
                yield text
