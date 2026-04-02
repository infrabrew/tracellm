"""Sampling strategies for text generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    """Generation sampling parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_tokens: int = 4096
    stop_sequences: list[str] | None = None
    seed: int | None = None


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Penalize tokens that have already appeared in the sequence."""
    if penalty == 1.0:
        return logits

    score = torch.gather(logits, 1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, input_ids, score)
    return logits


def apply_frequency_presence_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    """Apply frequency and presence penalties (OpenAI-style)."""
    if frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits

    # Count token frequencies
    vocab_size = logits.shape[-1]
    counts = torch.zeros_like(logits)
    counts.scatter_add_(1, input_ids, torch.ones_like(input_ids, dtype=logits.dtype))

    logits -= frequency_penalty * counts
    logits -= presence_penalty * (counts > 0).float()
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits below the top-k threshold."""
    if k <= 0:
        return logits
    top_k_val = torch.topk(logits, min(k, logits.size(-1))).values[..., -1, None]
    return logits.masked_fill(logits < top_k_val, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling — keep the smallest set of tokens with cumulative prob >= p."""
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Mask tokens beyond the nucleus
    mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[mask] = float("-inf")

    # Restore original order
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    input_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply all sampling transforms and pick a token.

    Args:
        logits: Raw logits of shape (batch, vocab_size).
        params: Sampling configuration.
        input_ids: Previous token IDs for repetition/frequency penalty.

    Returns:
        Selected token IDs of shape (batch, 1).
    """
    # Repetition penalty
    if input_ids is not None and params.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, input_ids, params.repetition_penalty)

    # Frequency / presence penalty
    if input_ids is not None:
        logits = apply_frequency_presence_penalty(
            logits, input_ids, params.frequency_penalty, params.presence_penalty
        )

    # Greedy
    if params.temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    # Temperature scaling
    logits = logits / max(params.temperature, 1e-7)

    # Top-k
    logits = top_k_filter(logits, params.top_k)

    # Top-p (nucleus)
    logits = top_p_filter(logits, params.top_p)

    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
