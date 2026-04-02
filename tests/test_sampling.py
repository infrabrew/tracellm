"""Tests for sampling strategies."""

import torch

from tracellm.inference.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    top_k_filter,
    top_p_filter,
    sample_token,
)


def test_greedy_sampling():
    """Temperature=0 should always pick the highest logit."""
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
    params = SamplingParams(temperature=0.0)
    token = sample_token(logits, params)
    assert token.item() == 1  # index of 5.0


def test_top_k_filter():
    """Top-k should zero out everything below top k values."""
    logits = torch.tensor([[1.0, 5.0, 2.0, 3.0, 4.0]])
    filtered = top_k_filter(logits.clone(), k=2)
    # Only indices 1 (5.0) and 4 (4.0) should remain
    assert filtered[0, 1] == 5.0
    assert filtered[0, 4] == 4.0
    assert filtered[0, 0] == float("-inf")
    assert filtered[0, 2] == float("-inf")


def test_top_p_filter():
    """Top-p should keep the minimum set of tokens with cumulative prob >= p."""
    # Strong preference for index 1 — with low p, only that should survive
    logits = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
    filtered = top_p_filter(logits.clone(), p=0.5)
    # The dominant token should remain non-inf
    assert filtered[0, 1] > float("-inf")


def test_repetition_penalty():
    """Repetition penalty should reduce probability of already-seen tokens."""
    logits = torch.tensor([[5.0, 5.0, 5.0]])
    input_ids = torch.tensor([[0, 1]])  # tokens 0 and 1 were seen

    penalized = apply_repetition_penalty(logits.clone(), input_ids, penalty=1.5)
    # Tokens 0 and 1 should now be lower than token 2 (unseen)
    assert penalized[0, 2] > penalized[0, 0]
    assert penalized[0, 2] > penalized[0, 1]


def test_sample_token_returns_valid_index():
    """sample_token should return indices within vocab range."""
    vocab_size = 100
    logits = torch.randn(1, vocab_size)
    params = SamplingParams(temperature=0.8, top_k=10, top_p=0.9)
    token = sample_token(logits, params)
    assert 0 <= token.item() < vocab_size


def test_sample_token_with_seed_is_deterministic():
    """Same seed + logits should produce same token."""
    logits = torch.randn(1, 1000)
    params = SamplingParams(temperature=0.8, top_k=50)

    torch.manual_seed(42)
    t1 = sample_token(logits.clone(), params)

    torch.manual_seed(42)
    t2 = sample_token(logits.clone(), params)

    assert t1.item() == t2.item()
