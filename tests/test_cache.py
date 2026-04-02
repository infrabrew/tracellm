"""Tests for KV cache management."""

import torch

from tracellm.inference.cache import KVCacheManager, CacheEntry


def test_cache_put_and_get():
    """Should store and retrieve KV pairs."""
    cache = KVCacheManager(max_size_gb=1.0, enabled=True)
    token_ids = [1, 2, 3, 4, 5]

    # Fake past_key_values — one layer with small tensors
    kv = ((torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)),)

    cache.put(token_ids, kv)
    entry = cache.get(token_ids)

    assert entry is not None
    assert entry.token_count == 5
    assert len(entry.past_key_values) == 1


def test_cache_miss():
    """Should return None for uncached token sequences."""
    cache = KVCacheManager(max_size_gb=1.0, enabled=True)
    assert cache.get([1, 2, 3]) is None


def test_cache_disabled():
    """Disabled cache should never store or return entries."""
    cache = KVCacheManager(max_size_gb=1.0, enabled=False)
    kv = ((torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)),)

    cache.put([1, 2, 3], kv)
    assert cache.get([1, 2, 3]) is None
    assert cache.entry_count == 0


def test_cache_eviction():
    """Exceeding size limit should evict LRU entries."""
    # Tiny cache — 1 KB
    cache = KVCacheManager(max_size_gb=1e-6, enabled=True)

    kv1 = ((torch.randn(1, 4, 10, 32), torch.randn(1, 4, 10, 32)),)
    kv2 = ((torch.randn(1, 4, 10, 32), torch.randn(1, 4, 10, 32)),)

    cache.put([1, 2], kv1)
    cache.put([3, 4], kv2)

    # At least one should have been evicted due to tiny size
    # (both entries are larger than 1KB)
    assert cache.entry_count <= 1


def test_cache_clear():
    """Clear should empty the cache."""
    cache = KVCacheManager(max_size_gb=1.0, enabled=True)
    kv = ((torch.randn(1, 4, 5, 32), torch.randn(1, 4, 5, 32)),)
    cache.put([1, 2], kv)
    assert cache.entry_count == 1

    cache.clear()
    assert cache.entry_count == 0
    assert cache.size_gb == 0.0


def test_cache_entry_size():
    """CacheEntry should report approximate memory size."""
    kv = ((torch.randn(1, 8, 128, 64), torch.randn(1, 8, 128, 64)),)
    entry = CacheEntry(key="test", past_key_values=kv, token_count=128)

    # 2 tensors * 1 * 8 * 128 * 64 * 4 bytes = ~524288 bytes per tensor
    expected_bytes = 2 * 1 * 8 * 128 * 64 * 4
    assert abs(entry.size_bytes - expected_bytes) < 100
