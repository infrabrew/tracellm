"""KV cache management for transformer inference."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from tracellm.utils.logging import get_logger

log = get_logger("tracellm.inference.cache")


@dataclass
class CacheEntry:
    """A single KV cache entry for one sequence."""
    key: str                      # hash of the prompt prefix
    past_key_values: Any          # tuple of (key, value) tensors per layer
    token_count: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    @property
    def size_bytes(self) -> int:
        """Estimate memory footprint of the cached KV pairs."""
        total = 0
        if self.past_key_values is None:
            return 0
        for layer_kv in self.past_key_values:
            if isinstance(layer_kv, (tuple, list)):
                for tensor in layer_kv:
                    if isinstance(tensor, torch.Tensor):
                        total += tensor.nelement() * tensor.element_size()
        return total


class KVCacheManager:
    """LRU cache for KV pairs to speed up re-prompting with shared prefixes."""

    def __init__(self, max_size_gb: float = 4.0, enabled: bool = True):
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.enabled = enabled
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size = 0

    @property
    def size_gb(self) -> float:
        return self._current_size / (1024**3)

    @property
    def entry_count(self) -> int:
        return len(self._cache)

    def _make_key(self, token_ids: list[int]) -> str:
        """Create a cache key from a list of token IDs."""
        import hashlib
        raw = ",".join(str(t) for t in token_ids)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, token_ids: list[int]) -> CacheEntry | None:
        """Look up cached KV pairs for a token prefix."""
        if not self.enabled:
            return None

        key = self._make_key(token_ids)
        entry = self._cache.get(key)
        if entry:
            entry.last_accessed = time.time()
            self._cache.move_to_end(key)
            return entry
        return None

    def put(self, token_ids: list[int], past_key_values: Any) -> None:
        """Store KV pairs in cache, evicting LRU entries if needed."""
        if not self.enabled:
            return

        key = self._make_key(token_ids)
        entry = CacheEntry(
            key=key,
            past_key_values=past_key_values,
            token_count=len(token_ids),
        )

        entry_size = entry.size_bytes

        # Evict until we have room
        while self._current_size + entry_size > self.max_size_bytes and self._cache:
            evict_key, evict_entry = self._cache.popitem(last=False)
            self._current_size -= evict_entry.size_bytes
            del evict_entry.past_key_values
            log.debug(f"Evicted cache entry {evict_key}")

        if entry_size <= self.max_size_bytes:
            self._cache[key] = entry
            self._current_size += entry_size

    def clear(self) -> None:
        """Flush the entire cache."""
        for entry in self._cache.values():
            del entry.past_key_values
        self._cache.clear()
        self._current_size = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("KV cache cleared")
