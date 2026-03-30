"""In-memory LRU cache for search results.

Key: SHA-256(query + str(top_n) + str(sorted(filters.items())))
TTL: 300 seconds, max 100 entries, eviction via OrderedDict LRU.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any

from ragcore.models import SearchResponse


_MAX_SIZE = 100
_TTL_SECONDS = 300.0


class SearchCache:
    """Thread-safe (GIL-protected) LRU cache for SearchResponse objects."""

    def __init__(self, max_size: int = _MAX_SIZE, ttl: float = _TTL_SECONDS) -> None:
        self._max_size = max_size
        self._ttl = ttl
        # OrderedDict: key -> (SearchResponse, inserted_at_monotonic)
        self._store: OrderedDict[str, tuple[SearchResponse, float]] = OrderedDict()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(query: str, top_n: int | None, filters: dict) -> str:
        raw = query + str(top_n) + str(sorted(filters.items()))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, ts) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> SearchResponse | None:
        """Return cached response or None if missing/expired."""
        self._evict_expired()
        if key not in self._store:
            return None
        response, ts = self._store[key]
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        # Move to end (most-recently-used)
        self._store.move_to_end(key)
        return response

    def set(self, key: str, response: SearchResponse) -> None:
        """Store a response; evicts LRU entry if at capacity."""
        self._evict_expired()
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (response, time.monotonic())
        while len(self._store) > self._max_size:
            # Evict least-recently-used (front of OrderedDict)
            self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Remove a specific key if present."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
