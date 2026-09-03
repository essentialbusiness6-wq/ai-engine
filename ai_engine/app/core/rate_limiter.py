"""
Simple in-memory sliding-window rate limiter, keyed per user. Swappable for
a Redis-backed implementation in multi-process deployments by implementing
the same `allow(key)` contract.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class SlidingWindowRateLimiter:
    def __init__(self, max_requests_per_minute: int = 60):
        self._max_requests = max_requests_per_minute
        self._window_seconds = 60.0
        self._hits: dict[str, deque] = {}
        self._lock = threading.RLock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            window = self._hits.setdefault(key, deque())
            while window and now - window[0] > self._window_seconds:
                window.popleft()
            if len(window) >= self._max_requests:
                return False
            window.append(now)
            return True
