import hashlib
import secrets
import threading
import time
from collections import defaultdict, deque

KEY_PREFIX = "rag_"


def generate_api_key() -> tuple[str, str, str]:
    """Return (plain_key, display_prefix, sha256_hash). Only the hash is stored."""
    plain = KEY_PREFIX + secrets.token_urlsafe(32)
    return plain, plain[:10], hash_api_key(plain)


def hash_api_key(plain: str) -> str:
    return hashlib.sha256(plain.encode()).hexdigest()


def admin_token_matches(expected: str, provided: str | None) -> bool:
    if not expected or not provided:
        return False
    return secrets.compare_digest(expected, provided)


class SlidingWindowLimiter:
    """In-memory per-key rate limiter (single-process; swap for Redis to scale out)."""

    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True
