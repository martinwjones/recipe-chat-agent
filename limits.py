"""Per-user usage limits: requests and LLM tokens per minute.

Each user's minute starts with their first request. A request is refused once the user has
used all their requests or tokens for the minute. Token use is only known after the model
replies, so a request may finish above the token limit; either way, going over the limit
blocks the user until their minute ends plus a penalty (one extra minute), after which both
counters start again from zero.

Counters live in memory, which is fine for a single-replica demo agent.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

WINDOW_SECONDS = 60


@dataclass
class _Usage:
    window_start: float
    requests: int = 0
    tokens: int = 0
    blocked_until: float = 0.0


class UsageLimiter:
    def __init__(self) -> None:
        self.max_requests = int(os.environ.get("USER_REQUESTS_PER_MINUTE", "20"))
        self.max_tokens = int(os.environ.get("USER_TOKENS_PER_MINUTE", "5000"))
        self.penalty_seconds = int(os.environ.get("USER_LIMIT_PENALTY_SECONDS", "60"))
        self._users: dict[str, _Usage] = {}
        self._lock = threading.Lock()

    def start(self, user: str) -> dict | None:
        """Counts a request for the user. Returns None if it may proceed, or the usage
        snapshot (with blocked=True) if the user is over a limit."""
        now = time.time()
        with self._lock:
            u = self._current(user, now)
            if u.blocked_until > now:
                return self._snapshot(u, now)
            if u.requests >= self.max_requests or u.tokens >= self.max_tokens:
                u.blocked_until = u.window_start + WINDOW_SECONDS + self.penalty_seconds
                return self._snapshot(u, now)
            u.requests += 1
            return None

    def finish(self, user: str, tokens: int) -> dict:
        """Adds the request's LLM tokens and returns the user's usage after it."""
        now = time.time()
        with self._lock:
            u = self._current(user, now)
            u.tokens += tokens
            if u.tokens > self.max_tokens and u.blocked_until <= now:
                u.blocked_until = u.window_start + WINDOW_SECONDS + self.penalty_seconds
            return self._snapshot(u, now)

    def usage(self, user: str) -> dict:
        now = time.time()
        with self._lock:
            return self._snapshot(self._current(user, now), now)

    def _current(self, user: str, now: float) -> _Usage:
        u = self._users.get(user)
        # A new minute starts once the old one (and any penalty) is over.
        if u is None or (now >= u.window_start + WINDOW_SECONDS and now >= u.blocked_until):
            u = self._users[user] = _Usage(window_start=now)
        return u

    def _snapshot(self, u: _Usage, now: float) -> dict:
        blocked = u.blocked_until > now
        resets_at = u.blocked_until if blocked else u.window_start + WINDOW_SECONDS
        return {
            "requests": {"used": u.requests, "limit": self.max_requests},
            "tokens": {"used": u.tokens, "limit": self.max_tokens},
            "blocked": blocked,
            "resetsInSeconds": max(0, round(resets_at - now)),
            "penaltySeconds": self.penalty_seconds,
        }
