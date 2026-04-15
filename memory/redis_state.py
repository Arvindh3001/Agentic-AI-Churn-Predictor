"""
Redis Short-Term State Manager
================================
Manages agent task state, partial results, and conversation context
in Redis with configurable TTL.

Key namespaces:
  churn:task:{task_id}        → full AgentState snapshot (JSON)
  churn:task:{task_id}:status → current step string
  churn:task:{task_id}:stream → list of streamed progress events
  churn:customer:{id}:context → last-seen context for a customer

Usage:
    from memory.redis_state import RedisStateManager

    store = RedisStateManager()
    store.save_state(task_id, agent_state)
    state = store.load_state(task_id)
    store.append_stream_event(task_id, {"step": "prediction", "status": "done"})
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import redis
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)

DEFAULT_TTL_SECONDS = 60 * 60 * 24  # 24 hours
STREAM_TTL_SECONDS = 60 * 60 * 2    # 2 hours for streaming events


class RedisStateManager:
    """
    Redis-backed short-term memory for agent task state.

    Falls back gracefully to an in-memory dict when Redis is unreachable
    (useful for development without Docker).
    """

    def __init__(self, ttl: int = DEFAULT_TTL_SECONDS) -> None:
        self.ttl = ttl
        self._client: redis.Redis | None = None
        self._fallback: dict[str, str] = {}
        self._connect()

    def _connect(self) -> None:
        try:
            client = redis.from_url(settings.db.redis_url, decode_responses=True)
            client.ping()
            self._client = client
            logger.info("Redis connected", url=settings.db.redis_url)
        except Exception as exc:
            logger.warning(
                "Redis unreachable — using in-memory fallback",
                error=str(exc),
            )
            self._client = None

    # ------------------------------------------------------------------ #
    # Task State
    # ------------------------------------------------------------------ #

    def save_state(self, task_id: str, state: dict[str, Any]) -> None:
        """Persist full AgentState snapshot for a task."""
        key = f"churn:task:{task_id}"
        payload = json.dumps(state, default=str)
        self._set(key, payload, self.ttl)

        # Also update quick-access status key
        status_key = f"churn:task:{task_id}:status"
        self._set(status_key, state.get("current_step", "unknown"), self.ttl)

    def load_state(self, task_id: str) -> dict[str, Any] | None:
        """Load AgentState snapshot for a task. Returns None if not found."""
        key = f"churn:task:{task_id}"
        raw = self._get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def get_status(self, task_id: str) -> str:
        """Fast status check without deserialising the full state."""
        key = f"churn:task:{task_id}:status"
        return self._get(key) or "not_found"

    def delete_task(self, task_id: str) -> None:
        """Clean up all keys for a completed task."""
        for suffix in ("", ":status", ":stream"):
            self._delete(f"churn:task:{task_id}{suffix}")

    # ------------------------------------------------------------------ #
    # Streaming Events
    # ------------------------------------------------------------------ #

    def append_stream_event(
        self,
        task_id: str,
        event: dict[str, Any],
    ) -> None:
        """
        Push a progress event to the task's stream list.
        Consumers (WebSocket handler) pop from this list in real-time.
        """
        key = f"churn:task:{task_id}:stream"
        event["timestamp"] = datetime.utcnow().isoformat()
        if self._client:
            self._client.rpush(key, json.dumps(event, default=str))
            self._client.expire(key, STREAM_TTL_SECONDS)
        else:
            existing = json.loads(self._fallback.get(key, "[]"))
            existing.append(event)
            self._fallback[key] = json.dumps(existing)

    def pop_stream_events(self, task_id: str, max_events: int = 50) -> list[dict[str, Any]]:
        """Pop all pending stream events (non-blocking)."""
        key = f"churn:task:{task_id}:stream"
        events: list[dict[str, Any]] = []

        if self._client:
            for _ in range(max_events):
                raw = self._client.lpop(key)
                if raw is None:
                    break
                events.append(json.loads(raw))
        else:
            all_events = json.loads(self._fallback.get(key, "[]"))
            events = all_events[:max_events]
            self._fallback[key] = json.dumps(all_events[max_events:])

        return events

    def get_all_stream_events(self, task_id: str) -> list[dict[str, Any]]:
        """Read all stream events without consuming them (for status polling)."""
        key = f"churn:task:{task_id}:stream"
        if self._client:
            raw_list = self._client.lrange(key, 0, -1)
            return [json.loads(r) for r in raw_list]
        return json.loads(self._fallback.get(key, "[]"))

    # ------------------------------------------------------------------ #
    # Customer Context Cache
    # ------------------------------------------------------------------ #

    def cache_customer_context(
        self,
        customer_id: str,
        context: dict[str, Any],
        ttl: int = 3600,
    ) -> None:
        """Cache enriched customer context for fast re-use within 1 hour."""
        key = f"churn:customer:{customer_id}:context"
        self._set(key, json.dumps(context, default=str), ttl)

    def get_customer_context(self, customer_id: str) -> dict[str, Any] | None:
        """Return cached customer context, or None if expired/missing."""
        key = f"churn:customer:{customer_id}:context"
        raw = self._get(key)
        return json.loads(raw) if raw else None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _set(self, key: str, value: str, ttl: int) -> None:
        if self._client:
            self._client.setex(key, ttl, value)
        else:
            self._fallback[key] = value

    def _get(self, key: str) -> str | None:
        if self._client:
            return self._client.get(key)
        return self._fallback.get(key)

    def _delete(self, key: str) -> None:
        if self._client:
            self._client.delete(key)
        else:
            self._fallback.pop(key, None)

    @property
    def is_connected(self) -> bool:
        return self._client is not None
