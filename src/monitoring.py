"""Shared monitoring metrics (prediction counts, latency, retraining status).

Backed by Redis when reachable so counters are shared across horizontally scaled
API containers; falls back to an in-process store (e.g. for unit tests, or when
Redis is unavailable) so the API still works standalone.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

from src.config import settings

_KEY_PREFIX = "rurangasort"
_LATENCY_KEY = f"{_KEY_PREFIX}:latencies"
_MAX_LATENCY_SAMPLES = 500


class _InMemoryBackend:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._latencies: list[float] = []
        self._class_counts: dict[str, int] = {}
        self._last_retraining: dict | None = None
        self._training_jobs: dict[str, dict] = {}

    def incr(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount

    def get_counter(self, key: str) -> int:
        with self._lock:
            return self._counters.get(key, 0)

    def push_latency(self, value: float) -> None:
        with self._lock:
            self._latencies.append(value)
            self._latencies = self._latencies[-_MAX_LATENCY_SAMPLES:]

    def get_latencies(self) -> list[float]:
        with self._lock:
            return list(self._latencies)

    def incr_class(self, class_name: str) -> None:
        with self._lock:
            self._class_counts[class_name] = self._class_counts.get(class_name, 0) + 1

    def get_class_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._class_counts)

    def set_last_retraining(self, info: dict) -> None:
        with self._lock:
            self._last_retraining = info

    def get_last_retraining(self) -> dict | None:
        with self._lock:
            return self._last_retraining


_backend_lock = threading.Lock()
_redis_client = None
_in_memory = _InMemoryBackend()


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    with _backend_lock:
        if _redis_client is None:
            try:
                import redis

                client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=0.5)
                client.ping()
                _redis_client = client
            except Exception:
                _redis_client = False  # sentinel meaning "unavailable"
    return _redis_client or None


_PROCESS_START = time.time()


def get_process_uptime_seconds() -> float:
    return time.time() - _PROCESS_START


def record_prediction(success: bool, latency_ms: float | None = None, predicted_class: str | None = None) -> None:
    redis_client = _get_redis()
    if redis_client:
        redis_client.incr(f"{_KEY_PREFIX}:predictions_total")
        if not success:
            redis_client.incr(f"{_KEY_PREFIX}:predictions_failed")
        if latency_ms is not None:
            redis_client.lpush(_LATENCY_KEY, latency_ms)
            redis_client.ltrim(_LATENCY_KEY, 0, _MAX_LATENCY_SAMPLES - 1)
        if predicted_class:
            redis_client.hincrby(f"{_KEY_PREFIX}:class_counts", predicted_class, 1)
    else:
        _in_memory.incr("predictions_total")
        if not success:
            _in_memory.incr("predictions_failed")
        if latency_ms is not None:
            _in_memory.push_latency(latency_ms)
        if predicted_class:
            _in_memory.incr_class(predicted_class)


def record_retraining_event(status: str, model_version: str | None = None, detail: str | None = None) -> None:
    info = {
        "status": status,
        "model_version": model_version,
        "detail": detail,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    redis_client = _get_redis()
    if redis_client:
        redis_client.set(f"{_KEY_PREFIX}:last_retraining", json.dumps(info))
    else:
        _in_memory.set_last_retraining(info)


def get_last_retraining_event() -> dict | None:
    redis_client = _get_redis()
    if redis_client:
        raw = redis_client.get(f"{_KEY_PREFIX}:last_retraining")
        return json.loads(raw) if raw else None
    return _in_memory.get_last_retraining()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(percentile * len(ordered)), len(ordered) - 1)
    return ordered[index]


def get_metrics_snapshot() -> dict:
    redis_client = _get_redis()

    if redis_client:
        total = int(redis_client.get(f"{_KEY_PREFIX}:predictions_total") or 0)
        failed = int(redis_client.get(f"{_KEY_PREFIX}:predictions_failed") or 0)
        latencies = [float(v) for v in redis_client.lrange(_LATENCY_KEY, 0, -1)]
        class_counts = {
            k.decode() if isinstance(k, bytes) else k: int(v)
            for k, v in redis_client.hgetall(f"{_KEY_PREFIX}:class_counts").items()
        }
    else:
        total = _in_memory.get_counter("predictions_total")
        failed = _in_memory.get_counter("predictions_failed")
        latencies = _in_memory.get_latencies()
        class_counts = _in_memory.get_class_counts()

    return {
        "predictions_total": total,
        "predictions_failed": failed,
        "predictions_succeeded": total - failed,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
        "class_prediction_distribution": class_counts,
        "last_retraining": get_last_retraining_event(),
        "process_uptime_seconds": round(get_process_uptime_seconds(), 1),
        "metrics_backend": "redis" if redis_client else "in-memory",
    }
