"""Thread-safe episodic and baseline memory for stream processing."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Optional


@dataclass(frozen=True)
class _CacheEntry:
    timestamp: float
    payload: Any
    context_vector: Optional[tuple[float, ...]]
    namespace: str


@dataclass(frozen=True)
class _BaselineEntry:
    timestamp: float
    payload: Any


class SemanticCache:
    """Stores short-lived execution results and separate long-term user baselines.

    Context-vector matching is intentionally local and dependency-free. Callers
    provide vectors from their own feature or embedding pipeline; vectors are
    compared only within the same namespace and dimensionality.
    """

    def __init__(
        self,
        ttl_seconds: float = 3600,
        *,
        baseline_ttl_seconds: Optional[float] = None,
        max_entries: int = 4096,
        similarity_threshold: float = 0.995,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        if baseline_ttl_seconds is not None and baseline_ttl_seconds < 0:
            raise ValueError("baseline_ttl_seconds must be non-negative")
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        self.ttl = float(ttl_seconds)
        self.baseline_ttl = baseline_ttl_seconds
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self._clock = clock
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._baselines: dict[tuple[Hashable, Hashable], _BaselineEntry] = {}
        self._metrics: dict[Hashable, deque[tuple[float, str, Any]]] = defaultdict(deque)
        self._lock = threading.RLock()

        # Keep the original attribute available for callers that inspected it.
        self.cache = self._entries

    @staticmethod
    def _vector(values: Optional[Iterable[float]]) -> Optional[tuple[float, ...]]:
        if values is None:
            return None
        vector = tuple(float(value) for value in values)
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError("context vectors must contain finite values")
        return vector

    @staticmethod
    def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right):
            return -1.0
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 1.0 if left == right else 0.0
        similarity = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
        return max(-1.0, min(1.0, similarity))

    def _is_expired(self, timestamp: float, ttl: Optional[float], now: float) -> bool:
        return ttl is not None and (ttl == 0 or now - timestamp >= ttl)

    def get(
        self,
        cache_key: str,
        context_vector: Optional[Iterable[float]] = None,
        *,
        namespace: str = "default",
        similarity_threshold: Optional[float] = None,
    ) -> Optional[Any]:
        """Return an exact or sufficiently similar episodic result, if present."""
        vector = self._vector(context_vector)
        threshold = self.similarity_threshold if similarity_threshold is None else similarity_threshold
        if not 0 <= threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        now = self._clock()
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is not None:
                if self._is_expired(entry.timestamp, self.ttl, now):
                    del self._entries[cache_key]
                elif entry.namespace == namespace and (
                    vector is None or (
                        entry.context_vector is not None
                        and self._cosine_similarity(vector, entry.context_vector) >= threshold
                    )
                ):
                    self._entries.move_to_end(cache_key)
                    return entry.payload

            if vector is None:
                return None

            best_key: Optional[str] = None
            best_similarity = -1.0
            for candidate_key, candidate in list(self._entries.items()):
                if self._is_expired(candidate.timestamp, self.ttl, now):
                    del self._entries[candidate_key]
                    continue
                if candidate.namespace != namespace or candidate.context_vector is None:
                    continue
                similarity = self._cosine_similarity(vector, candidate.context_vector)
                if similarity >= threshold and similarity > best_similarity:
                    best_key = candidate_key
                    best_similarity = similarity

            if best_key is None:
                return None
            self._entries.move_to_end(best_key)
            return self._entries[best_key].payload

    def set(
        self,
        cache_key: str,
        payload: Any,
        context_vector: Optional[Iterable[float]] = None,
        *,
        namespace: str = "default",
    ) -> None:
        """Store an episodic result, evicting the least-recently-used entry if full."""
        vector = self._vector(context_vector)
        with self._lock:
            self._entries[cache_key] = _CacheEntry(
                timestamp=self._clock(),
                payload=payload,
                context_vector=vector,
                namespace=namespace,
            )
            self._entries.move_to_end(cache_key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def set_baseline(self, user_id: Hashable, key: Hashable, payload: Any) -> None:
        """Store a long-term baseline independently of short-lived cache entries."""
        with self._lock:
            self._baselines[(user_id, key)] = _BaselineEntry(self._clock(), payload)

    def get_baseline(self, user_id: Hashable, key: Hashable) -> Optional[Any]:
        with self._lock:
            entry = self._baselines.get((user_id, key))
            if entry is None:
                return None
            if self._is_expired(entry.timestamp, self.baseline_ttl, self._clock()):
                del self._baselines[(user_id, key)]
                return None
            return entry.payload

    def record_metric(
        self,
        user_id: Hashable,
        metric: str,
        value: Any,
        *,
        max_samples: int = 1024,
    ) -> None:
        """Append a bounded short-term episodic measurement for one user."""
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        with self._lock:
            samples = self._metrics[user_id]
            if samples.maxlen != max_samples:
                samples = deque(samples, maxlen=max_samples)
                self._metrics[user_id] = samples
            samples.append((self._clock(), metric, value))

    def recent_metrics(
        self,
        user_id: Hashable,
        *,
        since: Optional[float] = None,
    ) -> list[tuple[float, str, Any]]:
        """Return a snapshot of recent episodic measurements."""
        with self._lock:
            samples = self._metrics.get(user_id, ())
            return [sample for sample in samples if since is None or sample[0] >= since]

    def clear(self) -> None:
        """Clear ephemeral cache entries without changing persisted baselines."""
        with self._lock:
            self._entries.clear()
            self._metrics.clear()
