"""Thread-safe client and execution performance metrics."""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from statistics import fmean
from typing import Any


class ClientPerformanceHook:
    """Collect execution latency and client frame-rate samples."""

    def __init__(self) -> None:
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._client_fps: list[float] = []
        self._token_counts: list[int] = []
        self._lock = threading.Lock()

    def record_trace_latency(self, trace_name: str, duration_ms: float) -> None:
        duration = float(duration_ms)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration_ms must be a finite non-negative number")
        with self._lock:
            self._latencies[trace_name].append(duration)

    def track_client_render_fps(self, frame_rate: float, token_count: int) -> None:
        fps = float(frame_rate)
        if not math.isfinite(fps) or fps < 0:
            raise ValueError("frame_rate must be a finite non-negative number")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise ValueError("token_count must be a non-negative integer")
        with self._lock:
            self._client_fps.append(fps)
            self._token_counts.append(token_count)

    def extract_performance_profile(self) -> dict[str, Any]:
        with self._lock:
            latencies = {
                name: list(samples) for name, samples in self._latencies.items()
            }
            client_fps = list(self._client_fps)
            token_counts = list(self._token_counts)

        all_latencies = [sample for samples in latencies.values() for sample in samples]
        return {
            "total_samples": len(all_latencies) + len(client_fps),
            "mean_client_fps": fmean(client_fps) if client_fps else 0.0,
            "mean_latency_ms": fmean(all_latencies) if all_latencies else 0.0,
            "latency_samples": len(all_latencies),
            "client_fps_samples": len(client_fps),
            "mean_token_count": fmean(token_counts) if token_counts else 0.0,
            "trace_latencies": latencies,
        }
