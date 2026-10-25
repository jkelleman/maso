"""Deterministic biometric processing and baseline comparison graph nodes."""

from __future__ import annotations

import math
from typing import Any, Hashable, Mapping

from src.memory import SemanticCache


class BiometricProcessor:
    """Collect metric summaries from the orchestrator's semantic tokens."""

    name = "BiometricProcessor"

    def process_tokens(
        self,
        tokens: list[dict[str, Any]],
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        metrics: dict[str, dict[str, Any]] = {}
        for token in tokens:
            if token.get("type") != "metric_summary":
                continue
            metric = token.get("metric")
            if not isinstance(metric, str) or not metric:
                raise ValueError("metric summaries require a non-empty metric name")
            metrics[metric] = {
                key: token[key]
                for key in (
                    "samples",
                    "mean",
                    "minimum",
                    "maximum",
                    "latest",
                    "trend",
                    "variability",
                )
                if key in token
            }
        return {"metrics": metrics}


class RecoveryEvaluator:
    """Compare current metric means with explicitly configured user baselines.

    Results are numeric comparisons only; this node does not provide a health
    assessment or infer recovery status.
    """

    name = "RecoveryEvaluator"

    def __init__(self, cache: SemanticCache, user_id: Hashable) -> None:
        self.cache = cache
        self.user_id = user_id

    @staticmethod
    def _baseline_mean(baseline: Any, metric: str) -> float:
        value = baseline.get("mean") if isinstance(baseline, Mapping) else baseline
        if isinstance(value, bool):
            raise ValueError(f"baseline for {metric!r} must have a finite numeric mean")
        try:
            mean = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"baseline for {metric!r} must have a finite numeric mean"
            ) from error
        if not math.isfinite(mean):
            raise ValueError(f"baseline for {metric!r} must have a finite numeric mean")
        return mean

    async def process_tokens(
        self,
        tokens: list[dict[str, Any]],
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        dependencies = state.get("dependencies", {})
        processed = dependencies.get("biometrics", {})
        metrics = processed.get("metrics", {}) if isinstance(processed, Mapping) else {}

        comparisons: dict[str, dict[str, float | None]] = {}
        for metric, summary in metrics.items():
            if not isinstance(summary, Mapping) or "mean" not in summary:
                continue
            baseline = self.cache.get_baseline(self.user_id, metric)
            if baseline is None:
                continue
            current_mean = float(summary["mean"])
            if not math.isfinite(current_mean):
                raise ValueError(f"current mean for {metric!r} must be finite")
            baseline_mean = self._baseline_mean(baseline, metric)
            comparisons[metric] = {
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "delta": current_mean - baseline_mean,
                "percent_change": (
                    None
                    if baseline_mean == 0
                    else (current_mean - baseline_mean) / abs(baseline_mean) * 100
                ),
            }
        return {"baseline_comparisons": comparisons}
