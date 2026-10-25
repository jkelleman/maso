"""Deterministic tests for stream processing, orchestration, memory, and judging."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from src.engine import StreamOrchestrator
from src.memory import SemanticCache


class MockBiometricProcessor:
    name = "BiometricSignalProcessor"

    def __init__(self):
        self.calls = 0

    def process_tokens(self, tokens):
        self.calls += 1
        assert isinstance(tokens, list)
        return "Analyzing tokenized stream -> Detected anomalies: 0."


@dataclass
class JudgeResult:
    passed: bool
    issues: list[str]


class AutomatedTrajectoryJudge:
    """Deterministic LLM-as-a-judge simulation for trajectory invariants."""

    @staticmethod
    def evaluate(
        tokens,
        trajectory,
        *,
        expected_metrics=(),
        expected_nodes=(),
        allowed_tools=(),
    ):
        issues = []
        metrics = {
            token["metric"]
            for token in tokens
            if token.get("type") == "metric_summary"
        }
        missing_metrics = set(expected_metrics) - metrics
        if missing_metrics:
            issues.append(f"context drift: missing metrics {sorted(missing_metrics)}")

        started = [event["node"] for event in trajectory if event["event"] == "node_started"]
        completed = [event["node"] for event in trajectory if event["event"] == "node_completed"]
        if len(started) != len(set(started)) or len(completed) != len(set(completed)):
            issues.append("trajectory stability: a node executed more than once")
        if set(started) != set(completed):
            issues.append("trajectory stability: node start/completion events do not match")
        if not set(expected_nodes).issubset(started):
            issues.append("trajectory stability: an expected graph node did not execute")

        tool_started = [
            event["tool"] for event in trajectory if event["event"] == "tool_started"
        ]
        tool_completed = [
            event["tool"] for event in trajectory if event["event"] == "tool_completed"
        ]
        if tool_started != tool_completed:
            issues.append("tool validation: tool start/completion events do not match")
        if not set(tool_started).issubset(set(allowed_tools)):
            issues.append("tool validation: an unapproved tool was called")
        return JudgeResult(passed=not issues, issues=issues)


def test_deterministic_trajectory_consistency():
    """Duplicate tracking inputs follow a deterministic cache pathway."""
    orchestrator = StreamOrchestrator(window_bounds="rolling")
    processor = MockBiometricProcessor()
    orchestrator.register_agent(processor)
    telemetry = {"metric": "heart_rate", "value": 72.5}

    initial_trajectory = orchestrator.execute_trajectory_step(telemetry)
    cached_trajectory = orchestrator.execute_trajectory_step(telemetry)

    assert "Analyzing tokenized stream" in initial_trajectory
    assert "[Cache Hit]" in cached_trajectory
    assert processor.calls == 1


def test_sliding_window_bounds_and_semantic_tokens():
    now = [100.0]
    orchestrator = StreamOrchestrator(
        window_size_seconds=10,
        clock=lambda: now[0],
    )
    orchestrator.ingest_telemetry({
        "timestamp": 89,
        "metrics": {"heart_rate": 70, "hrv": 42},
    })
    orchestrator.ingest_telemetry({
        "timestamp": 95,
        "metrics": {"heart_rate": 76, "hrv": 48},
    })
    orchestrator.ingest_telemetry({"timestamp": 101, "metric": "heart_rate", "value": 80})

    window = orchestrator.compute_sliding_window()
    tokens = orchestrator.compress_to_semantic_tokens(window)

    assert len(window) == 1
    assert [token["metric"] for token in tokens] == ["heart_rate", "hrv"]
    heart_rate = tokens[0]
    assert heart_rate["samples"] == 1
    assert heart_rate["mean"] == 76
    assert heart_rate["trend"] == 0


def test_async_graph_runs_parallel_nodes_then_dependencies():
    orchestrator = StreamOrchestrator()
    started = set()
    both_started = asyncio.Event()

    async def worker_a(tokens):
        started.add("a")
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return "a-result"

    async def worker_b(tokens):
        started.add("b")
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        return "b-result"

    async def evaluator(tokens, state):
        assert state["dependencies"] == {
            "signal": "a-result",
            "context": "b-result",
        }
        return "recovery-evaluated"

    orchestrator.register_node("signal", worker_a)
    orchestrator.register_node("context", worker_b)
    orchestrator.register_node("recovery", evaluator, depends_on=("signal", "context"))

    result = asyncio.run(orchestrator.execute_trajectory({"metric": "hrv", "value": 45}))

    assert started == {"a", "b"}
    assert "[recovery Output]: recovery-evaluated" in result
    completed = [
        event["node"]
        for event in orchestrator.last_trajectory
        if event["event"] == "node_completed"
    ]
    assert completed.index("signal") < completed.index("recovery")
    assert completed.index("context") < completed.index("recovery")


def test_tool_calls_are_allowlisted_validated_and_not_cached():
    orchestrator = StreamOrchestrator()
    invocations = []

    async def record_recovery(user_id, score):
        invocations.append((user_id, score))
        return "recorded"

    async def evaluator(tokens, state):
        return {
            "output": "recovery ready",
            "tool_calls": [{
                "name": "record_recovery",
                "arguments": {"user_id": "u1", "score": 0.8},
            }],
        }

    orchestrator.register_tool(
        "record_recovery",
        record_recovery,
        validator=lambda args: isinstance(args.get("score"), float) and 0 <= args["score"] <= 1,
    )
    orchestrator.register_node("recovery", evaluator)

    first = asyncio.run(orchestrator.execute_trajectory({"metric": "hrv", "value": 45}))
    second = asyncio.run(orchestrator.execute_trajectory({"metric": "hrv", "value": 45}))

    assert "recovery ready" in first and "recovery ready" in second
    assert invocations == [("u1", 0.8), ("u1", 0.8)]
    assert sum(event["event"] == "tool_completed" for event in orchestrator.last_trajectory) == 1


def test_unknown_tool_call_fails_closed():
    orchestrator = StreamOrchestrator()

    async def unsafe_worker(tokens):
        return {"tool_calls": [{"name": "unregistered", "arguments": {}}]}

    orchestrator.register_node("worker", unsafe_worker)
    with pytest.raises(ValueError, match="not registered"):
        asyncio.run(orchestrator.execute_trajectory({"metric": "hrv", "value": 45}))


def test_semantic_cache_tiers_similarity_and_expiration():
    now = [10.0]
    cache = SemanticCache(
        ttl_seconds=5,
        baseline_ttl_seconds=20,
        clock=lambda: now[0],
    )
    cache.set("query-a", "cached", (1.0, 0.0), namespace="recovery")
    cache.set_baseline("user-1", "hrv", {"mean": 45})
    cache.record_metric("user-1", "heart_rate", 72, max_samples=2)

    assert cache.get("query-b", (0.9999, 0.01), namespace="recovery") == "cached"
    assert cache.get("query-b", (0.0, 1.0), namespace="recovery") is None
    assert cache.get_baseline("user-1", "hrv") == {"mean": 45}
    assert cache.recent_metrics("user-1")[0][1:] == ("heart_rate", 72)

    now[0] += 6
    assert cache.get("query-a", (1.0, 0.0), namespace="recovery") is None
    assert cache.get_baseline("user-1", "hrv") == {"mean": 45}


def test_automated_judge_detects_drift_and_invalid_tool_trajectory():
    tokens = [{"type": "metric_summary", "metric": "heart_rate"}]
    trace = [
        {"event": "node_started", "node": "processor"},
        {"event": "node_completed", "node": "processor"},
        {"event": "tool_started", "node": "processor", "tool": "unapproved"},
    ]

    judged = AutomatedTrajectoryJudge.evaluate(
        tokens,
        trace,
        expected_metrics=("heart_rate", "hrv"),
        expected_nodes=("processor", "evaluator"),
        allowed_tools=("persist_baseline",),
    )

    assert not judged.passed
    assert len(judged.issues) == 4
    assert any("context drift" in issue for issue in judged.issues)
    assert any("trajectory stability" in issue for issue in judged.issues)
    assert any("tool validation" in issue for issue in judged.issues)


def test_automated_judge_accepts_stable_complete_trajectory():
    tokens = [
        {"type": "metric_summary", "metric": "heart_rate"},
        {"type": "metric_summary", "metric": "hrv"},
    ]
    trace = [
        {"event": "node_started", "node": "processor"},
        {"event": "tool_started", "node": "processor", "tool": "persist_baseline"},
        {"event": "tool_completed", "node": "processor", "tool": "persist_baseline"},
        {"event": "node_completed", "node": "processor"},
    ]

    judged = AutomatedTrajectoryJudge.evaluate(
        tokens,
        trace,
        expected_metrics=("heart_rate", "hrv"),
        expected_nodes=("processor",),
        allowed_tools=("persist_baseline",),
    )

    assert judged == JudgeResult(passed=True, issues=[])
