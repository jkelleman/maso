"""Deterministic tests for stream processing, orchestration, memory, and judging."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass

import pytest

from src.agents import BiometricProcessor, RecoveryEvaluator
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
        if Counter(tool_started) != Counter(tool_completed):
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


def test_sliding_window_orders_late_measurements_by_event_time():
    orchestrator = StreamOrchestrator(clock=lambda: 100.0)
    orchestrator.ingest_telemetry({"timestamp": 95, "metric": "heart_rate", "value": 80})
    orchestrator.ingest_telemetry({"timestamp": 90, "metric": "heart_rate", "value": 60})

    tokens = orchestrator.compress_to_semantic_tokens(orchestrator.compute_sliding_window())

    assert tokens[0]["latest"] == 80
    assert tokens[0]["trend"] == 20


def test_sliding_window_retains_future_measurements_until_active():
    now = [100.0]
    orchestrator = StreamOrchestrator(clock=lambda: now[0])
    orchestrator.ingest_telemetry({"timestamp": 101, "metric": "heart_rate", "value": 80})

    assert orchestrator.compute_sliding_window() == []
    now[0] = 101.0
    assert len(orchestrator.compute_sliding_window()) == 1


def test_shared_cache_isolated_between_orchestrators():
    cache = SemanticCache()
    first = StreamOrchestrator(cache=cache)
    second = StreamOrchestrator(cache=cache)
    first.register_node("worker", lambda tokens: "first")
    second.register_node("worker", lambda tokens: "second")

    assert first.execute_trajectory_step({"metric": "hrv", "value": 45}) == "[worker Output]: first"
    assert second.execute_trajectory_step({"metric": "hrv", "value": 45}) == "[worker Output]: second"


def test_state_aware_handlers_do_not_use_token_only_cache():
    now = [100.0]
    orchestrator = StreamOrchestrator(window_size_seconds=1, clock=lambda: now[0])

    def worker(tokens, state):
        return state["window"][0]["user_id"]

    orchestrator.register_node("worker", worker)
    first = orchestrator.execute_trajectory_step({
        "timestamp": 100,
        "metric": "hrv",
        "value": 45,
        "user_id": "first",
    })
    now[0] = 102.0
    second = orchestrator.execute_trajectory_step({
        "timestamp": 102,
        "metric": "hrv",
        "value": 45,
        "user_id": "second",
    })

    assert first == "[worker Output]: first"
    assert second == "[worker Output]: second"


def test_cancelled_trajectory_cancels_and_awaits_running_nodes():
    async def run():
        orchestrator = StreamOrchestrator()
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def worker(tokens):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        orchestrator.register_node("worker", worker)
        trajectory = asyncio.create_task(
            orchestrator.execute_trajectory({"metric": "hrv", "value": 45})
        )
        await started.wait()
        trajectory.cancel()
        with pytest.raises(asyncio.CancelledError):
            await trajectory
        assert cleaned.is_set()

    asyncio.run(run())


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


def test_biometric_nodes_compare_window_summaries_with_user_baselines():
    cache = SemanticCache()
    cache.set_baseline("user-1", "heart_rate", {"mean": 60})
    orchestrator = StreamOrchestrator(cache=cache, window_size_seconds=60)
    orchestrator.register_node("biometrics", BiometricProcessor())
    orchestrator.register_node(
        "recovery",
        RecoveryEvaluator(cache, "user-1"),
        depends_on=("biometrics",),
    )

    result = asyncio.run(orchestrator.execute_trajectory({
        "metrics": {"heart_rate": 72, "hrv": 45},
    }))

    assert "[biometrics Output]" in result
    assert "[recovery Output]" in result
    output = result.split("[recovery Output]: ", 1)[1]
    assert "'baseline_mean': 60.0" in output
    assert "'current_mean': 72.0" in output
    assert "'percent_change': 20.0" in output
    assert "hrv" not in output


def test_recovery_evaluator_handles_zero_baseline_without_division():
    cache = SemanticCache()
    cache.set_baseline("user-1", "heart_rate", 0)
    orchestrator = StreamOrchestrator(cache=cache)
    orchestrator.register_node("biometrics", BiometricProcessor())
    orchestrator.register_node(
        "recovery",
        RecoveryEvaluator(cache, "user-1"),
        depends_on=("biometrics",),
    )

    result = asyncio.run(orchestrator.execute_trajectory({
        "metric": "heart_rate",
        "value": 1,
    }))

    assert "'delta': 1.0" in result
    assert "'percent_change': None" in result


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
    assert orchestrator.last_trajectory == [
        {"event": "node_started", "node": "worker"}
    ]


def test_dependency_cycles_are_rejected():
    orchestrator = StreamOrchestrator()

    async def node(tokens):
        return "complete"

    orchestrator.register_node("first", node, depends_on=("second",))
    orchestrator.register_node("second", node, depends_on=("first",))

    with pytest.raises(ValueError, match="dependency cycle"):
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
    assert cache.get("query-c", (100.0, 0.0), namespace="recovery") is None
    assert cache.get_baseline("user-1", "hrv") == {"mean": 45}
    assert cache.recent_metrics("user-1")[0][1:] == ("heart_rate", 72)

    now[0] += 6
    assert cache.get("query-a", (1.0, 0.0), namespace="recovery") is None
    assert cache.get_baseline("user-1", "hrv") == {"mean": 45}


def test_semantic_cache_keys_and_lru_are_namespace_scoped():
    cache = SemanticCache(max_entries=2)
    cache.set("k", "A", namespace="A")
    cache.set("k", "B", namespace="B")

    assert cache.get("k", namespace="A") == "A"
    assert cache.get("k", namespace="B") == "B"
    assert len(cache._entries) == 2


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


def test_automated_judge_accepts_concurrent_tool_completion_order():
    trace = [
        {"event": "tool_started", "node": "first", "tool": "tool_a"},
        {"event": "tool_started", "node": "second", "tool": "tool_b"},
        {"event": "tool_completed", "node": "second", "tool": "tool_b"},
        {"event": "tool_completed", "node": "first", "tool": "tool_a"},
    ]

    judged = AutomatedTrajectoryJudge.evaluate(
        [],
        trace,
        allowed_tools=("tool_a", "tool_b"),
    )

    assert judged == JudgeResult(passed=True, issues=[])
