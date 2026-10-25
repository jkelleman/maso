"""Asynchronous orchestration for bounded, high-frequency telemetry streams."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Optional

from src.memory import SemanticCache
from src.telemetry import ClientPerformanceHook


@dataclass(frozen=True)
class _Node:
    name: str
    handler: Any
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class _Tool:
    handler: Callable[..., Any]
    validator: Optional[Callable[[Mapping[str, Any]], bool]]


class StreamOrchestrator:
    """Tokenizes telemetry windows and runs registered workers as an async DAG.

    Synchronous ``process_tokens`` workers remain supported, while async workers
    and tool handlers execute without blocking the event loop.
    """

    def __init__(
        self,
        window_bounds: str | float = "rolling",
        cache: Optional[SemanticCache] = None,
        *,
        window_size_seconds: float = 60.0,
        max_buffer_size: int = 100_000,
        max_concurrency: int = 8,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if window_size_seconds <= 0:
            raise ValueError("window_size_seconds must be positive")
        if max_buffer_size < 1 or max_concurrency < 1:
            raise ValueError("max_buffer_size and max_concurrency must be positive")
        if isinstance(window_bounds, (int, float)):
            window_size_seconds = float(window_bounds)
            window_bounds = "rolling"
        if window_bounds != "rolling":
            raise ValueError("window_bounds must be 'rolling' or a duration in seconds")

        self.window_bounds = window_bounds
        self.window_size_seconds = float(window_size_seconds)
        self.cache = cache if cache is not None else SemanticCache()
        self.telemetry = ClientPerformanceHook()
        self.agents: list[dict[str, Any]] = []
        self.stream_buffer: deque[dict[str, Any]] = deque(maxlen=max_buffer_size)
        self.max_concurrency = max_concurrency
        self._clock = clock
        self._nodes: dict[str, _Node] = {}
        self._tools: dict[str, _Tool] = {}
        self._buffer_lock = threading.RLock()
        self._graph_revision = 0
        self.last_trajectory: list[dict[str, Any]] = []

    def register_agent(
        self,
        agent_instance: Any,
        *,
        name: Optional[str] = None,
        depends_on: Iterable[str] = (),
    ) -> None:
        """Register an agent instance as a graph node."""
        agent_name = name or getattr(agent_instance, "name", None) or "AnonymousAgent"
        self.register_node(agent_name, agent_instance, depends_on=depends_on)
        self.agents.append({"name": agent_name, "execute": self._resolve_handler(agent_instance)})

    def register_node(
        self,
        name: str,
        handler: Any,
        *,
        depends_on: Iterable[str] = (),
    ) -> None:
        """Register a callable/worker node with optional upstream dependencies."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("node name must be a non-empty string")
        if name in self._nodes:
            raise ValueError(f"node {name!r} is already registered")
        resolved = self._resolve_handler(handler)
        if resolved is None:
            raise TypeError(f"node {name!r} must be callable or expose process_tokens, process, or run")
        dependencies = tuple(dict.fromkeys(depends_on))
        if name in dependencies:
            raise ValueError("a node cannot depend on itself")
        self._nodes[name] = _Node(name, handler, dependencies)
        self._graph_revision += 1

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        validator: Optional[Callable[[Mapping[str, Any]], bool]] = None,
    ) -> None:
        """Add an explicitly allowlisted tool available to agent tool calls."""
        if not isinstance(name, str) or not name.strip() or name in self._tools:
            raise ValueError("tool name must be non-empty and unique")
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        self._tools[name] = _Tool(handler, validator)
        self._graph_revision += 1

    def ingest_telemetry(self, raw_signal: Mapping[str, Any]) -> None:
        """Append a copy of an incoming measurement to the bounded stream buffer."""
        if not isinstance(raw_signal, Mapping):
            raise TypeError("telemetry points must be mappings")
        tick = dict(raw_signal)
        timestamp = tick.get("timestamp", self._clock())
        if isinstance(timestamp, datetime):
            timestamp = timestamp.timestamp()
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError) as error:
            raise ValueError("telemetry timestamp must be numeric or datetime") from error
        if not math.isfinite(timestamp):
            raise ValueError("telemetry timestamp must be finite")
        tick["processed_timestamp"] = timestamp
        with self._buffer_lock:
            self.stream_buffer.append(tick)

    def compute_sliding_window(
        self,
        window_size_seconds: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Return points within the current time bounds and discard stale points."""
        duration = self.window_size_seconds if window_size_seconds is None else window_size_seconds
        if duration <= 0:
            raise ValueError("window_size_seconds must be positive")
        now = self._clock()
        lower_bound = now - duration
        with self._buffer_lock:
            active = [
                tick for tick in self.stream_buffer
                if lower_bound <= tick["processed_timestamp"] <= now
            ]
            self.stream_buffer = deque(active, maxlen=self.stream_buffer.maxlen)
            return [dict(tick) for tick in active]

    @staticmethod
    def _measurement_values(tick: Mapping[str, Any]) -> list[tuple[str, float]]:
        metrics = tick.get("metrics")
        if isinstance(metrics, Mapping):
            candidates = metrics.items()
        elif "value" in tick:
            candidates = [(tick.get("metric", "signal"), tick["value"])]
        else:
            return []
        output: list[tuple[str, float]] = []
        for name, value in candidates:
            if isinstance(value, bool):
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                output.append((str(name), numeric_value))
        return output

    def compress_to_semantic_tokens(
        self,
        window_data: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Summarize telemetry into deterministic per-metric semantic token records."""
        grouped: dict[str, list[float]] = {}
        for tick in window_data:
            for metric, value in self._measurement_values(tick):
                grouped.setdefault(metric, []).append(value)
        if not grouped:
            return [{"type": "signal_status", "status": "idle"}]

        tokens: list[dict[str, Any]] = []
        for metric in sorted(grouped):
            values = grouped[metric]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            tokens.append({
                "type": "metric_summary",
                "metric": metric,
                "samples": len(values),
                "mean": round(mean, 6),
                "minimum": round(min(values), 6),
                "maximum": round(max(values), 6),
                "latest": round(values[-1], 6),
                "trend": round(values[-1] - values[0], 6),
                "variability": round(math.sqrt(variance), 6),
            })
        return tokens

    @staticmethod
    def _resolve_handler(agent: Any) -> Optional[Callable[..., Any]]:
        if callable(agent):
            return agent
        for method_name in ("process_tokens", "process", "run"):
            method = getattr(agent, method_name, None)
            if callable(method):
                return method
        return None

    @staticmethod
    def _context_vector(tokens: list[dict[str, Any]]) -> tuple[float, ...]:
        vector: list[float] = []
        for token in tokens:
            if token.get("type") != "metric_summary":
                continue
            vector.extend((
                float(token["mean"]),
                float(token["minimum"]),
                float(token["maximum"]),
                float(token["latest"]),
                float(token["trend"]),
                float(token["variability"]),
                float(token["samples"]),
            ))
        return tuple(vector) or (0.0,)

    @staticmethod
    def _invoke_arguments(handler: Callable[..., Any], tokens: list[dict[str, Any]], state: dict[str, Any]):
        try:
            parameters = list(inspect.signature(handler).parameters.values())
        except (TypeError, ValueError):
            return (tokens,), {}
        state_parameter = next((parameter for parameter in parameters if parameter.name == "state"), None)
        positional = [
            parameter for parameter in parameters
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if state_parameter is not None:
            if positional and positional[0].name == "state":
                return (state,), {}
            if state_parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                return (tokens,), {"state": state}
            return (tokens, state), {}
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
            return (tokens,), {"state": state}
        if len(positional) >= 2:
            return (tokens, state), {}
        if len(positional) == 1:
            return (tokens,), {}
        return (), {}

    async def _invoke(
        self,
        handler: Callable[..., Any],
        tokens: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> Any:
        args, kwargs = self._invoke_arguments(handler, tokens, state)
        if inspect.iscoroutinefunction(handler):
            return await handler(*args, **kwargs)
        result = await asyncio.to_thread(handler, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _invoke_tool(self, tool: _Tool, arguments: Mapping[str, Any]) -> Any:
        if tool.validator is not None and not tool.validator(arguments):
            raise ValueError("tool-call arguments failed validation")
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(**arguments)
        result = await asyncio.to_thread(tool.handler, **arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _execute_agent(
        self,
        node: _Node,
        tokens: list[dict[str, Any]],
        state: dict[str, Any],
        events: list[dict[str, Any]],
        semaphore: asyncio.Semaphore,
    ) -> tuple[Any, list[tuple[str, Any]]]:
        handler = self._resolve_handler(node.handler)
        if handler is None:
            raise TypeError(f"node {node.name!r} has no executable handler")
        async with semaphore:
            events.append({"event": "node_started", "node": node.name})
            response = await self._invoke(handler, tokens, state)

        tool_calls: Any = []
        output = response
        if isinstance(response, Mapping) and "tool_calls" in response:
            tool_calls = response["tool_calls"]
            output = response.get("output")
        if tool_calls is None:
            tool_calls = []
        if not isinstance(tool_calls, list):
            raise ValueError(f"node {node.name!r} returned invalid tool_calls")

        tool_results: list[tuple[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, Mapping):
                raise ValueError("tool calls must be mappings")
            tool_name = call.get("name")
            arguments = call.get("arguments", {})
            if not isinstance(tool_name, str) or not isinstance(arguments, Mapping):
                raise ValueError("tool calls require a name and mapping arguments")
            tool = self._tools.get(tool_name)
            if tool is None:
                raise ValueError(f"tool {tool_name!r} is not registered")
            events.append({"event": "tool_started", "node": node.name, "tool": tool_name})
            async with semaphore:
                result = await self._invoke_tool(tool, arguments)
            events.append({"event": "tool_completed", "node": node.name, "tool": tool_name})
            tool_results.append((tool_name, result))

        events.append({"event": "node_completed", "node": node.name})
        return output, tool_results

    def _cache_context(
        self,
        tokens: list[dict[str, Any]],
    ) -> tuple[str, str, tuple[float, ...]]:
        topology = ",".join(self._nodes)
        metric_scope = ",".join(
            token["metric"] for token in tokens if token.get("type") == "metric_summary"
        ) or "idle"
        namespace = f"graph:{self._graph_revision}:{topology}:metrics:{metric_scope}"
        canonical_tokens = json.dumps(tokens, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(canonical_tokens.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}", namespace, self._context_vector(tokens)

    def _validate_graph(self) -> None:
        for node in self._nodes.values():
            missing = set(node.dependencies) - self._nodes.keys()
            if missing:
                raise ValueError(f"node {node.name!r} has unknown dependencies: {sorted(missing)}")

    async def _run_graph(
        self,
        tokens: list[dict[str, Any]],
        active_window: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, list[tuple[str, Any]]], list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        outputs: dict[str, Any] = {}
        tool_results: dict[str, list[tuple[str, Any]]] = {}
        pending_nodes = set(self._nodes)
        running: dict[asyncio.Task[tuple[Any, list[tuple[str, Any]]]], str] = {}
        semaphore = asyncio.Semaphore(self.max_concurrency)

        while pending_nodes or running:
            ready = [
                self._nodes[name] for name in self._nodes
                if name in pending_nodes and set(self._nodes[name].dependencies) <= outputs.keys()
            ]
            for node in ready:
                pending_nodes.remove(node.name)
                node_state = {
                    "tokens": tokens,
                    "window": active_window,
                    "outputs": dict(outputs),
                    "tool_results": {key: list(value) for key, value in tool_results.items()},
                    "dependencies": {key: outputs[key] for key in node.dependencies},
                    "node": node.name,
                }
                task = asyncio.create_task(
                    self._execute_agent(node, tokens, node_state, events, semaphore),
                    name=f"maso:{node.name}",
                )
                running[task] = node.name

            if not running:
                raise ValueError("agent graph contains a dependency cycle")
            completed, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(completed, key=lambda item: running[item]):
                name = running.pop(task)
                try:
                    output, called_tools = task.result()
                except BaseException:
                    for other in running:
                        other.cancel()
                    if running:
                        await asyncio.gather(*running, return_exceptions=True)
                    self.last_trajectory = list(events)
                    raise
                outputs[name] = output
                tool_results[name] = called_tools
        return outputs, tool_results, events

    def _render_outputs(self, outputs: Mapping[str, Any]) -> str:
        return " ".join(
            f"[{name} Output]: {outputs[name]}"
            for name in self._nodes
            if outputs[name] is not None
        )

    async def execute_trajectory(
        self,
        metric_payload: Optional[Mapping[str, Any]] = None,
        *,
        simulated_frame_rate: float = 90.0,
    ) -> str:
        """Ingest an optional point and execute the registered graph asynchronously."""
        start_time = time.perf_counter()
        if metric_payload is not None:
            self.ingest_telemetry(metric_payload)
        active_window = self.compute_sliding_window()
        tokens = self.compress_to_semantic_tokens(active_window)
        cache_key, namespace, vector = self._cache_context(tokens)
        cached = self.cache.get(cache_key, vector, namespace=namespace)
        if cached is not None:
            self.last_trajectory = [{"event": "cache_hit"}]
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.telemetry.record_trace_latency("CacheLookup", duration_ms)
            self.telemetry.track_client_render_fps(simulated_frame_rate, len(tokens))
            return f"[Cache Hit] {cached}"

        self._validate_graph()
        outputs, tool_results, events = await self._run_graph(tokens, active_window)
        self.last_trajectory = list(events)
        rendered = self._render_outputs(outputs)
        if not any(tool_results.values()):
            self.cache.set(cache_key, rendered, vector, namespace=namespace)
        duration_ms = (time.perf_counter() - start_time) * 1000
        self.telemetry.record_trace_latency("FullPipelineStep", duration_ms)
        self.telemetry.track_client_render_fps(simulated_frame_rate, len(tokens))
        return rendered

    async def execute_trajectory_step_async(
        self,
        metric_payload: Mapping[str, Any],
        simulated_frame_rate: float = 90.0,
    ) -> str:
        """Async compatibility alias for the original single-step entry point."""
        return await self.execute_trajectory(
            metric_payload,
            simulated_frame_rate=simulated_frame_rate,
        )

    def execute_trajectory_step(
        self,
        metric_payload: Mapping[str, Any],
        simulated_frame_rate: float = 90.0,
    ) -> str:
        """Synchronous compatibility wrapper for legacy callers and tests."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute_trajectory(
                metric_payload,
                simulated_frame_rate=simulated_frame_rate,
            ))
        raise RuntimeError(
            "execute_trajectory_step cannot run inside an event loop; "
            "use execute_trajectory_step_async instead"
        )
