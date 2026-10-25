# Multi-Agent Stream Orchestration Framework (MASO)

MASO is a Python framework for bounded telemetry stream processing and asynchronous multi-agent orchestration. The core uses only the Python standard library; external model, embedding, and storage integrations can be supplied by applications.

## Capabilities

### Stream processing

- Maintains a bounded rolling window using event timestamps.
- Compresses high-frequency numeric measurements into deterministic per-metric summaries, including sample count, range, latest value, trend, and variability.
- Accepts a single `metric`/`value` point or a mapping of metrics in each telemetry point.

### Agent graph and tools

- Runs independent agent nodes concurrently and schedules dependent nodes when their inputs are ready.
- Supports synchronous and asynchronous handlers, with dependency outputs available in the node state.
- Executes only explicitly registered tools and supports per-tool argument validators.

### Memory and evaluation

- Provides bounded, TTL-based episodic caching with optional context-vector matching.
- Separates short-term runtime measurements from long-term user baseline values.
- Includes a deterministic trajectory judge for context drift, node execution stability, and tool-call validation. It does not make external foundation-model calls.

## Getting started

```python
from src.engine import StreamOrchestrator
from src.memory import SemanticCache


class BiometricSignalProcessor:
    name = "BiometricSignalProcessor"

    def process_tokens(self, tokens):
        return f"Processed {len(tokens)} semantic metric summaries."


orchestrator = StreamOrchestrator(cache=SemanticCache(ttl_seconds=3600))
orchestrator.register_agent(BiometricSignalProcessor())
result = orchestrator.execute_trajectory_step(
    {"metric": "heart_rate", "value": 72.5}
)
print(result)
```

The synchronous `execute_trajectory_step` method is provided for legacy callers. Async applications should await `execute_trajectory` or `execute_trajectory_step_async`.

## Tests

Install the test dependencies and run the deterministic validation suite:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/evaluation/test_trajectory.py -v
```
