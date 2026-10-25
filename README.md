# Multi-Agent Stream Orchestration Framework (MASO)

## TL;DR

MASO is a real-time multi-agent stream orchestration engine optimized for processing high-frequency data using stateful sliding window tokenization and multi-tier semantic memory networks. The framework operates with bounded context windows, deterministic signal compression, and concurrent agent graph execution, enabling sub-millisecond cache performance and quantifiable observability for production MLOps pipelines.

## Core Use Cases

- **High-Frequency Physiological Tracking**: Ingesting live biometric telemetry streams (HR, HRV, core strain) and downsampling signals into semantic token states for model routing.
- **Stateful Asynchronous Coaching Systems**: Managing cross-agent relational dependencies and memory persistence across long-running wellness or performance evaluation loops.
- **Deterministic MLOps Validation**: Running isolated agent trajectory tests under simulated device payloads to verify output consistency and eliminate runtime context drift.

## Target Audience

- **Foundation AI Researchers** building native multi-agent orchestration backends beyond standard stateless wrapper implementations.
- **Machine Learning Infrastructure Engineers** seeking optimized context structures, streaming token boundaries, and high-performance caching layers.
- **Data Platform Engineers** handling low-latency time-series integration pipelines across multi-node compute clusters.

## Key Benefits

- **Maximized Token Efficiency**: Specialized deserialization parsers minimize context bloating by mapping raw signals straight into dense data vectors.
- **Sub-Millisecond Cache Performance**: Multi-tier semantic caching layer routes recurrent states instantly, bypassing redundant deep model execution calls to reduce inference overhead.
- **Quantifiable Observability**: Inherent telemetry hooks actively track client-side compute latency and monitor for viewport or rendering drops under intense data strain.

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

## Production artifacts

Pushing a release tag in the form `vMAJOR.MINOR.PATCH` runs
`.github/workflows/deploy.yml`, independently of pull request CI. The workflow
checks source linting and security with Flake8 and Bandit, builds a source
distribution and wheel, and validates their metadata with Twine. Download the
`maso-production-dist` artifact from the workflow run within five days.

To build and validate the same distributions locally:

```bash
python -m pip install build==1.6.1 twine==7.0.0
python -m build
python -m twine check --strict dist/*
```

The workflow only stores build artifacts; it does not publish packages or sign
them. Artifacts can be passed to a separate signing or publishing process.
