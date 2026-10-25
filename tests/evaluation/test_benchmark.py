import random

from src.engine import StreamOrchestrator


class HeavyMockProcessor:
    def __init__(self):
        self.name = "BiometricSignalProcessor"

    def process_tokens(self, tokens: str) -> str:
        return "Processed metrics batch successfully."


def test_high_throughput_stream_performance():
    """Simulates a heavy real-time telemetry load to evaluate execution latency profiles."""
    orchestrator = StreamOrchestrator(window_bounds="rolling")
    orchestrator.register_agent(HeavyMockProcessor())

    metric_values = [60.0, 72.5, 85.0, 98.0, 110.5]

    for i in range(500):
        mock_payload = {
            "metric": "heart_rate",
            "value": random.choice(metric_values),
        }
        simulated_fps = 90.0 if i % 10 != 0 else 42.0
        orchestrator.execute_trajectory_step(
            mock_payload,
            simulated_frame_rate=simulated_fps,
        )

    profile = orchestrator.telemetry.extract_performance_profile()

    assert profile["total_samples"] > 0
    assert "mean_client_fps" in profile
    assert profile["total_samples"] == 1000
