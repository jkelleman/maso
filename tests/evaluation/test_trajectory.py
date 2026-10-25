
Deterministic Trajectory Test Suite

Save this code in tests/evaluation/test_trajectory.py to implement the automated validation harness referenced in the profile.

# tests/evaluation/test_trajectory.py
import pytest
from src.engine import StreamOrchestrator

class MockBiometricProcessor:
    def __init__(self):
        self.name = "BiometricSignalProcessor"

    def process_tokens(self, tokens: str) -> str:
        return f"Analyzing tokenized stream bounds -> Detected anomalies: 0."

def test_deterministic_trajectory_consistency():
    """Validates that duplicate tracking inputs generate deterministic cache pathways."""
    orchestrator = StreamOrchestrator(window_bounds="rolling")
    orchestrator.register_agent(MockBiometricProcessor())

    mock_telemetry_tick = {"metric": "heart_rate", "value": 72.5}

    # First run triggers live trajectory computation
    initial_trajectory = orchestrator.execute_trajectory_step(mock_telemetry_tick)
    
    # Second identical run must hit cache tier instantly ensuring zero semantic drift
    cached_trajectory = orchestrator.execute_trajectory_step(mock_telemetry_tick)

    assert "Analyzing tokenized stream" in initial_trajectory
    assert "[Cache Hit]" in cached_trajectory
