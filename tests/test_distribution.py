"""Checks for the production distribution configuration."""

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_distribution_preserves_runtime_package():
    with (ROOT / "pyproject.toml").open("rb") as source:
        config = tomllib.load(source)

    assert config["tool"]["setuptools"]["packages"] == ["src"]
    assert config["tool"]["setuptools"]["include-package-data"] is False
    for module in ("__init__", "agents", "engine", "memory"):
        assert (ROOT / "src" / f"{module}.py").is_file()


def test_distribution_has_no_runtime_dependencies():
    with (ROOT / "pyproject.toml").open("rb") as source:
        config = tomllib.load(source)

    assert config["project"]["dependencies"] == []
    assert config["project"]["requires-python"] == ">=3.11"
