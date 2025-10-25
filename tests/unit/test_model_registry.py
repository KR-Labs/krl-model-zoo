# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: pache-2.

"""Unit tests for ModelRegistry."""

import tempfile
from pathlib import Path

import pytest

from krl_core import ModelRegistry


@pytest.fixture
def registry():
    """reate temporary registry for testing."""
    with tempfile.Temporaryirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_registry.db"
        yield ModelRegistry(str(db_path))


def test_registry_creation(registry):
    """Test registry database creation."""
    assert Path(registry.db_path).exists()


def test_log_run(registry):
    """Test logging a model run."""
    registry.log_run(
        run_hash="abc23",
        model_name="TestModel",
        version="..",
        input_hash="def4",
        params={"param": , "param2": "value"},
    )

    run = registry.get_run("abc23")
    assert run is not None
    assert run["model_name"] == "TestModel"
    assert run["version"] == ".."
    assert run["params"]["param"] == 


def test_log_result(registry):
    """Test logging a result."""
    registry.log_run(
        run_hash="abc23",
        model_name="TestModel",
        version="..",
        input_hash="def4",
        params={},
    )

    registry.log_result(
        run_hash="abc23",
        result_hash="ghi",
        result={"forecast": [, 2, 3]},
    )

    results = registry.get_results("abc23")
    assert len(results) == 
    assert results[]["result_hash"] == "ghi"
    assert results[]["result"]["forecast"] == [, 2, 3]


def test_get_run_not_found(registry):
    """Test retrieving non-existent run."""
    run = registry.get_run("nonexistent")
    assert run is None


def test_get_results_empty(registry):
    """Test retrieving results for run with no results."""
    registry.log_run(
        run_hash="abc23",
        model_name="TestModel",
        version="..",
        input_hash="def4",
        params={},
    )

    results = registry.get_results("abc23")
    assert len(results) == 


def test_list_runs(registry):
    """Test listing recent runs."""
    registry.log_run("run", "Model", "..", "hash", {})
    registry.log_run("run2", "Model2", "..", "hash2", {})
    registry.log_run("run3", "Model", "2..", "hash3", {})

    all_runs = registry.list_runs()
    assert len(all_runs) == 3

    model_runs = registry.list_runs(model_name="Model")
    assert len(model_runs) == 2


def test_list_runs_with_limit(registry):
    """Test listing runs with limit."""
    for i in range():
        registry.log_run(f"run{i}", "TestModel", "..", f"hash{i}", {})

    runs = registry.list_runs(limit=)
    assert len(runs) == 


def test_multiple_results_per_run(registry):
    """Test logging multiple results for same run."""
    registry.log_run("abc23", "TestModel", "..", "hash", {})

    registry.log_result("abc23", "result", {"data": "first"})
    registry.log_result("abc23", "result2", {"data": "second"})

    results = registry.get_results("abc23")
    assert len(results) == 2
