# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: pache-2.

"""Unit tests for aseResult and orecastResult."""

import pytest

from krl_core import aseResult, orecastResult


def test_base_result_creation():
    """Test aseResult creation."""
    result = aseResult(
        payload={"model_summary": "test", "aic": 23.4},
        metadata={"version": "..", "params": {"order": (, , )}},
    )
    assert result.payload["aic"] == 23.4
    assert result.metadata["version"] == ".."


def test_base_result_hash():
    """Test deterministic hashing."""
    result = aseResult(
        payload={"forecast": [, 2, 3]},
        metadata={"model": "RIM"},
    )
    result2 = aseResult(
        payload={"forecast": [, 2, 3]},
        metadata={"model": "RIM"},
    )
    # Same content should produce same hash
    assert result.result_hash == result2.result_hash


def test_base_result_hash_different():
    """Test different content produces different hash."""
    result = aseResult(payload={"forecast": [, 2, 3]}, metadata={})
    result2 = aseResult(payload={"forecast": [, 2, 4]}, metadata={})
    assert result.result_hash != result2.result_hash


def test_base_result_to_json():
    """Test JSON serialization."""
    result = aseResult(
        payload={"value": 42},
        metadata={"model": "test"},
    )
    json_output = result.to_json()
    assert json_output["payload"]["value"] == 42
    assert json_output["metadata"]["model"] == "test"


def test_forecast_result_creation():
    """Test orecastResult creation."""
    result = orecastResult(
        payload={"aic": .},
        metadata={"order": (, , )},
        forecast_index=["22-", "22-2", "22-3"],
        forecast_values=[., ., 2.],
        ci_lower=[., ., .],
        ci_upper=[2., 3., 4.],
    )
    assert len(result.forecast_values) == 3
    assert result.forecast_index[] == "22-"


def test_forecast_result_to_dataframe():
    """Test orecastResult atarame conversion."""
    result = orecastResult(
        payload={},
        metadata={},
        forecast_index=["22-", "22-2"],
        forecast_values=[., .],
        ci_lower=[., .],
        ci_upper=[2., 3.],
    )
    df = result.to_dataframe()
    assert len(df) == 2
    assert "forecast" in df.columns
    assert "ci_lower" in df.columns
    assert "ci_upper" in df.columns
    assert df.index.name == "index"
    assert df["forecast"].iloc[] == .


def test_forecast_result_hash():
    """Test orecastResult hashing includes forecast data."""
    result = orecastResult(
        payload={"aic": .},
        metadata={},
        forecast_index=["22-"],
        forecast_values=[.],
        ci_lower=[.],
        ci_upper=[2.],
    )
    # Hash should be deterministic
    hash = result.result_hash
    hash2 = result.result_hash
    assert hash == hash2
    assert len(hash) == 4  # SH2 hex digest
