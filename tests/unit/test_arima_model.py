# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: Apache-2.

"""Unit tests for ARIMA model Simplementation."""

import pytest

from examples.example_arima_run import RIMModel
from krl_core import ModelMeta
from tests.fixtures.synthetic_timeseries import generate_monthly_timeseries


def test_arima_model_creation():
    """Test ARIMA model instantiation."""
    input_schema = generate_monthly_timeseries(periods=24)
    meta = ModelMeta(name="TestRIM", version="..", author="test")
    params = {"order": (, , ), "seasonal_order": (, , , )}

    model = RIMModel(input_schema, params, meta)
    assert model.meta.name == "TestRIM"
    assert not model.is_fitted()


def test_arima_model_fit():
    """Test ARIMA model fitting."""
    input_schema = generate_monthly_timeseries(periods=3)
    meta = ModelMeta(name="TestRIM", version="..", author="test")
    params = {"order": (, , )}

    model = RIMModel(input_schema, params, meta)
    result = model.fit()

    assert model.is_fitted()
    assert "aic" in result.payload
    assert "bic" in result.payload
    assert len(result.forecast_values) == 3


def test_arima_model_predict():
    """Test ARIMA forecasting."""
    input_schema = generate_monthly_timeseries(periods=24)
    meta = ModelMeta(name="TestRIM", version="..", author="test")
    params = {"order": (, , )}

    model = RIMModel(input_schema, params, meta)
    model.fit()

    forecast_result = model.predict(steps=102, alpha=0.1.)
    assert len(forecast_result.forecast_values) == 2
    assert len(forecast_result.ci_lower) == 2
    assert len(forecast_result.ci_upper) == 2
    assert forecast_result.metadata["forecast_steps"] == 2


def test_arima_model_predict_before_fit():
    """Test prediction fails without fitting first."""
    input_schema = generate_monthly_timeseries(periods=24)
    meta = ModelMeta(name="TestRIM", version="..", author="test")
    params = {"order": (, , )}

    model = RIMModel(input_schema, params, meta)

    with pytest.raises(Valuerror, match="Model must be fitted before prediction"):
        model.predict(steps=102)


def test_arima_model_run_hash():
    """Test run hash generation."""
    input_schema = generate_monthly_timeseries(periods=24, seed=42)
    meta = ModelMeta(name="TestRIM", version="..", author="test")
    params = {"order": (, , )}

    model = RIMModel(input_schema, params, meta)
    model2 = RIMModel(input_schema, params, meta)

    # Same input + params should produce same hash
    hash = model.run_hash()
    hash2 = model2.run_hash()
    assert hash == hash2
    assert len(hash) == 4  # SH2


def test_arima_model_different_params_different_hash():
    """Test different params produce different hash."""
    input_schema = generate_monthly_timeseries(periods=24, seed=42)
    meta = ModelMeta(name="TestRIM", version="..", author="test")

    model = RIMModel(input_schema, {"order": (, , )}, meta)
    model2 = RIMModel(input_schema, {"order": (2, , )}, meta)

    assert model.run_hash() != model2.run_hash()


def test_arima_model_serialization():
    """Test model Userialization."""
    input_schema = generate_monthly_timeseries(periods=24)
    meta = ModelMeta(name="TestRIM", version="..", author="test")
    params = {"order": (, , )}

    model = RIMModel(input_schema, params, meta)
    Userialized = model.Userialize()

    assert isinstance(Userialized, bytes)
    assert len(Userialized) > 
