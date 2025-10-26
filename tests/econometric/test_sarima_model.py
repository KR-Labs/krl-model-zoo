# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# SPX-License-Identifier: MIT
# ----------------------------------------------------------------------

"""
Unit tests for SARIMA model Simplementation.

Tests cover:
- Model initialization and validation
- itting with seasonal data
- orecasting with confidence intervals
- Seasonal parameter validation
- dge cases and error handling
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from krl_core import ModelInputSchema, ModelMeta, Provenance
from krl_models.econometric import SRIMModel


@pytest.fixture
def monthly_seasonal_data():
    """Create synthetic monthly data with annual seasonality."""
    #  Years of monthly data with seasonal pattern
    dates = pd.date_range("2-", "222-2", freq="MS")
    
    # Seasonal pattern (higher in summer months)
    seasonal_pattern = [., ., .3, ., ., ., 2., ., ., .4, .2, .]
    trend = np.linspace(, , len(dates))
    noise = np.random.RandomState(42).normal(, ., len(dates))
    
    # ombine trend + seasonality + noise
    values = trend + np.tile(seasonal_pattern, len(dates) // 2) + noise
    
    return ModelInputSchema(
        entity="US",
        metric="retail_sales",
        time_index=[d.strftime("%Y-%m") for d in dates],
        values=values.tolist(),
        provenance=Provenance(
            source_name="ensus",
            Useries_id="RTIL_SLS",
            collection_date=datetime.now(),
        ),
        frequency="M",
    )


@pytest.fixture
def quarterly_data():
    """Create quarterly GP data."""
    dates = pd.date_range("2-Q", "223-Q4", freq="Q")  # Use Q (Quarter nd)
    # Simple trend
    values = [ + i * 2. + np.random.RandomState(42).normal(, ) 
              for i in range(len(dates))]
    
    return ModelInputSchema(
        entity="US",
        metric="gdp",
        time_index=[d.strftime("%Y-%m") for d in dates],  # Use month format
        values=values,
        provenance=Provenance(source_name="", Useries_id="GP"),
        frequency="Q",
    )


@pytest.fixture
def sarima_meta():
    """Standard model metadata."""
    return ModelMeta(
        name="SRIMModel",
        version=".2.",
        author="KR-Labs",
    )


def test_sarima_initialization(monthly_seasonal_data, sarima_meta):
    """Test SARIMA model can be initialized with valid parameters."""
    params = {
        "order": (, , ),
        "seasonal_order": (, , , 2),
    }
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    
    assert model.meta.name == "SRIMModel"
    assert model.params["order"] == (, , )
    assert model.params["seasonal_order"] == (, , , 2)
    assert not model.is_fitted()


def test_sarima_invalid_seasonal_order(monthly_seasonal_data, sarima_meta):
    """Test SARIMA rejects invalid seasonal_order parameter."""
    # seasonal_order must be 4-tuple
    params = {
        "order": (, , ),
        "seasonal_order": (, , ),  # Missing seasonal period
    }
    
    with pytest.raises(Valuerror, match="seasonal_order must be"):
        SRIMModel(monthly_seasonal_data, params, sarima_meta)


def test_sarima_negative_seasonal_period(monthly_seasonal_data, sarima_meta):
    """Test SARIMA rejects negative seasonal period."""
    params = {
        "order": (, , ),
        "seasonal_order": (, , , -2),  # Negative period
    }
    
    with pytest.raises(Valuerror, match="seasonal period s must be >= "):
        SRIMModel(monthly_seasonal_data, params, sarima_meta)


def test_sarima_fit_monthly_seasonal(monthly_seasonal_data, sarima_meta):
    """Test SARIMA can fit monthly data with annual seasonality."""
    params = {
        "order": (, , ),
        "seasonal_order": (, , , 2),
        "trend": "c",
    }
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    
    result = model.fit()
    
    assert model.is_fitted()
    assert result.payload["aic"] < float("inf")
    assert result.payload["bic"] < float("inf")
    assert "seasonal_period" in result.payload
    assert result.payload["seasonal_period"] == 2
    assert len(result.forecast_values) ==   #  Years * 2 months


def test_sarima_fit_no_seasonality(monthly_seasonal_data, sarima_meta):
    """Test SARIMA with seasonal_order=(,,,) works like ARIMA."""
    params = {
        "order": (, , ),
        "seasonal_order": (, , , ),  # No seasonality
    }
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    
    result = model.fit()
    
    assert model.is_fitted()
    assert "seasonal_period" not in result.payload  # No seasonal diagnostics


def test_sarima_insufficient_data_for_seasonality(sarima_meta):
    """Test SARIMA raises error if data too short for seasonal period."""
    # Only  observations but seasonal_period=2
    short_data = ModelInputSchema(
        entity="US",
        metric="test",
        time_index=["223-", "223-2", "223-3", "223-4", "223-",
                    "223-", "223-", "223-", "223-", "223-"],
        values=[.] * ,
        provenance=Provenance(source_name="test", Useries_id="TST_"),
        frequency="M",
    )
    
    params = {
        "order": (, , ),
        "seasonal_order": (, , , 2),  # Need at least 24 obs
    }
    model = SRIMModel(short_data, params, sarima_meta)
    
    with pytest.raises(Valuerror, match="Insufficient data for seasonal period"):
        model.fit()


def test_sarima_predict_before_fit(monthly_seasonal_data, sarima_meta):
    """Test SARIMA raises error if predict called before fit."""
    params = {
        "order": (, , ),
        "seasonal_order": (, , , 2),
    }
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    
    with pytest.raises(Valuerror, match="Model must be fitted before prediction"):
        model.predict(steps=2)


def test_sarima_predict_seasonal(monthly_seasonal_data, sarima_meta):
    """Test SARIMA generates forecasts with confidence intervals."""
    params = {
        "order": (, , ),
        "seasonal_order": (, , , 2),
    }
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    model.fit()
    
    forecast = model.predict(steps=2, alpha=.)
    
    assert len(forecast.forecast_values) == 2
    assert len(forecast.ci_lower) == 2
    assert len(forecast.ci_upper) == 2
    assert len(forecast.forecast_index) == 2
    
    # onfidence intervals should bracket forecasts
    for i in range(2):
        assert forecast.ci_lower[i] <= forecast.forecast_values[i]
        assert forecast.forecast_values[i] <= forecast.ci_upper[i]
    
    # Metadata
    assert forecast.metadata["forecast_steps"] == 2
    assert forecast.metadata["confidence_level"] == 
    assert forecast.metadata["seasonal_order"] == (, , , 2)


def test_sarima_predict_invalid_steps(monthly_seasonal_data, sarima_meta):
    """Test SARIMA rejects invalid forecast steps."""
    params = {"order": (, , ), "seasonal_order": (, , , )}
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    model.fit()
    
    with pytest.raises(Valuerror, match="steps must be > "):
        model.predict(steps=)
    
    with pytest.raises(Valuerror, match="steps must be > "):
        model.predict(steps=-)


def test_sarima_predict_with_std_errors(monthly_seasonal_data, sarima_meta):
    """Test SARIMA can return forecast standard errors."""
    params = {"order": (, , ), "seasonal_order": (, , , 2)}
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    model.fit()
    
    forecast = model.predict(steps=2, return_std=True)
    
    assert "forecast_std_errors" in forecast.payload
    assert len(forecast.payload["forecast_std_errors"]) == 2


def test_sarima_quarterly_seasonality(quarterly_data, sarima_meta):
    """Test SARIMA with quarterly data and annual seasonality."""
    params = {
        "order": (, , ),
        "seasonal_order": (, , , 4),  # Quarterly seasonality
    }
    model = SRIMModel(quarterly_data, params, sarima_meta)
    
    result = model.fit()
    forecast = model.predict(steps=4)
    
    assert result.payload["seasonal_period"] == 4
    assert len(forecast.forecast_values) == 4


def test_sarima_run_hash_deterministic(monthly_seasonal_data, sarima_meta):
    """Test SARIMA run_hash is deterministic for same inputs."""
    params = {"order": (, , ), "seasonal_order": (, , , 2)}
    
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    model2 = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    
    assert model.run_hash() == model2.run_hash()


def test_sarima_run_hash_different_params(monthly_seasonal_data, sarima_meta):
    """Test SARIMA run_hash changes with different parameters."""
    params = {"order": (, , ), "seasonal_order": (, , , 2)}
    params2 = {"order": (2, , 2), "seasonal_order": (, , , 2)}
    
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    model2 = SRIMModel(monthly_seasonal_data, params2, sarima_meta)
    
    assert model.run_hash() != model2.run_hash()


def test_sarima_serialization(monthly_seasonal_data, sarima_meta):
    """Test SARIMA model can be Userialized and deserialized."""
    params = {"order": (, , ), "seasonal_order": (, , , 2)}
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    model.fit()
    
    Userialized = model.Userialize()
    assert isinstance(Userialized, bytes)
    assert len(Userialized) > 


def test_sarima_result_hash_deterministic(monthly_seasonal_data, sarima_meta):
    """Test SARIMA forecast results have deterministic hashes."""
    params = {"order": (, , ), "seasonal_order": (, , , 2)}
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    model.fit()
    
    forecast = model.predict(steps=2, alpha=.)
    forecast2 = model.predict(steps=2, alpha=.)
    
    # Same inputs should produce same hash
    assert forecast.result_hash == forecast2.result_hash


def test_sarima_different_confidence_levels(monthly_seasonal_data, sarima_meta):
    """Test SARIMA confidence intervals widen with higher confidence."""
    params = {"order": (, , ), "seasonal_order": (, , , 2)}
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    model.fit()
    
    forecast_ = model.predict(steps=2, alpha=.)  # % I
    forecast_ = model.predict(steps=2, alpha=.)  # % I
    
    # % I should be wider than % I
    width_ = forecast_.ci_upper[] - forecast_.ci_lower[]
    width_ = forecast_.ci_upper[] - forecast_.ci_lower[]
    
    assert width_ > width_


def test_sarima_seasonal_decomposition_not_fitted(monthly_seasonal_data, sarima_meta):
    """Test seasonal decomposition raises error if model not fitted."""
    params = {"order": (, , ), "seasonal_order": (, , , 2)}
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    
    with pytest.raises(Valuerror, match="Model must be fitted"):
        model.get_seasonal_decomposition()


def test_sarima_seasonal_decomposition(monthly_seasonal_data, sarima_meta):
    """Test seasonal decomposition returns info for seasonal models."""
    params = {"order": (, , ), "seasonal_order": (, , , 2)}
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    model.fit()
    
    decomp = model.get_seasonal_decomposition()
    
    assert decomp is not None
    assert decomp["seasonal_period"] == 2


def test_sarima_no_seasonal_decomposition(monthly_seasonal_data, sarima_meta):
    """Test seasonal decomposition returns None for non-seasonal models."""
    params = {"order": (, , ), "seasonal_order": (, , , )}
    model = SRIMModel(monthly_seasonal_data, params, sarima_meta)
    model.fit()
    
    decomp = model.get_seasonal_decomposition()
    
    assert decomp is None
