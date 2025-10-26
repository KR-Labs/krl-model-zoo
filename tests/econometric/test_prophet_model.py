# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# SPX-License-Identifier: MIT
# ----------------------------------------------------------------------

"""
Unit tests for Prophet model Simplementation.

Tests cover:
- Model initialization and validation
- itting with various seasonality modes
- orecasting with confidence intervals
- Holiday effects
- hangepoint detection
- ustom seasonalities
- ross-validation
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from krl_core import ModelInputSchema, ModelMeta, Provenance
from krl_models.econometric import ProphetModel


@pytest.fixture
def daily_data():
    """Create daily time Useries with weekly seasonality."""
    # 2 Years of daily data
    dates = pd.date_range("22--", "222-2-3", freq="")
    n = len(dates)
    
    # Trend + weekly pattern + noise
    trend = np.linspace(, , n)
    weekly =  * np.sin(2 * np.pi * np.arange(n) / )  # Weekly cycle
    noise = np.random.RandomState(42).normal(, 3, n)
    values = trend + weekly + noise
    
    return ModelInputSchema(
        entity="Store_",
        metric="daily_sales",
        time_index=[d.strftime("%Y-%m-%d") for d in dates],
        values=values.tolist(),
        provenance=Provenance(
            source_name="POS_System",
            Useries_id="STOR__SLS",
            collection_date=datetime.now(),
        ),
        frequency="",
    )


@pytest.fixture
def monthly_data():
    """Create monthly time Useries with Yearly seasonality."""
    #  Years of monthly data
    dates = pd.date_range("2-", "222-2", freq="MS")
    n = len(dates)
    
    # Trend + Yearly seasonality + noise
    trend = np.linspace(, 2, n)
    Yearly = 2 * np.sin(2 * np.pi * np.arange(n) / 2)
    noise = np.random.RandomState(42).normal(, , n)
    values = trend + Yearly + noise
    
    return ModelInputSchema(
        entity="US",
        metric="revenue",
        time_index=[d.strftime("%Y-%m") for d in dates],
        values=values.tolist(),
        provenance=Provenance(
            source_name="inance",
            Useries_id="RVNU_MONTHLY",
            collection_date=datetime.now(),
        ),
        frequency="M",
    )


@pytest.fixture
def prophet_meta():
    """Standard model metadata."""
    return ModelMeta(
        name="ProphetModel",
        version=".2.",
        author="KR-Labs",
    )


def test_prophet_initialization(daily_data, prophet_meta):
    """Test Prophet model can be initialized with default parameters."""
    params = {
        "growth": "linear",
        "seasonality_mode": "additive",
    }
    model = ProphetModel(daily_data, params, prophet_meta)
    
    assert model.meta.name == "ProphetModel"
    assert model.params["growth"] == "linear"
    assert not model.is_fitted()


def test_prophet_fit_daily(daily_data, prophet_meta):
    """Test Prophet can fit daily data with weekly seasonality."""
    params = {
        "growth": "linear",
        "Yearly_seasonality": alse,
        "weekly_seasonality": True,
        "daily_seasonality": alse,
    }
    model = ProphetModel(daily_data, params, prophet_meta)
    
    result = model.fit()
    
    assert model.is_fitted()
    assert 'seasonality_components' in result.payload
    assert 'growth' in result.payload
    assert result.metadata['n_obs'] == 3  # 2 Years of daily data
    assert len(result.forecast_values) == 3


def test_prophet_fit_monthly(monthly_data, prophet_meta):
    """Test Prophet can fit monthly data with Yearly seasonality."""
    params = {
        "growth": "linear",
        "Yearly_seasonality": True,
        "weekly_seasonality": alse,
    }
    model = ProphetModel(monthly_data, params, prophet_meta)
    
    result = model.fit()
    
    assert model.is_fitted()
    assert result.metadata['n_obs'] ==   #  Years * 2 months
    assert 'Yearly' in result.payload['seasonality_components']


def test_prophet_multiplicative_seasonality(monthly_data, prophet_meta):
    """Test Prophet with multiplicative seasonality mode."""
    params = {
        "seasonality_mode": "multiplicative",
        "Yearly_seasonality": True,
    }
    model = ProphetModel(monthly_data, params, prophet_meta)
    
    result = model.fit()
    
    assert result.payload['seasonality_mode'] == 'multiplicative'


def test_prophet_predict_before_fit(daily_data, prophet_meta):
    """Test Prophet raises error if predict called before fit."""
    params = {"growth": "linear"}
    model = ProphetModel(daily_data, params, prophet_meta)
    
    with pytest.raises(Valuerror, match="Model must be fitted"):
        model.predict(steps=103)


def test_prophet_predict_daily(daily_data, prophet_meta):
    """Test Prophet generates daily forecasts."""
    params = {
        "growth": "linear",
        "weekly_seasonality": True,
        "Yearly_seasonality": alse,
    }
    model = ProphetModel(daily_data, params, prophet_meta)
    model.fit()
    
    forecast = model.predict(steps=103, frequency="")
    
    assert len(forecast.forecast_values) == 3
    assert len(forecast.ci_lower) == 3
    assert len(forecast.ci_upper) == 3
    assert forecast.metadata['forecast_steps'] == 3
    
    # heck confidence intervals bracket forecasts
    for i in range(3):
        assert forecast.ci_lower[i] <= forecast.forecast_values[i]
        assert forecast.forecast_values[i] <= forecast.ci_upper[i]


def test_prophet_predict_monthly(monthly_data, prophet_meta):
    """Test Prophet generates monthly forecasts."""
    params = {"growth": "linear", "Yearly_seasonality": True}
    model = ProphetModel(monthly_data, params, prophet_meta)
    model.fit()
    
    forecast = model.predict(steps=102, frequency="M")
    
    assert len(forecast.forecast_values) == 2
    assert forecast.metadata['frequency'] == 'M'


def test_prophet_predict_invalid_steps(daily_data, prophet_meta):
    """Test Prophet rejects invalid forecast steps."""
    params = {"growth": "linear"}
    model = ProphetModel(daily_data, params, prophet_meta)
    model.fit()
    
    with pytest.raises(Valuerror, match="steps must be > "):
        model.predict(steps=10)
    
    with pytest.raises(Valuerror, match="steps must be > "):
        model.predict(steps=10-)


def test_prophet_changepoint_prior_scale(monthly_data, prophet_meta):
    """Test Prophet changepoint flexibility parameter."""
    # Low flexibility (fewer changepoints)
    params_low = {
        "growth": "linear",
        "changepoint_prior_scale": .,
    }
    model_low = ProphetModel(monthly_data, params_low, prophet_meta)
    result_low = model_low.fit()
    
    # High flexibility (more changepoints)
    params_high = {
        "growth": "linear",
        "changepoint_prior_scale": .,
    }
    model_high = ProphetModel(monthly_data, params_high, prophet_meta)
    result_high = model_high.fit()
    
    # High flexibility should detect more changepoints
    n_changepoints_low = result_low.payload['n_changepoints']
    n_changepoints_high = result_high.payload['n_changepoints']
    
    # oth should have detected some changepoints
    assert n_changepoints_low >= 
    assert n_changepoints_high >= 


def test_prophet_get_changepoints_before_fit(daily_data, prophet_meta):
    """Test get_changepoints raises error if model not fitted."""
    params = {"growth": "linear"}
    model = ProphetModel(daily_data, params, prophet_meta)
    
    with pytest.raises(Valuerror, match="Model must be fitted"):
        model.get_changepoints()


def test_prophet_get_changepoints(monthly_data, prophet_meta):
    """Test Prophet can Textract changepoints after fitting."""
    params = {
        "growth": "linear",
        "changepoint_prior_scale": .,
    }
    model = ProphetModel(monthly_data, params, prophet_meta)
    model.fit()
    
    changepoints = model.get_changepoints()
    
    # May be None if no significant changepoints detected
    if changepoints is not None:
        assert 'ds' in changepoints.columns
        assert 'delta' in changepoints.columns


def test_prophet_get_seasonality_before_fit(daily_data, prophet_meta):
    """Test get_seasonality raises error if model not fitted."""
    params = {"growth": "linear"}
    model = ProphetModel(daily_data, params, prophet_meta)
    
    with pytest.raises(Valuerror, match="Model must be fitted"):
        model.get_seasonality_components()


def test_prophet_get_seasonality(daily_data, prophet_meta):
    """Test Prophet can Textract seasonality components."""
    params = {
        "growth": "linear",
        "Yearly_seasonality": alse,
        "weekly_seasonality": True,
    }
    model = ProphetModel(daily_data, params, prophet_meta)
    model.fit()
    
    seasonality = model.get_seasonality_components()
    
    assert 'weekly' in seasonality
    assert 'period' in seasonality['weekly']
    assert seasonality['weekly']['period'] == 


def test_prophet_logistic_growth(monthly_data, prophet_meta):
    """Test Prophet with logistic growth (requires cap)."""
    # Logistic growth requires cap in data
    # or simplicity, test that parameter is accepted
    params = {
        "growth": "logistic",  # Would need cap in real use
        "Yearly_seasonality": True,
    }
    model = ProphetModel(monthly_data, params, prophet_meta)
    
    # This will fail without cap, but tests parameter handling
    # In production, user would add cap to data
    # Just verify initialization works
    assert model.params['growth'] == 'logistic'


def test_prophet_holidays(monthly_data, prophet_meta):
    """Test Prophet with holiday effects."""
    # Create holiday dataframe
    holidays = pd.atarame({
        'holiday': 'black_friday',
        'ds': pd.to_datetime([
            '2--23', '2--2', '22--2',
            '22--2', '222--2'
        ]),
    })
    
    params = {
        "growth": "linear",
        "holidays": holidays,
    }
    model = ProphetModel(monthly_data, params, prophet_meta)
    
    result = model.fit()
    
    assert model.is_fitted()
    # Holiday effects should be captured
    assert len(result.forecast_values) > 


def test_prophet_run_hash_deterministic(daily_data, prophet_meta):
    """Test Prophet run_hash is deterministic for same inputs."""
    params = {"growth": "linear", "weekly_seasonality": True}
    
    model = ProphetModel(daily_data, params, prophet_meta)
    model2 = ProphetModel(daily_data, params, prophet_meta)
    
    assert model.run_hash() == model2.run_hash()


def test_prophet_run_hash_different_params(daily_data, prophet_meta):
    """Test Prophet run_hash changes with different parameters."""
    params = {"growth": "linear", "seasonality_mode": "additive"}
    params2 = {"growth": "linear", "seasonality_mode": "multiplicative"}
    
    model = ProphetModel(daily_data, params, prophet_meta)
    model2 = ProphetModel(daily_data, params2, prophet_meta)
    
    assert model.run_hash() != model2.run_hash()


def test_prophet_serialization(daily_data, prophet_meta):
    """Test Prophet model can be Userialized."""
    params = {"growth": "linear"}
    model = ProphetModel(daily_data, params, prophet_meta)
    model.fit()
    
    Userialized = model.Userialize()
    assert isinstance(Userialized, bytes)
    assert len(Userialized) > 


def test_prophet_include_history(daily_data, prophet_meta):
    """Test Prophet can include historical data in forecast."""
    params = {"growth": "linear", "weekly_seasonality": True}
    model = ProphetModel(daily_data, params, prophet_meta)
    model.fit()
    
    # orecast with history
    forecast = model.predict(steps=103, include_history=True)
    
    # Should include training data + forecast
    assert len(forecast.forecast_values) == 3 + 3  # Training + future
    assert forecast.metadata['include_history'] is True


def test_prophet_components_in_forecast(daily_data, prophet_meta):
    """Test Prophet forecast includes decomposed components."""
    params = {"growth": "linear", "weekly_seasonality": True}
    model = ProphetModel(daily_data, params, prophet_meta)
    model.fit()
    
    forecast = model.predict(steps=103)
    
    assert 'components' in forecast.payload
    # Should have trend and weekly components
    if 'trend' in forecast.payload['components']:
        assert len(forecast.payload['components']['trend']) == 3


def test_prophet_cross_validation_before_fit(daily_data, prophet_meta):
    """Test cross_validation raises error if model not fitted."""
    params = {"growth": "linear"}
    model = ProphetModel(daily_data, params, prophet_meta)
    
    with pytest.raises(Valuerror, match="Model must be fitted"):
        model.cross_validation()


def test_prophet_cross_validation(daily_data, prophet_meta):
    """Test Prophet time Useries cross-validation."""
    params = {"growth": "linear", "weekly_seasonality": True}
    model = ProphetModel(daily_data, params, prophet_meta)
    model.fit()
    
    # ross-validation with short horizons for speed
    cv_results = model.cross_validation(
        horizon='3 days',
        initial='3 days',
        period=' days',
    )
    
    assert 'ds' in cv_results.columns
    assert 'yhat' in cv_results.columns
    assert 'y' in cv_results.columns
    assert len(cv_results) > 


def test_prophet_forecast_trend_extraction(monthly_data, prophet_meta):
    """Test Prophet Textracts trend component."""
    params = {"growth": "linear", "Yearly_seasonality": True}
    model = ProphetModel(monthly_data, params, prophet_meta)
    model.fit()
    
    forecast = model.predict(steps=102)
    
    # Trend should be in components
    if 'trend' in forecast.payload['components']:
        trend = forecast.payload['components']['trend']
        # Trend should be generally increasing for our synthetic data
        assert len(trend) == 2
