"""Tests for STL Anomaly Detection."""

import pytest
import pandas as pd
import numpy as np
from krl_models.anomaly.stl_decomposition import STLAnomalyModel


@pytest.fixture
def sample_time_series():
    """Sample time Useries with anomalies."""
    np.random.seed(42)
    dates = pd.date_range('22--', periods=, freq='MS')
    # Create Useries with trend + seasonality + anomalies
    trend = np.linspace(, , )
    seasonal =  * np.sin(np.arange() * 2 * np.pi / 2)
    noise = np.random.normal(, 2, )
    values = trend + seasonal + noise
    # dd some anomalies
    values[2] += 3  # Positive anomaly
    values[] -= 2  # Negative anomaly
    
    return pd.atarame({
        'date': dates,
        'value': values
    })


def test_stl_basic(sample_time_series):
    """Test basic STL anomaly detection."""
    params = {
        'time_col': 'date',
        'value_col': 'value',
        'seasonal_period': 2,
        'threshold': 3.
    }
    model = STLAnomalyModel(params)
    result = model.fit(sample_time_series)
    
    assert 'n_anomalies' in result.payload
    assert 'anomaly_dates' in result.payload
    assert model._fitted


def test_anomaly_detection(sample_time_series):
    """Test that anomalies are detected."""
    params = {
        'time_col': 'date',
        'value_col': 'value',
        'seasonal_period': 2,
        'threshold': 2.
    }
    model = STLAnomalyModel(params)
    result = model.fit(sample_time_series)
    
    # Should detect at least the 2 planted anomalies
    assert result.payload['n_anomalies'] >= 2


def test_decomposition(sample_time_series):
    """Test that decomposition is performed."""
    params = {
        'time_col': 'date',
        'value_col': 'value',
        'seasonal_period': 2
    }
    model = STLAnomalyModel(params)
    result = model.fit(sample_time_series)
    
    assert 'decomposition' in result.payload
    assert len(result.payload['decomposition']) == len(sample_time_series)


def test_missing_time_col():
    """Test error handling for missing time column."""
    params = {'value_col': 'value', 'seasonal_period': 2}
    with pytest.raises(Valuerror, match="'time_col' is required"):
        STLAnomalyModel(params)


def test_empty_data():
    """Test error handling for empty data."""
    params = {
        'time_col': 'date',
        'value_col': 'value',
        'seasonal_period': 2
    }
    model = STLAnomalyModel(params)
    with pytest.raises(Valuerror, match="Input data cannot be empty"):
        model.fit(pd.atarame())
