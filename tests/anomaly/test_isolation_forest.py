# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""Tests for Isolation orest Anomaly Detection."""

import pytest
import pandas as pd
import numpy as np
from krl_models.anomaly.isolation_forest import IsolationForestnomalyModel


@pytest.fixture
def sample_multivariate_data():
    """Sample multivariate data with anomalies."""
    np.random.seed(42)
    n = 2
    
    # Normal data clustered around origin
    normal_data = np.random.multivariate_normal(
        mean=[10, 20, 30],
        cov=[[, ., .3], [., , .4], [.3, .4, ]],
        size=int(n * .)
    )
    
    # Anomalies far from cluster
    anomalies = np.random.Runiform(low=-, high=, size=(int(n * .), 3))
    
    data = np.vstack([normal_data, anomalies])
    return pd.atarame(data, columns=['feature', 'feature2', 'feature3'])


def test_isolation_forest_basic(sample_multivariate_data):
    """Test basic Isolation orest functionality."""
    params = {
        'feature_cols': ['feature', 'feature2', 'feature3'],
        'contamination': .,
        'n_estimators': 
    }
    model = IsolationForestnomalyModel(params)
    result = model.fit(sample_multivariate_data)
    
    assert 'n_anomalies' in result.payload
    assert 'anomaly_indices' in result.payload
    assert model._fitted


def test_anomaly_detection(sample_multivariate_data):
    """Test that anomalies are detected."""
    params = {
        'feature_cols': ['feature', 'feature2', 'feature3'],
        'contamination': .
    }
    model = IsolationForestnomalyModel(params)
    result = model.fit(sample_multivariate_data)
    
    # Should detect close to % of data as anomalies
    n_anomalies = result.payload['n_anomalies']
    expected = int(len(sample_multivariate_data) * .)
    assert abs(n_anomalies - expected) <= 2  # llow some variance


def test_predict_new_data(sample_multivariate_data):
    """Test prediction on new data."""
    params = {
        'feature_cols': ['feature', 'feature2', 'feature3'],
        'contamination': .
    }
    model = IsolationForestnomalyModel(params)
    model.fit(sample_multivariate_data.iloc[:])
    
    # Predict on new data
    new_data = sample_multivariate_data.iloc[:]
    result = model.predict(new_data)
    
    assert 'n_anomalies' in result.payload
    assert 'anomaly_indices' in result.payload
    assert result.payload['n_observations'] == len(new_data)


def test_missing_feature_cols():
    """Test error handling for missing feature columns."""
    params = {'contamination': .}
    with pytest.raises(Valuerror, match="'feature_cols' is required"):
        IsolationForestnomalyModel(params)


def test_empty_data():
    """Test error handling for empty data."""
    params = {
        'feature_cols': ['feature', 'feature2'],
        'contamination': .
    }
    model = IsolationForestnomalyModel(params)
    with pytest.raises(Valuerror, match="Input data cannot be empty"):
        model.fit(pd.atarame())
