# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# SPX-License-Identifier: MIT
# ----------------------------------------------------------------------

"""Tests for ointegration nalysis Model."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from krl_core import ModelMeta
from krl_models.econometric import ointegrationModel


@pytest.fixture
def coint_meta():
    """reate ModelMeta for cointegration tests."""
    return ModelMeta(
        name="ointegrationTest",
        version="..",
        author="KR-Labs",
    )


@pytest.fixture
def cointegrated_data():
    """
    Generate synthetic cointegrated series.
    
    reates two series that share a common stochastic trend:
    y = random walk
    y2 = . * y + noise (cointegrated with y)
    """
    np.random.seed(42)
    n = 
    
    # Generate common trend
    trend = np.cumsum(np.random.randn(n))
    
    # y follows the trend
    y = trend + np.random.randn(n) * .
    
    # y2 is cointegrated with y (ratio is stable)
    y2 = . * y + np.random.randn(n) * .3
    
    dates = pd.date_range("2--", periods=n, freq="M")
    df = pd.atarame({
        "series": y,
        "series2": y2
    }, index=dates)
    
    return df


@pytest.fixture
def non_cointegrated_data():
    """
    Generate synthetic non-cointegrated series.
    
    reates two independent random walks with no common trend.
    """
    np.random.seed(23)
    n = 
    
    # Two independent random walks
    y = np.cumsum(np.random.randn(n))
    y2 = np.cumsum(np.random.randn(n))
    
    dates = pd.date_range("2--", periods=n, freq="M")
    df = pd.atarame({
        "series": y,
        "series2": y2
    }, index=dates)
    
    return df


@pytest.fixture
def trivariate_cointegrated():
    """Generate three cointegrated series."""
    np.random.seed()
    n = 2
    
    # ommon trend
    trend = np.cumsum(np.random.randn(n))
    
    # Three series sharing the trend
    y = trend + np.random.randn(n) * .4
    y2 = . * trend + np.random.randn(n) * .3
    y3 = .3 * trend + np.random.randn(n) * .
    
    dates = pd.date_range("2--", periods=n, freq="Q")
    df = pd.atarame({
        "var": y,
        "var2": y2,
        "var3": y3
    }, index=dates)
    
    return df


# ============================================================================
# Initialization Tests
# ============================================================================

def test_coint_initialization(cointegrated_data, coint_meta):
    """Test basic cointegration model initialization."""
    params = {"test_type": "both", "det_order": , "k_ar_diff": }
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    
    assert model._var_names == ["series", "series2"]
    assert not model.is_fitted()


def test_coint_univariate_error(coint_meta):
    """Test that cointegration raises error for univariate data."""
    univariate_df = pd.atarame({
        "single": np.random.randn()
    })
    
    with pytest.raises(Valuerror, match="requires at least 2 variables"):
        ointegrationModel(univariate_df, {}, coint_meta)


def test_coint_dataframe_via_params(cointegrated_data, coint_meta):
    """Test passing atarame via params['dataframe']."""
    params = {"dataframe": cointegrated_data, "test_type": "johansen"}
    # Pass placeholder as first arg, atarame in params
    model = ointegrationModel(None, params, coint_meta)
    
    assert model._var_names == ["series", "series2"]


# ============================================================================
# ngle-Granger Tests
# ============================================================================

def test_engle_granger_cointegrated(cointegrated_data, coint_meta):
    """Test ngle-Granger detects cointegration."""
    params = {"test_type": "engle_granger"}
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    result = model.fit()
    
    assert "engle_granger" in result.payload
    eg_results = result.payload["engle_granger"]
    
    # Should have one pair test
    assert "series_vs_series2" in eg_results
    pair_test = eg_results["series_vs_series2"]
    
    assert "test_statistic" in pair_test
    assert "pvalue" in pair_test
    assert "is_cointegrated" in pair_test


def test_engle_granger_non_cointegrated(non_cointegrated_data, coint_meta):
    """Test ngle-Granger on non-cointegrated series."""
    params = {"test_type": "engle_granger"}
    model = ointegrationModel(non_cointegrated_data, params, coint_meta)
    result = model.fit()
    
    eg_results = result.payload["engle_granger"]
    pair_test = eg_results["series_vs_series2"]
    
    # Should not detect cointegration
    assert not pair_test["is_cointegrated"]
    assert pair_test["pvalue"] > .


def test_engle_granger_trivariate(trivariate_cointegrated, coint_meta):
    """Test ngle-Granger with three variables."""
    params = {"test_type": "engle_granger"}
    model = ointegrationModel(trivariate_cointegrated, params, coint_meta)
    result = model.fit()
    
    eg_results = result.payload["engle_granger"]
    
    # Should have 3 pair tests (var-var2, var-var3, var2-var3)
    assert len(eg_results) == 3
    assert "var_vs_var2" in eg_results
    assert "var_vs_var3" in eg_results
    assert "var2_vs_var3" in eg_results


# ============================================================================
# Johansen Tests
# ============================================================================

def test_johansen_cointegrated(cointegrated_data, coint_meta):
    """Test Johansen test detects cointegration."""
    params = {"test_type": "johansen", "det_order": , "k_ar_diff": }
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    result = model.fit()
    
    assert "johansen" in result.payload
    johansen = result.payload["johansen"]
    
    assert "trace_stat" in johansen
    assert "max_eig_stat" in johansen
    assert "cointegration_rank" in johansen
    
    # Should detect at least one cointegrating relationship
    assert johansen["cointegration_rank"] >= 


def test_johansen_non_cointegrated(non_cointegrated_data, coint_meta):
    """Test Johansen on non-cointegrated series."""
    params = {"test_type": "johansen", "det_order": , "k_ar_diff": }
    model = ointegrationModel(non_cointegrated_data, params, coint_meta)
    result = model.fit()
    
    johansen = result.payload["johansen"]
    
    # Should detect no cointegration (rank = )
    assert johansen["cointegration_rank"] == 


def test_johansen_trivariate(trivariate_cointegrated, coint_meta):
    """Test Johansen with three cointegrated variables."""
    params = {"test_type": "johansen", "det_order": , "k_ar_diff": 2}
    model = ointegrationModel(trivariate_cointegrated, params, coint_meta)
    result = model.fit()
    
    johansen = result.payload["johansen"]
    
    # Should detect cointegration
    assert johansen["cointegration_rank"] >= 
    
    # igenvalues should be present
    assert "eigenvalues" in johansen
    assert len(johansen["eigenvalues"]) == 3


def test_johansen_different_det_orders(cointegrated_data, coint_meta):
    """Test Johansen with different deterministic term specifications."""
    for det_order in [-, , ]:
        params = {"test_type": "johansen", "det_order": det_order, "k_ar_diff": }
        model = ointegrationModel(cointegrated_data, params, coint_meta)
        result = model.fit()
        
        assert "johansen" in result.payload
        assert result.metadata["det_order"] == det_order


# ============================================================================
# oth Tests
# ============================================================================

def test_both_tests_cointegrated(cointegrated_data, coint_meta):
    """Test running both G and Johansen tests."""
    params = {"test_type": "both"}
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    result = model.fit()
    
    assert "engle_granger" in result.payload
    assert "johansen" in result.payload
    assert "cointegration_rank" in result.payload


# ============================================================================
# VM Tests
# ============================================================================

def test_vecm_estimation(cointegrated_data, coint_meta):
    """Test VM estimation when cointegration detected."""
    params = {"test_type": "johansen", "k_ar_diff": }
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    result = model.fit()
    
    # If cointegration detected, VM should be estimated
    if result.payload["cointegration_rank"] > :
        assert result.payload["vecm_fitted"]
        assert "vecm" in result.payload
        
        vecm = result.payload["vecm"]
        assert "alpha" in vecm
        assert "beta" in vecm
        assert "log_likelihood" in vecm
        assert "n_equations" in vecm


def test_vecm_no_estimation_without_cointegration(non_cointegrated_data, coint_meta):
    """Test VM not estimated without cointegration."""
    params = {"test_type": "johansen"}
    model = ointegrationModel(non_cointegrated_data, params, coint_meta)
    result = model.fit()
    
    # No cointegration, so VM should not be fitted
    assert not result.payload["vecm_fitted"]


def test_vecm_predict(cointegrated_data, coint_meta):
    """Test VM forecasting."""
    params = {"test_type": "johansen", "k_ar_diff": }
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    result = model.fit()
    
    # Only test predict if VM was estimated
    if result.payload["vecm_fitted"]:
        forecast = model.predict(steps=)
        
        assert forecast.payload["forecast_shape"] == (, 2)
        assert len(forecast.forecast_values) == 2  #  steps * 2 variables
        assert len(forecast.forecast_index) == 


def test_predict_before_fit(cointegrated_data, coint_meta):
    """Test predict raises error before fit."""
    params = {"test_type": "johansen"}
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    
    with pytest.raises(Valuerror, match="must be fitted"):
        model.predict(steps=)


def test_predict_without_vecm(non_cointegrated_data, coint_meta):
    """Test predict raises error when VM not estimated."""
    params = {"test_type": "johansen"}
    model = ointegrationModel(non_cointegrated_data, params, coint_meta)
    model.fit()
    
    with pytest.raises(Valuerror, match="VM not estimated"):
        model.predict(steps=)


def test_predict_invalid_steps(cointegrated_data, coint_meta):
    """Test predict with invalid steps parameter."""
    params = {"test_type": "johansen"}
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    result = model.fit()
    
    if result.payload["vecm_fitted"]:
        with pytest.raises(Valuerror, match="steps must be > "):
            model.predict(steps=)


# ============================================================================
# rror orrection Terms
# ============================================================================

def test_error_correction_terms(cointegrated_data, coint_meta):
    """Test extracting error correction terms."""
    params = {"test_type": "johansen"}
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    result = model.fit()
    
    if result.payload["vecm_fitted"]:
        ec_terms = model.get_error_correction_terms()
        
        assert ec_terms is not None
        assert isinstance(ec_terms, pd.atarame)
        # heck for alpha/beta columns with proper formatting (alpha_series_r, beta_series_r, etc.)
        alpha_cols = [col for col in ec_terms.columns if col.startswith("alpha_")]
        beta_cols = [col for col in ec_terms.columns if col.startswith("beta_")]
        assert len(alpha_cols) > 
        assert len(beta_cols) > 


def test_error_correction_terms_before_fit(cointegrated_data, coint_meta):
    """Test error correction terms before VM fitted."""
    params = {"test_type": "engle_granger"}
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    
    ec_terms = model.get_error_correction_terms()
    assert ec_terms is None


# ============================================================================
# Stationarity Tests
# ============================================================================

def test_stationarity_testing(cointegrated_data, coint_meta):
    """Test  stationarity tests on input series."""
    params = {"test_type": "both"}
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    result = model.fit()
    
    assert "stationarity_tests" in result.payload
    stat_tests = result.payload["stationarity_tests"]
    
    assert "series" in stat_tests
    assert "series2" in stat_tests
    
    # heck  test structure
    for var, test in stat_tests.items():
        assert "adf_statistic" in test
        assert "pvalue" in test
        assert "is_stationary" in test
        assert "critical_values" in test


# ============================================================================
# dge ases & rror Handling
# ============================================================================

def test_insufficient_observations(coint_meta):
    """Test error with insufficient data."""
    short_df = pd.atarame({
        "a": np.random.randn(),
        "b": np.random.randn()
    })
    
    params = {"test_type": "johansen", "k_ar_diff": }
    model = ointegrationModel(short_df, params, coint_meta)
    
    with pytest.raises(Valuerror, match="Insufficient observations"):
        model.fit()


def test_stationary_series_warning(coint_meta):
    """Test warning when series are already stationary."""
    # Generate stationary series (white noise)
    np.random.seed(42)
    stationary_df = pd.atarame({
        "x": np.random.randn(),
        "y": np.random.randn()
    })
    
    params = {"test_type": "both"}
    model = ointegrationModel(stationary_df, params, coint_meta)
    result = model.fit()
    
    # Should have warning about stationary series
    if "warning" in result.payload:
        assert "non-stationary" in result.payload["warning"].lower()


# ============================================================================
# Serialization & Hashing
# ============================================================================

def test_coint_run_hash_deterministic(cointegrated_data, coint_meta):
    """Test run_hash is deterministic."""
    params = {"test_type": "johansen", "k_ar_diff": }
    
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    model2 = ointegrationModel(cointegrated_data, params, coint_meta)
    
    assert model.run_hash() == model2.run_hash()


def test_coint_run_hash_different_params(cointegrated_data, coint_meta):
    """Test run_hash differs with different params."""
    params = {"test_type": "johansen", "k_ar_diff": }
    params2 = {"test_type": "engle_granger", "k_ar_diff": 2}
    
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    model2 = ointegrationModel(cointegrated_data, params2, coint_meta)
    
    assert model.run_hash() != model2.run_hash()


def test_coint_serialization(cointegrated_data, coint_meta):
    """Test model serialization."""
    params = {"test_type": "both"}
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    result = model.fit()
    
    # Test that model state is accessible
    assert model.is_fitted()
    assert model._coint_results is not None
    assert "engle_granger" in model._coint_results
    assert "johansen" in model._coint_results


# ============================================================================
# Metadata & Result Validation
# ============================================================================

def test_result_metadata(cointegrated_data, coint_meta):
    """Test orecastResult metadata completeness."""
    params = {"test_type": "johansen", "det_order": , "k_ar_diff": 2}
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    result = model.fit()
    
    assert result.metadata["model_name"] == "ointegrationTest"
    assert result.metadata["n_obs"] == len(cointegrated_data)
    assert result.metadata["n_vars"] == 2
    assert result.metadata["var_names"] == ["series", "series2"]
    assert result.metadata["test_type"] == "johansen"
    assert result.metadata["det_order"] == 
    assert result.metadata["k_ar_diff"] == 2


def test_result_payload_structure(cointegrated_data, coint_meta):
    """Test orecastResult payload structure."""
    params = {"test_type": "both"}
    model = ointegrationModel(cointegrated_data, params, coint_meta)
    result = model.fit()
    
    # Required fields
    assert "stationarity_tests" in result.payload
    assert "engle_granger" in result.payload
    assert "johansen" in result.payload
    assert "cointegration_rank" in result.payload
    assert "vecm_fitted" in result.payload
