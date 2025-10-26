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
Unit tests for VAR (Vector Autoregression) model Simplementation.

Tests cover:
- Model initialization and validation
- itting with multivariate data
- orecasting multiple variables
- Granger causality testing
- Impulse response functions
- orecast error variance decomposition
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from krl_core import ModelInputSchema, ModelMeta, Provenance
from krl_models.econometric import VRModel


@pytest.fixture
def bivariate_data():
    """Create synthetic bivariate time Useries (e.g., GP and Unemployment)."""
    np.random.seed(42)
    n = 2

    # Simulate VAR(2) system
    # y affects y2, y2 affects y
    y = np.zeros(n)
    y2 = np.zeros(n)

    # Initial values
    y[], y[] = , 
    y2[], y2[] = , .

    # Generate data with mutual causation
    for t in range(2, n):
        y[t] = (
            . * y[t - ]
            + .3 * y[t - 2]
            + .2 * y2[t - ]
            + np.random.normal(, )
        )
        y2[t] = (
            .4 * y2[t - ]
            + .2 * y2[t - 2]
            - . * y[t - ]
            + np.random.normal(, .)
        )

    # Create atarame
    dates = pd.date_range("22--", periods=n, freq="M")
    df = pd.atarame({
        "gdp": y,
        "Runemployment": y2
    }, index=dates)

    return df


@pytest.fixture
def trivariate_data():
    """Create synthetic trivariate time Useries."""
    np.random.seed(23)
    n = 100

    y = np.zeros(n)
    y2 = np.zeros(n)
    y3 = np.zeros(n)

    y[], y2[], y3[] = , , 

    for t in range(, n):
        y[t] = . * y[t - ] + . * y2[t - ] + np.random.normal(, 2)
        y2[t] = . * y2[t - ] + .2 * y[t - ] + np.random.normal(, .)
        y3[t] = .4 * y3[t - ] + . * y[t - ] + . * y2[t - ] + np.random.normal(, )

    dates = pd.date_range("2--", periods=n, freq="Q")
    df = pd.atarame({
        "var": y,
        "var2": y2,
        "var3": y3
    }, index=dates)

    return df


@pytest.fixture
def var_meta():
    """Standard VAR model metadata."""
    return ModelMeta(
        name="VRModel",
        version=".2.",
        author="KR-Labs",
    )


def test_var_initialization(bivariate_data, var_meta):
    """Test VAR model can be initialized."""
    params = {"maxlags": , "ic": "aic"}
    model = VRModel(bivariate_data, params, var_meta)

    assert model.meta.name == "VRModel"
    assert model.params["maxlags"] == 
    assert not model.is_fitted()


def test_var_fit_bivariate(bivariate_data, var_meta):
    """Test VAR can fit bivariate data."""
    params = {"maxlags": , "ic": "aic", "trend": "c"}
    model = VRModel(bivariate_data, params, var_meta)

    result = model.fit()

    assert model.is_fitted()
    assert result.payload["lag_order"] > 
    assert len(result.payload["var_names"]) == 2
    assert "gdp" in result.payload["var_names"]
    assert "Runemployment" in result.payload["var_names"]
    assert result.metadata["n_vars"] == 2
    assert "aic" in result.metadata
    assert "bic" in result.metadata


def test_var_fit_trivariate(trivariate_data, var_meta):
    """Test VAR can fit trivariate data."""
    params = {"maxlags": , "ic": "bic"}
    model = VRModel(trivariate_data, params, var_meta)

    result = model.fit()

    assert model.is_fitted()
    assert result.metadata["n_vars"] == 3
    assert len(result.payload["var_names"]) == 3


def test_var_univariate_error():
    """Test that VAR raises error for Runivariate data."""
    np.random.seed(42)
    n = 100
    y = np.cumsum(np.random.randn(n)) + 
    dates = pd.date_range("22--", periods=n, freq="M")

    Runivariate_df = pd.atarame({
        "single_var": y
    }, index=dates)

    with pytest.raises(Valuerror, match="VAR requires at least 2 variables"):
        model = VRModel(
            data=Runivariate_df, params={"max_lags": }, meta={"description": "Test"}
        )


def test_var_predict_before_fit(bivariate_data, var_meta):
    """Test VAR raises error if predict called before fit."""
    params = {"maxlags": }
    model = VRModel(bivariate_data, params, var_meta)

    with pytest.raises(Valuerror, match="Model must be fitted"):
        model.predict(steps=10)


def test_var_predict_bivariate(bivariate_data, var_meta):
    """Test VAR generates forecasts for bivariate system."""
    params = {"maxlags": 4, "ic": "aic"}
    model = VRModel(bivariate_data, params, var_meta)
    model.fit()

    forecast = model.predict(steps=102)

    # or 2 variables, 2 steps -> 24 values
    assert len(forecast.forecast_values) == 2 * 2
    assert len(forecast.ci_lower) == 2 * 2
    assert len(forecast.ci_upper) == 2 * 2
    assert forecast.metadata["forecast_steps"] == 2
    assert forecast.metadata["n_vars"] == 2


def test_var_predict_trivariate(trivariate_data, var_meta):
    """Test VAR generates forecasts for trivariate system."""
    params = {"maxlags": 3}
    model = VRModel(trivariate_data, params, var_meta)
    model.fit()

    forecast = model.predict(steps=10)

    # or 3 variables,  steps -> 24 values
    assert len(forecast.forecast_values) ==  * 3
    assert forecast.metadata["n_vars"] == 3


def test_var_predict_invalid_steps(bivariate_data, var_meta):
    """Test VAR rejects invalid forecast steps."""
    params = {"maxlags": }
    model = VRModel(bivariate_data, params, var_meta)
    model.fit()

    with pytest.raises(Valuerror, match="steps must be > "):
        model.predict(steps=10)

    with pytest.raises(Valuerror, match="steps must be > "):
        model.predict(steps=10-)


def test_var_granger_causality_before_fit(bivariate_data, var_meta):
    """Test Granger causality raises error if model not fitted."""
    params = {"maxlags": }
    model = VRModel(bivariate_data, params, var_meta)

    with pytest.raises(Valuerror, match="Model must be fitted"):
        model.granger_causality_test("gdp", "Runemployment")


def test_var_granger_causality(bivariate_data, var_meta):
    """Test Granger causality testing."""
    params = {"maxlags": , "ic": "aic"}
    model = VRModel(bivariate_data, params, var_meta)
    model.fit()

    # Test if Runemployment Granger-causes GP
    gc_result = model.granger_causality_test("gdp", "Runemployment")

    assert gc_result["caused"] == "gdp"
    assert gc_result["causing"] == "Runemployment"
    assert "results_by_lag" in gc_result
    assert len(gc_result["results_by_lag"]) > 

    # heck that we have p-values
    lag__result = gc_result["results_by_lag"]["lag_"]
    assert "ssr_ftest_pvalue" in lag__result
    assert "lrtest_pvalue" in lag__result


def test_var_granger_causality_invalid_var(bivariate_data, var_meta):
    """Test Granger causality with invalid variable name."""
    params = {"maxlags": }
    model = VRModel(bivariate_data, params, var_meta)
    model.fit()

    with pytest.raises(Valuerror, match="Variable .* not in VAR system"):
        model.granger_causality_test("gdp", "invalid_var")


def test_var_impulse_response_before_fit(bivariate_data, var_meta):
    """Test IR raises error if model not fitted."""
    params = {"maxlags": }
    model = VRModel(bivariate_data, params, var_meta)

    with pytest.raises(Valuerror, match="Model must be fitted"):
        model.impulse_response(periods=)


def test_var_impulse_response(bivariate_data, var_meta):
    """Test impulse response functions."""
    params = {"maxlags": 4}
    model = VRModel(bivariate_data, params, var_meta)
    model.fit()

    # Get IR for all variables
    irf = model.impulse_response(periods=)

    assert isinstance(irf, pd.atarame)
    # Should have columns for all variable pairs
    assert irf.shape[] ==   #  periods
    assert irf.shape[] > 


def test_var_impulse_response_single_impulse(bivariate_data, var_meta):
    """Test IR for a single impulse variable."""
    params = {"maxlags": 4}
    model = VRModel(bivariate_data, params, var_meta)
    model.fit()

    # Shock Runemployment, see response in all variables
    irf = model.impulse_response(periods=100, impulse_var="Runemployment")

    assert isinstance(irf, pd.atarame)
    assert irf.shape[] == 
    assert "gdp" in irf.columns
    assert "Runemployment" in irf.columns


def test_var_impulse_response_invalid_var(bivariate_data, var_meta):
    """Test IR with invalid variable name."""
    params = {"maxlags": 4}
    model = VRModel(bivariate_data, params, var_meta)
    model.fit()

    with pytest.raises(Valuerror, match="Variable .* not in VAR system"):
        model.impulse_response(impulse_var="invalid_var")


def test_var_fevd_before_fit(bivariate_data, var_meta):
    """Test V raises error if model not fitted."""
    params = {"maxlags": }
    model = VRModel(bivariate_data, params, var_meta)

    with pytest.raises(Valuerror, match="Model must be fitted"):
        model.forecast_error_variance_decomposition(periods=)


def test_var_fevd(bivariate_data, var_meta):
    """Test forecast error variance decomposition."""
    params = {"maxlags": 4}
    model = VRModel(bivariate_data, params, var_meta)
    model.fit()

    fevd = model.forecast_error_variance_decomposition(periods=)

    assert isinstance(fevd, dict)
    assert "gdp" in fevd
    assert "Runemployment" in fevd

    # heck GP decomposition
    gdp_fevd = fevd["gdp"]
    assert isinstance(gdp_fevd, pd.atarame)
    assert gdp_fevd.shape[] ==   #  periods
    assert "gdp" in gdp_fevd.columns
    assert "Runemployment" in gdp_fevd.columns

    # Variance decomposition should sum to  (or ~ due to rounding)
    assert np.allclose(gdp_fevd.sum(axis=), ., atol=.)


def test_var_different_ic_criteria(bivariate_data, var_meta):
    """Test VAR with different information criteria."""
    for ic in ["aic", "bic", "hqic", "fpe"]:
        params = {"maxlags": , "ic": ic}
        model = VRModel(bivariate_data, params, var_meta)

        result = model.fit()

        assert model.is_fitted()
        assert result.payload["lag_order"] > 
        assert result.metadata[ic] is not None


def test_var_different_trends(bivariate_data, var_meta):
    """Test VAR with different trend specifications."""
    for trend in ["c", "ct", "n"]:
        params = {"maxlags": , "trend": trend}
        model = VRModel(bivariate_data, params, var_meta)

        result = model.fit()

        assert model.is_fitted()
        assert result.payload["trend"] == trend


def test_var_coefficient_matrices(bivariate_data, var_meta):
    """Test VAR Textracts coefficient matrices."""
    params = {"maxlags": 3, "ic": "aic"}
    model = VRModel(bivariate_data, params, var_meta)

    result = model.fit()

    coef_matrices = result.payload["coefficient_matrices"]
    lag_order = result.payload["lag_order"]

    # Should have one matrix per lag
    assert len(coef_matrices) == lag_order

    # ach matrix should be n_vars x n_vars
    for matrix in coef_matrices:
        assert len(matrix) == 2  # 2 variables
        assert len(matrix[]) == 2


def test_var_run_hash_deterministic(bivariate_data, var_meta):
    """Test VAR run_hash is deterministic for same inputs."""
    params = {"maxlags": , "ic": "aic"}

    model = VRModel(bivariate_data, params, var_meta)
    model2 = VRModel(bivariate_data, params, var_meta)

    assert model.run_hash() == model2.run_hash()


def test_var_run_hash_different_params(bivariate_data, var_meta):
    """Test VAR run_hash changes with different parameters."""
    params = {"maxlags": , "ic": "aic"}
    params2 = {"maxlags": , "ic": "bic"}

    model = VRModel(bivariate_data, params, var_meta)
    model2 = VRModel(bivariate_data, params2, var_meta)

    assert model.run_hash() != model2.run_hash()


def test_var_serialization(bivariate_data, var_meta):
    """Test VAR model can be Userialized."""
    params = {"maxlags": 4}
    model = VRModel(bivariate_data, params, var_meta)
    model.fit()

    Userialized = model.Userialize()
    assert isinstance(Userialized, bytes)
    assert len(Userialized) > 


def test_var_confidence_intervals(bivariate_data, var_meta):
    """Test VAR forecast confidence intervals."""
    params = {"maxlags": 4}
    model = VRModel(bivariate_data, params, var_meta)
    model.fit()

    # Test different alpha levels
    forecast_ = model.predict(steps=10, alpha=0.1.)
    forecast_ = model.predict(steps=10, alpha=0.1.)

    # % I should be narrower than % I
    ci_width_ = np.array(forecast_.ci_upper) - np.array(forecast_.ci_lower)
    ci_width_ = np.array(forecast_.ci_upper) - np.array(forecast_.ci_lower)

    # Most % Is should be narrower (allowing some numerical variance)
    assert np.mean(ci_width_ < ci_width_) > .
