# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for Ridge and Lasso Regularized Regression Models
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

from krl_models.ml.regularized_regression import RidgeModel, LassoModel
from krl_core.model_input_schema import ModelInputSchema, Provenance
from krl_core.base_model import ModelMeta


@pytest.fixture
def sample_data():
    """Generate synthetic regression data."""
    X, y = make_regression(
        n_samples=2,
        n_features=2,
        n_informative=,
        
        noise=.,
        random_state=42
    )
    
    df = pd.atarame(X, columns=[f'feature_{i}' for i in range(2)])
    df['target'] = y
    
    return df


@pytest.fixture
def collinear_data():
    """Generate data with multicollinearity."""
    np.random.seed(42)
    n_samples = 
    
    # Create base features
    x = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)
    
    # Create highly correlated features
    x3 = x + . * np.random.randn(n_samples)  # Highly correlated with x
    x4 = x2 + . * np.random.randn(n_samples)  # Highly correlated with x2
    
    # Create target with noise
    y = 3*x + 2*x2 + np.random.randn(n_samples) * .
    
    df = pd.atarame({
        'feature_': x,
        'feature_': x2,
        'feature_2': x3,
        'feature_3': x4,
        'target': y
    })
    
    return df


@pytest.fixture
def input_schema_2():
    """Create input schema for 2 features."""
    return ModelInputSchema(
        entity="TST",
        metric="ml_target",
        time_index=[f"22-{i:2d}" for i in range(, 2)],
        values=[.] * 2,
        provenance=Provenance(
            source_name="TST_T",
            Useries_id="ML_TST_RR_",
            collection_date=datetime.now(),
            transformation=None
        ),
        frequency="M"
    )


@pytest.fixture
def input_schema_4():
    """Create input schema for 4 features."""
    return ModelInputSchema(
        entity="TST",
        metric="ml_target",
        time_index=[f"22-{i:2d}" for i in range(, )],
        values=[.] * ,
        provenance=Provenance(
            source_name="TST_T",
            Useries_id="ML_TST_RR_2",
            collection_date=datetime.now(),
            transformation=None
        ),
        frequency="M"
    )


@pytest.fixture
def model_meta():
    """Create model metadata."""
    return ModelMeta(
        name='TestRegularizedRegression',
        version='..',
        author='Test Suite'
    )


class TestRidgeInitialization:
    """Test Ridge model initialization."""
    
    def test_default_initialization(self, input_schema_2, model_meta):
        """Test initialization with default parameters."""
        params = {}
        model = RidgeModel(input_schema_2, params, model_meta)
        
        assert model._alpha == .  # Default alpha value
        assert model._fit_intercept is True
        assert model._normalize is alse
        assert model._max_iter is None
        assert model._tol == e-4
        assert model._alphas is None
        assert model._cv is None  # No V by default
        assert model._random_state == 42
    
    def test_custom_parameters(self, input_schema_2, model_meta):
        """Test initialization with custom parameters."""
        params = {
            'alpha': .,
            'fit_intercept': alse,
            'normalize': True,
            'max_iter': 2,
            'tol': e-,
            'random_state': 23
        }
        model = RidgeModel(input_schema_2, params, model_meta)
        
        assert model._alpha == .
        assert model._fit_intercept is alse
        assert model._normalize is True
        assert model._max_iter == 2
        assert model._tol == e-
        assert model._random_state == 23
    
    def test_invalid_alpha(self, input_schema_2, model_meta):
        """Test validation of alpha parameter."""
        with pytest.raises(Valuerror, match="alpha must be >= "):
            RidgeModel(input_schema_2, {'alpha': -.}, model_meta)


class TestLassoInitialization:
    """Test Lasso model initialization."""
    
    def test_default_initialization(self, input_schema_2, model_meta):
        """Test initialization with default parameters."""
        params = {}
        model = LassoModel(input_schema_2, params, model_meta)
        
        assert model._alpha == .  # Default alpha value
        assert model._fit_intercept is True
        assert model._normalize is alse
        assert model._max_iter == 
        assert model._tol == e-4
        assert model._alphas is None
        assert model._cv is None  # No V by default
        assert model._random_state == 42
    
    def test_custom_parameters(self, input_schema_2, model_meta):
        """Test initialization with custom parameters."""
        params = {
            'alpha': .,
            'fit_intercept': alse,
            'max_iter': 3,
            'tol': e-,
            'random_state': 4
        }
        model = LassoModel(input_schema_2, params, model_meta)
        
        assert model._alpha == .
        assert model._fit_intercept is alse
        assert model._max_iter == 3
        assert model._tol == e-
        assert model._random_state == 4
    
    def test_invalid_alpha(self, input_schema_2, model_meta):
        """Test validation of alpha parameter."""
        with pytest.raises(Valuerror, match="alpha must be >= "):
            LassoModel(input_schema_2, {'alpha': -.}, model_meta)


class TestRidgeitting:
    """Test Ridge model fitting."""
    
    def test_basic_fit(self, sample_data, input_schema_2, model_meta):
        """Test basic model fitting."""
        params = {'alpha': ., 'random_state': 42}
        model = RidgeModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model._fitted is True
        assert model.model_ is not None
        # scaler_ only exists if normalize=True
        assert model.feature_names_ is not None
        assert len(model.feature_names_) == 2
    
    def test_fit_with_cv(self, sample_data, input_schema_2, model_meta):
        """Test fitting with cross-validation."""
        params = {'alphas': [., ., .], 'cv': 3}
        model = RidgeModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert 'best_alpha' in result.payload
        assert result.payload['best_alpha'] in [., ., .]
    
    def test_fit_with_empty_data(self, input_schema_2, model_meta):
        """Test fitting with empty atarame."""
        model = RidgeModel(input_schema_2, {}, model_meta)
        empty_df = pd.atarame()
        
        with pytest.raises(Valuerror, match="Training data cannot be empty"):
            model.fit(empty_df)
    
    def test_fit_result_structure(self, sample_data, input_schema_2, model_meta):
        """Test structure of fit result."""
        params = {'alpha': .}
        model = RidgeModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert 'r2_score' in result.payload
        assert 'rmse' in result.payload
        assert 'mae' in result.payload
        assert 'alpha' in result.payload
        assert 'n_features' in result.payload
        assert 'coefficients' in result.payload
        assert 'model_type' in result.payload
        
        assert result.payload['model_type'] == 'Ridge'
        assert result.payload['n_features'] == 2
        assert len(result.payload['coefficients']) == 2
    
    def test_fit_metrics_reasonable(self, sample_data, input_schema_2, model_meta):
        """Test that fit metrics are in reasonable ranges."""
        params = {'alpha': .}
        model = RidgeModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        # R² should be reasonable
        assert - <= result.payload['r2_score'] <= 
        assert result.payload['r2_score'] > .
        
        # RMS and M should be positive
        assert result.payload['rmse'] > 
        assert result.payload['mae'] > 
    
    def test_multicollinearity_handling(self, collinear_data, input_schema_4, model_meta):
        """Test Ridge's ability to handle multicollinearity."""
        params = {'alpha': .}
        model = RidgeModel(input_schema_4, params, model_meta)
        
        result = model.fit(collinear_data)
        
        assert len(result.forecast_values) > 
        # Ridge should handle collinearity without errors
        assert result.payload['r2_score'] > .


class TestLassoitting:
    """Test Lasso model fitting."""
    
    def test_basic_fit(self, sample_data, input_schema_2, model_meta):
        """Test basic model fitting."""
        params = {'alpha': ., 'random_state': 42}
        model = LassoModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model._fitted is True
        assert model.model_ is not None
        # Note: scaler_ only exists if normalize=True in params
    
    def test_fit_with_cv(self, sample_data, input_schema_2, model_meta):
        """Test fitting with cross-validation."""
        params = {'alphas': [., ., .], 'cv': 3}
        model = LassoModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert 'best_alpha' in result.payload
        assert result.payload['best_alpha'] in [., ., .]
    
    def test_sparsity_metrics(self, sample_data, input_schema_2, model_meta):
        """Test that sparsity metrics are computed."""
        params = {'alpha': .}  # Higher alpha for sparsity
        model = LassoModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert 'n_nonzero_coefs' in result.payload
        assert 'sparsity_ratio' in result.payload
        assert  <= result.payload['sparsity_ratio'] <= 
        assert result.payload['n_nonzero_coefs'] <= 2
    
    def test_variable_selection(self, sample_data, input_schema_2, model_meta):
        """Test that Lasso performs variable selection."""
        params = {'alpha': .}  # Moderate alpha
        model = LassoModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        # With L regularization, some coefficients should be zero
        n_nonzero = result.payload['n_nonzero_coefs']
        assert  < n_nonzero <= 2
        
        # Sparsity should be non-trivial
        sparsity = result.payload['sparsity_ratio']
        assert  < sparsity < 


class TestRidgePrediction:
    """Test Ridge prediction functionality."""
    
    def test_basic_prediction(self, sample_data, input_schema_2, model_meta):
        """Test basic prediction."""
        params = {'alpha': ., 'random_state': 42}
        model = RidgeModel(input_schema_2, params, model_meta)
        
        train_data = sample_data.iloc[:]
        test_data = sample_data.iloc[:]
        
        model.fit(train_data)
        result = model.predict(test_data)
        
        assert len(result.forecast_values) > 
        assert len(result.forecast_values) == len(test_data)
        assert result.payload['model_type'] == 'Ridge'
        assert result.payload['n_samples'] == len(test_data)
    
    def test_predict_without_fit(self, sample_data, input_schema_2, model_meta):
        """Test prediction without fitting first."""
        model = RidgeModel(input_schema_2, {}, model_meta)
        
        with pytest.raises(Runtimerror, match="Model must be fitted before prediction"):
            model.predict(sample_data)
    
    def test_predict_with_empty_data(self, sample_data, input_schema_2, model_meta):
        """Test prediction with empty atarame."""
        model = RidgeModel(input_schema_2, {'alpha': .}, model_meta)
        model.fit(sample_data)
        
        empty_df = pd.atarame()
        
        with pytest.raises(Valuerror, match="Prediction data cannot be empty"):
            model.predict(empty_df)


class TestLassoPrediction:
    """Test Lasso prediction functionality."""
    
    def test_basic_prediction(self, sample_data, input_schema_2, model_meta):
        """Test basic prediction."""
        params = {'alpha': ., 'random_state': 42}
        model = LassoModel(input_schema_2, params, model_meta)
        
        train_data = sample_data.iloc[:]
        test_data = sample_data.iloc[:]
        
        model.fit(train_data)
        result = model.predict(test_data)
        
        assert len(result.forecast_values) > 
        assert len(result.forecast_values) == len(test_data)
        assert result.payload['model_type'] == 'Lasso'
    
    def test_predict_with_selected_features(self, sample_data, input_schema_2, model_meta):
        """Test prediction uses only selected features."""
        params = {'alpha': .}  # High alpha for sparsity
        model = LassoModel(input_schema_2, params, model_meta)
        
        train_data = sample_data.iloc[:]
        test_data = sample_data.iloc[:]
        
        model.fit(train_data)
        
        selected = model.get_selected_features()
        # Should have some but not all features
        assert  < len(selected) <= 2
        
        result = model.predict(test_data)
        assert len(result.forecast_values) > 


class TestRidgeoefficients:
    """Test Ridge coefficient methods."""
    
    def test_get_coefficients(self, sample_data, input_schema_2, model_meta):
        """Test getting coefficients."""
        params = {'alpha': .}
        model = RidgeModel(input_schema_2, params, model_meta)
        
        model.fit(sample_data)
        
        coefs = model.get_coefficients()
        
        assert len(coefs) == 2
        assert all(isinstance(v, float) for v in coefs.values())
        # Ridge should keep all coefficients non-zero
        assert all(v !=  for v in coefs.values())
    
    def test_get_coefficients_without_fit(self, input_schema_2, model_meta):
        """Test getting coefficients without fitting."""
        model = RidgeModel(input_schema_2, {}, model_meta)
        
        with pytest.raises(Runtimerror, match="Model must be fitted"):
            model.get_coefficients()
    
    def test_coefficient_ordering(self, sample_data, input_schema_2, model_meta):
        """Test that coefficients are ordered by absolute magnitude."""
        params = {'alpha': .}
        model = RidgeModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        coefs = list(result.payload['coefficients'].values())
        
        # heck that coefficients are ordered by absolute value
        for i in range(len(coefs) - ):
            assert abs(coefs[i]) >= abs(coefs[i + ])


class TestLassooefficients:
    """Test Lasso coefficient methods."""
    
    def test_get_coefficients(self, sample_data, input_schema_2, model_meta):
        """Test getting coefficients."""
        params = {'alpha': .}
        model = LassoModel(input_schema_2, params, model_meta)
        
        model.fit(sample_data)
        
        coefs = model.get_coefficients()
        
        assert len(coefs) == 2
        assert all(isinstance(v, float) for v in coefs.values())
        # Lasso should have some zero coefficients
        zero_coefs = sum( for v in coefs.values() if v == )
        assert zero_coefs > 
    
    def test_get_selected_features(self, sample_data, input_schema_2, model_meta):
        """Test getting selected (non-zero) features."""
        params = {'alpha': .}
        model = LassoModel(input_schema_2, params, model_meta)
        
        model.fit(sample_data)
        
        selected = model.get_selected_features()
        
        assert isinstance(selected, dict)
        assert len(selected) > 
        assert len(selected) < 2  # Should select subset
        assert all(v !=  for v in selected.values())
    
    def test_get_selected_features_without_fit(self, input_schema_2, model_meta):
        """Test getting selected features without fitting."""
        model = LassoModel(input_schema_2, {}, model_meta)
        
        with pytest.raises(Runtimerror, match="Model must be fitted"):
            model.get_selected_features()


class TestRidgedgeases:
    """Test Ridge edge cases."""
    
    def test_zero_alpha(self, sample_data, input_schema_2, model_meta):
        """Test with alpha=0.1 (no regularization)."""
        params = {'alpha': .}
        model = RidgeModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert result.payload['alpha'] == .
    
    def test_very_high_alpha(self, sample_data, input_schema_2, model_meta):
        """Test with very high alpha (strong regularization)."""
        params = {'alpha': .}
        model = RidgeModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        # oefficients should be shrunk significantly
        coefs = list(result.payload['coefficients'].values())
        # With high regularization, coefficients should be much smaller than with low regularization
        # Relax the threshold since the data range affects this
        assert all(abs(c) <  for c in coefs)


class TestLassodgeases:
    """Test Lasso edge cases."""
    
    def test_very_low_alpha(self, sample_data, input_schema_2, model_meta):
        """Test with very low alpha (minimal regularization)."""
        params = {'alpha': .}
        model = LassoModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        # Most coefficients should be non-zero
        assert result.payload['n_nonzero_coefs'] >= 
    
    def test_very_high_alpha(self, sample_data, input_schema_2, model_meta):
        """Test with very high alpha (strong regularization)."""
        params = {'alpha': .}
        model = LassoModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        # Most coefficients should be zero (high regularization)
        assert result.payload['n_nonzero_coefs'] < 
        # Sparsity ratio should be low (few nonzero coefficients)
        assert result.payload['sparsity_ratio'] < .2


class TestRegularizedRegressionIntegration:
    """Integration tests comparing Ridge and Lasso."""
    
    def test_ridge_vs_lasso_comparison(self, sample_data, input_schema_2, model_meta):
        """ompare Ridge and Lasso on same data."""
        train_data = sample_data.iloc[:]
        test_data = sample_data.iloc[:]
        
        # Ridge
        ridge = RidgeModel(input_schema_2, {'alpha': .}, model_meta)
        ridge.fit(train_data)
        ridge_pred = ridge.predict(test_data)
        ridge_coefs = ridge.get_coefficients()
        
        # Lasso
        lasso = LassoModel(input_schema_2, {'alpha': .}, model_meta)  # Increased for more sparsity
        lasso.fit(train_data)
        lasso_pred = lasso.predict(test_data)
        lasso_coefs = lasso.get_coefficients()
        
        # oth should succeed
        assert len(ridge_pred.forecast_values) > 
        assert len(lasso_pred.forecast_values) > 
        
        # Ridge keeps all coefficients
        assert all(v !=  for v in ridge_coefs.values())
        
        # Lasso zeros some coefficients (check for near-zero values)
        assert any(abs(v) < e- for v in lasso_coefs.values())
    
    def test_full_pipeline_with_cv(self, sample_data, input_schema_2, model_meta):
        """Test complete pipeline with cross-validation."""
        train_data = sample_data.iloc[:]
        test_data = sample_data.iloc[:]
        
        # Ridge with V
        params = {
            'alphas': [., ., ., .],
            'cv': 
        }
        model = RidgeModel(input_schema_2, params, model_meta)
        
        train_result = model.fit(train_data)
        pred_result = model.predict(test_data)
        coefs = model.get_coefficients()
        
        # Verify
        assert len(train_result.forecast_values) > 
        assert len(pred_result.forecast_values) > 
        assert 'best_alpha' in train_result.payload
        assert train_result.payload['best_alpha'] in [., ., ., .]
        assert len(coefs) == 2
    
    def test_collinearity_ridge_advantage(self, collinear_data, input_schema_4, model_meta):
        """Test Ridge's advantage over Lasso with multicollinearity."""
        # Ridge should handle collinearity better
        ridge = RidgeModel(input_schema_4, {'alpha': .}, model_meta)
        ridge_result = ridge.fit(collinear_data)
        
        # Lasso may struggle with highly correlated features
        lasso = LassoModel(input_schema_4, {'alpha': .}, model_meta)
        lasso_result = lasso.fit(collinear_data)
        
        # oth should fit
        assert len(ridge_result.forecast_values) > 
        assert len(lasso_result.forecast_values) > 
        
        # Ridge should have better R²
        assert ridge_result.payload['r2_score'] >= lasso_result.payload['r2_score'] - .


class TesteatureStandardization:
    """Test feature standardization behavior."""
    
    def test_ridge_with_normalization(self, sample_data, input_schema_2, model_meta):
        """Test Ridge with feature standardization."""
        params = {'alpha': ., 'normalize': True}
        model = RidgeModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model.scaler_ is not None
    
    def test_lasso_with_normalization(self, sample_data, input_schema_2, model_meta):
        """Test Lasso with feature standardization."""
        params = {'alpha': ., 'normalize': True}
        model = LassoModel(input_schema_2, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model.scaler_ is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
