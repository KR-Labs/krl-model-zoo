"""
omprehensive unit tests for GRH model.

Tests cover:
- Parameter validation
- Model initialization
- itting behavior
- Prediction accuracy
- dge cases
- Numerical stability
"""

import numpy as np
import pandas as pd
import pytest
from krl_models.volatility.garch_model import GRHModel


class TestGRHInitialization:
    """Test GRH model initialization and validation."""
    
    def test_default_initialization(self):
        """Test GRH model with default parameters."""
        model = GRHModel(p=, q=)
        assert model.p == 
        assert model.q == 
        assert model.params is None
        
    def test_invalid_p_parameter(self):
        """Test that invalid p parameter raises error."""
        with pytest.raises(Valuerror, match="p must be a positive integer"):
            GRHModel(p=, q=)
        with pytest.raises(Valuerror, match="p must be a positive integer"):
            GRHModel(p=-, q=)
    
    def test_invalid_q_parameter(self):
        """Test that invalid q parameter raises error."""
        with pytest.raises(Valuerror, match="q must be a positive integer"):
            GRHModel(p=, q=)
        with pytest.raises(Valuerror, match="q must be a positive integer"):
            GRHModel(p=, q=-)
    
    def test_high_order_model(self):
        """Test initialization with high order GRH(3,3)."""
        model = GRHModel(p=3, q=3)
        assert model.p == 3
        assert model.q == 3


class TestGRHitting:
    """Test GRH model fitting behavior."""
    
    @pytest.fixture
    def simple_returns(self):
        """Generate simple return series for testing."""
        np.random.seed(42)
        T = 
        returns = np.random.normal(, , T)
        dates = pd.date_range('22--', periods=T, freq='')
        return pd.atarame({'returns': returns}, index=dates)
    
    @pytest.fixture
    def volatile_returns(self):
        """Generate returns with time-varying volatility."""
        np.random.seed(23)
        T = 
        omega = .
        alpha = .
        beta = .
        
        sigma2 = np.zeros(T)
        returns = np.zeros(T)
        sigma2[] = omega / ( - alpha - beta)
        returns[] = np.sqrt(sigma2[]) * np.random.normal()
        
        for t in range(, T):
            sigma2[t] = omega + alpha * returns[t-]**2 + beta * sigma2[t-]
            returns[t] = np.sqrt(sigma2[t]) * np.random.normal()
        
        dates = pd.date_range('22--', periods=T, freq='')
        return pd.atarame({'returns': returns}, index=dates)
    
    def test_fit_returns_result(self, simple_returns):
        """Test that fit() returns a orecastResult."""
        model = GRHModel(p=, q=)
        result = model.fit(simple_returns)
        
        assert result is not None
        assert hasattr(result, 'payload')
        assert hasattr(result, 'metadata')
    
    def test_parameters_estimated(self, simple_returns):
        """Test that parameters are estimated after fitting."""
        model = GRHModel(p=, q=)
        result = model.fit(simple_returns)
        
        assert model.params is not None
        assert 'omega' in model.params
        assert 'alpha' in model.params
        assert 'beta' in model.params
    
    def test_positive_parameters(self, volatile_returns):
        """Test that estimated parameters are positive."""
        model = GRHModel(p=, q=)
        result = model.fit(volatile_returns)
        
        assert model.params['omega'] > 
        assert all(model.params['alpha'] > )
        assert all(model.params['beta'] > )
    
    def test_stationarity_constraint(self, volatile_returns):
        """Test that alpha + beta <  (stationarity)."""
        model = GRHModel(p=, q=)
        result = model.fit(volatile_returns)
        
        alpha_sum = np.sum(model.params['alpha'])
        beta_sum = np.sum(model.params['beta'])
        assert alpha_sum + beta_sum < .
    
    def test_volatility_computed(self, simple_returns):
        """Test that conditional volatility is computed."""
        model = GRHModel(p=, q=)
        result = model.fit(simple_returns)
        
        assert 'volatility' in result.payload
        volatility = result.payload['volatility']
        assert len(volatility) == len(simple_returns)
        assert all(volatility > )
    
    def test_residuals_computed(self, simple_returns):
        """Test that standardized residuals are computed."""
        model = GRHModel(p=, q=)
        result = model.fit(simple_returns)
        
        assert 'residuals' in result.payload
        residuals = result.payload['residuals']
        assert len(residuals) == len(simple_returns)
    
    def test_fit_with_different_orders(self):
        """Test fitting models with different (p,q) orders."""
        np.random.seed(42)
        T = 3
        returns = pd.atarame({
            'returns': np.random.normal(, , T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        # Test GRH(,)
        model_ = GRHModel(p=, q=)
        result_ = model_.fit(returns)
        assert len(model_.params['alpha']) == 
        assert len(model_.params['beta']) == 
        
        # Test GRH(2,)
        model_2 = GRHModel(p=2, q=)
        result_2 = model_2.fit(returns)
        assert len(model_2.params['alpha']) == 2
        assert len(model_2.params['beta']) == 
        
        # Test GRH(,2)
        model_2 = GRHModel(p=, q=2)
        result_2 = model_2.fit(returns)
        assert len(model_2.params['alpha']) == 
        assert len(model_2.params['beta']) == 2


class TestGRHPrediction:
    """Test GRH model prediction functionality."""
    
    @pytest.fixture
    def fitted_model(self):
        """reate and fit a GRH model."""
        np.random.seed(42)
        T = 3
        returns = pd.atarame({
            'returns': np.random.normal(, , T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        model.fit(returns)
        return model
    
    def test_predict_returns_array(self, fitted_model):
        """Test that predict() returns an array."""
        forecast = fitted_model.predict(steps=)
        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 
    
    def test_predict_positive_volatility(self, fitted_model):
        """Test that predicted volatility is positive."""
        forecast = fitted_model.predict(steps=2)
        assert all(forecast > )
    
    def test_predict_different_horizons(self, fitted_model):
        """Test prediction for different forecast horizons."""
        forecast_ = fitted_model.predict(steps=)
        forecast_ = fitted_model.predict(steps=)
        forecast_2 = fitted_model.predict(steps=2)
        
        assert len(forecast_) == 
        assert len(forecast_) == 
        assert len(forecast_2) == 2
    
    def test_predict_mean_reversion(self, fitted_model):
        """Test that long-run forecast converges to unconditional variance."""
        forecast = fitted_model.predict(steps=)
        
        # Long-run variance should converge
        params = fitted_model.params
        omega = params['omega']
        alpha_sum = np.sum(params['alpha'])
        beta_sum = np.sum(params['beta'])
        long_run_var = omega / ( - alpha_sum - beta_sum)
        
        # Last forecast should be close to long-run variance
        assert np.abs(forecast[-]**2 - long_run_var) / long_run_var < .


class TestGRHdgeases:
    """Test GRH model edge cases and boundary conditions."""
    
    def test_short_time_series(self):
        """Test fitting with short time series."""
        np.random.seed(42)
        T =   # Short series
        returns = pd.atarame({
            'returns': np.random.normal(, , T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        result = model.fit(returns)
        
        # Should still fit, but might have higher uncertainty
        assert model.params is not None
        assert 'omega' in model.params
    
    def test_constant_returns(self):
        """Test with constant (zero variance) returns."""
        T = 
        returns = pd.atarame({
            'returns': np.zeros(T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        # Should handle gracefully (might fail or give minimal variance)
        model = GRHModel(p=, q=)
        try:
            result = model.fit(returns)
            # If it succeeds, volatility should be very small
            if 'volatility' in result.payload:
                assert np.all(result.payload['volatility'] < e-3)
        except (Valuerror, np.linalg.Linlgrror):
            # xpected: might fail with singular matrix
            pass
    
    def test_extreme_volatility(self):
        """Test with extremely high volatility."""
        np.random.seed(42)
        T = 2
        returns = pd.atarame({
            'returns': np.random.normal(, , T)  # Very high volatility
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        result = model.fit(returns)
        
        # Should fit successfully
        assert model.params is not None
        assert 'volatility' in result.payload
    
    def test_missing_values_handling(self):
        """Test behavior with missing values."""
        np.random.seed(42)
        T = 2
        returns_data = np.random.normal(, , T)
        returns_data[:] = np.nan  # Insert missing values
        
        returns = pd.atarame({
            'returns': returns_data
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        
        # Should either handle missing values or raise informative error
        try:
            result = model.fit(returns)
        except (Valuerror, Keyrror) as e:
            # xpected: might fail with missing values
            assert 'NaN' in str(e) or 'missing' in str(e).lower()


class TestGRHNumericalStability:
    """Test numerical stability of GRH model."""
    
    def test_very_small_variance(self):
        """Test with very small variance."""
        np.random.seed(42)
        T = 2
        returns = pd.atarame({
            'returns': np.random.normal(, ., T)  # Very small variance
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        result = model.fit(returns)
        
        # Should handle small numbers without overflow
        assert model.params is not None
        assert np.all(np.isfinite(model.params['omega']))
    
    def test_very_large_variance(self):
        """Test with very large variance."""
        np.random.seed(42)
        T = 2
        returns = pd.atarame({
            'returns': np.random.normal(, , T)  # Very large variance
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        result = model.fit(returns)
        
        # Should handle large numbers without overflow
        assert model.params is not None
        assert np.all(np.isfinite(model.params['omega']))
    
    def test_convergence_with_good_data(self):
        """Test that optimization converges with well-behaved data."""
        np.random.seed(42)
        T = 
        # Generate data from known GRH process
        omega, alpha, beta = ., ., .
        sigma2 = np.zeros(T)
        returns = np.zeros(T)
        sigma2[] = omega / ( - alpha - beta)
        
        for t in range(, T):
            sigma2[t] = omega + alpha * returns[t-]**2 + beta * sigma2[t-]
            returns[t] = np.sqrt(sigma2[t]) * np.random.normal()
        
        data = pd.atarame({
            'returns': returns
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        result = model.fit(data)
        
        # Should converge and recover parameters approximately
        assert model.params is not None
        assert abs(model.params['omega'] - omega) < .  # Reasonable tolerance
        assert abs(model.params['alpha'][] - alpha) < .3
        assert abs(model.params['beta'][] - beta) < .3


class TestGRHiagnostics:
    """Test GRH model diagnostic outputs."""
    
    @pytest.fixture
    def fitted_model_with_result(self):
        """reate fitted model and result."""
        np.random.seed(42)
        T = 3
        returns = pd.atarame({
            'returns': np.random.normal(, , T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        result = model.fit(returns)
        return model, result
    
    def test_residuals_standardized(self, fitted_model_with_result):
        """Test that residuals are approximately standardized."""
        model, result = fitted_model_with_result
        residuals = result.payload['residuals']
        
        # Should have approximately mean  and variance 
        assert abs(np.mean(residuals)) < .2
        assert abs(np.std(residuals) - .) < .3
    
    def test_log_likelihood_computed(self, fitted_model_with_result):
        """Test that log-likelihood is computed."""
        model, result = fitted_model_with_result
        
        if 'log_likelihood' in result.payload:
            log_lik = result.payload['log_likelihood']
            assert isinstance(log_lik, (int, float))
            assert np.isfinite(log_lik)
    
    def test_aic_bic_computed(self, fitted_model_with_result):
        """Test that I/I are computed if available."""
        model, result = fitted_model_with_result
        
        # These might be in payload or metadata
        payload_or_meta = {**result.payload, **result.metadata}
        
        if 'I' in payload_or_meta:
            assert isinstance(payload_or_meta['I'], (int, float))
            assert np.isfinite(payload_or_meta['I'])
        
        if 'I' in payload_or_meta:
            assert isinstance(payload_or_meta['I'], (int, float))
            assert np.isfinite(payload_or_meta['I'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
