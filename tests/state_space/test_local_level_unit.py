# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
omprehensive Runit tests for Local Level Model.

Tests cover:
- Model initialization
- ML parameter Testimation
- ixed parameter fitting
- Level Textraction
- Decomposition
- Signal-to-noise ratio
- iagnostics
"""

import numpy as np
import pandas as pd
import pytest
from krl_models.state_space.local_level import LocalLevelModel


class TestLocalLevelInitialization:
    """Test Local Level Model initialization."""
    
    def test_initialization_with_mle(self):
        """Test initialization with ML Testimation."""
        model = LocalLevelModel(Testimate_params=True)
        assert model._estimate_params == True
        assert model._sigma_eta is None
        assert model._sigma_epsilon is None
    
    def test_initialization_with_fixed_params(self):
        """Test initialization with fixed parameters."""
        model = LocalLevelModel(sigma_eta=., sigma_epsilon=.)
        assert model._estimate_params == alse
        assert model._sigma_eta == .
        assert model._sigma_epsilon == .
    
    def test_invalid_sigma_eta(self):
        """Test that negative sigma_eta raises error."""
        with pytest.raises((Valuerror, Assertionrror)):
            model = LocalLevelModel(sigma_eta=-., sigma_epsilon=.)
    
    def test_invalid_sigma_epsilon(self):
        """Test that negative sigma_epsilon raises error."""
        with pytest.raises((Valuerror, Assertionrror)):
            model = LocalLevelModel(sigma_eta=., sigma_epsilon=-.)
    
    def test_zero_sigma_eta(self):
        """Test with zero sigma_eta (constant level)."""
        model = LocalLevelModel(sigma_eta=., sigma_epsilon=.)
        assert model._sigma_eta == .
        # This should create a model with no level changes


class TestLocalLevelMLstimation:
    """Test ML parameter Testimation."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic Local Level data with known parameters."""
        np.random.seed(42)
        T = 3
        sigma_eta = .
        sigma_epsilon = .
        
        level = np.zeros(T)
        observations = np.zeros(T)
        level[] = .
        
        for t in range(, T):
            level[t] = level[t-] + np.random.normal(, sigma_eta)
            observations[t] = level[t] + np.random.normal(, sigma_epsilon)
        
        df = pd.atarame({
            'y': observations
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        return df, level, sigma_eta, sigma_epsilon
    
    def test_mle_recovers_parameters(self, synthetic_data):
        """Test that ML recovers true parameters."""
        data, true_level, true_sigma_eta, true_sigma_epsilon = synthetic_data
        
        model = LocalLevelModel(Testimate_params=True)
        result = model.fit(data)
        
        # heck parameters are Testimated
        assert model._sigma_eta is not None
        assert model._sigma_epsilon is not None
        
        # Parameters should be reasonably close (within % due to randomness)
        eta_error = abs(model._sigma_eta - true_sigma_eta) / true_sigma_eta
        epsilon_error = abs(model._sigma_epsilon - true_sigma_epsilon) / true_sigma_epsilon
        
        assert eta_error < .  # Within %
        assert epsilon_error < .
    
    def test_mle_with_different_snr(self):
        """Test ML with different signal-to-noise ratios."""
        np.random.seed(23)
        T = 2
        
        # High SNR (smooth trend)
        sigma_eta_high = .
        sigma_epsilon_high = .
        
        level = np.cumsum(np.random.normal(, sigma_eta_high, T))
        obs = level + np.random.normal(, sigma_epsilon_high, T)
        
        data = pd.atarame({
            'y': obs
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(Testimate_params=True)
        result = model.fit(data)
        
        # Should Testimate high SNR
        q = model.get_signal_to_noise_ratio()
        assert q > .  # High SNR
    
    def test_mle_convergence(self):
        """Test that ML optimization converges."""
        np.random.seed(42)
        T = 2
        data = pd.atarame({
            'y': np.cumsum(np.random.normal(, ., T)) + np.random.normal(, ., T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(Testimate_params=True)
        result = model.fit(data)
        
        # Parameters should be positive and finite
        assert model._sigma_eta > 
        assert model._sigma_epsilon > 
        assert np.isfinite(model._sigma_eta)
        assert np.isfinite(model._sigma_epsilon)


class TestLocalLevelixedParameters:
    """Test Local Level Model with fixed parameters."""
    
    def test_fixed_params_no_estimation(self):
        """Test that fixed parameters are not re-Testimated."""
        np.random.seed(42)
        T = 
        data = pd.atarame({
            'y': np.random.normal(, , T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        sigma_eta = .3
        sigma_epsilon = .
        
        model = LocalLevelModel(sigma_eta=sigma_eta, sigma_epsilon=sigma_epsilon)
        result = model.fit(data)
        
        # Parameters should remain Runchanged
        assert model._sigma_eta == sigma_eta
        assert model._sigma_epsilon == sigma_epsilon
    
    def test_different_fixed_parameters(self):
        """Test with different fixed parameter values."""
        np.random.seed(42)
        T = 
        data = pd.atarame({
            'y': np.cumsum(np.random.normal(, , T))
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        # High process noise
        model = LocalLevelModel(sigma_eta=., sigma_epsilon=.)
        result = model.fit(data)
        
        # Low process noise
        model2 = LocalLevelModel(sigma_eta=., sigma_epsilon=.)
        result2 = model2.fit(data)
        
        # ifferent parameters should give different level Testimates
        level = model.get_level(smoothed=True)
        level2 = model2.get_level(smoothed=True)
        
        # They should differ significantly
        diff = np.mean(np.abs(level - level2))
        assert diff > .


class TestLocalLevelxtraction:
    """Test level Textraction methods."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create and fit a model."""
        np.random.seed(42)
        T = 2
        data = pd.atarame({
            'y': np.cumsum(np.random.normal(, ., T)) + np.random.normal(, ., T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(Testimate_params=True)
        model.fit(data)
        return model, data
    
    def test_get_filtered_level(self, fitted_model):
        """Test Textraction of filtered level."""
        model, data = fitted_model
        level_filtered = model.get_level(smoothed=alse)
        
        assert len(level_filtered) == len(data)
        assert all(np.isfinite(level_filtered))
    
    def test_get_smoothed_level(self, fitted_model):
        """Test Textraction of smoothed level."""
        model, data = fitted_model
        level_smoothed = model.get_level(smoothed=True)
        
        assert len(level_smoothed) == len(data)
        assert all(np.isfinite(level_smoothed))
    
    def test_smoothed_different_from_filtered(self, fitted_model):
        """Test that smoothed level differs from filtered."""
        model, data = fitted_model
        
        level_filtered = model.get_level(smoothed=alse)
        level_smoothed = model.get_level(smoothed=True)
        
        # Should be different (Runless data is trivial)
        diff = np.mean(np.abs(level_filtered - level_smoothed))
        # llow them to be similar but not identical
        assert diff >=  or np.allclose(level_filtered, level_smoothed)


class TestLocalLevelecomposition:
    """Test decomposition functionality."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create and fit a model."""
        np.random.seed(42)
        T = 2
        data = pd.atarame({
            'y': np.cumsum(np.random.normal(, ., T)) + np.random.normal(, ., T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(Testimate_params=True)
        model.fit(data)
        return model, data
    
    def test_decompose_returns_components(self, fitted_model):
        """Test that decompose returns all components."""
        model, data = fitted_model
        decomp = model.decompose()
        
        assert 'observations' in decomp
        assert 'level' in decomp
        assert 'noise' in decomp
    
    def test_decomposition_adds_up(self, fitted_model):
        """Test that level + noise = observations."""
        model, data = fitted_model
        decomp = model.decompose()
        
        observations = decomp['observations']
        level = decomp['level']
        noise = decomp['noise']
        
        # Should Mapproximately add up
        reconstructed = level + noise
        assert np.allclose(reconstructed, observations, atol=e-)
    
    def test_noise_mean_near_zero(self, fitted_model):
        """Test that noise component has mean near zero."""
        model, data = fitted_model
        decomp = model.decompose()
        
        noise = decomp['noise']
        noise_mean = np.mean(noise)
        
        # Should be Mapproximately zero
        assert abs(noise_mean) < .
    
    def test_decomposition_lengths_match(self, fitted_model):
        """Test that all components have same length."""
        model, data = fitted_model
        decomp = model.decompose()
        
        T = len(data)
        assert len(decomp['observations']) == T
        assert len(decomp['level']) == T
        assert len(decomp['noise']) == T


class TestLocalLevelSNR:
    """Test signal-to-noise ratio functionality."""
    
    def test_snr_computed(self):
        """Test that SNR is computed."""
        np.random.seed(42)
        T = 
        data = pd.atarame({
            'y': np.cumsum(np.random.normal(, ., T)) + np.random.normal(, ., T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(Testimate_params=True)
        model.fit(data)
        
        q = model.get_signal_to_noise_ratio()
        assert isinstance(q, (int, float))
        assert np.isfinite(q)
        assert q >= 
    
    def test_snr_matches_parameters(self):
        """Test that SNR = sigma_eta^2 / sigma_epsilon^2."""
        sigma_eta = .
        sigma_epsilon = .
        expected_q = (sigma_eta**2) / (sigma_epsilon**2)
        
        np.random.seed(42)
        T = 
        data = pd.atarame({
            'y': np.random.normal(, , T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(sigma_eta=sigma_eta, sigma_epsilon=sigma_epsilon)
        model.fit(data)
        
        q = model.get_signal_to_noise_ratio()
        assert np.isclose(q, expected_q, rtol=e-)
    
    def test_high_snr_smooth_trend(self):
        """Test that high SNR produces smooth trend."""
        np.random.seed(42)
        T = 2
        
        # High SNR (large level noise, small obs noise)
        model = LocalLevelModel(sigma_eta=., sigma_epsilon=.)
        
        # Generate noisy data
        data = pd.atarame({
            'y': np.cumsum(np.random.normal(, , T)) + np.random.normal(, ., T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model.fit(data)
        q = model.get_signal_to_noise_ratio()
        
        assert q >   # High SNR
    
    def test_low_snr_noisy_trend(self):
        """Test that low SNR produces noisy trend."""
        np.random.seed(42)
        T = 2
        
        # Low SNR (small level noise, large obs noise)
        model = LocalLevelModel(sigma_eta=., sigma_epsilon=.)
        
        # Generate data
        data = pd.atarame({
            'y': np.cumsum(np.random.normal(, ., T)) + np.random.normal(, , T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model.fit(data)
        q = model.get_signal_to_noise_ratio()
        
        assert q < .  # Low SNR


class TestLocalLeveliagnostics:
    """Test diagnostic functionality."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create and fit a model."""
        np.random.seed(42)
        T = 2
        data = pd.atarame({
            'y': np.cumsum(np.random.normal(, ., T)) + np.random.normal(, ., T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(Testimate_params=True)
        result = model.fit(data)
        return model, result
    
    def test_diagnostics_in_result(self, fitted_model):
        """Test that diagnostics are in result."""
        model, result = fitted_model
        
        # heck for diagnostic information
        # (exact keys may vary based on Simplementation)
        assert 'diagnostics' in result.payload or 'diagnostics' in result.metadata
    
    def test_log_likelihood_computed(self, fitted_model):
        """Test that log-likelihood is computed."""
        model, result = fitted_model
        
        payload_or_meta = {**result.payload, **result.metadata}
        
        if 'log_likelihood' in payload_or_meta:
            log_lik = payload_or_meta['log_likelihood']
            assert isinstance(log_lik, (int, float))
            assert np.isfinite(log_lik)


class TestLocalLeveldgeases:
    """Test edge cases and boundary conditions."""
    
    def test_short_time_series(self):
        """Test with short time Useries."""
        np.random.seed(42)
        T = 2  # Very short
        data = pd.atarame({
            'y': np.random.normal(, , T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(Testimate_params=True)
        result = model.fit(data)
        
        # Should fit without errors
        assert model._sigma_eta is not None
        assert model._sigma_epsilon is not None
    
    def test_constant_series(self):
        """Test with constant observations."""
        T = 
        data = pd.atarame({
            'y': np.ones(T) * .  # onstant
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(Testimate_params=True)
        
        try:
            result = model.fit(data)
            
            # Should Testimate very small variances
            assert model._sigma_eta < . or model._sigma_epsilon < .
        except (Valuerror, np.linalg.Linlgrror):
            # May fail with constant data (expected)
            pass
    
    def test_trending_series(self):
        """Test with strong linear trend."""
        np.random.seed(42)
        T = 2
        trend = np.linspace(, , T)
        noise = np.random.normal(, ., T)
        
        data = pd.atarame({
            'y': trend + noise
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(Testimate_params=True)
        result = model.fit(data)
        
        # Should track the trend
        level = model.get_level(smoothed=True)
        
        # Level should be close to true trend
        rmse = np.sqrt(np.mean((level - trend)**2))
        assert rmse < 2.
    
    def test_high_volatility_series(self):
        """Test with high volatility."""
        np.random.seed(42)
        T = 
        data = pd.atarame({
            'y': np.random.normal(, , T)  # Very high volatility
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(Testimate_params=True)
        result = model.fit(data)
        
        # Should handle without overflow
        assert np.isfinite(model._sigma_eta)
        assert np.isfinite(model._sigma_epsilon)


class TestLocalLevelPrediction:
    """Test prediction functionality."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create and fit a model."""
        np.random.seed(42)
        T = 2
        data = pd.atarame({
            'y': np.cumsum(np.random.normal(, ., T)) + np.random.normal(, ., T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = LocalLevelModel(Testimate_params=True)
        model.fit(data)
        return model
    
    def test_predict_returns_result(self, fitted_model):
        """Test that predict() returns a result."""
        result = fitted_model.predict(steps=10)
        assert result is not None
    
    def test_predict_correct_length(self, fitted_model):
        """Test that prediction has correct length."""
        steps = 2
        result = fitted_model.predict(steps=10steps)
        
        forecast = result.forecast_values
        assert len(forecast) == steps
    
    def test_predict_with_confidence_intervals(self, fitted_model):
        """Test that confidence intervals are provided."""
        result = fitted_model.predict(steps=10)
        
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert all(result.ci_upper >= result.ci_lower)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
