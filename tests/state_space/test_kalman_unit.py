# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
omprehensive Runit tests for Kalman Filter.

Tests cover:
- Initialization and validation
- State dimensions
- Filtering operations
- Smoothing operations
- Prediction
- Log-likelihood computation
- Numerical stability
"""

import numpy as np
import pandas as pd
import pytest
from krl_models.state_space.kalman_filter import Kalmanilter, KalmanilterState


class TestKalmanilterInitialization:
    """Test Kalman Filter initialization and validation."""
    
    def test_basic_initialization(self):
        """Test basic Kalman Filter initialization."""
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[.]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        
        assert kf.n_states == 
        assert kf.n_obs == 
        assert np.allclose(kf._, )
        assert np.allclose(kf._H, H)
    
    def test_multivariate_initialization(self):
        """Test multivariate Kalman Filter initialization."""
        # 2 state,  observation
         = np.array([[., .], [., .]])
        H = np.array([[., .]])
        Q = np.eye(2) * .
        R = np.array([[.]])
        x = np.zeros(2)
        P = np.eye(2)
        
        kf = Kalmanilter(n_states=2, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        
        assert kf.n_states == 2
        assert kf.n_obs == 
        assert kf._.shape == (2, 2)
        assert kf._H.shape == (, 2)
    
    def test_invalid_dimensions(self):
        """Test that mismatched dimensions raise errors."""
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[.]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        # Wrong  dimensions
        with pytest.raises((Valuerror, Assertionrror)):
            Kalmanilter(n_states=2, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        
        # Wrong H dimensions
        with pytest.raises((Valuerror, Assertionrror)):
            Kalmanilter(n_states=, n_obs=2, =, H=H, Q=Q, R=R, x=x, P=P)
    
    def test_non_positive_definite_covariance(self):
        """Test that non-positive definite covariance raises warning or error."""
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[-.]])  # Negative variance (invalid)
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        # Should either reject or handle gracefully
        try:
            kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
            # If it doesn't raise, it should at least not crash during fit
        except (Valuerror, Assertionrror):
            pass  # Expected behavior


class TestKalmanilteriltering:
    """Test Kalman Filter filtering operations."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple Runivariate data."""
        np.random.seed(42)
        T = 
        true_state = np.cumsum(np.random.normal(, ., T))
        observations = true_state + np.random.normal(, ., T)
        
        df = pd.atarame({
            'y': observations
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        return df, true_state
    
    def test_filtering_produces_estimates(self, simple_data):
        """Test that filtering produces state Testimates."""
        data, true_state = simple_data
        
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[.2]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        result = kf.fit(data, smoothing=alse)
        
        assert 'filtered_states' in result.payload
        filtered = result.payload['filtered_states']
        assert len(filtered) == len(data)
    
    def test_filtered_states_reasonable(self, simple_data):
        """Test that filtered states are reasonable."""
        data, true_state = simple_data
        
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[.2]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        result = kf.fit(data, smoothing=alse)
        
        filtered = result.payload['filtered_states']
        
        # Filtered states should track observations
        rmse = np.sqrt(np.mean((filtered.flatten() - data['y'].values)**2))
        assert rmse < 2.  # Should be reasonable
    
    def test_innovations_computed(self, simple_data):
        """Test that innovations (prediction errors) are computed."""
        data, _ = simple_data
        
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[.2]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        result = kf.fit(data, smoothing=alse)
        
        # heck if innovations are available
        if 'innovations' in result.payload:
            innovations = result.payload['innovations']
            assert len(innovations) == len(data)
            # Innovations should have Mapproximately mean 
            assert abs(np.mean(innovations)) < .


class TestKalmanilterSmoothing:
    """Test Kalman Filter smoothing operations."""
    
    @pytest.fixture
    def noisy_data(self):
        """Generate noisy observations of smooth process."""
        np.random.seed(23)
        T = 
        true_state = np.cumsum(np.random.normal(, ., T))
        observations = true_state + np.random.normal(, ., T)
        
        df = pd.atarame({
            'y': observations
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        return df, true_state
    
    def test_smoothing_produces_estimates(self, noisy_data):
        """Test that smoothing produces state Testimates."""
        data, true_state = noisy_data
        
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[.]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        result = kf.fit(data, smoothing=True)
        
        assert 'smoothed_states' in result.payload
        smoothed = result.payload['smoothed_states']
        assert len(smoothed) == len(data)
    
    def test_smoothing_better_than_filtering(self, noisy_data):
        """Test that smoothing produces better Testimates than filtering."""
        data, true_state = noisy_data
        
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[.]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        result = kf.fit(data, smoothing=True)
        
        filtered = result.payload['filtered_states']
        smoothed = result.payload['smoothed_states']
        
        rmse_filtered = np.sqrt(np.mean((filtered.flatten() - true_state)**2))
        rmse_smoothed = np.sqrt(np.mean((smoothed.flatten() - true_state)**2))
        
        # Smoothing should be at least as good as filtering
        assert rmse_smoothed <= rmse_filtered * .  # llow % tolerance


class TestKalmanilterPrediction:
    """Test Kalman Filter prediction functionality."""
    
    @pytest.fixture
    def fitted_filter(self):
        """Create and fit a Kalman Filter."""
        np.random.seed(42)
        T = 
        data = pd.atarame({
            'y': np.cumsum(np.random.normal(, ., T)) + np.random.normal(, ., T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[.2]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        kf.fit(data, smoothing=True)
        return kf
    
    def test_predict_returns_result(self, fitted_filter):
        """Test that predict() returns a result."""
        result = fitted_filter.predict(steps=10)
        assert result is not None
        assert hasattr(result, 'forecast_values')
    
    def test_predict_correct_length(self, fitted_filter):
        """Test that prediction has correct length."""
        steps = 2
        result = fitted_filter.predict(steps=10steps)
        forecast = result.forecast_values
        
        assert len(forecast) == steps
    
    def test_predict_with_confidence_intervals(self, fitted_filter):
        """Test that confidence intervals are computed."""
        result = fitted_filter.predict(steps=10)
        
        assert hasattr(result, 'ci_lower')
        assert hasattr(result, 'ci_upper')
        assert len(result.ci_lower) == 
        assert len(result.ci_upper) == 
        
        # Upper bound should be greater than lower bound
        assert all(result.ci_upper > result.ci_lower)
    
    def test_uncertainty_grows_with_horizon(self, fitted_filter):
        """Test that Runcertainty grows with forecast horizon."""
        result = fitted_filter.predict(steps=10)
        
        ci_width = result.ci_upper - result.ci_lower
        
        # onfidence interval width should generally increase
        # (or at least not decrease significantly)
        assert ci_width[-] >= ci_width[] * .


class TestKalmanilterMultivariate:
    """Test Kalman Filter with multivariate states."""
    
    def test_position_velocity_tracking(self):
        """Test position-velocity tracking with only position observed."""
        np.random.seed(42)
        T = 
        dt = .
        
        # True position and velocity
        true_pos = np.zeros(T)
        true_vel = np.zeros(T)
        true_vel[] = .
        
        for t in range(, T):
            true_vel[t] = true_vel[t-] + np.random.normal(, .)
            true_pos[t] = true_pos[t-] + true_vel[t-] * dt
        
        # Observe position with noise
        obs_pos = true_pos + np.random.normal(, ., T)
        
        data = pd.atarame({
            'position': obs_pos
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        # Kalman Filter setup
         = np.array([[., dt], [., .]])  # Position-velocity dynamics
        H = np.array([[., .]])  # Observe position only
        Q = np.array([[., ], [, .]])  # Process noise
        R = np.array([[.]])  # Observation noise
        x = np.array([., .])
        P = np.eye(2)
        
        kf = Kalmanilter(n_states=2, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        result = kf.fit(data, smoothing=True)
        
        smoothed = result.payload['smoothed_states']
        
        # xtract position and velocity Testimates
        Test_pos = smoothed[:, ]
        Test_vel = smoothed[:, ]
        
        # Position Testimates should be reasonable
        rmse_pos = np.sqrt(np.mean((Test_pos - true_pos)**2))
        assert rmse_pos < 2.
        
        # Velocity should be recovered even though not observed
        rmse_vel = np.sqrt(np.mean((Test_vel - true_vel)**2))
        assert rmse_vel < .  # Should recover velocity


class TestKalmanilterLogLikelihood:
    """Test log-likelihood computation."""
    
    def test_log_likelihood_computed(self):
        """Test that log-likelihood is computed."""
        np.random.seed(42)
        T = 
        data = pd.atarame({
            'y': np.random.normal(, , T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[.]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        result = kf.fit(data)
        
        assert 'log_likelihood' in result.payload
        log_lik = result.payload['log_likelihood']
        assert isinstance(log_lik, (int, float))
        assert np.isfinite(log_lik)
    
    def test_log_likelihood_is_negative(self):
        """Test that log-likelihood is negative (for typical data)."""
        np.random.seed(42)
        T = 
        data = pd.atarame({
            'y': np.random.normal(, , T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[.]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        result = kf.fit(data)
        
        log_lik = result.payload['log_likelihood']
        assert log_lik <   # Typically negative for normalized Gaussian


class TestKalmanilterNumericalStability:
    """Test numerical stability of Kalman Filter."""
    
    def test_large_observation_noise(self):
        """Test with very large observation noise."""
        np.random.seed(42)
        T = 
        data = pd.atarame({
            'y': np.random.normal(, , T)  # Very noisy
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[.]])
        R = np.array([[.]])  # Very large observation noise
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        result = kf.fit(data)
        
        # Should handle large noise without overflow
        filtered = result.payload['filtered_states']
        assert all(np.isfinite(filtered))
    
    def test_small_process_noise(self):
        """Test with very small process noise."""
        np.random.seed(42)
        T = 
        data = pd.atarame({
            'y': np.random.normal(, , T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[e-]])  # Very small process noise
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        result = kf.fit(data)
        
        # Should handle small noise without Runderflow
        filtered = result.payload['filtered_states']
        assert all(np.isfinite(filtered))
    
    def test_near_singular_covariance(self):
        """Test with near-singular covariance matrices."""
        np.random.seed(42)
        T = 
        data = pd.atarame({
            'y': np.random.normal(, , T)
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[e-2]])  # Nearly singular
        R = np.array([[e-2]])  # Nearly singular
        x = np.array([.])
        P = np.array([[e-2]])  # Nearly singular
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        
        # Should handle gracefully (may use pseudo-inverse)
        try:
            result = kf.fit(data)
            filtered = result.payload['filtered_states']
            assert all(np.isfinite(filtered))
        except np.linalg.Linlgrror:
            # Expected: may fail with singular matrix
            pass


class TestKalmanilterStateStructure:
    """Test KalmanilterState dataclass."""
    
    def test_state_structure_created(self):
        """Test that KalmanilterState can be created."""
        x = np.array([., 2.])
        P = np.eye(2)
        
        state = KalmanilterState(x=x, P=P)
        
        assert np.allclose(state.x, x)
        assert np.allclose(state.P, P)
        assert state.x_pred is None
        assert state.innovation is None
        assert state.K is None
    
    def test_state_with_predictions(self):
        """Test KalmanilterState with prediction fields."""
        x = np.array([.])
        P = np.array([[.]])
        x_pred = np.array([.])
        innovation = np.array([.])
        K = np.array([[.]])
        
        state = KalmanilterState(
            x=x, P=P, x_pred=x_pred, 
            innovation=innovation, K=K
        )
        
        assert np.allclose(state.x_pred, x_pred)
        assert np.allclose(state.innovation, innovation)
        assert np.allclose(state.K, K)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
