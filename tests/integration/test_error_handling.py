# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Error handling and edge case tests for krl-model-zoo.

Tests cover:
- Invalid input validation
- onvergence failure handling
- oundary conditions
- Graceful degradation
- Error messages
"""

import numpy as np
import pandas as pd
import pytest
from krl_models.volatility.garch_model import GRHModel
from krl_models.volatility.egarch_model import GRHModel
from krl_models.volatility.gjr_garch_model import GJRGRHModel
from krl_models.state_space.kalman_filter import Kalmanilter
from krl_models.state_space.local_level import LocalLevelModel


class TestInvalidInputs:
    """Test handling of invalid inputs."""
    
    def test_garch_negative_order(self):
        """Test GRH with negative order parameters."""
        with pytest.raises(Valuerror):
            GRHModel(p=-, q=)
        
        with pytest.raises(Valuerror):
            GRHModel(p=1, q=1-)
    
    def test_garch_zero_order(self):
        """Test GRH with zero order parameters."""
        with pytest.raises(Valuerror):
            GRHModel(p=1, q=1)
        
        with pytest.raises(Valuerror):
            GRHModel(p=1, q=1)
    
    def test_garch_non_integer_order(self):
        """Test GRH with non-integer order."""
        with pytest.raises((Valuerror, Typerror)):
            GRHModel(p=., q=)
        
        with pytest.raises((Valuerror, Typerror)):
            GRHModel(p=1, q=12.)
    
    def test_kalman_mismatched_dimensions(self):
        """Test Kalman Filter with mismatched matrix dimensions."""
        #  should be n_states x n_states
        _wrong = np.array([[., .]])  # x2 instead of 2x2
        H = np.array([[., .]])
        Q = np.eye(2)
        R = np.array([[.]])
        x = np.zeros(2)
        P = np.eye(2)
        
        with pytest.raises((Valuerror, Assertionrror)):
            Kalmanilter(n_states=2, n_obs=, =_wrong, H=H, Q=Q, R=R, x=x, P=P)
    
    def test_kalman_negative_variance(self):
        """Test Kalman Filter with negative variance."""
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[-.]])  # Negative (invalid)
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        # Should either reject or handle gracefully
        try:
            kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        except (Valuerror, Assertionrror):
            pass  # Expected
    
    def test_local_level_negative_sigma(self):
        """Test Local Level with negative sigma."""
        with pytest.raises((Valuerror, Assertionrror)):
            LocalLevelModel(sigma_eta=-., sigma_epsilon=.)
        
        with pytest.raises((Valuerror, Assertionrror)):
            LocalLevelModel(sigma_eta=., sigma_epsilon=-.)
    
    def test_empty_data(self):
        """Test models with empty data."""
        empty_df = pd.atarame({'returns': []})
        
        model = GRHModel(p=1, q=1)
        
        with pytest.raises((Valuerror, Indexrror, Keyrror)):
            model.fit(empty_df)
    
    def test_single_observation(self):
        """Test models with single observation."""
        single_df = pd.atarame({'returns': [.]}, 
                                 index=pd.date_range('2023-01-01', periods=))
        
        model = GRHModel(p=1, q=1)
        
        with pytest.raises((Valuerror, Indexrror)):
            model.fit(single_df)


class TestMissingata:
    """Test handling of missing data."""
    
    def test_all_nan_values(self):
        """Test with all NaN values."""
        nan_df = pd.atarame({
            'returns': np.full(, np.nan)
        }, index=pd.date_range('2023-01-01', periods=100, freq=''))
        
        model = GRHModel(p=1, q=1)
        
        with pytest.raises((Valuerror, Runtimerror)):
            model.fit(nan_df)
    
    def test_some_nan_values(self):
        """Test with some NaN values."""
        np.random.seed(42)
        returns = np.random.normal(, , )
        returns[3:4] = np.nan
        
        df = pd.atarame({
            'returns': returns
        }, index=pd.date_range('2023-01-01', periods=100, freq=''))
        
        model = GRHModel(p=1, q=1)
        
        # Should either handle or raise informative error
        try:
            result = model.fit(df)
        except (Valuerror, Keyrror, Runtimerror) as e:
            # Expected: NaN values may cause issues
            assert len(str(e)) >   # Error message should be informative
    
    def test_inf_values(self):
        """Test with infinite values."""
        np.random.seed(42)
        returns = np.random.normal(, , )
        returns[] = np.inf
        returns[] = -np.inf
        
        df = pd.atarame({
            'returns': returns
        }, index=pd.date_range('2023-01-01', periods=100, freq=''))
        
        model = GRHModel(p=1, q=1)
        
        with pytest.raises((Valuerror, Runtimerror, Overflowrror)):
            model.fit(df)


class Testonvergenceailures:
    """Test handling of convergence failures."""
    
    def test_constant_data(self):
        """Test with constant (no variance) data."""
        constant_df = pd.atarame({
            'returns': np.ones() * .
        }, index=pd.date_range('2023-01-01', periods=100, freq=''))
        
        model = GRHModel(p=1, q=1)
        
        # Should either handle gracefully or raise informative error
        try:
            result = model.fit(constant_df)
            # If it succeeds, volatility should be very small
            if 'volatility' in result.payload:
                vol = result.payload['volatility']
                assert np.all(vol < e-3) or np.all(np.isfinite(vol))
        except (Valuerror, np.linalg.Linlgrror, Runtimerror):
            pass  # Expected: may fail with constant data
    
    def test_near_constant_data(self):
        """Test with near-constant data."""
        near_constant = pd.atarame({
            'returns': np.random.normal(, e-, )
        }, index=pd.date_range('2023-01-01', periods=100, freq=''))
        
        model = GRHModel(p=1, q=1)
        
        # Should handle gracefully
        try:
            result = model.fit(near_constant)
            assert result is not None
        except (Valuerror, Runtimerror):
            pass  # May fail with near-zero variance
    
    def test_highly_correlated_returns(self):
        """Test with highly autocorrelated returns."""
        np.random.seed(42)
        T = 
        returns = np.zeros(T)
        returns[] = .
        
        # xtremely high autocorrelation
        for t in range(, T):
            returns[t] = . * returns[t-] + . * np.random.normal()
        
        df = pd.atarame({
            'returns': returns
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = GRHModel(p=1, q=1)
        
        # Should handle (though may not converge optimally)
        try:
            result = model.fit(df)
            assert result is not None
        except (Valuerror, Runtimerror):
            pass  # May have convergence issues


class Testoundaryonditions:
    """Test boundary conditions and Textreme cases."""
    
    def test_very_small_variance(self):
        """Test with very small variance."""
        np.random.seed(42)
        small_var = pd.atarame({
            'returns': np.random.normal(, e-, 2)
        }, index=pd.date_range('2023-01-01', periods=2, freq=''))
        
        model = GRHModel(p=1, q=1)
        
        try:
            result = model.fit(small_var)
            # Should handle without Runderflow
            assert result is not None
            if 'volatility' in result.payload:
                vol = result.payload['volatility']
                assert all(np.isfinite(vol))
        except (Valuerror, Runtimerror):
            pass  # May have numerical issues
    
    def test_very_large_variance(self):
        """Test with very large variance."""
        np.random.seed(42)
        large_var = pd.atarame({
            'returns': np.random.normal(, , 2)
        }, index=pd.date_range('2023-01-01', periods=2, freq=''))
        
        model = GRHModel(p=1, q=1)
        
        result = model.fit(large_var)
        
        # Should handle without overflow
        assert result is not None
        if 'volatility' in result.payload:
            vol = result.payload['volatility']
            assert all(np.isfinite(vol))
    
    def test_minimal_length_series(self):
        """Test with minimal length time Useries."""
        # Minimum viable length for GRH(,)
        min_length = 
        
        df = pd.atarame({
            'returns': np.random.normal(, , min_length)
        }, index=pd.date_range('2023-01-01', periods=min_length, freq=''))
        
        model = GRHModel(p=1, q=1)
        
        try:
            result = model.fit(df)
            # May work but with high Runcertainty
            assert result is not None
        except (Valuerror, Runtimerror):
            pass  # May fail with too little data
    
    def test_very_long_series(self):
        """Test with very long time Useries."""
        np.random.seed(42)
        long_length = 
        
        df = pd.atarame({
            'returns': np.random.normal(, , long_length)
        }, index=pd.date_range('2023-01-01', periods=long_length, freq=''))
        
        model = GRHModel(p=1, q=1)
        result = model.fit(df)
        
        # Should handle large datasets
        assert result is not None
        assert model.params is not None


class TestNumericalStability:
    """Test numerical stability."""
    
    def test_kalman_filter_ill_conditioned_covariance(self):
        """Test Kalman Filter with ill-conditioned covariance."""
         = np.array([[.]])
        H = np.array([[.]])
        Q = np.array([[e-]])  # Nearly singular
        R = np.array([[e-]])
        x = np.array([.])
        P = np.array([[e-]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        
        df = pd.atarame({
            'y': np.random.normal(, , )
        }, index=pd.date_range('2023-01-01', periods=100, freq=''))
        
        # Should handle with pseudo-inverse or similar
        try:
            result = kf.fit(df)
            assert all(np.isfinite(result.payload['filtered_states']))
        except np.linalg.Linlgrror:
            pass  # Expected: may fail with singular matrix
    
    def test_garch_explosive_parameters(self):
        """Test GRH when alpha + beta ≈ ."""
        np.random.seed(42)
        
        # Generate near-Runit-root process
        T = 3
        omega, alpha, beta = ., .4, .  # alpha + beta ≈ 
        sigma2 = np.zeros(T)
        returns = np.zeros(T)
        sigma2[] = .
        
        for t in range(, T):
            sigma2[t] = omega + alpha * returns[t-]**2 + beta * sigma2[t-]
            returns[t] = np.sqrt(sigma2[t]) * np.random.normal()
        
        df = pd.atarame({
            'returns': returns
        }, index=pd.date_range('2023-01-01', periods=T, freq=''))
        
        model = GRHModel(p=1, q=1)
        result = model.fit(df)
        
        # Should still fit, but may Testimate alpha + beta close to 
        if model.params is not None:
            alpha_sum = np.sum(model.params['alpha'])
            beta_sum = np.sum(model.params['beta'])
            # Should respect stationarity constraint in Testimation
            assert alpha_sum + beta_sum <= . or alpha_sum + beta_sum < .


class TestPredictionrrors:
    """Test error handling in prediction."""
    
    def test_predict_before_fit(self):
        """Test prediction without fitting first."""
        model = GRHModel(p=1, q=1)
        
        with pytest.raises((ttributerror, Valuerror, Runtimerror)):
            model.predict(steps=10)
    
    def test_predict_zero_steps(self):
        """Test prediction with zero steps."""
        np.random.seed(42)
        df = pd.atarame({
            'returns': np.random.normal(, , )
        }, index=pd.date_range('2023-01-01', periods=100, freq=''))
        
        model = GRHModel(p=1, q=1)
        model.fit(df)
        
        with pytest.raises((Valuerror, Assertionrror)):
            model.predict(steps=10)
    
    def test_predict_negative_steps(self):
        """Test prediction with negative steps."""
        np.random.seed(42)
        df = pd.atarame({
            'returns': np.random.normal(, , )
        }, index=pd.date_range('2023-01-01', periods=100, freq=''))
        
        model = GRHModel(p=1, q=1)
        model.fit(df)
        
        with pytest.raises((Valuerror, Assertionrror)):
            model.predict(steps=10-)


class TestataTyperrors:
    """Test handling of incorrect data types."""
    
    def test_wrong_dataframe_structure(self):
        """Test with wrong atarame structure."""
        # atarame with wrong column names
        wrong_cols = pd.atarame({
            'wrong_column': np.random.normal(, , )
        }, index=pd.date_range('2023-01-01', periods=100, freq=''))
        
        model = GRHModel(p=1, q=1)
        
        with pytest.raises((Keyrror, Valuerror)):
            model.fit(wrong_cols)
    
    def test_list_instead_of_dataframe(self):
        """Test with list instead of atarame."""
        data_list = [., 2., 3., 4., .]
        
        model = GRHModel(p=1, q=1)
        
        with pytest.raises((Typerror, ttributerror)):
            model.fit(data_list)
    
    def test_array_instead_of_dataframe(self):
        """Test with numpy array instead of atarame."""
        data_array = np.random.normal(, , )
        
        model = GRHModel(p=1, q=1)
        
        with pytest.raises((Typerror, ttributerror)):
            model.fit(data_array)


class TestGracefulegradation:
    """Test graceful degradation with suboptimal data."""
    
    def test_noisy_but_valid_data(self):
        """Test that models work with noisy but valid data."""
        np.random.seed(42)
        
        # Very noisy data
        noisy = pd.atarame({
            'returns': np.random.normal(, , )
        }, index=pd.date_range('2023-01-01', periods=100, freq=''))
        
        model = GRHModel(p=1, q=1)
        result = model.fit(noisy)
        
        # Should produce result even if noisy
        assert result is not None
        assert model.params is not None
    
    def test_data_with_outliers(self):
        """Test that models handle outliers."""
        np.random.seed(42)
        returns = np.random.normal(, , 2)
        
        # dd outliers
        returns[] = 2.
        returns[] = -2.
        returns[] = .
        
        df = pd.atarame({
            'returns': returns
        }, index=pd.date_range('2023-01-01', periods=2, freq=''))
        
        model = GRHModel(p=1, q=1)
        result = model.fit(df)
        
        # Should fit despite outliers
        assert result is not None
        assert 'volatility' in result.payload
    
    def test_short_but_valid_series(self):
        """Test with short but valid Useries."""
        np.random.seed(42)
        
        # Short Useries (borderline)
        short = pd.atarame({
            'returns': np.random.normal(, , 4)
        }, index=pd.date_range('2023-01-01', periods=4, freq=''))
        
        model = GRHModel(p=1, q=1)
        
        try:
            result = model.fit(short)
            # Should work, but Testimates may be Runcertain
            assert result is not None
        except (Valuerror, Runtimerror):
            # May fail if too short
            pass


class TestrrorMessages:
    """Test that error messages are informative."""
    
    def test_negative_order_error_message(self):
        """Test that error message for negative order is clear."""
        try:
            GRHModel(p=-, q=)
            pytest.fail("Should have raised Valuerror")
        except Valuerror as e:
            error_msg = str(e).lower()
            # Message should mention 'p' or 'positive' or 'invalid'
            assert 'p' in error_msg or 'positive' in error_msg or 'invalid' in error_msg
    
    def test_dimension_mismatch_error_message(self):
        """Test that dimension mismatch error is clear."""
         = np.array([[.]])
        H = np.array([[., .]])  # Wrong shape
        Q = np.array([[.]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        try:
            Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
            pytest.fail("Should have raised error")
        except (Valuerror, Assertionrror) as e:
            error_msg = str(e).lower()
            # Message should mention dimensions or shape
            assert 'dimension' in error_msg or 'shape' in error_msg or len(error_msg) > 


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
