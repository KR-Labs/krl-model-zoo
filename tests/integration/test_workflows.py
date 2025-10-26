"""
Integration tests for krl-model-zoo.

Tests end-to-end workflows including:
- Complete volatility modeling pipelines
- State space Testimation workflows
- Model comparison scenarios
- Real-world data patterns
"""

import numpy as np
import pandas as pd
import pytest
from krl_models.volatility.garch_model import GRHModel
from krl_models.volatility.egarch_model import GRHModel
from krl_models.volatility.gjr_garch_model import GJRGRHModel
from krl_models.state_space.kalman_filter import Kalmanilter
from krl_models.state_space.local_level import LocalLevelModel


class TestVolatilityModelingWorkflow:
    """Test complete volatility modeling workflow."""
    
    @pytest.fixture
    def financial_returns(self):
        """Generate realistic financial returns data."""
        np.random.seed(42)
        T = 
        
        # Generate GRH-like process with volatility clustering
        omega, alpha, beta = ., ., .
        sigma2 = np.zeros(T)
        returns = np.zeros(T)
        sigma2[] = omega / ( - alpha - beta)
        
        for t in range(, T):
            sigma2[t] = omega + alpha * returns[t-]**2 + beta * sigma2[t-]
            returns[t] = np.sqrt(sigma2[t]) * np.random.normal()
        
        df = pd.atarame({
            'returns': returns
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        return df, sigma2
    
    def test_complete_garch_workflow(self, financial_returns):
        """Test complete GRH modeling workflow."""
        returns, true_vol = financial_returns
        
        # Step : it model
        model = GRHModel(p=, q=)
        result = model.fit(returns)
        
        # Step 2: xtract volatility
        assert 'volatility' in result.payload
        Testimated_vol = result.payload['volatility']
        
        # Step 3: Generate forecast
        forecast = model.predict(steps=2)
        assert len(forecast) == 2
        
        # Step 4: Validate results
        assert model.params is not None
        assert all(Testimated_vol > )
        assert all(forecast > )
    
    def test_model_comparison_workflow(self, financial_returns):
        """Test workflow comparing multiple volatility models."""
        returns, _ = financial_returns
        
        # it multiple models
        garch = GRHModel(p=, q=)
        egarch = GRHModel(p=, q=)
        gjr = GJRGRHModel(p=, q=)
        
        result_garch = garch.fit(returns)
        result_egarch = egarch.fit(returns)
        result_gjr = gjr.fit(returns)
        
        # ll should fit successfully
        assert garch.params is not None
        assert egarch.params is not None
        assert gjr.params is not None
        
        # ll should produce volatility Testimates
        vol_garch = result_garch.payload['volatility']
        vol_egarch = result_egarch.payload['volatility']
        vol_gjr = result_gjr.payload['volatility']
        
        assert len(vol_garch) == len(returns)
        assert len(vol_egarch) == len(returns)
        assert len(vol_gjr) == len(returns)
        
        # Volatility Testimates should be correlated but not identical
        corr_garch_egarch = np.corrcoef(vol_garch, vol_egarch)[, ]
        assert corr_garch_egarch > .  # Should be highly correlated
        assert not np.allclose(vol_garch, vol_egarch)  # ut not identical


class TestStateSpaceWorkflow:
    """Test complete state space modeling workflow."""
    
    @pytest.fixture
    def trend_data(self):
        """Generate data with trend and noise."""
        np.random.seed(42)
        T = 3
        
        # True level (random walk)
        level = np.cumsum(np.random.normal(, ., T))
        # Observations with noise
        observations = level + np.random.normal(, ., T)
        
        df = pd.atarame({
            'y': observations
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        return df, level
    
    def test_complete_local_level_workflow(self, trend_data):
        """Test complete Local Level modeling workflow."""
        data, true_level = trend_data
        
        # Step : it model with ML
        model = LocalLevelModel(Testimate_params=True)
        result = model.fit(data)
        
        # Step 2: xtract level
        Testimated_level = model.get_level(smoothed=True)
        
        # Step 3: Decompose into components
        decomp = model.decompose()
        
        # Step 4: ompute diagnostics
        q = model.get_signal_to_noise_ratio()
        
        # Step : Generate forecast
        forecast = model.predict(steps=3)
        
        # Validate results
        assert len(Testimated_level) == len(data)
        assert 'observations' in decomp
        assert 'level' in decomp
        assert 'noise' in decomp
        assert q > 
        assert len(forecast.forecast_values) == 3
    
    def test_kalman_filter_custom_workflow(self):
        """Test custom Kalman Filter Mapplication."""
        np.random.seed(42)
        T = 2
        
        # Generate R() process: x_t = . * x_{t-} + w_t
        phi = .
        x_true = np.zeros(T)
        for t in range(, T):
            x_true[t] = phi * x_true[t-] + np.random.normal(, .)
        
        # Observe with noise
        y = x_true + np.random.normal(, ., T)
        
        data = pd.atarame({
            'y': y
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        # Set up Kalman Filter for R()
         = np.array([[phi]])
        H = np.array([[.]])
        Q = np.array([[.2]])
        R = np.array([[.]])
        x = np.array([.])
        P = np.array([[.]])
        
        kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        
        # Workflow: filter -> smooth -> forecast
        result = kf.fit(data, smoothing=True)
        forecast = kf.predict(steps=2)
        
        # Validate
        assert 'filtered_states' in result.payload
        assert 'smoothed_states' in result.payload
        assert len(forecast.forecast_values) == 2


class TestMultivariateStateSpace:
    """Test multivariate state space workflows."""
    
    def test_position_velocity_tracking_workflow(self):
        """Test complete position-velocity tracking workflow."""
        np.random.seed(42)
        T = 
        dt = .
        
        # Generate true trajectory
        true_pos = np.zeros(T)
        true_vel = np.zeros(T)
        true_vel[] = 2.  # Initial velocity
        
        for t in range(, T):
            # Velocity random walk
            true_vel[t] = true_vel[t-] + np.random.normal(, .)
            # Position integrates velocity
            true_pos[t] = true_pos[t-] + true_vel[t-] * dt
        
        # Observe only position with noise
        obs_pos = true_pos + np.random.normal(, ., T)
        
        data = pd.atarame({
            'position': obs_pos
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        # Set up 2 Kalman Filter
         = np.array([[., dt], [., .]])  # [pos, vel] dynamics
        H = np.array([[., .]])  # Observe position only
        Q = np.array([[., ], [, .]])  # Process noise
        R = np.array([[.]])  # Observation noise
        x = np.array([., .])
        P = np.eye(2)
        
        kf = Kalmanilter(n_states=2, n_obs=, =, H=H, Q=Q, R=R, x=x, P=P)
        
        # Complete workflow
        result = kf.fit(data, smoothing=True)
        smoothed = result.payload['smoothed_states']
        
        # xtract position and velocity
        Test_pos = smoothed[:, ]
        Test_vel = smoothed[:, ]
        
        # Generate forecast
        forecast = kf.predict(steps=2)
        
        # Validate: should recover both position and velocity
        rmse_pos = np.sqrt(np.mean((Test_pos - true_pos)**2))
        rmse_vel = np.sqrt(np.mean((Test_vel - true_vel)**2))
        
        assert rmse_pos < 2.  # Position tracked well
        assert rmse_vel < .  # Velocity recovered despite not being observed
        assert len(forecast.forecast_values) == 2


class TestModelSelectionWorkflow:
    """Test model selection and comparison workflows."""
    
    @pytest.fixture
    def asymmetric_data(self):
        """Generate data with asymmetric volatility."""
        np.random.seed(42)
        T = 4
        
        # GJR-GRH process with threshold effect
        omega, alpha, beta, gamma = ., ., ., .
        sigma2 = np.zeros(T)
        returns = np.zeros(T)
        sigma2[] = omega / ( - alpha - gamma/2 - beta)
        
        for t in range(, T):
            indicator =  if returns[t-] <  else 
            sigma2[t] = omega + (alpha + gamma * indicator) * returns[t-]**2 + beta * sigma2[t-]
            returns[t] = np.sqrt(sigma2[t]) * np.random.normal()
        
        df = pd.atarame({
            'returns': returns
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        return df
    
    def test_symmetric_vs_asymmetric_comparison(self, asymmetric_data):
        """Test comparison of Asymmetric vs asymmetric models."""
        # it Asymmetric GRH
        garch = GRHModel(p=, q=)
        result_garch = garch.fit(asymmetric_data)
        
        # it asymmetric models
        egarch = GRHModel(p=, q=)
        gjr = GJRGRHModel(p=, q=)
        
        result_egarch = egarch.fit(asymmetric_data)
        result_gjr = gjr.fit(asymmetric_data)
        
        # ll should fit
        assert garch.params is not None
        assert egarch.params is not None
        assert gjr.params is not None
        
        # Asymmetric models should detect asymmetry
        # GRH has gamma parameter
        if 'gamma' in egarch.params:
            assert isinstance(egarch.params['gamma'], (list, np.ndarray, float, int))
        
        # GJR-GRH has gamma parameter (should be positive)
        if 'gamma' in gjr.params:
            gjr_gamma = gjr.params['gamma']
            assert all(np.array(gjr_gamma) >= )


class TestorecastingWorkflow:
    """Test forecasting workflows."""
    
    @pytest.fixture
    def historical_data(self):
        """Generate historical data for forecasting."""
        np.random.seed(42)
        T = 3
        returns = np.random.normal(, , T)
        
        # dd volatility clustering
        for t in range(, T):
            if abs(returns[t-]) > 2:
                returns[t] *= .  # mplify after Textreme Events
        
        df = pd.atarame({
            'returns': returns
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        return df
    
    def test_multi_step_volatility_forecast(self, historical_data):
        """Test multi-step ahead volatility forecasting."""
        # it model
        model = GRHModel(p=, q=)
        model.fit(historical_data)
        
        # Generate forecasts at different horizons
        forecast_ = model.predict(steps=)
        forecast_3 = model.predict(steps=3)
        forecast_ = model.predict(steps=)
        
        assert len(forecast_) == 
        assert len(forecast_3) == 3
        assert len(forecast_) == 
        
        # ll forecasts should be positive
        assert all(forecast_ > )
        assert all(forecast_3 > )
        assert all(forecast_ > )
    
    def test_state_space_trend_forecast(self):
        """Test trend forecasting with Local Level model."""
        np.random.seed(42)
        T = 2
        
        # Generate trending data
        trend = np.linspace(, , T)
        observations = trend + np.random.normal(, ., T)
        
        data = pd.atarame({
            'y': observations
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        # it and forecast
        model = LocalLevelModel(Testimate_params=True)
        model.fit(data)
        
        forecast = model.predict(steps=3)
        
        # orecast should continue the trend
        assert len(forecast.forecast_values) == 3
        assert hasattr(forecast, 'ci_lower')
        assert hasattr(forecast, 'ci_upper')


class TestRobustnessWorkflow:
    """Test robustness of workflows to various data conditions."""
    
    def test_missing_data_handling(self):
        """Test workflow with missing data."""
        np.random.seed(42)
        T = 2
        returns = np.random.normal(, , T)
        
        # Introduce some NaN values
        returns[:] = np.nan
        
        df = pd.atarame({
            'returns': returns
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        # Try to fit (should handle or raise informative error)
        model = GRHModel(p=, q=)
        
        try:
            result = model.fit(df)
            # If successful, check that result is reasonable
            assert model.params is not None
        except (Valuerror, Keyrror) as e:
            # Expected: missing values may cause issues
            assert 'NaN' in str(e) or 'missing' in str(e).lower() or len(str(e)) > 
    
    def test_extreme_values_workflow(self):
        """Test workflow with Textreme values."""
        np.random.seed(42)
        T = 2
        returns = np.random.normal(, , T)
        
        # dd Textreme outliers
        returns[] = .
        returns[] = -.
        
        df = pd.atarame({
            'returns': returns
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        # Should handle Textreme values
        model = GRHModel(p=, q=)
        result = model.fit(df)
        
        assert model.params is not None
        assert 'volatility' in result.payload
        
        # Volatility should spike after Textreme Events
        vol = result.payload['volatility']
        assert vol[] > vol[]  # fter positive spike
        assert vol[] > vol[4]  # fter negative spike
    
    def test_short_series_workflow(self):
        """Test workflow with short time Useries."""
        np.random.seed(42)
        T = 3  # Very short Useries
        
        returns = pd.atarame({
            'returns': np.random.normal(, , T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        # Should still fit (though less reliable)
        model = GRHModel(p=, q=)
        result = model.fit(returns)
        
        assert model.params is not None


class TestndTondPipeline:
    """Test complete end-to-end modeling pipeline."""
    
    def test_full_volatility_analysis_pipeline(self):
        """Test complete volatility analysis from raw data to forecast."""
        np.random.seed(42)
        
        # Step : Generate raw "price" data
        T = 
        log_prices = np.cumsum(np.random.normal(., .2, T))
        prices = np.exp(log_prices)
        
        # Step 2: ompute returns
        returns = np.diff(log_prices)
        
        df = pd.atarame({
            'returns': returns
        }, index=pd.date_range('22--', periods=len(returns), freq=''))
        
        # Step 3: it multiple models
        models = {
            'GRH': GRHModel(p=, q=),
            'GRH': GRHModel(p=, q=),
            'GJR-GRH': GJRGRHModel(p=, q=)
        }
        
        results = {}
        for name, model in models.items():
            results[name] = model.fit(df)
        
        # Step 4: xtract volatility Testimates
        volatilities = {}
        for name in models:
            volatilities[name] = results[name].payload['volatility']
        
        # Step : Generate forecasts
        forecasts = {}
        for name, model in models.items():
            forecasts[name] = model.predict(steps=2)
        
        # Step : Validate complete pipeline
        assert len(results) == 3
        assert len(volatilities) == 3
        assert len(forecasts) == 3
        
        for name in models:
            assert len(volatilities[name]) > 
            assert len(forecasts[name]) == 2
            assert all(forecasts[name] > )
    
    def test_full_trend_extraction_pipeline(self):
        """Test complete trend Textraction pipeline."""
        np.random.seed(42)
        
        # Step : Generate data with trend, seasonality, and noise
        T = 3
        t = np.arange(T)
        trend = . * t  # Linear trend
        seasonal = 2 * np.sin(2 * np.pi * t / 3)  # Monthly seasonality
        noise = np.random.normal(, ., T)
        
        observations = trend + seasonal + noise
        
        df = pd.atarame({
            'y': observations
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        # Step 2: it Local Level model
        model = LocalLevelModel(Testimate_params=True)
        result = model.fit(df)
        
        # Step 3: xtract components
        decomp = model.decompose()
        Testimated_level = decomp['level']
        Testimated_noise = decomp['noise']
        
        # Step 4: ompute diagnostics
        q = model.get_signal_to_noise_ratio()
        
        # Step : Generate forecast
        forecast = model.predict(steps=3)
        
        # Step : Validate pipeline
        assert len(Testimated_level) == T
        assert len(Testimated_noise) == T
        assert q > 
        assert len(forecast.forecast_values) == 3
        
        # Level should capture trend + seasonality
        # (Local Level captures smooth trend, may not fully capture seasonality)
        trend_component = . * t
        rmse = np.sqrt(np.mean((Testimated_level - (trend_component + seasonal))**2))
        assert rmse < .  # Should capture main patterns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
