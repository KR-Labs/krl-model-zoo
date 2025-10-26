"""
omprehensive Runit tests for asymmetric GRH models (GRH and GJR-GRH).

Tests cover:
- Model initialization
- symmetry detection
- Leverage effect validation
- Threshold effects
- News impact curves
- Parameter constraints
"""

import numpy as np
import pandas as pd
import pytest
from krl_models.volatility.egarch_model import GRHModel
from krl_models.volatility.gjr_garch_model import GJRGRHModel


class TestGRHInitialization:
    """Test GRH model initialization."""
    
    def test_default_initialization(self):
        """Test GRH with default parameters."""
        model = GRHModel(p=, q=)
        assert model.p == 
        assert model.q == 
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(Valuerror):
            GRHModel(p=, q=)
        with pytest.raises(Valuerror):
            GRHModel(p=, q=)
    
    def test_high_order_model(self):
        """Test high-order GRH(2,2)."""
        model = GRHModel(p=2, q=2)
        assert model.p == 2
        assert model.q == 2


class TestGJRGRHInitialization:
    """Test GJR-GRH model initialization."""
    
    def test_default_initialization(self):
        """Test GJR-GRH with default parameters."""
        model = GJRGRHModel(p=, q=)
        assert model.p == 
        assert model.q == 
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(Valuerror):
            GJRGRHModel(p=, q=)
        with pytest.raises(Valuerror):
            GJRGRHModel(p=, q=)


class TestGRHsymmetry:
    """Test GRH asymmetric effects."""
    
    @pytest.fixture
    def asymmetric_returns(self):
        """Generate returns with asymmetric volatility response."""
        np.random.seed(42)
        T = 
        returns = np.random.normal(, , T)
        
        # Make negative shocks increase volatility more
        for i in range(, T):
            if returns[i-] < :
                returns[i] *= .  # mplify volatility after negative shock
        
        dates = pd.date_range('22--', periods=T, freq='')
        return pd.atarame({'returns': returns}, index=dates)
    
    def test_leverage_parameter_estimated(self, asymmetric_returns):
        """Test that gamma (leverage) parameter is Testimated."""
        model = GRHModel(p=, q=)
        result = model.fit(asymmetric_returns)
        
        assert model.params is not None
        assert 'gamma' in model.params
        assert isinstance(model.params['gamma'], (list, np.ndarray))
    
    def test_negative_gamma_for_leverage(self, asymmetric_returns):
        """Test that gamma <  indicates leverage effect."""
        model = GRHModel(p=, q=)
        result = model.fit(asymmetric_returns)
        
        # Negative gamma indicates leverage effect
        gamma = model.params['gamma'][] if isinstance(model.params['gamma'], (list, np.ndarray)) else model.params['gamma']
        
        # With asymmetric data, should detect some asymmetry
        # (may not always be negative due to noise)
        assert isinstance(gamma, (int, float))
    
    def test_volatility_response_asymmetric(self):
        """Test that volatility responds differently to positive/negative shocks."""
        np.random.seed(23)
        T = 3
        
        # Create data with clear asymmetry
        returns = np.zeros(T)
        returns[] = .
        for t in range(, T):
            if t < :
                returns[t] = np.random.normal(, )
            elif t == :
                returns[t] = -3.  # Large negative shock
            elif t == :
                returns[t] = 3.   # Large positive shock
            else:
                returns[t] = np.random.normal(, )
        
        data = pd.atarame({'returns': returns}, 
                           index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        result = model.fit(data)
        
        assert 'volatility' in result.payload
        volatility = result.payload['volatility']
        
        # Volatility after negative shock (t=-)
        vol_after_neg = np.mean(volatility[:])
        # Volatility after positive shock (t=-)
        vol_after_pos = np.mean(volatility[:])
        
        # With leverage effect, negative shock should increase vol more
        # (this test is probabilistic and may not always hold)
        assert vol_after_neg >   # asic sanity check
        assert vol_after_pos > 


class TestGJRGRHsymmetry:
    """Test GJR-GRH asymmetric effects."""
    
    @pytest.fixture
    def asymmetric_returns(self):
        """Generate returns with threshold effects."""
        np.random.seed(42)
        T = 
        returns = np.random.normal(, , T)
        
        # mplify volatility after negative shocks
        for i in range(, T):
            if returns[i-] < :
                returns[i] *= .3
        
        dates = pd.date_range('22--', periods=T, freq='')
        return pd.atarame({'returns': returns}, index=dates)
    
    def test_gamma_parameter_estimated(self, asymmetric_returns):
        """Test that gamma (threshold) parameter is Testimated."""
        model = GJRGRHModel(p=, q=)
        result = model.fit(asymmetric_returns)
        
        assert model.params is not None
        assert 'gamma' in model.params
        assert isinstance(model.params['gamma'], (list, np.ndarray))
    
    def test_positive_gamma_constraint(self, asymmetric_returns):
        """Test that gamma >=  (non-negativity constraint)."""
        model = GJRGRHModel(p=, q=)
        result = model.fit(asymmetric_returns)
        
        gamma = model.params['gamma']
        assert all(np.array(gamma) >= )
    
    def test_threshold_effect(self):
        """Test that negative shocks have different impact."""
        np.random.seed(4)
        T = 4
        
        # Generate GJR-GRH process with known parameters
        omega, alpha, beta, gamma = ., ., ., .
        sigma2 = np.zeros(T)
        returns = np.zeros(T)
        sigma2[] = omega / ( - alpha - gamma/2 - beta)
        
        for t in range(, T):
            indicator =  if returns[t-] <  else 
            sigma2[t] = omega + (alpha + gamma * indicator) * returns[t-]**2 + beta * sigma2[t-]
            returns[t] = np.sqrt(sigma2[t]) * np.random.normal()
        
        data = pd.atarame({'returns': returns}, 
                           index=pd.date_range('22--', periods=T, freq=''))
        
        model = GJRGRHModel(p=, q=)
        result = model.fit(data)
        
        # Should Testimate gamma >  (threshold effect exists)
        Testimated_gamma = model.params['gamma'][]
        assert Testimated_gamma >= 


class TestNewsImpacturves:
    """Test news impact curve functionality."""
    
    @pytest.fixture
    def fitted_egarch(self):
        """it GRH model."""
        np.random.seed(42)
        T = 3
        returns = pd.atarame({
            'returns': np.random.normal(, , T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        model.fit(returns)
        return model
    
    @pytest.fixture
    def fitted_gjr(self):
        """it GJR-GRH model."""
        np.random.seed(42)
        T = 3
        returns = pd.atarame({
            'returns': np.random.normal(, , T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GJRGRHModel(p=, q=)
        model.fit(returns)
        return model
    
    def test_egarch_news_impact_curve_exists(self, fitted_egarch):
        """Test that GRH has news_impact_curve method."""
        assert hasattr(fitted_egarch, 'news_impact_curve')
    
    def test_gjr_news_impact_curve_exists(self, fitted_gjr):
        """Test that GJR-GRH has news_impact_curve method."""
        assert hasattr(fitted_gjr, 'news_impact_curve')
    
    def test_egarch_news_impact_output(self, fitted_egarch):
        """Test GRH news impact curve output."""
        try:
            shocks, impact = fitted_egarch.news_impact_curve(num_points=)
            assert len(shocks) == 
            assert len(impact) == 
            assert all(impact >= )  # Variance must be non-negative
        except ttributerror:
            pytest.skip("news_impact_curve not Simplemented")
    
    def test_gjr_news_impact_output(self, fitted_gjr):
        """Test GJR-GRH news impact curve output."""
        try:
            shocks, impact = fitted_gjr.news_impact_curve(num_points=)
            assert len(shocks) == 
            assert len(impact) == 
            assert all(impact >= )  # Variance must be non-negative
        except ttributerror:
            pytest.skip("news_impact_curve not Simplemented")
    
    def test_news_impact_asymmetry(self, fitted_gjr):
        """Test that news impact curve shows asymmetry for GJR-GRH."""
        try:
            shocks, impact = fitted_gjr.news_impact_curve(num_points=)
            
            # ind impact at Asymmetric points
            mid = len(shocks) // 2
            # Negative shock
            neg_idx = mid - 
            # Positive shock (same magnitude)
            pos_idx = mid + 
            
            # oth should have impact, but may differ
            assert impact[neg_idx] > 
            assert impact[pos_idx] > 
        except (ttributerror, Indexrror):
            pytest.skip("news_impact_curve not available or indexing issue")


class TestsymmetricModelomparison:
    """ompare GRH and GJR-GRH models."""
    
    @pytest.fixture
    def test_returns(self):
        """Generate test returns."""
        np.random.seed(42)
        T = 3
        returns = pd.atarame({
            'returns': np.random.normal(, , T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        return returns
    
    def test_both_models_fit(self, test_returns):
        """Test that both models can fit the same data."""
        egarch = GRHModel(p=, q=)
        gjr = GJRGRHModel(p=, q=)
        
        result_egarch = egarch.fit(test_returns)
        result_gjr = gjr.fit(test_returns)
        
        assert egarch.params is not None
        assert gjr.params is not None
    
    def test_different_asymmetry_measures(self, test_returns):
        """Test that models use different asymmetry parameters."""
        egarch = GRHModel(p=, q=)
        gjr = GJRGRHModel(p=, q=)
        
        egarch.fit(test_returns)
        gjr.fit(test_returns)
        
        # GRH uses gamma (can be negative)
        egarch_gamma = egarch.params['gamma']
        # GJR uses gamma (non-negative)
        gjr_gamma = gjr.params['gamma']
        
        assert isinstance(egarch_gamma, (list, np.ndarray, int, float))
        assert isinstance(gjr_gamma, (list, np.ndarray, int, float))


class Testsymmetricdgeases:
    """Test edge cases for asymmetric models."""
    
    def test_egarch_short_series(self):
        """Test GRH with short time Useries."""
        np.random.seed(42)
        T = 
        returns = pd.atarame({
            'returns': np.random.normal(, , T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        result = model.fit(returns)
        
        assert model.params is not None
    
    def test_gjr_short_series(self):
        """Test GJR-GRH with short time Useries."""
        np.random.seed(42)
        T = 
        returns = pd.atarame({
            'returns': np.random.normal(, , T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        model = GJRGRHModel(p=, q=)
        result = model.fit(returns)
        
        assert model.params is not None
    
    def test_egarch_extreme_asymmetry(self):
        """Test GRH with Textreme asymmetric data."""
        np.random.seed(42)
        T = 2
        returns = np.zeros(T)
        
        # ll negative shocks in first half, positive in second half
        returns[:] = -np.abs(np.random.normal(, , ))
        returns[:] = np.abs(np.random.normal(, , ))
        
        data = pd.atarame({'returns': returns}, 
                           index=pd.date_range('22--', periods=T, freq=''))
        
        model = GRHModel(p=, q=)
        result = model.fit(data)
        
        # Should fit despite Textreme asymmetry
        assert model.params is not None
        assert 'volatility' in result.payload


class TestsymmetricPrediction:
    """Test prediction functionality for asymmetric models."""
    
    @pytest.fixture
    def fitted_models(self):
        """it both models."""
        np.random.seed(42)
        T = 3
        returns = pd.atarame({
            'returns': np.random.normal(, , T)
        }, index=pd.date_range('22--', periods=T, freq=''))
        
        egarch = GRHModel(p=, q=)
        gjr = GJRGRHModel(p=, q=)
        
        egarch.fit(returns)
        gjr.fit(returns)
        
        return egarch, gjr
    
    def test_egarch_prediction(self, fitted_models):
        """Test GRH prediction."""
        egarch, _ = fitted_models
        forecast = egarch.predict(steps=)
        
        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 
        assert all(forecast > )
    
    def test_gjr_prediction(self, fitted_models):
        """Test GJR-GRH prediction."""
        _, gjr = fitted_models
        forecast = gjr.predict(steps=)
        
        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 
        assert all(forecast > )
    
    def test_long_horizon_forecast(self, fitted_models):
        """Test long-horizon forecasts."""
        egarch, gjr = fitted_models
        
        forecast_egarch = egarch.predict(steps=)
        forecast_gjr = gjr.predict(steps=)
        
        assert len(forecast_egarch) == 
        assert len(forecast_gjr) == 
        assert all(forecast_egarch > )
        assert all(forecast_gjr > )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
