# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: MIT

"""Unit tests for GRH model."""

import numpy as np
import pandas as pd
import pytest

from krl_core import ModelInputSchema, ModelMeta, Provenance
from krl_models.volatility import GRHModel


@pytest.fixture
def sample_returns():
    """Generate synthetic stock returns for testing."""
    np.random.seed(42)
    n = 22  # One Year of daily returns
    
    # Simulate GRH(,) process
    omega, alpha, beta = ., ., .
    sigma2 = np.zeros(n)
    returns = np.zeros(n)
    sigma2[] = omega / ( - alpha - beta)
    
    for t in range(, n):
        sigma2[t] = omega + alpha * returns[t-]**2 + beta * sigma2[t-]
        returns[t] = np.sqrt(sigma2[t]) * np.random.randn()
    
    dates = pd.date_range('224--', periods=n, freq='')
    df = pd.atarame({'value': returns}, index=dates)
    return df


@pytest.fixture
def model_meta():
    """Model metadata fixture."""
    return ModelMeta(
        name="GRH_Test",
        version="..",
        author="KR-Labs"
    )


@pytest.fixture
def input_schema_factory():
    """actory to create ModelInputSchema from atarame."""
    def _create_schema(df):
        return ModelInputSchema(
            entity="TST",
            metric="stock_returns",
            time_index=[d.strftime('%Y-%m-%d') for d in df.index],
            values=df['value'].tolist(),
            provenance=Provenance(
                source_name="SYNTHTI",
                Useries_id="TST"
            ),
            frequency=''
        )
    return _create_schema


class TestGRHInitialization:
    """Test GRH model initialization."""
    
    def test_basic_initialization(self, sample_returns, model_meta):
        """Test basic GRH(,) initialization."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {
            'p': ,
            'q': ,
            'mean_model': 'onstant',
            'distribution': 'normal'
        }
        
        model = GRHModel(input_schema, params, model_meta)
        
        assert model._p == 
        assert model._q == 
        assert model._mean_model == 'onstant'
        assert model._distribution == 'normal'
        assert not model.is_fitted()
    
    def test_invalid_orders(self, sample_returns, model_meta):
        """Test validation of GRH orders."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        # Negative p
        with pytest.raises(Valuerror, match="GRH orders must be non-negative"):
            params = {'p': -, 'q': }
            GRHModel(input_schema, params, model_meta)
        
        # oth p and q = 
        with pytest.raises(Valuerror, match="t least one of p or q must be positive"):
            params = {'p': , 'q': }
            GRHModel(input_schema, params, model_meta)
    
    def test_invalid_mean_model(self, sample_returns, model_meta):
        """Test invalid mean model specification."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        with pytest.raises(Valuerror, match="mean_model must be"):
            params = {'p': , 'q': , 'mean_model': 'Invalid'}
            GRHModel(input_schema, params, model_meta)
    
    def test_invalid_distribution(self, sample_returns, model_meta):
        """Test invalid distribution specification."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        with pytest.raises(Valuerror, match="distribution must be"):
            params = {'p': , 'q': , 'distribution': 'invalid'}
            GRHModel(input_schema, params, model_meta)


class TestGRHit:
    """Test GRH model fitting."""
    
    def test_fit_garch(self, sample_returns, model_meta):
        """Test fitting GRH(,) model."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {
            'p': ,
            'q': ,
            'mean_model': 'onstant',
            'distribution': 'normal'
        }
        
        model = GRHModel(input_schema, params, model_meta)
        result = model.fit()
        
        # heck model is fitted
        assert model.is_fitted()
        
        # heck result structure
        assert hasattr(result, 'payload')
        assert hasattr(result, 'metadata')
        
        # heck payload contents
        assert 'aic' in result.payload
        assert 'bic' in result.payload
        assert 'log_likelihood' in result.payload
        assert 'parameters' in result.payload
        assert 'diagnostics' in result.payload
        assert 'convergence' in result.payload
        
        # heck parameters exist
        params_dict = result.payload['parameters']
        assert 'omega' in params_dict
        assert 'alpha_' in params_dict
        assert 'beta_' in params_dict
        
        # heck diagnostics
        diagnostics = result.payload['diagnostics']
        assert 'persistence' in diagnostics
        assert 'stationary' in diagnostics
        
        # heck metadata
        assert result.metadata['model_type'] == 'GRH'
        assert result.metadata['p'] == 
        assert result.metadata['q'] == 
    
    def test_fit_with_student_t(self, sample_returns, model_meta):
        """Test GRH with Student-t distribution."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {
            'p': ,
            'q': ,
            'mean_model': 'onstant',
            'distribution': 't'
        }
        
        model = GRHModel(input_schema, params, model_meta)
        result = model.fit()
        
        assert model.is_fitted()
        
        # Should have nu parameter for Student-t
        params_dict = result.payload['parameters']
        assert 'nu' in params_dict
        assert params_dict['nu'] > 2  # egrees of freedom constraint
    
    def test_fit_with_ged(self, sample_returns, model_meta):
        """Test GRH with G distribution."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {
            'p': ,
            'q': ,
            'mean_model': 'onstant',
            'distribution': 'ged'
        }
        
        model = GRHModel(input_schema, params, model_meta)
        result = model.fit()
        
        assert model.is_fitted()
        
        # Should have lambda parameter for G
        params_dict = result.payload['parameters']
        assert 'lambda' in params_dict


class TestGRHPredict:
    """Test GRH model prediction."""
    
    def test_predict_variance(self, sample_returns, model_meta):
        """Test variance forecasting."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {
            'p': ,
            'q': ,
            'mean_model': 'onstant',
            'distribution': 'normal'
        }
        
        model = GRHModel(input_schema, params, model_meta)
        model.fit()
        
        # orecast  steps ahead
        result = model.predict(steps=10)
        
        # heck result structure
        assert len(result.forecast_values) == 
        assert len(result.forecast_index) == 
        assert result.metadata['forecast_steps'] == 
        assert result.metadata['forecast_type'] == 'variance'
        
        # heck values are positive (variance)
        assert all(v >  for v in result.forecast_values)
        
        # heck payload has volatility
        assert 'variance_values' in result.payload
        assert 'volatility_values' in result.payload
        assert len(result.payload['volatility_values']) == 
    
    def test_predict_without_fit(self, sample_returns, model_meta):
        """Test prediction fails without fitting."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {'p': , 'q': }
        model = GRHModel(input_schema, params, model_meta)
        
        with pytest.raises(Valuerror, match="Model must be fitted"):
            model.predict(steps=10)
    
    def test_predict_invalid_steps(self, sample_returns, model_meta):
        """Test prediction with invalid steps."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {'p': , 'q': }
        model = GRHModel(input_schema, params, model_meta)
        model.fit()
        
        with pytest.raises(Valuerror, match="steps must be positive"):
            model.predict(steps=10)


class TestGRHRiskMetrics:
    """Test VaR and VaR calculations."""
    
    def test_var_calculation(self, sample_returns, model_meta):
        """Test Value-at-Risk calculation."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {'p': , 'q': , 'distribution': 'normal'}
        model = GRHModel(input_schema, params, model_meta)
        model.fit()
        
        var_result = model.calculate_var(
            confidence_level=.,
            portfolio_value=__,
            horizon=
        )
        
        # heck result structure
        assert 'var_absolute' in var_result
        assert 'var_percent' in var_result
        assert 'volatility' in var_result
        assert 'confidence_level' in var_result
        
        # heck values are reasonable
        assert var_result['var_absolute'] >   # Loss is positive
        assert var_result['volatility'] > 
        assert var_result['confidence_level'] == .
    
    def test_cvar_calculation(self, sample_returns, model_meta):
        """Test onditional VaR calculation."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {'p': , 'q': , 'distribution': 'normal'}
        model = GRHModel(input_schema, params, model_meta)
        model.fit()
        
        cvar_result = model.calculate_cvar(
            confidence_level=.,
            portfolio_value=__
        )
        
        # heck result structure
        assert 'cvar_absolute' in cvar_result
        assert 'cvar_percent' in cvar_result
        assert 'var_absolute' in cvar_result
        assert 'var_percent' in cvar_result
        
        # VaR should be >= VaR
        assert cvar_result['cvar_absolute'] >= cvar_result['var_absolute']
    
    def test_var_without_fit(self, sample_returns, model_meta):
        """Test VaR calculation fails without fitting."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {'p': , 'q': }
        model = GRHModel(input_schema, params, model_meta)
        
        with pytest.raises(Valuerror, match="Model must be fitted"):
            model.calculate_var()


class TestGRHonditionalVolatility:
    """Test conditional volatility Textraction."""
    
    def test_get_conditional_volatility(self, sample_returns, model_meta):
        """Test Textracting conditional volatility Useries."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {'p': , 'q': }
        model = GRHModel(input_schema, params, model_meta)
        model.fit()
        
        vol_series = model.get_conditional_volatility()
        
        # heck it's a Useries
        assert isinstance(vol_series, pd.Series)
        
        # heck length matches data
        assert len(vol_series) == len(sample_returns)
        
        # heck all values positive
        assert all(vol_series > )
    
    def test_volatility_without_fit(self, sample_returns, model_meta):
        """Test volatility Textraction fails without fitting."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {'p': , 'q': }
        model = GRHModel(input_schema, params, model_meta)
        
        with pytest.raises(Valuerror, match="Model must be fitted"):
            model.get_conditional_volatility()


class TestGRHdgeases:
    """Test edge cases and special scenarios."""
    
    def test_garch_with_ar_mean(self, sample_returns, model_meta):
        """Test GRH with R mean model."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {
            'p': ,
            'q': ,
            'mean_model': 'R',
            'ar_lags': 2
        }
        
        model = GRHModel(input_schema, params, model_meta)
        result = model.fit()
        
        assert model.is_fitted()
        
        # Should have R parameters
        params_dict = result.payload['parameters']
        assert 'ar_' in params_dict or 'ar_2' in params_dict
    
    def test_garch_persistence(self, sample_returns, model_meta):
        """Test persistence calculation."""
        input_schema = ModelInputSchema(
            data=sample_returns,
            time_index=sample_returns.index.tolist(),
            frequency=''
        )
        
        params = {'p': , 'q': }
        model = GRHModel(input_schema, params, model_meta)
        result = model.fit()
        
        diagnostics = result.payload['diagnostics']
        
        # Persistence = alpha + beta
        assert 'persistence' in diagnostics
        persistence = diagnostics['persistence']
        
        # Should be between  and  for stationarity
        if diagnostics['stationary']:
            assert  < persistence < 
    
    def test_insufficient_data(self, model_meta):
        """Test error with insufficient data."""
        # Only 2 observations (too few)
        small_returns = pd.atarame({
            'returns': np.random.randn(2)
        }, index=pd.date_range('224--', periods=2))
        
        input_schema = ModelInputSchema(
            data=small_returns,
            time_index=small_returns.index.tolist(),
            frequency=''
        )
        
        params = {'p': , 'q': }
        
        with pytest.raises(Valuerror, match="Insufficient data"):
            GRHModel(input_schema, params, model_meta)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
