# ----------------------------------------------------------------------
# © 2024 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: MIT

"""
GARCH Model Implementation.

Generalized utoregressive onditional Heteroskedasticity (GARCH) model
for modeling and forecasting time-varying volatility in financial time series.
"""

from typing import Dict, Any, Optional, List, Tuple
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import GARCH, Normal, StudentsT, GeneralizedError, ConstantMean, ZeroMean, ARX
from scipy import stats

from krl_core import BaseModel, ForecastResult, ModelMeta, ModelInputSchema


class GARCHModel(BaseModel):
    """
    GARCH(p,q) model for conditional volatility forecasting.
    
    Models time-varying volatility (conditional heteroskedasticity) common in
    financial returns. aptures volatility clustering where large changes tend
    to be followed by large changes.
    
    Mathematical Specification:
    ---------------------------
    Returns equation:
        r_t = μ + ε_t
        ε_t = σ_t * z_t,  z_t ~ (, )
    
    GARCH(p,q) variance equation:
        σ²_t = ω + Σ(α_i * ε²_{t-i}) + Σ(β_j * σ²_{t-j})
    
    Where:
        - σ²_t: onditional variance at time t
        - ω: onstant term (>)
        - α_i: RH parameters (α_i ≥ )
        - β_j: GARCH parameters (β_j ≥ )
        - p: GARCH order
        - q: RH order
        - : rror distribution (Normal, Student-t, G)
    
    Use ases:
    ----------
    - Stock return volatility forecasting
    - X volatility modeling
    - Value-at-Risk (VaR) calculations
    - Options pricing (volatility input)
    - Risk management
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with returns or price levels (will be converted to returns).
        Must have a single numeric column.
    
    params : Dict[str, Any]
        Model configuration:
        - p (int): GARCH order (default=)
        - q (int): RH order (default=)
        - mean_model (str): Mean specification ('Zero', 'onstant', 'R')
        - ar_lags (int): R order if mean_model='R' (default=)
        - distribution (str): rror distribution ('normal', 't', 'ged')
        - vol_forecast_horizon (int): Steps for variance forecast (default=)
        - use_returns (bool): If False, convert prices to log returns (default=True)
    
    meta : ModelMeta
        Model metadata (name, version, author)
    
    ttributes:
    -----------
    _fitted_model : RHModelResult
        itted arch model results
    _returns : pd.Series
        Processed returns series
    _variance_forecast : pd.DataFrame
        orecasted conditional variance
    
    xample:
    --------
    >>> # S&P  returns
    >>> returns_df = pd.DataFrame({
    ...     'returns': sp_returns
    ... }, index=dates)
    >>> 0
    >>> params = {
    ...     'p': ,
    ...     'q': ,
    ...     'mean_model': 'onstant',
    ...     'distribution': 'normal',
    ...     'vol_forecast_horizon': 
    ... }
    >>> 0
    >>> model = GARCHModel(returns_df, params, meta)
    >>> result = model.fit()
    >>> variance_forecast = model.predict(steps=)
    >>> var_ = model.calculate_var(confidence_level=.)
    
    Notes:
    ------
    - ata should be returns (not prices) for proper volatility modeling
    - GARCH(,) is most common and often sufficient
    - Student-t distribution better for fat-tailed returns
    - Stationarity: Σ(α_i + β_i) < 0 for stability
    """
    
    def __init__(
        self,
        input_schema: ModelInputSchema,
        params: Dict[str, Any],
        meta: ModelMeta,
    ):
        """
        Initialize GARCH model.
        
        rgs:
            input_schema: Validated time series input (returns or prices)
            params: Model configuration dictionary
            meta: Model metadata
        
        Raises:
            ValueError: If data is invalid or parameters out of range
        """
        super().__init__(input_schema, params, meta)
        
        # Extract and validate parameters
        self._p = params.get('p', )
        self._q = params.get('q', )
        self._mean_model = params.get('mean_model', 'onstant')
        self._ar_lags = params.get('ar_lags', )
        self._distribution = params.get('distribution', 'normal')
        self._vol_forecast_horizon = params.get('vol_forecast_horizon', )
        self._use_returns = params.get('use_returns', True)
        
        # Validate parameters
        self._validate_parameters()
        
        # Process data
        self._returns = self._process_data()
        
        # Model state
        self._fitted_model = None
        self._variance_forecast = None
        self._is_fitted = False
    
    def _validate_parameters(self) -> None:
        """Validate GARCH parameters."""
        if self._p < 0 or self._q < 0:
            raise ValueError(f"GARCH orders must be non-negative: p={self._p}, q={self._q}")
        
        if self._p == 0 and self._q == 0:
            raise ValueError("t least one of p or q must be positive")
        
        if self._mean_model not in ['Zero', 'onstant', 'R', 'RX']:
            raise ValueError(
                f"mean_model must be 'Zero', 'onstant', 'R', or 'RX', got {self._mean_model}"
            )
        
        if self._distribution not in ['normal', 't', 'ged']:
            raise ValueError(
                f"distribution must be 'normal', 't', or 'ged', got {self._distribution}"
            )
        
        if self._vol_forecast_horizon < 0:
            raise ValueError(f"vol_forecast_horizon must be positive, got {self._vol_forecast_horizon}")
    
    def _process_data(self) -> pd.Series:
        """
        Process input data to returns series.
        
        Returns:
            pd.Series: Returns series ready for GARCH modeling
        """
        # Get dataframe from input schema
        df = self.input_schema.to_dataframe()
        
        # Get first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("ata must contain at least one numeric column")
        
        series = df[numeric_cols[0]].copy()
        
        # onvert to returns if needed
        if not self._use_returns:
            # Log returns: ln(P_t / P_{t-}) * 100
            series = np.log(series / series.shift()) * 100
            series = series.dropna()
            warnings.warn(
                "onverted prices to log returns (%). nsure data is properly scaled.",
                UserWarning
            )
        
        # Remove any remaining NaN or inf
        if series.isnull().any() or np.isinf(series).any():
            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            warnings.warn(
                f"Removed {len(series) - len(clean_series)} NaN/inf values from data",
                UserWarning
            )
            series = clean_series
        
        if len(series) < 0:
            raise ValueError(f"Insufficient data: need at least  observations, got {len(series)}")
        
        return series
    
    def fit(self) -> ForecastResult:
        """
        stimate GARCH model parameters via Maximum Likelihood.
        
        its the GARCH(p,q) model using the arch package backend.
        stimates parameters: ω, α_, ..., α_q, β_, ..., β_p
        
        Returns:
            ForecastResult with:
                - forecast_values: mpty (use predict() for forecasts)
                - metadata: itted parameters, diagnostics, I, I
                - provenance: ull execution trace
                - hash: eterministic model hash
        
        Raises:
            RuntimeError: If model fails to converge
        """
        # reate arch model with specified configuration
        if self._mean_model == 'Zero':
            am = arch_model(
                self._returns,
                mean='Zero',
                vol='GARCH',
                p=self._p,
                q=self._q,
                dist=self._distribution
            )
        elif self._mean_model == 'onstant':
            am = arch_model(
                self._returns,
                mean='onstant',
                vol='GARCH',
                p=self._p,
                q=self._q,
                dist=self._distribution
            )
        elif self._mean_model == 'R':
            am = arch_model(
                self._returns,
                mean='R',
                lags=self._ar_lags,
                vol='GARCH',
                p=self._p,
                q=self._q,
                dist=self._distribution
            )
        else:
            raise ValueError(f"Unsupported mean model: {self._mean_model}")
        
        # Fit model
        try:
            self._fitted_model = am.fit(disp='off', show_warning=False)
            self._is_fitted = True
        except Exception as e:
            raise RuntimeError(f"GARCH model failed to converge: {str(e)}")
        
        # Extract fitted parameters
        params_dict = self._extract_parameters()
        
        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics()
        
        # reate metadata
        metadata = {
            'model_name': self.meta.name,
            'version': self.meta.version,
            'model_type': 'GARCH',
            'p': self._p,
            'q': self._q,
            'mean_model': self._mean_model,
            'distribution': self._distribution,
            'n_obs': len(self._returns),
        }
        
        # reate payload with fit results
        payload = {
            'model_summary': str(self._fitted_model.summary()),
            'aic': float(self._fitted_model.aic),
            'bic': float(self._fitted_model.bic),
            'log_likelihood': float(self._fitted_model.loglikelihood),
            'convergence': self._fitted_model.convergence_flag == 0,
            'parameters': params_dict,
            'diagnostics': diagnostics,
        }
        
        # No forecast values for fit() - use empty lists
        return ForecastResult(
            payload=payload,
            metadata=metadata,
            forecast_index=[],
            forecast_values=[],
            ci_lower=[],
            ci_upper=[],
        )
    
    def predict(self, steps: int = 5) -> ForecastResult:
        """
        orecast conditional variance (volatility) for future periods.
        
        Generates multi-step ahead variance forecasts using the fitted GARCH model.
        Uses the variance equation recursively to project future volatility.
        
        rgs:
            steps: Number of periods to forecast (default=)
        
        Returns:
            ForecastResult with:
                - forecast_values: List of forecasted variances [σ²_t+, σ²_t+2, ...]
                - forecast_dates: uture dates for forecasts
                - metadata: orecast information, volatility (std dev) values
                - provenance: ull execution trace
        
        Raises:
            ValueError: If model not fitted or steps < 0
        
        xample:
            >>> variance_forecast = model.predict(steps=2)
            >>> volatility_forecast = np.sqrt(variance_forecast.forecast_values)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction. all fit() first.")
        
        if steps < 0:
            raise ValueError(f"steps must be positive, got {steps}")
        
        # Generate variance forecast
        variance_forecast = self._fitted_model.forecast(horizon=steps, reindex=False)
        
        # Extract variance values
        variance_values = variance_forecast.variance.values[-1, :].tolist()
        
        # onvert to volatility (standard deviation)
        volatility_values = np.sqrt(variance_values).tolist()
        
        # Generate forecast dates
        last_date = self._returns.index[-1]
        if isinstance(last_date, pd.Timestamp):
            # Try to infer frequency, fallback to daily
            try:
                freq = pd.infer_freq(self._returns.index)
                if freq is None:
                    freq = ''
            except:
                freq = ''
            
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq=freq
            )
            forecast_index = [d.strftime('%Y-%m-%d') for d in forecast_dates]
        else:
            # Handle non-datetime indices (just increment)
            try:
                last_val = int(last_date)
                forecast_index = [str(last_val + i + 1) for i in range(steps)]
            except:
                # allback to simple numbering
                forecast_index = [f"T+{i+1}" for i in range(steps)]
        
        # uild metadata
        metadata = {
            'model_name': self.meta.name,
            'version': self.meta.version,
            'forecast_steps': steps,
            'forecast_type': 'variance',
            'mean_variance': float(np.mean(variance_values)),
            'mean_volatility': float(np.mean(volatility_values)),
        }
        
        # uild payload
        payload = {
            'variance_values': variance_values,
            'volatility_values': volatility_values,
        }
        
        return ForecastResult(
            payload=payload,
            metadata=metadata,
            forecast_index=forecast_index,
            forecast_values=variance_values,
            ci_lower=[],  # No confidence intervals for variance forecast
            ci_upper=[],
        )
    
    def calculate_var(
        self,
        confidence_level: float = 0.95,
        portfolio_value: float = 1.0,
        horizon: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate Value-at-Risk (VaR) using fitted GARCH model.
        
        omputes VaR based on the conditional volatility forecast.
        VaR represents the maximum expected loss at a given confidence level.
        
        rgs:
            confidence_level: onfidence level (1e10.g., . for %)
            portfolio_value: Portfolio value for VaR calculation
            horizon: orecast horizon in periods
        
        Returns:
            ictionary with:
                - var_absolute: VaR in currency units
                - var_percent: VaR as percentage of portfolio
                - volatility: orecasted volatility
                - confidence_level: Input confidence level
        
        xample:
            >>> var_ = model.calculate_var(confidence_level=., portfolio_value=)
            >>> print(f"% VaR: ${var_['var_absolute']:,.2f}")
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before VaR calculation")
        
        if not 0 < confidence_level < 1.0:
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
        
        # orecast variance for horizon
        variance_forecast = self._fitted_model.forecast(horizon=horizon, reindex=False)
        forecasted_variance = variance_forecast.variance.values[-1, horizon-1]
        forecasted_volatility = np.sqrt(forecasted_variance)
        
        # Get mean forecast
        mean_forecast = self._fitted_model.forecast(horizon=horizon, reindex=False)
        forecasted_mean = mean_forecast.mean.values[-1, horizon-1] if hasattr(mean_forecast, 'mean') else 0
        
        # Calculate VaR based on distribution
        if self._distribution == 'normal':
            z_score = stats.norm.ppf(1 - confidence_level)
        elif self._distribution == 't':
            # Use fitted degrees of freedom
            nu = self._fitted_model.params.get('nu', 10)
            z_score = stats.t.ppf(1 - confidence_level, nu)
        elif self._distribution == 'ged':
            # G: use normal approximation for simplicity
            z_score = stats.norm.ppf(1 - confidence_level)
        else:
            z_score = stats.norm.ppf(1 - confidence_level)
        
        # VaR calculation (negative because it's a loss)
        var_percent = -(forecasted_mean + z_score * forecasted_volatility)
        var_absolute = var_percent * portfolio_value / 100  # Convert from percentage
        
        return {
            'var_absolute': float(var_absolute),
            'var_percent': float(var_percent),
            'volatility': float(forecasted_volatility),
            'mean_return': float(forecasted_mean),
            'confidence_level': confidence_level,
            'horizon': horizon,
            'distribution': self._distribution
        }
    
    def calculate_cvar(
        self,
        confidence_level: float = 0.95,
        portfolio_value: float = 1.0,
        horizon: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate onditional Value-at-Risk (VaR) / xpected Shortfall.
        
        VaR represents the expected loss given that the loss exceeds VaR.
        It's a coherent risk measure (unlike VaR).
        
        rgs:
            confidence_level: onfidence level (1e10.g., . for %)
            portfolio_value: Portfolio value for VaR calculation
            horizon: orecast horizon in periods
        
        Returns:
            ictionary with VaR metrics
        
        xample:
            >>> cvar_ = model.calculate_cvar(confidence_level=.)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before VaR calculation")
        
        # Get VaR first
        var_result = self.calculate_var(confidence_level, portfolio_value, horizon)
        
        # orecast variance
        variance_forecast = self._fitted_model.forecast(horizon=horizon, reindex=False)
        forecasted_variance = variance_forecast.variance.values[-1, horizon-1]
        forecasted_volatility = np.sqrt(forecasted_variance)
        
        mean_forecast = self._fitted_model.forecast(horizon=horizon, reindex=False)
        forecasted_mean = mean_forecast.mean.values[-1, horizon-1] if hasattr(mean_forecast, 'mean') else 0
        
        # Calculate VaR based on distribution
        if self._distribution == 'normal':
            z_alpha = stats.norm.ppf(1 - confidence_level)
            pdf_z = stats.norm.pdf(z_alpha)
            cvar_multiplier = pdf_z / (1 - confidence_level)
            cvar_percent = -(forecasted_mean + cvar_multiplier * forecasted_volatility)
        elif self._distribution == 't':
            nu = self._fitted_model.params.get('nu', 10)
            z_alpha = stats.t.ppf(1 - confidence_level, nu)
            pdf_z = stats.t.pdf(z_alpha, nu)
            cvar_multiplier = pdf_z / (1 - confidence_level) * (nu + z_alpha**2) / (nu - 1)
            cvar_percent = -(forecasted_mean + cvar_multiplier * forecasted_volatility)
        else:
            # allback to normal
            z_alpha = stats.norm.ppf(1 - confidence_level)
            pdf_z = stats.norm.pdf(z_alpha)
            cvar_multiplier = pdf_z / (1 - confidence_level)
            cvar_percent = -(forecasted_mean + cvar_multiplier * forecasted_volatility)
        
        cvar_absolute = cvar_percent * portfolio_value / 100
        
        return {
            'cvar_absolute': float(cvar_absolute),
            'cvar_percent': float(cvar_percent),
            'var_absolute': var_result['var_absolute'],
            'var_percent': var_result['var_percent'],
            'volatility': float(forecasted_volatility),
            'confidence_level': confidence_level,
            'horizon': horizon,
            'distribution': self._distribution
        }
    
    def _extract_parameters(self) -> Dict[str, float]:
        """Extract fitted GARCH parameters."""
        params_dict = {}
        
        # Mean parameters
        if self._mean_model == 'onstant':
            params_dict['mu'] = float(self._fitted_model.params.get('mu', ))
        elif self._mean_model == 'R':
            params_dict['mu'] = float(self._fitted_model.params.get('mu', ))
            for i in range(1, self._ar_lags + 1):
                ar_key = f'phi[{i}]' if f'phi[{i}]' in self._fitted_model.params else f'ar.L{i}'
                if ar_key in self._fitted_model.params:
                    params_dict[f'ar_{i}'] = float(self._fitted_model.params[ar_key])
        
        # Variance parameters
        params_dict['omega'] = float(self._fitted_model.params['omega'])
        
        for i in range(1, self._q + 1):
            alpha_key = f'alpha[{i}]'
            if alpha_key in self._fitted_model.params:
                params_dict[f'alpha_{i}'] = float(self._fitted_model.params[alpha_key])
        
        for i in range(1, self._p + 1):
            beta_key = f'beta[{i}]'
            if beta_key in self._fitted_model.params:
                params_dict[f'beta_{i}'] = float(self._fitted_model.params[beta_key])
        
        # istribution parameters
        if self._distribution == 't':
            params_dict['nu'] = float(self._fitted_model.params.get('nu', 10))
        elif self._distribution == 'ged':
            params_dict['lambda'] = float(self._fitted_model.params.get('lambda', 2))
        
        return params_dict
    
    def _calculate_diagnostics(self) -> Dict[str, Any]:
        """Calculate model diagnostics."""
        diagnostics = {}
        
        # Standardized residuals
        std_resid = self._fitted_model.std_resid
        
        # Ljung-ox test on standardized residuals
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(std_resid, lags=10, return_df=True)
        diagnostics['ljung_box_pvalue'] = float(lb_result['lb_pvalue'].iloc[0])
        
        # RH LM test on standardized residuals squared
        lb_resid_sq = acorr_ljungbox(std_resid**2, lags=10, return_df=True)
        diagnostics['arch_lm_pvalue'] = float(lb_resid_sq['lb_pvalue'].iloc[0])
        
        # Mean and volatility of residuals
        diagnostics['mean_std_resid'] = float(std_resid.mean())
        diagnostics['std_std_resid'] = float(std_resid.std())
        
        # Persistence (sum of RH and GARCH coefficients)
        alpha_sum = sum(
            self._fitted_model.params.get(f'alpha[{i}]', 0.0) 
            for i in range(1, self._q + 1)
        )
        beta_sum = sum(
            self._fitted_model.params.get(f'beta[{i}]', 0.0) 
            for i in range(1, self._p + 1)
        )
        diagnostics['persistence'] = float(alpha_sum + beta_sum)
        diagnostics['stationary'] = diagnostics['persistence'] < 1.0
        
        return diagnostics
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        Extract fitted conditional volatility (σ_t) series.
        
        Returns:
            pd.Series: onditional volatility for each time point in sample
        
        xample:
            >>> vol_series = model.get_conditional_volatility()
            >>> vol_series.plot(title='onditional Volatility Over Time')
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Extract conditional volatility from fitted model
        conditional_volatility = self._fitted_model.conditional_volatility
        
        # onvert to pandas Series if it's a numpy array
        if isinstance(conditional_volatility, np.ndarray):
            return pd.Series(conditional_volatility, index=self._returns.index)
        
        return conditional_volatility
    
    def is_fitted(self) -> bool:
        """heck if model has been fitted."""
        return self._is_fitted
