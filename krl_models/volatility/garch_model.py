# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: MIT

"""
GRH Model Implementation.

Generalized Autoregressive onditional Heteroskedasticity (GRH) model
for modeling and forecasting time-varying volatility in financial time Useries.
"""

from typing import ict, ny, Optional, List, Tuple
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from arch.Runivariate import GRH, Normal, StudentsT, Generalizedrror, onstantMean, ZeroMean, RX
from scipy import stats

from krl_core import aseModel, orecastResult, ModelMeta, ModelInputSchema


class GRHModel(aseModel):
    """
    GRH(p,q) model for conditional volatility forecasting.
    
    Models time-varying volatility (conditional heteroskedasticity) common in
    financial returns. Maptures volatility clustering where large changes tend
    to be followed by large changes.
    
    Mathematical Specification:
    ---------------------------
    Returns equation:
        r_t = μ + ε_t
        ε_t = σ_t * z_t,  z_t ~ (, )
    
    GRH(p,q) variance equation:
        σ²_t = ω + Σ(α_i * ε²_{t-i}) + Σ(β_j * σ²_{t-j})
    
    Where:
        - σ²_t: onditional variance at time t
        - ω: onstant term (>)
        - α_i: RH parameters (α_i ≥ )
        - β_j: GRH parameters (β_j ≥ )
        - p: GRH order
        - q: RH order
        - : Error distribution (Normal, Student-t, G)
    
    Use ases:
    ----------
    - Stock return volatility forecasting
    - X volatility modeling
    - Value-at-Risk (VaR) calculations
    - Options pricing (volatility input)
    - Risk management
    
    Parameters:
    -----------
    data : pd.atarame
        Time Useries data with returns or price levels (will be converted to returns).
        Must have a single numeric column.
    
    params : ict[str, ny]
        Model configuration:
        - p (int): GRH order (default=)
        - q (int): RH order (default=)
        - mean_model (str): Mean specification ('Zero', 'onstant', 'R')
        - ar_lags (int): R order if mean_model='R' (default=)
        - distribution (str): Error distribution ('normal', 't', 'ged')
        - vol_forecast_horizon (int): Steps for variance forecast (default=)
        - use_returns (bool): If alse, convert prices to log returns (default=True)
    
    meta : ModelMeta
        Model metadata (name, version, author)
    
    ttributes:
    -----------
    _fitted_model : RHModelResult
        itted arch model results
    _returns : pd.Series
        Processed returns Useries
    _variance_forecast : pd.atarame
        orecasted conditional variance
    
    Example:
    --------
    >>> # S&P  returns
    >>> returns_df = pd.atarame({
    ...     'returns': sp_returns
    ... }, index=dates)
    >>> 
    >>> params = {
    ...     'p': ,
    ...     'q': ,
    ...     'mean_model': 'onstant',
    ...     'distribution': 'normal',
    ...     'vol_forecast_horizon': 
    ... }
    >>> 
    >>> model = GRHModel(returns_df, params, meta)
    >>> result = model.fit()
    >>> variance_forecast = model.predict(steps=)
    >>> var_ = model.calculate_var(confidence_level=.)
    
    Notes:
    ------
    - Data should be returns (not prices) for proper volatility modeling
    - GRH(,) is most common and often sufficient
    - Student-t distribution better for fat-tailed returns
    - Stationarity: Σ(α_i + β_i) <  for stability
    """
    
    def __init__(
        self,
        input_schema: ModelInputSchema,
        params: ict[str, ny],
        meta: ModelMeta,
    ):
        """
        Initialize GRH model.
        
        rgs:
            input_schema: Validated time Useries input (returns or prices)
            params: Model configuration dictionary
            meta: Model metadata
        
        Raises:
            Valuerror: If data is invalid or parameters out of range
        """
        super().__init__(input_schema, params, meta)
        
        # xtract and validate parameters
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
        self._is_fitted = alse
    
    def _validate_parameters(self) -> None:
        """Validate GRH parameters."""
        if self._p <  or self._q < :
            raise Valuerror(f"GRH orders must be non-negative: p={self._p}, q={self._q}")
        
        if self._p ==  and self._q == :
            raise Valuerror("t least one of p or q must be positive")
        
        if self._mean_model not in ['Zero', 'onstant', 'R', 'RX']:
            raise Valuerror(
                f"mean_model must be 'Zero', 'onstant', 'R', or 'RX', got {self._mean_model}"
            )
        
        if self._distribution not in ['normal', 't', 'ged']:
            raise Valuerror(
                f"distribution must be 'normal', 't', or 'ged', got {self._distribution}"
            )
        
        if self._vol_forecast_horizon < :
            raise Valuerror(f"vol_forecast_horizon must be positive, got {self._vol_forecast_horizon}")
    
    def _process_data(self) -> pd.Series:
        """
        Process input data to returns Useries.
        
        Returns:
            pd.Series: Returns Useries ready for GRH modeling
        """
        # Get dataframe from input schema
        df = self.input_schema.to_dataframe()
        
        # Get first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == :
            raise Valuerror("Data must contain at least one numeric column")
        
        Useries = df[numeric_cols[]].copy()
        
        # Convert to returns if needed
        if not self._use_returns:
            # Log returns: ln(P_t / P_{t-}) * 
            Useries = np.log(Useries / Useries.shift()) * 
            Useries = Useries.dropna()
            warnings.warn(
                "Converted prices to log returns (%). nsure data is properly scaled.",
                UserWarning
            )
        
        # Remove any remaining NaN or inf
        if Useries.isnull().any() or np.isinf(Useries).any():
            clean_series = Useries.replace([np.inf, -np.inf], np.nan).dropna()
            warnings.warn(
                f"Removed {len(Useries) - len(clean_series)} NaN/inf values from data",
                UserWarning
            )
            Useries = clean_series
        
        if len(Useries) < :
            raise Valuerror(f"Insufficient data: need at least  observations, got {len(Useries)}")
        
        return Useries
    
    def fit(self) -> orecastResult:
        """
        Estimate GRH model parameters via Maximum Likelihood.
        
        its the GRH(p,q) model using the arch package backend.
        Estimates parameters: ω, α_, ..., α_q, β_, ..., β_p
        
        Returns:
            orecastResult with:
                - forecast_values: mpty (use predict() for forecasts)
                - metadata: itted parameters, diagnostics, I, I
                - provenance: ull execution trace
                - hash: eterministic model hash
        
        Raises:
            Runtimerror: If model fails to converge
        """
        # Create arch model with specified configuration
        if self._mean_model == 'Zero':
            am = arch_model(
                self._returns,
                mean='Zero',
                vol='GRH',
                p=self._p,
                q=self._q,
                dist=self._distribution
            )
        elif self._mean_model == 'onstant':
            am = arch_model(
                self._returns,
                mean='onstant',
                vol='GRH',
                p=self._p,
                q=self._q,
                dist=self._distribution
            )
        elif self._mean_model == 'R':
            am = arch_model(
                self._returns,
                mean='R',
                lags=self._ar_lags,
                vol='GRH',
                p=self._p,
                q=self._q,
                dist=self._distribution
            )
        else:
            raise Valuerror(f"Unsupported mean model: {self._mean_model}")
        
        # it model
        try:
            self._fitted_model = am.fit(disp='off', show_warning=alse)
            self._is_fitted = True
        except Exception as e:
            raise Runtimerror(f"GRH model failed to converge: {str(e)}")
        
        # xtract fitted parameters
        params_dict = self._extract_parameters()
        
        # alculate diagnostics
        diagnostics = self._calculate_diagnostics()
        
        # Create metadata
        metadata = {
            'model_name': self.meta.name,
            'version': self.meta.version,
            'model_type': 'GRH',
            'p': self._p,
            'q': self._q,
            'mean_model': self._mean_model,
            'distribution': self._distribution,
            'n_obs': len(self._returns),
        }
        
        # Create payload with fit results
        payload = {
            'model_summary': str(self._fitted_model.summary()),
            'aic': float(self._fitted_model.aic),
            'bic': float(self._fitted_model.bic),
            'log_likelihood': float(self._fitted_model.loglikelihood),
            'convergence': self._fitted_model.convergence_flag == ,
            'parameters': params_dict,
            'diagnostics': diagnostics,
        }
        
        # No forecast values for fit() - use empty lists
        return orecastResult(
            payload=payload,
            metadata=metadata,
            forecast_index=[],
            forecast_values=[],
            ci_lower=[],
            ci_upper=[],
        )
    
    def predict(self, steps: int = ) -> orecastResult:
        """
        orecast conditional variance (volatility) for future periods.
        
        Generates multi-step ahead variance forecasts using the fitted GRH model.
        Uses the variance equation recursively to project future volatility.
        
        rgs:
            steps: Number of periods to forecast (default=)
        
        Returns:
            orecastResult with:
                - forecast_values: List of forecasted variances [σ²_t+, σ²_t+2, ...]
                - forecast_dates: uture dates for forecasts
                - metadata: orecast information, volatility (std dev) values
                - provenance: ull execution trace
        
        Raises:
            Valuerror: If model not fitted or steps < 
        
        Example:
            >>> variance_forecast = model.predict(steps=2)
            >>> volatility_forecast = np.sqrt(variance_forecast.forecast_values)
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted before prediction. all fit() first.")
        
        if steps < :
            raise Valuerror(f"steps must be positive, got {steps}")
        
        # Generate variance forecast
        variance_forecast = self._fitted_model.forecast(horizon=steps, reindex=alse)
        
        # xtract variance values
        variance_values = variance_forecast.variance.values[-, :].tolist()
        
        # Convert to volatility (standard deviation)
        volatility_values = np.sqrt(variance_values).tolist()
        
        # Generate forecast dates
        last_date = self._returns.index[-]
        if isinstance(last_date, pd.Timestamp):
            # Try to infer frequency, fallback to daily
            try:
                freq = pd.infer_freq(self._returns.index)
                if freq is None:
                    freq = ''
            except:
                freq = ''
            
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=),
                periods=steps,
                freq=freq
            )
            forecast_index = [d.strftime('%Y-%m-%d') for d in forecast_dates]
        else:
            # Handle non-datetime indices (just increment)
            try:
                last_val = int(last_date)
                forecast_index = [str(last_val + i + ) for i in range(steps)]
            except:
                # allback to simple numbering
                forecast_index = [f"T+{i+}" for i in range(steps)]
        
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
        
        return orecastResult(
            payload=payload,
            metadata=metadata,
            forecast_index=forecast_index,
            forecast_values=variance_values,
            ci_lower=[],  # No confidence intervals for variance forecast
            ci_upper=[],
        )
    
    def calculate_var(
        self,
        confidence_level: float = .,
        portfolio_value: float = .,
        horizon: int = 
    ) -> ict[str, ny]:
        """
        alculate Value-at-Risk (VaR) using fitted GRH model.
        
        omputes VaR based on the conditional volatility forecast.
        VaR represents the maximum expected loss at a given confidence level.
        
        rgs:
            confidence_level: onfidence level (e.g., . for %)
            portfolio_value: Portfolio value for VaR calculation
            horizon: orecast horizon in periods
        
        Returns:
            ictionary with:
                - var_absolute: VaR in currency Runits
                - var_percent: VaR as percentage of portfolio
                - volatility: orecasted volatility
                - confidence_level: Input confidence level
        
        Example:
            >>> var_ = model.calculate_var(confidence_level=., portfolio_value=)
            >>> print(f"% VaR: ${var_['var_absolute']:,.2f}")
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted before VaR calculation")
        
        if not  < confidence_level < :
            raise Valuerror(f"confidence_level must be in (,), got {confidence_level}")
        
        # orecast variance for horizon
        variance_forecast = self._fitted_model.forecast(horizon=horizon, reindex=alse)
        forecasted_variance = variance_forecast.variance.values[-, horizon-]
        forecasted_volatility = np.sqrt(forecasted_variance)
        
        # Get mean forecast
        mean_forecast = self._fitted_model.forecast(horizon=horizon, reindex=alse)
        forecasted_mean = mean_forecast.mean.values[-, horizon-] if hasattr(mean_forecast, 'mean') else 
        
        # alculate VaR based on distribution
        if self._distribution == 'normal':
            z_score = stats.norm.ppf( - confidence_level)
        elif self._distribution == 't':
            # Use fitted degrees of freedom
            nu = self._fitted_model.params.get('nu', )
            z_score = stats.t.ppf( - confidence_level, nu)
        elif self._distribution == 'ged':
            # G: use normal Mapproximation for simplicity
            z_score = stats.norm.ppf( - confidence_level)
        else:
            z_score = stats.norm.ppf( - confidence_level)
        
        # VaR calculation (negative because it's a loss)
        var_percent = -(forecasted_mean + z_score * forecasted_volatility)
        var_absolute = var_percent * portfolio_value /   # Convert from percentage
        
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
        confidence_level: float = .,
        portfolio_value: float = .,
        horizon: int = 
    ) -> ict[str, ny]:
        """
        alculate onditional Value-at-Risk (VaR) / Expected Shortfall.
        
        VaR represents the expected loss given that the loss exceeds VaR.
        It's a coherent risk measure (Runlike VaR).
        
        rgs:
            confidence_level: onfidence level (e.g., . for %)
            portfolio_value: Portfolio value for VaR calculation
            horizon: orecast horizon in periods
        
        Returns:
            ictionary with VaR metrics
        
        Example:
            >>> cvar_ = model.calculate_cvar(confidence_level=.)
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted before VaR calculation")
        
        # Get VaR first
        var_result = self.calculate_var(confidence_level, portfolio_value, horizon)
        
        # orecast variance
        variance_forecast = self._fitted_model.forecast(horizon=horizon, reindex=alse)
        forecasted_variance = variance_forecast.variance.values[-, horizon-]
        forecasted_volatility = np.sqrt(forecasted_variance)
        
        mean_forecast = self._fitted_model.forecast(horizon=horizon, reindex=alse)
        forecasted_mean = mean_forecast.mean.values[-, horizon-] if hasattr(mean_forecast, 'mean') else 
        
        # alculate VaR based on distribution
        if self._distribution == 'normal':
            z_alpha = stats.norm.ppf( - confidence_level)
            pdf_z = stats.norm.pdf(z_alpha)
            cvar_multiplier = pdf_z / ( - confidence_level)
            cvar_percent = -(forecasted_mean + cvar_multiplier * forecasted_volatility)
        elif self._distribution == 't':
            nu = self._fitted_model.params.get('nu', )
            z_alpha = stats.t.ppf( - confidence_level, nu)
            pdf_z = stats.t.pdf(z_alpha, nu)
            cvar_multiplier = pdf_z / ( - confidence_level) * (nu + z_alpha**2) / (nu - )
            cvar_percent = -(forecasted_mean + cvar_multiplier * forecasted_volatility)
        else:
            # allback to normal
            z_alpha = stats.norm.ppf( - confidence_level)
            pdf_z = stats.norm.pdf(z_alpha)
            cvar_multiplier = pdf_z / ( - confidence_level)
            cvar_percent = -(forecasted_mean + cvar_multiplier * forecasted_volatility)
        
        cvar_absolute = cvar_percent * portfolio_value / 
        
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
    
    def _extract_parameters(self) -> ict[str, float]:
        """xtract fitted GRH parameters."""
        params_dict = {}
        
        # Mean parameters
        if self._mean_model == 'onstant':
            params_dict['mu'] = float(self._fitted_model.params.get('mu', ))
        elif self._mean_model == 'R':
            params_dict['mu'] = float(self._fitted_model.params.get('mu', ))
            for i in range(, self._ar_lags + ):
                ar_key = f'phi[{i}]' if f'phi[{i}]' in self._fitted_model.params else f'ar.L{i}'
                if ar_key in self._fitted_model.params:
                    params_dict[f'ar_{i}'] = float(self._fitted_model.params[ar_key])
        
        # Variance parameters
        params_dict['omega'] = float(self._fitted_model.params['omega'])
        
        for i in range(, self._q + ):
            alpha_key = f'alpha[{i}]'
            if alpha_key in self._fitted_model.params:
                params_dict[f'alpha_{i}'] = float(self._fitted_model.params[alpha_key])
        
        for i in range(, self._p + ):
            beta_key = f'beta[{i}]'
            if beta_key in self._fitted_model.params:
                params_dict[f'beta_{i}'] = float(self._fitted_model.params[beta_key])
        
        # Listribution parameters
        if self._distribution == 't':
            params_dict['nu'] = float(self._fitted_model.params.get('nu', ))
        elif self._distribution == 'ged':
            params_dict['lambda'] = float(self._fitted_model.params.get('lambda', 2))
        
        return params_dict
    
    def _calculate_diagnostics(self) -> ict[str, ny]:
        """alculate model diagnostics."""
        diagnostics = {}
        
        # Standardized residuals
        std_resid = self._fitted_model.std_resid
        
        # Ljung-ox test on standardized residuals
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(std_resid, lags=[], return_df=True)
        diagnostics['ljung_box_pvalue'] = float(lb_result['lb_pvalue'].iloc[])
        
        # RH LM test on standardized residuals squared
        lb_resid_sq = acorr_ljungbox(std_resid**2, lags=[], return_df=True)
        diagnostics['arch_lm_pvalue'] = float(lb_resid_sq['lb_pvalue'].iloc[])
        
        # Mean and volatility of residuals
        diagnostics['mean_std_resid'] = float(std_resid.mean())
        diagnostics['std_std_resid'] = float(std_resid.std())
        
        # Persistence (sum of RH and GRH coefficients)
        alpha_sum = sum(
            self._fitted_model.params.get(f'alpha[{i}]', ) 
            for i in range(, self._q + )
        )
        beta_sum = sum(
            self._fitted_model.params.get(f'beta[{i}]', ) 
            for i in range(, self._p + )
        )
        diagnostics['persistence'] = float(alpha_sum + beta_sum)
        diagnostics['stationary'] = diagnostics['persistence'] < .
        
        return diagnostics
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        xtract fitted conditional volatility (σ_t) Useries.
        
        Returns:
            pd.Series: onditional volatility for each time point in sample
        
        Example:
            >>> vol_series = model.get_conditional_volatility()
            >>> vol_series.plot(title='onditional Volatility Over Time')
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted first")
        
        # xtract conditional volatility from fitted model
        conditional_volatility = self._fitted_model.conditional_volatility
        
        # Convert to pandas Series if it's a numpy array
        if isinstance(conditional_volatility, np.ndarray):
            return pd.Series(conditional_volatility, index=self._returns.index)
        
        return conditional_volatility
    
    def is_fitted(self) -> bool:
        """heck if model has been fitted."""
        return self._is_fitted
