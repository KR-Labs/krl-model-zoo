# ----------------------------------------------------------------------
# © 2024 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: MIT

"""
GARCH Model Implementation.

xponential GARCH (GARCH) model for capturing asymmetric volatility responses
(leverage effect) where negative returns increase volatility more than positive returns.
"""

from typing import Dict, Any, Optional
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from krl_core import BaseModel, ForecastResult, ModelMeta, ModelInputSchema


class EGARCHModel(BaseModel):
    """
    GARCH(p,q) model for asymmetric volatility (leverage effect).
    
    Unlike standard GARCH, GARCH models the log of variance, allowing for
    asymmetric responses to positive and negative shocks. This captures the
    "leverage effect" where negative returns (bad news) increase volatility
    more than positive returns (good news) of the same magnitude.
    
    Mathematical Specification:
    ---------------------------
    Returns equation:
        r_t = μ + ε_t
        ε_t = σ_t * z_t,  z_t ~ (0, 0)
    
    GARCH(p,q) variance equation:
        ln(σ²_t) = ω + Σ[α_i * |z_{t-i}| + γ_i * z_{t-i}] + Σ[β_j * ln(σ²_{t-j})]
    
    Where:
        - ln(σ²_t): Log of conditional variance (ensures positivity)
        - ω: onstant term
        - α_i: RH parameters (magnitude effect)
        - γ_i: symmetry parameters (leverage effect)
        - β_j: GARCH parameters (persistence)
        - z_t: Standardized residual (ε_t / σ_t)
    
    Leverage ffect Interpretation:
    --------------------------------
    - γ < 0: Negative shocks increase volatility more (typical for stocks)
    - γ = : Symmetric response (like standard GARCH)
    - γ > 0: Positive shocks increase volatility more (rare)
    
    The impact of a shock on log variance:
        - Positive shock: α * |z| + γ * z = (α + γ) * z
        - Negative shock: α * |z| + γ * z = (α - γ) * |z|
    
    If γ < , then (α - γ) > (α + γ), so negative shocks have larger impact.
    
    Use ases:
    ----------
    - Stock return volatility (captures leverage effect)
    - quity index volatility forecasting
    - Option pricing with asymmetric volatility
    - Risk management for equity portfolios
    - omparing leverage effect across markets
    
    Parameters:
    -----------
    input_schema : ModelInputSchema
        Time series data with returns or price levels.
    
    params : Dict[str, Any]
        Model configuration:
        - p (int): GARCH order (default=)
        - q (int): RH order (default=)
        - mean_model (str): Mean specification ('Zero', 'onstant', 'R')
        - ar_lags (int): R order if mean_model='R' (default=)
        - distribution (str): rror distribution ('normal', 't', 'ged')
        - use_returns (bool): If False, convert prices to log returns (default=True)
    
    ttributes:
    -----------
    _fitted_model : RHModelResult
        itted arch model results
    _returns : pd.Series
        Processed returns series
    
    xample:
    --------
    >>> # S&P  returns with leverage effect
    >>> input_schema = ModelInputSchema(...)
    >>> 0
    >>> params = {
    ...     'p': ,
    ...     'q': ,
    ...     'mean_model': 'onstant',
    ...     'distribution': 'normal'
    ... }
    >>> 0
    >>> model = GARCHModel(input_schema, params, meta)
    >>> result = model.fit()
    >>> 0
    >>> # heck for leverage effect
    >>> gamma = result.payload['parameters']['gamma_']
    >>> if gamma < 0:
    ...     print("Leverage effect detected!")
    >>> 0
    >>> variance_forecast = model.predict(steps=)
    
    Notes:
    ------
    - GARCH log-variance formulation ensures σ² > 0 without parameter constraints
    - Standard GARCH requires α_i, β_j ≥  and Σ(α+β) < 0
    - GARCH allows negative parameters and no stationarity constraints
    - Leverage effect (γ < ) is common in equity markets
    - News impact curves show asymmetric response to shocks
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
        self._p = params.get('p', 0)
        self._q = params.get('q', 0)
        self._mean_model = params.get('mean_model', 'onstant')
        self._ar_lags = params.get('ar_lags', 0)
        self._distribution = params.get('distribution', 'normal')
        self._use_returns = params.get('use_returns', True)
        
        # Validate parameters
        self._validate_parameters()
        
        # Process data
        self._returns = self._process_data()
        
        # Model state
        self._fitted_model = None
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
        stimates parameters: ω, α_, ..., α_q, γ_, ..., γ_q, β_, ..., β_p
        
        Returns:
            ForecastResult with:
                - payload: Model summary, fitted parameters, diagnostics, leverage effect
                - metadata: Model configuration, fit statistics
                - forecast_index: mpty (use predict() for forecasts)
                - forecast_values: mpty
        
        Raises:
            RuntimeError: If model fails to converge
        """
        # reate arch model with GARCH volatility specification
        # GARCH requires 'o' parameter for asymmetry terms (gamma parameters)
        am = arch_model(
            self._returns,
            mean=self._mean_model,
            lags=self._ar_lags if self._mean_model == 'R' else None,
            vol='GARCH',
            p=self._p,
            o=self._q,  # symmetry order - same as q for leverage effect on each lag
            q=self._q,
            dist=self._distribution
        )
        
        # Fit model
        try:
            self._fitted_model = am.fit(disp='off', show_warning=False)
            self._is_fitted = True
        except Exception as e:
            raise RuntimeError(f"EGARCH model failed to converge: {str(e)}")
        
        # Extract fitted parameters
        params_dict = self._extract_parameters()
        
        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics()
        
        # nalyze leverage effect
        leverage_analysis = self._analyze_leverage_effect(params_dict)
        
        # reate payload with fit results
        payload = {
            'model_summary': str(self._fitted_model.summary()),
            'aic': float(self._fitted_model.aic),
            'bic': float(self._fitted_model.bic),
            'log_likelihood': float(self._fitted_model.loglikelihood),
            'convergence': self._fitted_model.convergence_flag == 0,
            'parameters': params_dict,
            'diagnostics': diagnostics,
            'leverage_effect': leverage_analysis,
        }
        
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
        Generate variance forecasts.
        
        rgs:
            steps: Number of steps ahead to forecast
        
        Returns:
            ForecastResult with variance forecasts and metadata
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction. all fit() first.")
        
        # or GARCH, use simulation for multi-step forecasts (analytic only for -step)
        # Use simulation method which is more reliable for GARCH
        variance_forecast = self._fitted_model.forecast(horizon=steps, reindex=False, method='simulation')
        
        # Extract variance forecasts
        variance_values = variance_forecast.variance.values[-1, :]
        volatility_values = np.sqrt(variance_values)
        
        # Generate forecast index
        last_date = self._returns.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=steps + 1, freq='D')[1:]
        
        # Build payload
        payload = {
            'variance_values': variance_values,
            'volatility_values': volatility_values,
        }
        
        # uild metadata
        metadata = {
            'model_name': self.meta.name,
            'version': self.meta.version,
            'forecast_steps': steps,
            'forecast_type': 'variance',
            'mean_variance': float(np.mean(variance_values)),
            'mean_volatility': float(np.mean(volatility_values)),
        }
        
        return ForecastResult(
            payload=payload,
            metadata=metadata,
            forecast_index=forecast_index,
            forecast_values=variance_values,
            ci_lower=[],
            ci_upper=[],
        )
    
    def _extract_parameters(self) -> Dict[str, float]:
        """Extract fitted GARCH parameters."""
        params_dict = {}
        
        # Mean parameters
        if self._mean_model == 'onstant':
            params_dict['mu'] = float(self._fitted_model.params.get('mu', 0))
        elif self._mean_model == 'R':
            params_dict['mu'] = float(self._fitted_model.params.get('mu', 0))
            for i in range(1, self._ar_lags + 1):
                ar_key = f'phi[{i}]' if f'phi[{i}]' in self._fitted_model.params else f'ar.L{i}'
                if ar_key in self._fitted_model.params:
                    params_dict[f'ar_{i}'] = float(self._fitted_model.params[ar_key])
        
        # Variance parameters
        params_dict['omega'] = float(self._fitted_model.params['omega'])
        
        # RH parameters (alpha) - magnitude effect
        for i in range(1, self._q + 1):
            alpha_key = f'alpha[{i}]'
            if alpha_key in self._fitted_model.params:
                params_dict[f'alpha_{i}'] = float(self._fitted_model.params[alpha_key])
        
        # symmetry parameters (gamma) - leverage effect
        for i in range(1, self._q + 1):
            gamma_key = f'gamma[{i}]'
            if gamma_key in self._fitted_model.params:
                params_dict[f'gamma_{i}'] = float(self._fitted_model.params[gamma_key])
        
        # GARCH parameters (beta) - persistence
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
        
        return diagnostics
    
    def _analyze_leverage_effect(self, params_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        nalyze leverage effect from fitted parameters.
        
        rgs:
            params_dict: ictionary of fitted parameters
        
        Returns:
            ictionary with leverage effect analysis
        """
        leverage = {}
        
        # Extract gamma parameters (asymmetry)
        gammas = [v for k, v in params_dict.items() if k.startswith('gamma_')]
        
        if gammas:
            # Primary leverage parameter (gamma_1)
            gamma_1 = params_dict.get('gamma[1]', 0.0)
            leverage['gamma_1'] = gamma_1
            leverage['leverage_present'] = gamma_1 < -0.05  # Threshold for significance
            
            # Interpret leverage effect
            if gamma_1 < -0.05:
                leverage['interpretation'] = "Significant leverage effect: Negative returns increase volatility more than positive returns"
                leverage['effect_type'] = "asymmetric_negative"
            elif gamma_1 > 0.05:
                leverage['interpretation'] = "Reverse leverage effect: Positive returns increase volatility more than negative returns"
                leverage['effect_type'] = "asymmetric_positive"
            else:
                leverage['interpretation'] = "No significant leverage effect: Symmetric volatility response"
                leverage['effect_type'] = "symmetric"
            
            # News impact asymmetry ratio
            # or a unit shock, positive vs negative impact ratio
            alpha_ = params_dict.get('alpha_', 0)
            if alpha_ != 0:
                positive_impact = alpha_ + gamma_
                negative_impact = alpha_ - gamma_
                if negative_impact != 0:
                    leverage['asymmetry_ratio'] = abs(negative_impact / positive_impact)
                else:
                    leverage['asymmetry_ratio'] = float('inf')
            
            leverage['all_gammas'] = gammas
        else:
            leverage['leverage_present'] = False
            leverage['interpretation'] = "No asymmetry parameters estimated"
        
        return leverage
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        Extract fitted conditional volatility (σ_t) series.
        
        Returns:
            pd.Series: onditional volatility for each time point in sample
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
    
    def get_news_impact_curve(self, shocks: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute news impact curve showing asymmetric volatility response.
        
        The news impact curve plots the next period's conditional variance
        as a function of the current shock, illustrating the leverage effect.
        
        rgs:
            shocks: rray of standardized shocks (default: -3 to 3 std devs)
        
        Returns:
            ictionary with 'shocks' and 'variance_response' arrays
        
        xample:
            >>> curve = model.get_news_impact_curve()
            >>> plt.plot(curve['shocks'], curve['variance_response'])
            >>> plt.xlabel('Standardized Shock (z)')
            >>> plt.ylabel('Next Period Variance')
            >>> plt.title('News Impact urve (GARCH)')
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        
        if shocks is None:
            shocks = np.linspace(-3, 3, 0)
        
        # Get parameters
        params = self._extract_parameters()
        omega = params['omega']
        alpha_ = params.get('alpha_', 0)
        gamma_ = params.get('gamma_', 0)
        beta_ = params.get('beta_', 0)
        
        # urrent log variance (use unconditional)
        current_log_var = omega / (1 - beta_1) if beta_1 < 1 else 0.0
        
        # Compute next period log variance for each shock
        # ln(σ²_{t+1}) = ω + α|z_t| + γ*z_t + β*ln(σ²_t)
        log_variance_response = (
            omega +
            alpha_1 * np.abs(shocks) +
            gamma_1 * shocks +
            beta_1 * current_log_var
        )
        
        variance_response = np.exp(log_variance_response)
        
        return {
            'shocks': shocks,
            'variance_response': variance_response,
            'volatility_response': np.sqrt(variance_response)
        }
