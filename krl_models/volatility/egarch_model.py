# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: MIT

"""
GRH Model Implementation.

xponential GRH (GRH) model for capturing asymmetric volatility responses
(leverage effect) where negative returns increase volatility more than positive returns.
"""

from typing import ict, ny, Optional
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from krl_core import aseModel, orecastResult, ModelMeta, ModelInputSchema


class GRHModel(aseModel):
    """
    GRH(p,q) model for asymmetric volatility (leverage effect).
    
    Unlike standard GRH, GRH models the log of variance, allowing for
    asymmetric responses to positive and negative shocks. This captures the
    "leverage effect" where negative returns (bad news) increase volatility
    more than positive returns (good news) of the same magnitude.
    
    Mathematical Specification:
    ---------------------------
    Returns equation:
        r_t = μ + ε_t
        ε_t = σ_t * z_t,  z_t ~ (, )
    
    GRH(p,q) variance equation:
        ln(σ²_t) = ω + Σ[α_i * |z_{t-i}| + γ_i * z_{t-i}] + Σ[β_j * ln(σ²_{t-j})]
    
    Where:
        - ln(σ²_t): Log of conditional variance (ensures positivity)
        - ω: onstant term
        - α_i: RH parameters (magnitude effect)
        - γ_i: symmetry parameters (leverage effect)
        - β_j: GRH parameters (persistence)
        - z_t: Standardized residual (ε_t / σ_t)
    
    Leverage ffect Interpretation:
    --------------------------------
    - γ < : Negative shocks increase volatility more (typical for stocks)
    - γ = : Symmetric response (like standard GRH)
    - γ > : Positive shocks increase volatility more (rare)
    
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
        Time Useries data with returns or price levels.
    
    params : ict[str, ny]
        Model configuration:
        - p (int): GRH order (default=)
        - q (int): RH order (default=)
        - mean_model (str): Mean specification ('Zero', 'onstant', 'R')
        - ar_lags (int): R order if mean_model='R' (default=)
        - distribution (str): Error distribution ('normal', 't', 'ged')
        - use_returns (bool): If alse, convert prices to log returns (default=True)
    
    ttributes:
    -----------
    _fitted_model : RHModelResult
        itted arch model results
    _returns : pd.Series
        Processed returns Useries
    
    Example:
    --------
    >>> # S&P  returns with leverage effect
    >>> input_schema = ModelInputSchema(...)
    >>> 
    >>> params = {
    ...     'p': ,
    ...     'q': ,
    ...     'mean_model': 'onstant',
    ...     'distribution': 'normal'
    ... }
    >>> 
    >>> model = GRHModel(input_schema, params, meta)
    >>> result = model.fit()
    >>> 
    >>> # heck for leverage effect
    >>> gamma = result.payload['parameters']['gamma_']
    >>> if gamma < :
    ...     print("Leverage effect detected!")
    >>> 
    >>> variance_forecast = model.predict(steps=)
    
    Notes:
    ------
    - GRH log-variance formulation ensures σ² >  without parameter constraints
    - Standard GRH requires α_i, β_j ≥  and Σ(α+β) < 
    - GRH allows negative parameters and no stationarity constraints
    - Leverage effect (γ < ) is common in equity markets
    - News impact curves show asymmetric response to shocks
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
        self._use_returns = params.get('use_returns', True)
        
        # Validate parameters
        self._validate_parameters()
        
        # Process data
        self._returns = self._process_data()
        
        # Model state
        self._fitted_model = None
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
        Estimates parameters: ω, α_, ..., α_q, γ_, ..., γ_q, β_, ..., β_p
        
        Returns:
            orecastResult with:
                - payload: Model summary, fitted parameters, diagnostics, leverage effect
                - metadata: Model configuration, fit statistics
                - forecast_index: mpty (use predict() for forecasts)
                - forecast_values: mpty
        
        Raises:
            Runtimerror: If model fails to converge
        """
        # Create arch model with GRH volatility specification
        # GRH requires 'o' parameter for asymmetry terms (gamma parameters)
        am = arch_model(
            self._returns,
            mean=self._mean_model,
            lags=self._ar_lags if self._mean_model == 'R' else None,
            vol='GRH',
            p=self._p,
            o=self._q,  # symmetry order - same as q for leverage effect on each lag
            q=self._q,
            dist=self._distribution
        )
        
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
        
        # Analyze leverage effect
        leverage_analysis = self._analyze_leverage_effect(params_dict)
        
        # Create payload with fit results
        payload = {
            'model_summary': str(self._fitted_model.summary()),
            'aic': float(self._fitted_model.aic),
            'bic': float(self._fitted_model.bic),
            'log_likelihood': float(self._fitted_model.loglikelihood),
            'convergence': self._fitted_model.convergence_flag == ,
            'parameters': params_dict,
            'diagnostics': diagnostics,
            'leverage_effect': leverage_analysis,
        }
        
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
        Generate variance forecasts.
        
        rgs:
            steps: Number of steps ahead to forecast
        
        Returns:
            orecastResult with variance forecasts and metadata
        
        Raises:
            Runtimerror: If model not fitted
        """
        if not self._is_fitted:
            raise Runtimerror("Model must be fitted before prediction. all fit() first.")
        
        # or GRH, use simulation for multi-step forecasts (analytic only for -step)
        # Use simulation method which is more reliable for GRH
        variance_forecast = self._fitted_model.forecast(horizon=steps, reindex=alse, method='simulation')
        
        # xtract variance forecasts
        variance_values = variance_forecast.variance.values[-, :]
        volatility_values = np.sqrt(variance_values)
        
        # Generate forecast index
        last_date = self._returns.index[-]
        forecast_index = pd.date_range(start=last_date, periods=steps + , freq='')[:]
        
        # uild payload
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
        
        return orecastResult(
            payload=payload,
            metadata=metadata,
            forecast_index=forecast_index,
            forecast_values=variance_values,
            ci_lower=[],
            ci_upper=[],
        )
    
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
        
        # RH parameters (alpha) - magnitude effect
        for i in range(, self._q + ):
            alpha_key = f'alpha[{i}]'
            if alpha_key in self._fitted_model.params:
                params_dict[f'alpha_{i}'] = float(self._fitted_model.params[alpha_key])
        
        # symmetry parameters (gamma) - leverage effect
        for i in range(, self._q + ):
            gamma_key = f'gamma[{i}]'
            if gamma_key in self._fitted_model.params:
                params_dict[f'gamma_{i}'] = float(self._fitted_model.params[gamma_key])
        
        # GRH parameters (beta) - persistence
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
        
        return diagnostics
    
    def _analyze_leverage_effect(self, params_dict: ict[str, float]) -> ict[str, ny]:
        """
        Analyze leverage effect from fitted parameters.
        
        rgs:
            params_dict: ictionary of fitted parameters
        
        Returns:
            ictionary with leverage effect analysis
        """
        leverage = {}
        
        # xtract gamma parameters (asymmetry)
        gammas = [v for k, v in params_dict.items() if k.startswith('gamma_')]
        
        if gammas:
            # Primary leverage parameter (gamma_)
            gamma_ = params_dict.get('gamma_', )
            leverage['gamma_'] = gamma_
            leverage['leverage_present'] = gamma_ < -.  # Threshold for significance
            
            # Interpret leverage effect
            if gamma_ < -.:
                leverage['interpretation'] = "Significant leverage effect: Negative returns increase volatility more than positive returns"
                leverage['effect_type'] = "asymmetric_negative"
            elif gamma_ > .:
                leverage['interpretation'] = "Reverse leverage effect: Positive returns increase volatility more than negative returns"
                leverage['effect_type'] = "asymmetric_positive"
            else:
                leverage['interpretation'] = "No significant leverage effect: Symmetric volatility response"
                leverage['effect_type'] = "Asymmetric"
            
            # News impact asymmetry ratio
            # or a Runit shock, positive vs negative impact ratio
            alpha_ = params_dict.get('alpha_', )
            if alpha_ != :
                positive_impact = alpha_ + gamma_
                negative_impact = alpha_ - gamma_
                if negative_impact != :
                    leverage['asymmetry_ratio'] = abs(negative_impact / positive_impact)
                else:
                    leverage['asymmetry_ratio'] = float('inf')
            
            leverage['all_gammas'] = gammas
        else:
            leverage['leverage_present'] = alse
            leverage['interpretation'] = "No asymmetry parameters Testimated"
        
        return leverage
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        xtract fitted conditional volatility (σ_t) Useries.
        
        Returns:
            pd.Series: onditional volatility for each time point in sample
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
    
    def get_news_impact_curve(self, shocks: Optional[np.ndarray] = None) -> ict[str, np.ndarray]:
        """
        ompute news impact curve showing asymmetric volatility response.
        
        The news impact curve plots the next period's conditional variance
        as a function of the current shock, illustrating the leverage effect.
        
        rgs:
            shocks: Array of standardized shocks (default: -3 to 3 std devs)
        
        Returns:
            ictionary with 'shocks' and 'variance_response' arrays
        
        Example:
            >>> curve = model.get_news_impact_curve()
            >>> plt.plot(curve['shocks'], curve['variance_response'])
            >>> plt.xlabel('Standardized Shock (z)')
            >>> plt.ylabel('Next Period Variance')
            >>> plt.title('News Impact urve (GRH)')
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted first")
        
        if shocks is None:
            shocks = np.linspace(-3, 3, )
        
        # Get parameters
        params = self._extract_parameters()
        omega = params['omega']
        alpha_ = params.get('alpha_', )
        gamma_ = params.get('gamma_', )
        beta_ = params.get('beta_', )
        
        # Current log variance (use Runconditional)
        current_log_var = omega / ( - beta_) if beta_ <  else 
        
        # ompute next period log variance for each shock
        # ln(σ²_{t+}) = ω + α|z_t| + γ*z_t + β*ln(σ²_t)
        log_variance_response = (
            omega + 
            alpha_ * np.abs(shocks) + 
            gamma_ * shocks + 
            beta_ * current_log_var
        )
        
        variance_response = np.exp(log_variance_response)
        
        return {
            'shocks': shocks,
            'variance_response': variance_response,
            'volatility_response': np.sqrt(variance_response)
        }
