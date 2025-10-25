# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: MIT

"""
GJR-GRH Model Implementation.

Glosten-Jagannathan-Runkle GRH (GJR-GRH) model for capturing asymmetric
volatility through threshold effects where negative returns trigger
additional volatility increases.
"""

from typing import ict, ny, Optional
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from krl_core import aseModel, orecastResult, ModelMeta, ModelInputSchema


class GJRGRHModel(aseModel):
    """
    GJR-GRH(p,o,q) model for threshold asymmetric volatility.
    
    GJR-GRH extends standard GRH by adding threshold effects that allow
    negative shocks to have different impacts than positive shocks. This
    captures asymmetric volatility common in financial returns where bad news
    increases volatility more than good news.
    
    Mathematical Specification:
    ---------------------------
    Returns equation:
        r_t = μ + ε_t
        ε_t = σ_t * z_t,  z_t ~ (, )
    
    GJR-GRH(p,o,q) variance equation:
        σ²_t = ω + Σ[α_i * ε²_{t-i}] + Σ[γ_j * I_{t-j} * ε²_{t-j}] + Σ[β_k * σ²_{t-k}]
    
    Where:
        - σ²_t: onditional variance
        - ω: onstant term (>)
        - α_i: Symmetric RH parameters (≥)
        - γ_j: symmetric threshold parameters (≥)
        - β_k: GRH parameters (≥)
        - I_t: Indicator function =  if ε_t <  (negative shock),  otherwise
        - p: Number of GRH terms
        - o: Number of asymmetric (threshold) terms
        - q: Number of RH terms
    
    symmetric Response:
    --------------------
    Impact of a shock on variance:
        - Positive shock (ε_t > ): α * ε²_t
        - Negative shock (ε_t < ): (α + γ) * ε²_t
    
    If γ > , negative shocks have (α + γ) impact vs α for positive shocks.
    This creates a "threshold" or "step" response at zero.
    
    omparison with GRH:
    -----------------------
    - GJR-GRH: Threshold/step response, variance in levels
    - GRH: Smooth exponential response, log variance
    - GJR-GRH: Requires parameter constraints (α, γ, β ≥ )
    - GRH: No parameter sign restrictions
    - GJR-GRH: asier interpretation (α + γ for negative)
    - GRH: More flexible functional form
    
    Use ases:
    ----------
    - quity return volatility modeling
    - Option pricing with leverage effect
    - Risk management (VaR/VaR with asymmetry)
    - Volatility forecasting for long positions
    - omparing asymmetric effects across assets
    
    Parameters:
    -----------
    input_schema : ModelInputSchema
        Time series data with returns or price levels.
    
    params : ict[str, ny]
        Model configuration:
        - p (int): GRH order (default=)
        - o (int): symmetric order (default=)
        - q (int): RH order (default=)
        - mean_model (str): Mean specification ('Zero', 'onstant', 'R')
        - ar_lags (int): R order if mean_model='R' (default=)
        - distribution (str): rror distribution ('normal', 't', 'ged')
        - use_returns (bool): If alse, convert prices to log returns (default=True)
    
    ttributes:
    -----------
    _fitted_model : RHModelResult
        itted arch model results
    _returns : pd.Series
        Processed returns series
    
    xample:
    --------
    >>> # quity returns with threshold leverage effect
    >>> input_schema = ModelInputSchema(...)
    >>> 
    >>> params = {
    ...     'p': ,
    ...     'o': ,
    ...     'q': ,
    ...     'mean_model': 'onstant',
    ...     'distribution': 'normal'
    ... }
    >>> 
    >>> model = GJRGRHModel(input_schema, params, meta)
    >>> result = model.fit()
    >>> 
    >>> # heck for threshold effect
    >>> gamma = result.payload['parameters']['gamma_']
    >>> alpha = result.payload['parameters']['alpha_']
    >>> negative_impact = alpha + gamma
    >>> print(f"Positive shock impact: {alpha:.4f}")
    >>> print(f"Negative shock impact: {negative_impact:.4f}")
    >>> 
    >>> variance_forecast = model.predict(steps=)
    
    Notes:
    ------
    - Standard specification: GJR-GRH(,,) with p=o=q=
    - Stationarity: α + .*γ + β <  (different from standard GRH)
    - The "." factor comes from [I_t] = . assuming symmetry of z_t
    - GJR-GRH nests standard GRH when γ = 
    - Threshold effect at zero creates discontinuity in news impact curve
    """
    
    def __init__(
        self,
        input_schema: ModelInputSchema,
        params: ict[str, ny],
        meta: ModelMeta,
    ):
        """
        Initialize GJR-GRH model.
        
        rgs:
            input_schema: Validated time series input (returns or prices)
            params: Model configuration dictionary
            meta: Model metadata
        
        Raises:
            Valuerror: If data is invalid or parameters out of range
        """
        super().__init__(input_schema, params, meta)
        
        # xtract and validate parameters
        self._p = params.get('p', )
        self._o = params.get('o', )
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
        """Validate GJR-GRH parameters."""
        if self._p <  or self._o <  or self._q < :
            raise Valuerror(
                f"GJR-GRH orders must be non-negative: p={self._p}, o={self._o}, q={self._q}"
            )
        
        if self._p ==  and self._o ==  and self._q == :
            raise Valuerror("t least one of p, o, or q must be positive")
        
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
        Process input data to returns series.
        
        Returns:
            pd.Series: Returns series ready for GJR-GRH modeling
        """
        # Get dataframe from input schema
        df = self.input_schema.to_dataframe()
        
        # Get first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == :
            raise Valuerror("ata must contain at least one numeric column")
        
        series = df[numeric_cols[]].copy()
        
        # onvert to returns if needed
        if not self._use_returns:
            # Log returns: ln(P_t / P_{t-}) * 
            series = np.log(series / series.shift()) * 
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
        
        if len(series) < :
            raise Valuerror(f"Insufficient data: need at least  observations, got {len(series)}")
        
        return series
    
    def fit(self) -> orecastResult:
        """
        stimate GJR-GRH model parameters via Maximum Likelihood.
        
        its the GJR-GRH(p,o,q) model using the arch package backend.
        stimates parameters: ω, α_, ..., α_q, γ_, ..., γ_o, β_, ..., β_p
        
        Returns:
            orecastResult with:
                - payload: Model summary, fitted parameters, diagnostics, asymmetry analysis
                - metadata: Model configuration, fit statistics
                - forecast_index: mpty (use predict() for forecasts)
                - forecast_values: mpty
        
        Raises:
            Runtimerror: If model fails to converge
        """
        # reate arch model with GJR-GRH (TRH in arch package) specification
        # Note: arch package calls it TRH (Threshold RH)
        am = arch_model(
            self._returns,
            mean=self._mean_model,
            lags=self._ar_lags if self._mean_model == 'R' else None,
            vol='GRH',  # Will use power parameter to get GJR
            p=self._p,
            o=self._o,
            q=self._q,
            dist=self._distribution
        )
        
        # it model
        try:
            self._fitted_model = am.fit(disp='off', show_warning=alse)
            self._is_fitted = True
        except xception as e:
            raise Runtimerror(f"GJR-GRH model failed to converge: {str(e)}")
        
        # xtract fitted parameters
        params_dict = self._extract_parameters()
        
        # alculate diagnostics
        diagnostics = self._calculate_diagnostics()
        
        # nalyze threshold asymmetry
        asymmetry_analysis = self._analyze_threshold_effect(params_dict)
        
        # reate payload with fit results
        payload = {
            'model_summary': str(self._fitted_model.summary()),
            'aic': float(self._fitted_model.aic),
            'bic': float(self._fitted_model.bic),
            'log_likelihood': float(self._fitted_model.loglikelihood),
            'convergence': self._fitted_model.convergence_flag == ,
            'parameters': params_dict,
            'diagnostics': diagnostics,
            'asymmetry': asymmetry_analysis,
        }
        
        # reate metadata
        metadata = {
            'model_name': self.meta.name,
            'version': self.meta.version,
            'model_type': 'GJR-GRH',
            'p': self._p,
            'o': self._o,
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
        orecast conditional variance (volatility) for future periods.
        
        Generates multi-step ahead variance forecasts using the fitted GJR-GRH model.
        
        rgs:
            steps: Number of periods to forecast (default=)
        
        Returns:
            orecastResult with variance and volatility forecasts
        
        Raises:
            Valuerror: If model not fitted or steps < 
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted before prediction. all fit() first.")
        
        if steps < :
            raise Valuerror(f"steps must be positive, got {steps}")
        
        # Generate variance forecast
        variance_forecast = self._fitted_model.forecast(horizon=steps, reindex=alse)
        
        # xtract variance values
        variance_values = variance_forecast.variance.values[-, :].tolist()
        
        # onvert to volatility (standard deviation)
        volatility_values = np.sqrt(variance_values).tolist()
        
        # Generate forecast dates
        last_date = self._returns.index[-]
        if isinstance(last_date, pd.Timestamp):
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
            try:
                last_val = int(last_date)
                forecast_index = [str(last_val + i + ) for i in range(steps)]
            except:
                forecast_index = [f"T+{i+}" for i in range(steps)]
        
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
        """xtract fitted GJR-GRH parameters."""
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
        
        # RH parameters (alpha) - symmetric effect
        for i in range(, self._q + ):
            alpha_key = f'alpha[{i}]'
            if alpha_key in self._fitted_model.params:
                params_dict[f'alpha_{i}'] = float(self._fitted_model.params[alpha_key])
        
        # symmetry/threshold parameters (gamma) - additional effect for negative shocks
        for i in range(, self._o + ):
            gamma_key = f'gamma[{i}]'
            if gamma_key in self._fitted_model.params:
                params_dict[f'gamma_{i}'] = float(self._fitted_model.params[gamma_key])
        
        # GRH parameters (beta) - persistence
        for i in range(, self._p + ):
            beta_key = f'beta[{i}]'
            if beta_key in self._fitted_model.params:
                params_dict[f'beta_{i}'] = float(self._fitted_model.params[beta_key])
        
        # istribution parameters
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
        
        # Persistence calculation (different from standard GRH)
        # or GJR: persistence ≈ α + .*γ + β
        return diagnostics
    
    def _analyze_threshold_effect(self, params_dict: ict[str, float]) -> ict[str, ny]:
        """
        nalyze threshold asymmetry from fitted parameters.
        
        rgs:
            params_dict: ictionary of fitted parameters
        
        Returns:
            ictionary with threshold effect analysis
        """
        asymmetry = {}
        
        # xtract alpha (symmetric) and gamma (asymmetric) parameters
        alpha_ = params_dict.get('alpha_', )
        gamma_ = params_dict.get('gamma_', )
        beta_ = params_dict.get('beta_', )
        
        # Impact of shocks
        positive_impact = alpha_
        negative_impact = alpha_ + gamma_
        
        asymmetry['alpha_'] = alpha_
        asymmetry['gamma_'] = gamma_
        asymmetry['positive_shock_impact'] = positive_impact
        asymmetry['negative_shock_impact'] = negative_impact
        
        # heck for significant threshold effect
        asymmetry['threshold_present'] = gamma_ > .
        
        # Interpret threshold effect
        if gamma_ > .:
            ratio = negative_impact / positive_impact if positive_impact >  else float('inf')
            asymmetry['interpretation'] = (
                f"Significant threshold effect: Negative shocks have {ratio:.2f}x "
                f"the impact of positive shocks"
            )
            asymmetry['effect_type'] = "threshold_asymmetric"
            asymmetry['impact_ratio'] = ratio
        else:
            asymmetry['interpretation'] = "No significant threshold effect: Symmetric response"
            asymmetry['effect_type'] = "symmetric"
            asymmetry['impact_ratio'] = .
        
        # Persistence calculation (α + .*γ + β)
        persistence = alpha_ + . * gamma_ + beta_
        asymmetry['persistence'] = persistence
        asymmetry['stationary'] = persistence < .
        
        return asymmetry
    
    def get_conditional_volatility(self) -> pd.Series:
        """
        xtract fitted conditional volatility (σ_t) series.
        
        Returns:
            pd.Series: onditional volatility for each time point in sample
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted first")
        
        # xtract conditional volatility from fitted model
        conditional_volatility = self._fitted_model.conditional_volatility
        
        # onvert to pandas Series if it's a numpy array
        if isinstance(conditional_volatility, np.ndarray):
            return pd.Series(conditional_volatility, index=self._returns.index)
        
        return conditional_volatility
    
    def is_fitted(self) -> bool:
        """heck if model has been fitted."""
        return self._is_fitted
    
    def get_news_impact_curve(self, shocks: Optional[np.ndarray] = None) -> ict[str, np.ndarray]:
        """
        ompute news impact curve showing threshold asymmetric response.
        
        The news impact curve for GJR-GRH shows a "step" or discontinuity
        at zero, where negative shocks trigger additional volatility increases.
        
        rgs:
            shocks: rray of shocks in units of current volatility (default: -3 to 3)
        
        Returns:
            ictionary with 'shocks' and 'variance_response' arrays
        
        xample:
            >>> curve = model.get_news_impact_curve()
            >>> plt.plot(curve['shocks'], curve['variance_response'])
            >>> plt.axvline(x=, color='r', linestyle='--', label='Threshold')
            >>> plt.xlabel('Shock (ε_t)')
            >>> plt.ylabel('Next Period Variance')
            >>> plt.title('News Impact urve (GJR-GRH)')
            >>> plt.legend()
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted first")
        
        if shocks is None:
            # Use unconditional volatility to scale shocks
            uncond_vol = np.sqrt(self._returns.var())
            shocks = np.linspace(-3 * uncond_vol, 3 * uncond_vol, )
        
        # Get parameters
        params = self._extract_parameters()
        omega = params['omega']
        alpha_ = params.get('alpha_', )
        gamma_ = params.get('gamma_', )
        beta_ = params.get('beta_', )
        
        # urrent variance (use unconditional)
        alpha = alpha_
        gamma = gamma_
        beta = beta_
        
        # Unconditional variance: ω / ( - α - .*γ - β)
        if (alpha + . * gamma + beta) < :
            current_var = omega / ( - alpha - . * gamma - beta)
        else:
            current_var = self._returns.var()
        
        # ompute next period variance for each shock
        # σ²_{t+} = ω + α*ε²_t + γ*I(ε_t<)*ε²_t + β*σ²_t
        variance_response = np.zeros_like(shocks)
        
        for i, shock in enumerate(shocks):
            indicator = . if shock <  else .
            variance_response[i] = (
                omega + 
                alpha * shock**2 + 
                gamma * indicator * shock**2 + 
                beta * current_var
            )
        
        return {
            'shocks': shocks,
            'variance_response': variance_response,
            'volatility_response': np.sqrt(variance_response)
        }
