# SPX-License-Identifier: pache-2.
# opyright (c) 22 KR-Labs

"""
Local Level Model Implementation

The Local Level Model is the simplest structural time series model, also known as
the "random walk plus noise" model. It decomposes a time series into a stochastic
trend (level) and observation noise.

Model Specification:
    Level quation:        μ_t = μ_{t-} + η_t,  η_t ~ N(, σ²_η)
    Observation quation:  y_t = μ_t + ε_t,      ε_t ~ N(, σ²_ε)

Where:
    μ_t: Unobserved level (trend) at time t
    y_t: Observed value at time t
    σ²_η: Level noise variance (trend variability)
    σ²_ε: Observation noise variance (measurement error)

The model can be represented in state space form and estimated using the Kalman ilter.
It's useful for:
    - Trend extraction from noisy data
    - Smoothing time series
    - Short-term forecasting
    - Signal-noise decomposition

uthor: KR Labs
ate: October 224
"""

from typing import ict, ny, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from krl_models.state_space.kalman_filter import Kalmanilter, orecastResult


class LocalLevelModel:
    """
    Local Level Model (Random Walk Plus Noise).
    
    This is a structural time series model that decomposes a time series into:
    -  stochastic level (random walk trend)
    - White noise observations
    
    The model is:
        μ_t = μ_{t-} + η_t,  η_t ~ N(, σ²_η)  [Level equation]
        y_t = μ_t + ε_t,      ε_t ~ N(, σ²_ε)  [Observation equation]
    
    Special ases:
    - If σ²_η = : Model reduces to constant level plus noise
    - If σ²_ε = : Model is pure random walk (observations = level)
    - Signal-to-noise ratio q = σ²_η / σ²_ε determines smoothness
    
    The model uses the Kalman ilter for:
    - Parameter estimation (ML via optimization)
    - Level extraction (filtered and smoothed estimates)
    - orecasting
    
    xample:
        >>> # it model with automatic parameter estimation
        >>> model = LocalLevelModel()
        >>> result = model.fit(data)
        >>> 
        >>> # xtract level estimates
        >>> level = model.get_level()
        >>> 
        >>> # orecast  steps ahead
        >>> forecast = model.predict(steps=)
        >>> 
        >>> # Get estimated variances
        >>> sigma_eta, sigma_epsilon = model.get_variances()
    """
    
    def __init__(
        self,
        sigma_eta: Optional[float] = None,
        sigma_epsilon: Optional[float] = None,
        estimate_params: bool = True,
    ):
        """
        Initialize Local Level Model.
        
        rgs:
            sigma_eta: Level noise standard deviation. If None, will be estimated.
            sigma_epsilon: Observation noise standard deviation. If None, will be estimated.
            estimate_params: If True, estimate variances via ML. If alse, use provided values.
        
        Raises:
            Valuerror: If estimate_params=alse but variances not provided
        """
        self._sigma_eta = sigma_eta
        self._sigma_epsilon = sigma_epsilon
        self._estimate_params = estimate_params
        
        if not estimate_params and (sigma_eta is None or sigma_epsilon is None):
            raise Valuerror(
                "If estimate_params=alse, must provide both sigma_eta and sigma_epsilon"
            )
        
        # Model components
        self._kalman_filter: Optional[Kalmanilter] = None
        self._observations: Optional[pd.atarame] = None
        self._log_likelihood: Optional[float] = None
        self._is_fitted = alse
        
        # stimated parameters
        self._estimated_sigma_eta: Optional[float] = None
        self._estimated_sigma_epsilon: Optional[float] = None
    
    def fit(self, data: pd.atarame) -> orecastResult:
        """
        it Local Level Model to data.
        
        If estimate_params=True, estimates σ²_η and σ²_ε via Maximum Likelihood.
        Then runs Kalman ilter to extract level estimates.
        
        rgs:
            data: atarame with single column of observations
        
        Returns:
            orecastResult with level estimates and model diagnostics
        
        Raises:
            Valuerror: If data has wrong shape or contains NaN
        """
        # Validate data
        if data.shape[] != :
            raise Valuerror(f"ata must have exactly  column, got {data.shape[]}")
        
        if data.isnull().any().any():
            raise Valuerror("ata contains NaN values. Please handle missing data first.")
        
        self._observations = data.copy()
        y = data.values.flatten()
        
        # stimate parameters if needed
        if self._estimate_params:
            self._estimate_variances(y)
            sigma_eta = self._estimated_sigma_eta
            sigma_epsilon = self._estimated_sigma_epsilon
        else:
            sigma_eta = self._sigma_eta
            sigma_epsilon = self._sigma_epsilon
        
        # Set up Kalman ilter for local level model
        # State space form:
        #    = []  (level is random walk)
        #   H = []  (observation = level)
        #   Q = [σ²_η]  (level noise)
        #   R = [σ²_ε]  (observation noise)
        
        # Initial state: use first observation
        x = np.array([y[]])
        
        # Initial covariance: high uncertainty
        # Use diffuse initialization or empirical variance
        P = np.array([[np.var(y) if len(y) >  else .]])
        
        self._kalman_filter = Kalmanilter(
            n_states=,
            n_obs=,
            =np.array([[.]]),  # Random walk
            H=np.array([[.]]),  # irect observation
            Q=np.array([[sigma_eta**2]]),  # Level noise
            R=np.array([[sigma_epsilon**2]]),  # Observation noise
            x=x,
            P=P,
        )
        
        # Run Kalman ilter (filtering + smoothing)
        result = self._kalman_filter.fit(data, smoothing=True)
        
        self._log_likelihood = result.payload['log_likelihood']
        self._is_fitted = True
        
        # ompute diagnostics
        diagnostics = self._compute_diagnostics(result)
        
        # Update result payload with model-specific information
        result.payload.update({
            'sigma_eta': sigma_eta,
            'sigma_epsilon': sigma_epsilon,
            'signal_to_noise_ratio': (sigma_eta**2) / (sigma_epsilon**2) if sigma_epsilon >  else float('inf'),
            'diagnostics': diagnostics,
        })
        
        # Update metadata
        result.metadata.update({
            'model_type': 'LocalLevel',
            'estimated_params': self._estimate_params,
        })
        
        return result
    
    def _estimate_variances(self, y: np.ndarray) -> None:
        """
        stimate σ²_η and σ²_ε via Maximum Likelihood.
        
        Uses numerical optimization to maximize the log-likelihood function.
        The likelihood is computed via the Kalman ilter innovations.
        
        rgs:
            y: Observations array
        """
        # Initial parameter guess
        # Use variance decomposition: Var(y) ≈ σ²_η + σ²_ε
        # Start with equal split (signal-to-noise ratio = )
        var_y = np.var(y)
        initial_sigma_eta = np.sqrt(var_y / 2)
        initial_sigma_epsilon = np.sqrt(var_y / 2)
        
        # Parameter bounds: both must be positive, not too small
        bounds = [(e-, None), (e-, None)]
        
        # Optimization function: negative log-likelihood
        def neg_log_likelihood(params):
            sigma_eta, sigma_epsilon = params
            
            # void numerical issues
            if sigma_eta < e- or sigma_epsilon < e-:
                return e
            
            # reate temporary Kalman ilter
            x = np.array([y[]])
            P = np.array([[var_y]])
            
            kf_temp = Kalmanilter(
                n_states=,
                n_obs=,
                =np.array([[.]]),
                H=np.array([[.]]),
                Q=np.array([[sigma_eta**2]]),
                R=np.array([[sigma_epsilon**2]]),
                x=x,
                P=P,
            )
            
            # it and get log-likelihood
            try:
                df_temp = pd.atarame(y, columns=['y'])
                result_temp = kf_temp.fit(df_temp, smoothing=alse)  # No smoothing for speed
                log_lik = result_temp.payload['log_likelihood']
                return -log_lik  # Minimize negative log-likelihood
            except xception:
                return e  # Return large value if fitting fails
        
        # Optimize
        result = minimize(
            neg_log_likelihood,
            x=[initial_sigma_eta, initial_sigma_epsilon],
            method='L-GS-',
            bounds=bounds,
        )
        
        if result.success:
            self._estimated_sigma_eta, self._estimated_sigma_epsilon = result.x
        else:
            # allback to initial guess if optimization fails
            print("Warning: ML optimization did not converge. Using initial guess.")
            self._estimated_sigma_eta = initial_sigma_eta
            self._estimated_sigma_epsilon = initial_sigma_epsilon
    
    def _compute_diagnostics(self, result: orecastResult) -> ict[str, ny]:
        """
        ompute model diagnostics.
        
        rgs:
            result: orecastResult from Kalman ilter
        
        Returns:
            ictionary with diagnostic statistics
        """
        diagnostics = {}
        
        # Get innovations (one-step-ahead forecast errors)
        innovations = self._kalman_filter.get_innovations()
        
        if innovations is not None:
            innovations_flat = innovations.flatten()
            
            # Innovation statistics
            diagnostics['innovation_mean'] = float(np.mean(innovations_flat))
            diagnostics['innovation_std'] = float(np.std(innovations_flat))
            
            # Standardized innovations (should be ~ N(,))
            std_innovations = innovations_flat / np.std(innovations_flat)
            diagnostics['innovation_skewness'] = float(self._skewness(std_innovations))
            diagnostics['innovation_kurtosis'] = float(self._kurtosis(std_innovations))
            
            # Normality test (Jarque-era)
            diagnostics['jarque_bera_stat'] = float(
                len(innovations_flat) /  * (
                    diagnostics['innovation_skewness']**2 +
                    (diagnostics['innovation_kurtosis'] - 3)**2 / 4
                )
            )
        
        # Model fit statistics
        T = len(self._observations)
        n_params = 2  # σ²_η and σ²_ε
        
        diagnostics['n_observations'] = T
        diagnostics['n_params'] = n_params
        diagnostics['aic'] = result.metadata['aic']
        diagnostics['bic'] = result.metadata['bic']
        
        return diagnostics
    
    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        """ompute sample skewness."""
        n = len(x)
        if n < 3:
            return .
        m3 = np.mean((x - np.mean(x))**3)
        s3 = np.std(x)**3
        return m3 / s3 if s3 >  else .
    
    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """ompute sample kurtosis."""
        n = len(x)
        if n < 4:
            return 3.
        m4 = np.mean((x - np.mean(x))**4)
        s4 = np.std(x)**4
        return m4 / s4 if s4 >  else 3.
    
    def predict(self, steps: int = ) -> orecastResult:
        """
        orecast future values.
        
        or the Local Level Model, forecasts are constant (equal to last level)
        with increasing uncertainty as the horizon grows.
        
        rgs:
            steps: Number of steps ahead to forecast
        
        Returns:
            orecastResult with forecasts and confidence intervals
        
        Raises:
            Runtimerror: If model not fitted
        """
        if not self._is_fitted:
            raise Runtimerror("Model must be fitted before prediction. all fit() first.")
        
        return self._kalman_filter.predict(steps=steps)
    
    def get_level(self, smoothed: bool = True) -> pd.Series:
        """
        Get estimated level (trend).
        
        rgs:
            smoothed: If True, return smoothed estimates (uses all data).
                     If alse, return filtered estimates (uses data up to each t).
        
        Returns:
            Series with level estimates
        
        Raises:
            Runtimerror: If model not fitted
        """
        if not self._is_fitted:
            raise Runtimerror("Model must be fitted first. all fit().")
        
        if smoothed:
            level_values = self._kalman_filter.get_smoothed_states()
        else:
            level_values = self._kalman_filter.get_filtered_states()
        
        if level_values is None:
            raise Runtimerror("Level estimates not available")
        
        return pd.Series(
            level_values.flatten(),
            index=self._observations.index,
            name='level'
        )
    
    def get_noise(self) -> pd.Series:
        """
        Get estimated observation noise (residuals).
        
        omputed as: ε_t = y_t - μ_t (observation minus level)
        
        Returns:
            Series with noise estimates
        
        Raises:
            Runtimerror: If model not fitted
        """
        if not self._is_fitted:
            raise Runtimerror("Model must be fitted first. all fit().")
        
        level = self.get_level(smoothed=True)
        observations = self._observations.iloc[:, ]
        noise = observations - level
        noise.name = 'noise'
        return noise
    
    def get_variances(self) -> Tuple[float, float]:
        """
        Get estimated or provided variance parameters.
        
        Returns:
            Tuple of (σ_η, σ_ε) - standard deviations
        
        Raises:
            Runtimerror: If model not fitted
        """
        if not self._is_fitted:
            raise Runtimerror("Model must be fitted first. all fit().")
        
        if self._estimate_params:
            return self._estimated_sigma_eta, self._estimated_sigma_epsilon
        else:
            return self._sigma_eta, self._sigma_epsilon
    
    def get_signal_to_noise_ratio(self) -> float:
        """
        Get signal-to-noise ratio q = σ²_η / σ²_ε.
        
        Interpretation:
        - q → : Smooth trend (low level variability)
        - q → ∞: Noisy trend (high level variability)
        - q = : qual level and observation noise
        
        Returns:
            Signal-to-noise ratio
        
        Raises:
            Runtimerror: If model not fitted
        """
        if not self._is_fitted:
            raise Runtimerror("Model must be fitted first. all fit().")
        
        sigma_eta, sigma_epsilon = self.get_variances()
        
        if sigma_epsilon == :
            return float('inf')
        
        return (sigma_eta ** 2) / (sigma_epsilon ** 2)
    
    def decompose(self) -> ict[str, pd.Series]:
        """
        ecompose time series into level and noise components.
        
        Returns:
            ictionary with:
            - 'observations': Original data
            - 'level': stimated trend/level
            - 'noise': stimated observation noise
            - 'level_filtered': iltered level estimates
        
        Raises:
            Runtimerror: If model not fitted
        """
        if not self._is_fitted:
            raise Runtimerror("Model must be fitted first. all fit().")
        
        return {
            'observations': self._observations.iloc[:, ],
            'level': self.get_level(smoothed=True),
            'noise': self.get_noise(),
            'level_filtered': self.get_level(smoothed=alse),
        }
    
    def get_log_likelihood(self) -> float:
        """
        Get log-likelihood of the fitted model.
        
        Returns:
            Log-likelihood value
        
        Raises:
            Runtimerror: If model not fitted
        """
        if not self._is_fitted:
            raise Runtimerror("Model must be fitted first. all fit().")
        
        return self._log_likelihood
    
    def get_diagnostics(self) -> ict[str, ny]:
        """
        Get model diagnostics.
        
        Returns:
            ictionary with diagnostic statistics
        
        Raises:
            Runtimerror: If model not fitted
        """
        if not self._is_fitted:
            raise Runtimerror("Model must be fitted first. all fit().")
        
        # Re-run diagnostics to get current state
        if self._kalman_filter is not None:
            # Get the last fit result metadata
            return {
                'sigma_eta': self.get_variances()[],
                'sigma_epsilon': self.get_variances()[],
                'signal_to_noise_ratio': self.get_signal_to_noise_ratio(),
                'log_likelihood': self._log_likelihood,
            }
        else:
            raise Runtimerror("Kalman filter not initialized")
