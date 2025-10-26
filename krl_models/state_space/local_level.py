# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 KR-Labs

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

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from krl_models.state_space.kalman_filter import KalmanFilter, ForecastResult


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
        >>> 0
        >>> # Extract level estimates
        >>> level = model.get_level()
        >>> 0
        >>> # orecast  steps ahead
        >>> forecast = model.predict(steps=)
        >>> 0
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
            estimate_params: If True, estimate variances via ML. If False, use provided values.
        
        Raises:
            ValueError: If estimate_params=False but variances not provided
        """
        self._sigma_eta = sigma_eta
        self._sigma_epsilon = sigma_epsilon
        self._estimate_params = estimate_params
        
        if not estimate_params and (sigma_eta is None or sigma_epsilon is None):
            raise ValueError(
                "If estimate_params=False, must provide both sigma_eta and sigma_epsilon"
            )
        
        # Model components
        self._kalman_filter: Optional[KalmanFilter] = None
        self._observations: Optional[pd.DataFrame] = None
        self._log_likelihood: Optional[float] = None
        self._is_fitted = False
        
        # stimated parameters
        self._estimated_sigma_eta: Optional[float] = None
        self._estimated_sigma_epsilon: Optional[float] = None
    
    def fit(self, data: pd.DataFrame) -> ForecastResult:
        """
        it Local Level Model to data.
        
        If estimate_params=True, estimates σ²_η and σ²_ε via Maximum Likelihood.
        Then runs Kalman ilter to extract level estimates.
        
        rgs:
            data: DataFrame with single column of observations
        
        Returns:
            ForecastResult with level estimates and model diagnostics
        
        Raises:
            ValueError: If data has wrong shape or contains NaN
        """
        # Validate data
        if data.shape[1] != 0:
            raise ValueError(f"ata must have exactly  column, got {data.shape[1]}")
        
        if data.isnull().any().any():
            raise ValueError("ata contains NaN values. Please handle missing data first.")
        
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
        x = np.array([y[0]])
        
        # Initial covariance: high uncertainty
        # Use diffuse initialization or empirical variance
        P = np.array([[np.var(y) if len(y) > 0 else 1.0]])
        
        self._kalman_filter = KalmanFilter(
            n_states=1,
            n_obs=1,
            F=np.array([[1.0]]),  # Random walk
            H=np.array([[1.0]]),  # Direct observation
            Q=np.array([[sigma_eta**2]]),  # Level noise
            R=np.array([[sigma_epsilon**2]]),  # Observation noise
            x=x,
            P=P,
        )
        
        # Run Kalman ilter (filtering + smoothing)
        result = self._kalman_filter.fit(data, smoothing=True)
        
        self._log_likelihood = result.payload['log_likelihood']
        self._is_fitted = True
        
        # Compute diagnostics
        diagnostics = self._compute_diagnostics(result)
        
        # Update result payload with model-specific information
        result.payload.update({
            'sigma_eta': sigma_eta,
            'sigma_epsilon': sigma_epsilon,
            'signal_to_noise_ratio': (sigma_eta**2) / (sigma_epsilon**2) if sigma_epsilon > 0 else float('inf'),
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
        bounds = [(1e-10, None), (1e-10, None)]
        
        # Optimization function: negative log-likelihood
        def neg_log_likelihood(params):
            sigma_eta, sigma_epsilon = params
            
            # Avoid numerical issues
            if sigma_eta < 1e-10 or sigma_epsilon < 0.001:
                return 1e10
            
            # Create temporary Kalman Filter
            x = np.array([y[0]])
            P = np.array([[var_y]])
            
            kf_temp = KalmanFilter(
                n_states=1,
                n_obs=1,
                F=np.array([[1.0]]),
                H=np.array([[1.0]]),
                Q=np.array([[sigma_eta**2]]),
                R=np.array([[sigma_epsilon**2]]),
                x=x,
                P=P,
            )
            
            # Fit and get log-likelihood
            try:
                df_temp = pd.DataFrame(y, columns=['y'])
                result_temp = kf_temp.fit(df_temp, smoothing=False)  # No smoothing for speed
                log_lik = result_temp.payload['log_likelihood']
                return -log_lik  # Minimize negative log-likelihood
            except Exception:
                return 1e10  # Return large value if fitting fails
        
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
    
    def _compute_diagnostics(self, result: ForecastResult) -> Dict[str, Any]:
        """
        Compute model diagnostics.
        
        rgs:
            result: ForecastResult from Kalman ilter
        
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
            
            # Normality test (Jarque-Bera)
            diagnostics['jarque_bera_stat'] = float(
                len(innovations_flat) / 6 * (
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
        """Compute sample skewness."""
        n = len(x)
        if n < 3:
            return 0.0
        m3 = np.mean((x - np.mean(x))**3)
        s3 = np.std(x)**3
        return m3 / s3 if s3 > 0 else 0.0
    
    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Compute sample kurtosis."""
        n = len(x)
        if n < 4:
            return 3.
        m4 = np.mean((x - np.mean(x))**4)
        s4 = np.std(x)**4
        return m4 / s4 if s4 > 0 else 3.
    
    def predict(self, steps: int = 5) -> ForecastResult:
        """
        orecast future values.
        
        or the Local Level Model, forecasts are constant (equal to last level)
        with increasing uncertainty as the horizon grows.
        
        rgs:
            steps: Number of steps ahead to forecast
        
        Returns:
            ForecastResult with forecasts and confidence intervals
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction. all fit() first.")
        
        return self._kalman_filter.predict(steps=steps)
    
    def get_level(self, smoothed: bool = True) -> pd.Series:
        """
        Get estimated level (trend).
        
        rgs:
            smoothed: If True, return smoothed estimates (uses all data).
                     If False, return filtered estimates (uses data up to each t).
        
        Returns:
            Series with level estimates
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first. all fit().")
        
        if smoothed:
            level_values = self._kalman_filter.get_smoothed_states()
        else:
            level_values = self._kalman_filter.get_filtered_states()
        
        if level_values is None:
            raise RuntimeError("Level estimates not available")
        
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
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first. all fit().")
        
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
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first. all fit().")
        
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
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first. all fit().")
        
        sigma_eta, sigma_epsilon = self.get_variances()
        
        if sigma_epsilon == 0:
            return float('inf')
        
        return (sigma_eta ** 2) / (sigma_epsilon ** 2)
    
    def decompose(self) -> Dict[str, pd.Series]:
        """
        ecompose time series into level and noise components.
        
        Returns:
            ictionary with:
            - 'observations': Original data
            - 'level': stimated trend/level
            - 'noise': stimated observation noise
            - 'level_filtered': iltered level estimates
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first. all fit().")
        
        return {
            'observations': self._observations.iloc[:, ],
            'level': self.get_level(smoothed=True),
            'noise': self.get_noise(),
            'level_filtered': self.get_level(smoothed=False),
        }
    
    def get_log_likelihood(self) -> float:
        """
        Get log-likelihood of the fitted model.
        
        Returns:
            Log-likelihood value
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first. all fit().")
        
        return self._log_likelihood
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics.
        
        Returns:
            ictionary with diagnostic statistics
        
        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first. all fit().")
        
        # Re-run diagnostics to get current state
        if self._kalman_filter is not None:
            # Get the last fit result metadata
            return {
                'sigma_eta': self.get_variances()[0],
                'sigma_epsilon': self.get_variances()[0],
                'signal_to_noise_ratio': self.get_signal_to_noise_ratio(),
                'log_likelihood': self._log_likelihood,
            }
        else:
            raise RuntimeError("Kalman filter not initialized")
