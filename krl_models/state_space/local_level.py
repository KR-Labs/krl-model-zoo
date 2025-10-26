# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC, a subsidiary of Sudiata Giddasira, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Local Level Model Implementation.

This module implements the Local Level (random walk plus noise) state space model
with maximum likelihood parameter estimation and trend extraction capabilities.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult
from krl_models.state_space.kalman_filter import KalmanFilter


class LocalLevelModel(BaseModel):
    """Local Level State Space Model.
    
    The local level model assumes the observed series consists of a random walk
    level component plus white noise:
    
        Level equation:  mu_t = mu_{t-1} + eta_t,  eta_t ~ N(0, sigma_eta^2)
        Obs equation:    y_t = mu_t + epsilon_t,   epsilon_t ~ N(0, sigma_epsilon^2)
    
    Parameters are estimated via maximum likelihood using the Kalman filter.
    
    Attributes:
        sigma_eta: Standard deviation of level innovations
        sigma_epsilon: Standard deviation of observation noise
        signal_to_noise: Ratio sigma_eta / sigma_epsilon
    """
    
    def __init__(self):
        """Initialize Local Level Model."""
        super().__init__()
        
        self.sigma_eta: Optional[float] = None
        self.sigma_epsilon: Optional[float] = None
        self.signal_to_noise: Optional[float] = None
        
        self._kf: Optional[KalmanFilter] = None
        self._y: Optional[np.ndarray] = None
        self._log_likelihood: Optional[float] = None
        self._level: Optional[np.ndarray] = None
    
    def fit(
        self,
        y: np.ndarray,
        method: str = "mle",
        initial_params: Optional[Tuple[float, float]] = None,
    ) -> "LocalLevelModel":
        """Fit the local level model.
        
        Args:
            y: Observed time series (T,)
            method: Parameter estimation method ("mle" or "fixed")
            initial_params: Initial (sigma_eta, sigma_epsilon) for optimization
            
        Returns:
            self (for method chaining)
        """
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        
        self._y = y.copy()
        T = len(y)
        
        if method == "mle":
            # Maximum likelihood estimation
            if initial_params is None:
                # Initialize with fraction of sample variance
                var_y = np.var(y)
                initial_params = (np.sqrt(0.1 * var_y), np.sqrt(0.9 * var_y))
            
            result = minimize(
                self._neg_log_likelihood,
                x0=initial_params,
                args=(y,),
                method="L-BFGS-B",
                bounds=[(1e-6, None), (1e-6, None)],
            )
            
            self.sigma_eta, self.sigma_epsilon = result.x
            self._log_likelihood = -result.fun
            
        elif method == "fixed":
            if initial_params is None:
                raise ValueError("initial_params required for fixed method")
            self.sigma_eta, self.sigma_epsilon = initial_params
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.signal_to_noise = self.sigma_eta / self.sigma_epsilon
        
        # Run Kalman filter with estimated parameters
        self._kf = self._build_kalman_filter(y)
        self._kf.fit(y.reshape(-1, 1))
        
        # Extract smoothed level
        smoothed_states = self._kf.smooth()
        self._level = np.array([state.x[0] for state in smoothed_states])
        
        return self
    
    def _neg_log_likelihood(
        self,
        params: Tuple[float, float],
        y: np.ndarray,
    ) -> float:
        """Compute negative log-likelihood for optimization.
        
        Args:
            params: (sigma_eta, sigma_epsilon)
            y: Observed time series
            
        Returns:
            Negative log-likelihood
        """
        sigma_eta, sigma_epsilon = params
        
        if sigma_eta <= 0 or sigma_epsilon <= 0:
            return 1e10
        
        try:
            kf = self._build_kalman_filter_from_params(y, sigma_eta, sigma_epsilon)
            kf.fit(y.reshape(-1, 1))
            return -kf.log_likelihood
        except:
            return 1e10
    
    def _build_kalman_filter(self, y: np.ndarray) -> KalmanFilter:
        """Build Kalman filter with current parameters."""
        return self._build_kalman_filter_from_params(
            y, self.sigma_eta, self.sigma_epsilon
        )
    
    def _build_kalman_filter_from_params(
        self,
        y: np.ndarray,
        sigma_eta: float,
        sigma_epsilon: float,
    ) -> KalmanFilter:
        """Build Kalman filter from parameters.
        
        Args:
            y: Observed time series
            sigma_eta: Level innovation std dev
            sigma_epsilon: Observation noise std dev
            
        Returns:
            Configured Kalman filter
        """
        T = len(y)
        
        # State space matrices
        F = np.array([[1.0]])  # mu_t = mu_{t-1}
        H = np.array([[1.0]])  # y_t = mu_t
        Q = np.array([[sigma_eta ** 2]])
        R = np.array([[sigma_epsilon ** 2]])
        
        # Initialize with first observation
        x0 = np.array([y[0]])
        P0 = np.array([[sigma_epsilon ** 2]])
        
        return KalmanFilter(
            n_states=1,
            n_obs=1,
            F=F,
            H=H,
            Q=Q,
            R=R,
            x=x0,
            P=P0,
        )
    
    def forecast(self, steps: int) -> ForecastResult:
        """Generate forecasts.
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            ForecastResult with forecasts and confidence intervals
        """
        if self._kf is None:
            raise RuntimeError("Model must be fitted before forecasting")
        
        # Use Kalman filter for forecasting
        kf_result = self._kf.forecast(steps)
        
        # Flatten to 1D
        point_forecast = kf_result.point_forecast.flatten()
        lower_bound = kf_result.lower_bound.flatten()
        upper_bound = kf_result.upper_bound.flatten()
        
        return ForecastResult(
            model_name="LocalLevelModel",
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            metadata={
                "sigma_eta": self.sigma_eta,
                "sigma_epsilon": self.sigma_epsilon,
                "signal_to_noise": self.signal_to_noise,
                "log_likelihood": self._log_likelihood,
                "forecast_steps": steps,
            },
        )
    
    def decompose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose series into level and irregular components.
        
        Returns:
            Tuple of (level, irregular) arrays
        """
        if self._level is None:
            raise RuntimeError("Model must be fitted before decomposition")
        
        irregular = self._y - self._level
        
        return self._level, irregular
    
    @property
    def level(self) -> Optional[np.ndarray]:
        """Get extracted level component."""
        return self._level
    
    @property
    def log_likelihood(self) -> Optional[float]:
        """Get log-likelihood."""
        return self._log_likelihood
