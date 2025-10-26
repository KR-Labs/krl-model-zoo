# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC, a subsidiary of Sudiata Giddasira, Inc.
# SPDX-License-Identifier: Apache-2.0

"""EGARCH Model - Exponential GARCH with leverage effects."""

from typing import Optional
import numpy as np
from scipy.optimize import minimize

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult


class EGARCHModel(BaseModel):
    """EGARCH model for asymmetric volatility.
    
    Log-variance specification:
        log(sigma_t^2) = omega + sum(alpha_i * abs(z_{t-i})) + sum(gamma_i * z_{t-i}) 
                         + sum(beta_j * log(sigma_{t-j}^2))
    
    where z_t = epsilon_t / sigma_t is the standardized residual.
    gamma captures leverage effects (negative shocks increase volatility more).
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        super().__init__()
        self.p = p
        self.q = q
        self.omega: Optional[float] = None
        self.alpha: Optional[np.ndarray] = None
        self.gamma: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.mu: Optional[float] = None
        self._conditional_volatility: Optional[np.ndarray] = None
        self._log_likelihood: Optional[float] = None
    
    def fit(self, y: np.ndarray) -> "EGARCHModel":
        """Fit EGARCH model."""
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        
        self.mu = np.mean(y)
        var_y = np.var(y)
        
        # Initial parameters: omega, alpha, gamma, beta
        params_init = np.concatenate([
            [self.mu, -0.1],  # mu, omega
            np.full(self.q, 0.1),  # alpha
            np.full(self.q, -0.05),  # gamma (leverage)
            np.full(self.p, 0.95),  # beta
        ])
        
        result = minimize(
            self._neg_log_likelihood,
            x0=params_init,
            args=(y,),
            method='L-BFGS-B',
            bounds=self._get_bounds(),
        )
        
        self._extract_parameters(result.x)
        self._log_likelihood = -result.fun
        self._conditional_volatility = self._compute_volatility(y)
        
        return self
    
    def _get_bounds(self) -> list:
        bounds = [
            (-np.inf, np.inf),  # mu
            (-np.inf, np.inf),  # omega
        ]
        # alpha: positive
        for _ in range(self.q):
            bounds.append((0, np.inf))
        # gamma: can be negative (leverage)
        for _ in range(self.q):
            bounds.append((-1, 1))
        # beta: near 1 for persistence
        for _ in range(self.p):
            bounds.append((0, 0.999))
        return bounds
    
    def _extract_parameters(self, params: np.ndarray) -> None:
        idx = 0
        self.mu = params[idx]; idx += 1
        self.omega = params[idx]; idx += 1
        self.alpha = params[idx:idx + self.q]; idx += self.q
        self.gamma = params[idx:idx + self.q]; idx += self.q
        self.beta = params[idx:idx + self.p]
    
    def _neg_log_likelihood(self, params: np.ndarray, y: np.ndarray) -> float:
        try:
            idx = 0
            mu = params[idx]; idx += 1
            omega = params[idx]; idx += 1
            alpha = params[idx:idx + self.q]; idx += self.q
            gamma = params[idx:idx + self.q]; idx += self.q
            beta = params[idx:idx + self.p]
            
            residuals = y - mu
            T = len(residuals)
            log_sigma2 = np.zeros(T)
            
            # Initialize
            log_sigma2[0] = np.log(np.var(residuals))
            
            for t in range(1, T):
                log_sigma2[t] = omega
                
                for i in range(min(self.q, t)):
                    z_lag = residuals[t-i-1] / np.exp(0.5 * log_sigma2[t-i-1])
                    log_sigma2[t] += alpha[i] * np.abs(z_lag) + gamma[i] * z_lag
                
                for j in range(min(self.p, t)):
                    log_sigma2[t] += beta[j] * log_sigma2[t-j-1]
            
            sigma2 = np.exp(log_sigma2)
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + residuals**2 / sigma2)
            return -ll
        except:
            return 1e10
    
    def _compute_volatility(self, y: np.ndarray) -> np.ndarray:
        residuals = y - self.mu
        T = len(residuals)
        log_sigma2 = np.zeros(T)
        log_sigma2[0] = np.log(np.var(residuals))
        
        for t in range(1, T):
            log_sigma2[t] = self.omega
            for i in range(min(self.q, t)):
                z_lag = residuals[t-i-1] / np.exp(0.5 * log_sigma2[t-i-1])
                log_sigma2[t] += self.alpha[i] * np.abs(z_lag) + self.gamma[i] * z_lag
            for j in range(min(self.p, t)):
                log_sigma2[t] += self.beta[j] * log_sigma2[t-j-1]
        
        return np.exp(0.5 * log_sigma2)
    
    def forecast(self, steps: int = 1) -> ForecastResult:
        if self._conditional_volatility is None:
            raise RuntimeError("Model must be fitted first")
        
        # Simplified forecast: assume z_t -> E[|z|] and z_t -> 0
        last_log_sigma2 = np.log(self._conditional_volatility[-1]**2)
        forecasts = []
        
        for h in range(steps):
            log_sigma2_h = self.omega
            # Use E[|z|] = sqrt(2/pi) for standard normal
            for i in range(self.q):
                log_sigma2_h += self.alpha[i] * np.sqrt(2 / np.pi)
            if h == 0:
                log_sigma2_h += self.beta[0] * last_log_sigma2
            else:
                log_sigma2_h += self.beta[0] * np.log(forecasts[-1]**2)
            
            forecasts.append(np.exp(0.5 * log_sigma2_h))
        
        forecast_array = np.array(forecasts)
        std_error = 0.1 * forecast_array
        
        return ForecastResult(
            model_name=f"EGARCH({self.p},{self.q})",
            point_forecast=forecast_array,
            lower_bound=forecast_array - 1.96 * std_error,
            upper_bound=forecast_array + 1.96 * std_error,
            metadata={
                "omega": float(self.omega),
                "alpha": self.alpha.tolist(),
                "gamma": self.gamma.tolist(),
                "beta": self.beta.tolist(),
                "log_likelihood": self._log_likelihood,
            },
        )
    
    @property
    def conditional_volatility(self) -> Optional[np.ndarray]:
        return self._conditional_volatility
    
    @property
    def log_likelihood(self) -> Optional[float]:
        return self._log_likelihood
