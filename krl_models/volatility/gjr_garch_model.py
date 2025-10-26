# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC, a subsidiary of Sudiata Giddasira, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GJR-GARCH Model - Threshold GARCH with asymmetric effects."""

from typing import Optional
import numpy as np
from scipy.optimize import minimize

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult


class GJRGARCHModel(BaseModel):
    """GJR-GARCH model with threshold effects.
    
    Variance equation:
        sigma_t^2 = omega + sum(alpha_i * epsilon_{t-i}^2) 
                    + sum(gamma_i * I_{t-i} * epsilon_{t-i}^2)
                    + sum(beta_j * sigma_{t-j}^2)
    
    where I_t = 1 if epsilon_t < 0 (negative shock), 0 otherwise.
    gamma > 0 means negative shocks increase volatility more (leverage effect).
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
    
    def fit(self, y: np.ndarray) -> "GJRGARCHModel":
        """Fit GJR-GARCH model."""
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        
        self.mu = np.mean(y)
        var_y = np.var(y)
        
        params_init = np.concatenate([
            [self.mu, 0.01 * var_y],  # mu, omega
            np.full(self.q, 0.05),  # alpha
            np.full(self.q, 0.1),   # gamma (leverage)
            np.full(self.p, 0.85),  # beta
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
            (1e-6, None),       # omega
        ]
        for _ in range(self.q):
            bounds.append((1e-6, 1))  # alpha
        for _ in range(self.q):
            bounds.append((0, 1))  # gamma (leverage)
        for _ in range(self.p):
            bounds.append((1e-6, 1))  # beta
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
            
            # Stationarity check
            if np.sum(alpha) + 0.5 * np.sum(gamma) + np.sum(beta) >= 1:
                return 1e10
            
            residuals = y - mu
            T = len(residuals)
            sigma2 = np.zeros(T)
            
            # Initialize
            uncond_var = omega / (1 - np.sum(alpha) - 0.5 * np.sum(gamma) - np.sum(beta))
            sigma2[:max(self.p, self.q)] = uncond_var
            
            for t in range(max(self.p, self.q), T):
                sigma2[t] = omega
                
                for i in range(self.q):
                    if t - i - 1 >= 0:
                        eps_sq = residuals[t - i - 1]**2
                        sigma2[t] += alpha[i] * eps_sq
                        
                        # Threshold effect
                        if residuals[t - i - 1] < 0:
                            sigma2[t] += gamma[i] * eps_sq
                
                for j in range(self.p):
                    if t - j - 1 >= 0:
                        sigma2[t] += beta[j] * sigma2[t - j - 1]
                
                sigma2[t] = max(sigma2[t], 1e-6)
            
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + residuals**2 / sigma2)
            return -ll
        except:
            return 1e10
    
    def _compute_volatility(self, y: np.ndarray) -> np.ndarray:
        residuals = y - self.mu
        T = len(residuals)
        sigma2 = np.zeros(T)
        
        uncond_var = self.omega / (1 - np.sum(self.alpha) - 0.5 * np.sum(self.gamma) - np.sum(self.beta))
        sigma2[:max(self.p, self.q)] = uncond_var
        
        for t in range(max(self.p, self.q), T):
            sigma2[t] = self.omega
            
            for i in range(self.q):
                if t - i - 1 >= 0:
                    eps_sq = residuals[t - i - 1]**2
                    sigma2[t] += self.alpha[i] * eps_sq
                    if residuals[t - i - 1] < 0:
                        sigma2[t] += self.gamma[i] * eps_sq
            
            for j in range(self.p):
                if t - j - 1 >= 0:
                    sigma2[t] += self.beta[j] * sigma2[t - j - 1]
            
            sigma2[t] = max(sigma2[t], 1e-6)
        
        return np.sqrt(sigma2)
    
    def forecast(self, steps: int = 1) -> ForecastResult:
        if self._conditional_volatility is None:
            raise RuntimeError("Model must be fitted first")
        
        # Get last values
        last_residuals = (self._y - self.mu)[-(self.q):]
        last_volatility = self._conditional_volatility[-(self.p):]
        
        forecasts = []
        
        for h in range(steps):
            sigma2_h = self.omega
            
            # ARCH + GJR terms
            for i in range(self.q):
                if i < len(last_residuals):
                    eps_sq = last_residuals[-(i+1)]**2
                    sigma2_h += self.alpha[i] * eps_sq
                    # For forecasts, assume E[I_t] = 0.5 (symmetric future)
                    sigma2_h += 0.5 * self.gamma[i] * eps_sq
            
            # GARCH terms
            for j in range(self.p):
                if h == 0 and j < len(last_volatility):
                    sigma2_h += self.beta[j] * last_volatility[-(j+1)]**2
                elif h > j:
                    sigma2_h += self.beta[j] * forecasts[h - j - 1]**2
            
            forecasts.append(np.sqrt(sigma2_h))
        
        forecast_array = np.array(forecasts)
        std_error = 0.1 * forecast_array
        
        return ForecastResult(
            model_name=f"GJR-GARCH({self.p},{self.q})",
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
    
    @property
    def _y(self) -> Optional[np.ndarray]:
        return getattr(self, '__y', None)
    
    @_y.setter
    def _y(self, value: np.ndarray) -> None:
        self.__y = value
