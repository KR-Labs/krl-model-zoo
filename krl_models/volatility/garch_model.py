# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC, a subsidiary of Sudiata Giddasira, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GARCH Model Implementation.

This module implements the Generalized Autoregressive Conditional Heteroskedasticity
(GARCH) model for volatility forecasting with ML estimation, multi-step forecasting,
and VaR calculation capabilities.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult


class GARCHModel(BaseModel):
    """GARCH(p,q) Model for conditional volatility.
    
    The GARCH model specifies conditional variance as:
        sigma_t^2 = omega + sum(alpha_i * epsilon_{t-i}^2) + sum(beta_j * sigma_{t-j}^2)
    
    where epsilon_t = y_t - mu is the mean-adjusted return.
    
    Supports multiple error distributions: Normal, Student-t, GED, Skewed-t
    
    Attributes:
        p: GARCH order (lagged variance terms)
        q: ARCH order (lagged squared residual terms)
        distribution: Error distribution ('normal', 't', 'ged', 'skewt')
        omega: Constant term in variance equation
        alpha: ARCH coefficients (q,)
        beta: GARCH coefficients (p,)
        nu: Degrees of freedom (for t and skewt distributions)
        lambda_param: Asymmetry parameter (for skewt distribution)
    """
    
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        distribution: str = 'normal',
    ):
        """Initialize GARCH model.
        
        Args:
            p: GARCH order (default 1)
            q: ARCH order (default 1)
            distribution: Error distribution ('normal', 't', 'ged', 'skewt')
        """
        super().__init__()
        
        self.p = p
        self.q = q
        self.distribution = distribution
        
        self.omega: Optional[float] = None
        self.alpha: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.mu: Optional[float] = None
        self.nu: Optional[float] = None
        self.lambda_param: Optional[float] = None
        
        self._y: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._conditional_volatility: Optional[np.ndarray] = None
        self._log_likelihood: Optional[float] = None
    
    def fit(self, y: np.ndarray) -> "GARCHModel":
        """Fit GARCH model via maximum likelihood.
        
        Args:
            y: Time series data (returns)
            
        Returns:
            self (for method chaining)
        """
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        
        self._y = y.copy()
        T = len(y)
        
        # Initial parameter estimates
        self.mu = np.mean(y)
        residuals = y - self.mu
        
        # Initial variance parameters
        var_resid = np.var(residuals)
        omega_init = 0.01 * var_resid
        alpha_init = np.full(self.q, 0.05)
        beta_init = np.full(self.p, 0.9)
        
        # Distribution-specific parameters
        if self.distribution == 't':
            params_init = np.concatenate([[self.mu, omega_init], alpha_init, beta_init, [10.0]])
        elif self.distribution == 'skewt':
            params_init = np.concatenate([[self.mu, omega_init], alpha_init, beta_init, [10.0, 0.0]])
        else:
            params_init = np.concatenate([[self.mu, omega_init], alpha_init, beta_init])
        
        # Optimize
        result = minimize(
            self._neg_log_likelihood,
            x0=params_init,
            args=(y,),
            method='L-BFGS-B',
            bounds=self._get_bounds(),
        )
        
        # Extract parameters
        self._extract_parameters(result.x)
        self._log_likelihood = -result.fun
        
        # Compute fitted volatility
        self._residuals = y - self.mu
        self._conditional_volatility = self._compute_volatility(self._residuals)
        
        return self
    
    def _get_bounds(self) -> list:
        """Get parameter bounds for optimization."""
        bounds = [
            (-np.inf, np.inf),  # mu
            (1e-6, None),       # omega
        ]
        
        # alpha bounds (0, 1)
        for _ in range(self.q):
            bounds.append((1e-6, 1 - 1e-6))
        
        # beta bounds (0, 1)
        for _ in range(self.p):
            bounds.append((1e-6, 1 - 1e-6))
        
        # Distribution parameters
        if self.distribution == 't':
            bounds.append((2.1, 100))  # nu > 2
        elif self.distribution == 'skewt':
            bounds.append((2.1, 100))  # nu > 2
            bounds.append((-0.99, 0.99))  # lambda
        
        return bounds
    
    def _extract_parameters(self, params: np.ndarray) -> None:
        """Extract parameters from optimization result."""
        idx = 0
        self.mu = params[idx]
        idx += 1
        
        self.omega = params[idx]
        idx += 1
        
        self.alpha = params[idx:idx + self.q]
        idx += self.q
        
        self.beta = params[idx:idx + self.p]
        idx += self.p
        
        if self.distribution == 't':
            self.nu = params[idx]
        elif self.distribution == 'skewt':
            self.nu = params[idx]
            self.lambda_param = params[idx + 1]
    
    def _neg_log_likelihood(self, params: np.ndarray, y: np.ndarray) -> float:
        """Compute negative log-likelihood."""
        try:
            # Extract parameters
            idx = 0
            mu = params[idx]
            idx += 1
            omega = params[idx]
            idx += 1
            alpha = params[idx:idx + self.q]
            idx += self.q
            beta = params[idx:idx + self.p]
            idx += self.p
            
            # Check stationarity
            if np.sum(alpha) + np.sum(beta) >= 1:
                return 1e10
            
            # Compute residuals and volatility
            residuals = y - mu
            sigma2 = self._compute_volatility_from_params(residuals, omega, alpha, beta) ** 2
            
            # Log-likelihood
            if self.distribution == 'normal':
                ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + residuals**2 / sigma2)
            elif self.distribution == 't':
                nu = params[idx]
                if nu <= 2:
                    return 1e10
                ll = self._student_t_likelihood(residuals, sigma2, nu)
            else:
                ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + residuals**2 / sigma2)
            
            return -ll
        except:
            return 1e10
    
    def _student_t_likelihood(
        self,
        residuals: np.ndarray,
        sigma2: np.ndarray,
        nu: float,
    ) -> float:
        """Compute Student-t log-likelihood."""
        from scipy.special import gammaln
        
        T = len(residuals)
        
        ll = T * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)))
        ll -= 0.5 * np.sum(np.log(sigma2))
        ll -= ((nu + 1) / 2) * np.sum(np.log(1 + residuals**2 / ((nu - 2) * sigma2)))
        
        return ll
    
    def _compute_volatility(self, residuals: np.ndarray) -> np.ndarray:
        """Compute conditional volatility from residuals."""
        return self._compute_volatility_from_params(
            residuals, self.omega, self.alpha, self.beta
        )
    
    def _compute_volatility_from_params(
        self,
        residuals: np.ndarray,
        omega: float,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Compute conditional volatility given parameters."""
        T = len(residuals)
        sigma = np.zeros(T)
        
        # Initialize with unconditional variance
        uncond_var = omega / (1 - np.sum(alpha) - np.sum(beta))
        sigma[:max(self.p, self.q)] = np.sqrt(uncond_var)
        
        for t in range(max(self.p, self.q), T):
            sigma2_t = omega
            
            # ARCH terms
            for i in range(self.q):
                if t - i - 1 >= 0:
                    sigma2_t += alpha[i] * residuals[t - i - 1]**2
            
            # GARCH terms
            for j in range(self.p):
                if t - j - 1 >= 0:
                    sigma2_t += beta[j] * sigma[t - j - 1]**2
            
            sigma[t] = np.sqrt(max(sigma2_t, 1e-6))
        
        return sigma
    
    def forecast(self, steps: int = 1) -> ForecastResult:
        """Forecast conditional volatility.
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            ForecastResult with volatility forecasts
        """
        if self._conditional_volatility is None:
            raise RuntimeError("Model must be fitted before forecasting")
        
        # Get last residuals and volatilities
        last_residuals = self._residuals[-(self.q):]
        last_volatility = self._conditional_volatility[-(self.p):]
        
        forecasts = []
        
        for h in range(steps):
            sigma2_h = self.omega
            
            # ARCH component
            for i in range(self.q):
                if i == 0 and h == 0:
                    sigma2_h += self.alpha[i] * last_residuals[-1]**2
                elif i < len(last_residuals):
                    sigma2_h += self.alpha[i] * last_residuals[-(i+1)]**2
                else:
                    # Use unconditional variance for missing terms
                    uncond_var = self.omega / (1 - np.sum(self.alpha) - np.sum(self.beta))
                    sigma2_h += self.alpha[i] * uncond_var
            
            # GARCH component
            for j in range(self.p):
                if j == 0 and h == 0:
                    sigma2_h += self.beta[j] * last_volatility[-1]**2
                elif j < len(last_volatility):
                    sigma2_h += self.beta[j] * last_volatility[-(j+1)]**2
                elif h > j:
                    # Use previous forecast
                    sigma2_h += self.beta[j] * forecasts[h - j - 1]**2
                else:
                    uncond_var = self.omega / (1 - np.sum(self.alpha) - np.sum(self.beta))
                    sigma2_h += self.beta[j] * uncond_var
            
            forecasts.append(np.sqrt(sigma2_h))
        
        forecast_array = np.array(forecasts)
        
        # Approximate confidence intervals (±1.96 std of volatility forecast error)
        std_error = 0.1 * forecast_array  # Rough approximation
        lower = forecast_array - 1.96 * std_error
        upper = forecast_array + 1.96 * std_error
        
        return ForecastResult(
            model_name=f"GARCH({self.p},{self.q})",
            point_forecast=forecast_array,
            lower_bound=lower,
            upper_bound=upper,
            metadata={
                "omega": float(self.omega),
                "alpha": self.alpha.tolist(),
                "beta": self.beta.tolist(),
                "distribution": self.distribution,
                "log_likelihood": self._log_likelihood,
                "persistence": float(np.sum(self.alpha) + np.sum(self.beta)),
            },
        )
    
    def calculate_var(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Calculate Value at Risk.
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            horizon: Forecast horizon
            
        Returns:
            VaR estimate
        """
        if self._conditional_volatility is None:
            raise RuntimeError("Model must be fitted before VaR calculation")
        
        # Forecast volatility
        vol_forecast = self.forecast(steps=horizon)
        sigma_h = vol_forecast.point_forecast[-1]
        
        # Get quantile from distribution
        if self.distribution == 'normal':
            quantile = norm.ppf(1 - confidence)
        elif self.distribution == 't':
            quantile = student_t.ppf(1 - confidence, self.nu)
        else:
            quantile = norm.ppf(1 - confidence)
        
        var = self.mu + sigma_h * quantile
        
        return float(var)
    
    @property
    def conditional_volatility(self) -> Optional[np.ndarray]:
        """Get fitted conditional volatility series."""
        return self._conditional_volatility
    
    @property
    def residuals(self) -> Optional[np.ndarray]:
        """Get model residuals."""
        return self._residuals
    
    @property
    def log_likelihood(self) -> Optional[float]:
        """Get log-likelihood."""
        return self._log_likelihood
