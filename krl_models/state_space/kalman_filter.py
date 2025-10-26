# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC, a subsidiary of Sudiata Giddasira, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kalman Filter Implementation for State Space Models.

This module provides a linear Gaussian Kalman Filter with forward filtering,
RTS smoothing, and multi-step forecasting capabilities.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult


@dataclass
class KalmanFilterState:
    """State information at a single time step."""
    x: np.ndarray
    P: np.ndarray
    x_pred: Optional[np.ndarray] = None
    P_pred: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    innovation: Optional[np.ndarray] = None
    innovation_cov: Optional[np.ndarray] = None


class KalmanFilter(BaseModel):
    """Linear Gaussian Kalman Filter.
    
    State space model:
        x_t = F @ x_{t-1} + B @ u_t + w_t,  w_t ~ N(0, Q)
        y_t = H @ x_t + D @ u_t + v_t,      v_t ~ N(0, R)
    """
    
    def __init__(
        self,
        n_states: int,
        n_obs: int,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x: np.ndarray,
        P: np.ndarray,
        B: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
    ):
        """Initialize Kalman Filter."""
        super().__init__()
        
        self._validate_dimensions(n_states, n_obs, F, H, Q, R, x, P)
        
        self._n_states = n_states
        self._n_obs = n_obs
        
        self._F = F.copy()
        self._H = H.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._x = x.copy()
        self._P = P.copy()
        
        self._B = B.copy() if B is not None else None
        self._D = D.copy() if D is not None else None
        
        self._filtered_states: List[KalmanFilterState] = []
        self._smoothed_states: List[KalmanFilterState] = []
        self._observations: Optional[np.ndarray] = None
        self._log_likelihood: Optional[float] = None
    
    def _validate_dimensions(
        self,
        n_states: int,
        n_obs: int,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x: np.ndarray,
        P: np.ndarray,
    ) -> None:
        """Validate that all matrix dimensions are consistent."""
        if F.shape != (n_states, n_states):
            raise ValueError(f"F must be ({n_states}, {n_states}), got {F.shape}")
        if H.shape != (n_obs, n_states):
            raise ValueError(f"H must be ({n_obs}, {n_states}), got {H.shape}")
        if Q.shape != (n_states, n_states):
            raise ValueError(f"Q must be ({n_states}, {n_states}), got {Q.shape}")
        if R.shape != (n_obs, n_obs):
            raise ValueError(f"R must be ({n_obs}, {n_obs}), got {R.shape}")
        if x.shape != (n_states,):
            raise ValueError(f"x must be ({n_states},), got {x.shape}")
        if P.shape != (n_states, n_states):
            raise ValueError(f"P must be ({n_states}, {n_states}), got {P.shape}")
    
    def fit(
        self,
        y: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> "KalmanFilter":
        """Run forward filtering pass."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        T = len(y)
        
        if y.shape[1] != self._n_obs:
            raise ValueError(f"Data must have {self._n_obs} columns, got {y.shape[1]}")
        
        self._observations = y
        self._filtered_states = self._filter(y, controls)
        
        self._log_likelihood = sum(
            self._compute_log_likelihood_step(state)
            for state in self._filtered_states
        )
        
        return self
    
    def _filter(
        self,
        y: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> List[KalmanFilterState]:
        """Forward filtering algorithm."""
        T = len(y)
        filtered_states = []
        
        x = self._x.copy()
        P = self._P.copy()
        
        for t in range(T):
            u_t = controls[t] if controls is not None else None
            
            x_pred = self._F @ x
            if self._B is not None and u_t is not None:
                x_pred += self._B @ u_t
            
            P_pred = self._F @ P @ self._F.T + self._Q
            
            y_t = y[t]
            y_pred = self._H @ x_pred
            if self._D is not None and u_t is not None:
                y_pred += self._D @ u_t
            
            innovation = y_t - y_pred
            innovation_cov = self._H @ P_pred @ self._H.T + self._R
            
            try:
                K = P_pred @ self._H.T @ np.linalg.inv(innovation_cov)
            except np.linalg.LinAlgError:
                K = P_pred @ self._H.T @ np.linalg.pinv(innovation_cov)
            
            x = x_pred + K @ innovation
            P = (np.eye(self._n_states) - K @ self._H) @ P_pred
            
            filtered_states.append(
                KalmanFilterState(
                    x=x.copy(),
                    P=P.copy(),
                    x_pred=x_pred.copy(),
                    P_pred=P_pred.copy(),
                    K=K.copy(),
                    innovation=innovation.copy(),
                    innovation_cov=innovation_cov.copy(),
                )
            )
        
        return filtered_states
    
    def smooth(self) -> List[KalmanFilterState]:
        """Run Rauch-Tung-Striebel backward smoothing pass."""
        if not self._filtered_states:
            raise RuntimeError("Model must be fitted before smoothing. Call fit() first.")
        
        T = len(self._filtered_states)
        smoothed = []
        
        smoothed.append(
            KalmanFilterState(
                x=self._filtered_states[-1].x.copy(),
                P=self._filtered_states[-1].P.copy(),
            )
        )
        
        for t in range(T - 2, -1, -1):
            x_filt = self._filtered_states[t].x
            P_filt = self._filtered_states[t].P
            x_pred_next = self._filtered_states[t + 1].x_pred
            P_pred_next = self._filtered_states[t + 1].P_pred
            
            try:
                J = P_filt @ self._F.T @ np.linalg.inv(P_pred_next)
            except np.linalg.LinAlgError:
                J = P_filt @ self._F.T @ np.linalg.pinv(P_pred_next)
            
            x_smooth_next = smoothed[-1].x
            P_smooth_next = smoothed[-1].P
            
            x_smooth = x_filt + J @ (x_smooth_next - x_pred_next)
            P_smooth = P_filt + J @ (P_smooth_next - P_pred_next) @ J.T
            
            smoothed.append(
                KalmanFilterState(
                    x=x_smooth.copy(),
                    P=P_smooth.copy(),
                )
            )
        
        smoothed.reverse()
        self._smoothed_states = smoothed
        
        return smoothed
    
    def forecast(
        self,
        steps: int,
        controls: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        """Generate multi-step forecasts."""
        if not self._filtered_states:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        x = self._filtered_states[-1].x.copy()
        P = self._filtered_states[-1].P.copy()
        
        forecasts = []
        covariances = []
        
        for i in range(steps):
            u_t = controls[i] if controls is not None else None
            
            x = self._F @ x
            if self._B is not None and u_t is not None:
                x += self._B @ u_t
            
            P = self._F @ P @ self._F.T + self._Q
            
            y_pred = self._H @ x
            if self._D is not None and u_t is not None:
                y_pred += self._D @ u_t
            
            y_cov = self._H @ P @ self._H.T + self._R
            
            forecasts.append(y_pred)
            covariances.append(y_cov)
        
        forecasts_array = np.array(forecasts)
        std_errors = np.array([np.sqrt(np.diag(cov)) for cov in covariances])
        
        lower_bound = forecasts_array - 1.96 * std_errors
        upper_bound = forecasts_array + 1.96 * std_errors
        
        return ForecastResult(
            model_name="KalmanFilter",
            point_forecast=forecasts_array,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            metadata={
                "n_states": self._n_states,
                "n_obs": self._n_obs,
                "log_likelihood": self._log_likelihood,
                "forecast_steps": steps,
            },
        )
    
    def _compute_log_likelihood_step(self, state: KalmanFilterState) -> float:
        """Compute log-likelihood contribution for a single time step."""
        n = len(state.innovation)
        
        try:
            sign, logdet = np.linalg.slogdet(state.innovation_cov)
            if sign <= 0:
                logdet = np.log(np.linalg.det(state.innovation_cov + 1e-8 * np.eye(n)))
        except:
            logdet = np.log(np.linalg.det(state.innovation_cov + 1e-8 * np.eye(n)))
        
        inv_cov = np.linalg.inv(state.innovation_cov)
        mahalanobis = state.innovation @ inv_cov @ state.innovation
        
        ll = -0.5 * (n * np.log(2 * np.pi) + logdet + mahalanobis)
        
        return ll
    
    @property
    def filtered_states(self) -> List[KalmanFilterState]:
        """Get filtered states."""
        return self._filtered_states
    
    @property
    def smoothed_states(self) -> List[KalmanFilterState]:
        """Get smoothed states."""
        return self._smoothed_states
    
    @property
    def log_likelihood(self) -> Optional[float]:
        """Get log-likelihood."""
        return self._log_likelihood
