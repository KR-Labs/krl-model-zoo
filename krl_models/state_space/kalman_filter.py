# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# SPX-License-Identifier: pache-2.
# opyright (c) 22 KR-Labs

"""
Kalman ilter Implementation for State Space Models

This module implements the Kalman ilter algorithm for linear Gaussian state space models,
providing filtering, smoothing, and parameter estimation capabilities.

State Space Representation:
    State quation:     x_t = _t * x_{t-} + w_t,  w_t ~ N(, Q_t)
    Observation quation: y_t = H_t * x_t + v_t,  v_t ~ N(, R_t)

uthor: KR Labs
ate: October 224
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class KalmanilterState:
    """
    ontainer for Kalman ilter state estimates and covariances.
    
    ttributes:
        x: State estimate (n_states,)
        P: State covariance matrix (n_states, n_states)
        x_pred: Predicted state (before update)
        P_pred: Predicted covariance (before update)
        innovation: Measurement innovation (y - H*x_pred)
        innovation_cov: Innovation covariance (H*P_pred*H' + R)
        K: Kalman gain matrix
    """
    x: np.ndarray  # State estimate
    P: np.ndarray  # State covariance
    x_pred: Optional[np.ndarray] = None  # Predicted state
    P_pred: Optional[np.ndarray] = None  # Predicted covariance
    innovation: Optional[np.ndarray] = None  # y - H*x_pred
    innovation_cov: Optional[np.ndarray] = None  # S = H*P_pred*H' + R
    K: Optional[np.ndarray] = None  # Kalman gain


@dataclass
class ForecastResult:
    """Simple forecast result container."""
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    forecast_index: Any
    forecast_values: Any
    ci_lower: List
    ci_upper: List


class Kalmanilter:
    """
    Kalman ilter for Linear Gaussian State Space Models.
    
    Implements the complete Kalman ilter algorithm including:
    - orward filtering (estimate states given observations up to t)
    - Prediction (forecast future states)
    - ackward smoothing (optimal state estimates given all data)
    - Log-likelihood computation (for ML parameter estimation)
    
    xample:
        >>> # reate Kalman ilter for local level model
        >>> kf = Kalmanilter(
        ...     n_states=,
        ...     n_obs=,
        ...     =np.array([[.]]),  # Random walk
        ...     H=np.array([[.]]),  # irect observation
        ...     Q=np.array([[.]]),  # Process noise
        ...     R=np.array([[.]]),  # Observation noise
        ...     x=np.array([.]),
        ...     P=np.array([[.]])
        ... )
        >>> result = kf.fit(data)
        >>> forecast = kf.predict(steps=)
    """
    
    def __init__(
        self,
        n_states: int,
        n_obs: int,
        : np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x: np.ndarray,
        P: np.ndarray,
        : Optional[np.ndarray] = None,
        : Optional[np.ndarray] = None,
    ):
        """
        Initialize Kalman ilter with system matrices.
        
        rgs:
            n_states: Number of state variables
            n_obs: Number of observation variables
            : State transition matrix (n_states, n_states)
            H: Observation matrix (n_obs, n_states)
            Q: Process noise covariance (n_states, n_states)
            R: Observation noise covariance (n_obs, n_obs)
            x: Initial state estimate (n_states,)
            P: Initial state covariance (n_states, n_states)
            : ontrol input matrix (optional)
            : Observation control matrix (optional)
        """
        self._validate_dimensions(n_states, n_obs, , H, Q, R, x, P)
        
        self._n_states = n_states
        self._n_obs = n_obs
        
        self._ = .copy()
        self._H = H.copy()
        self._Q = Q.copy()
        self._R = R.copy()
        self._x = x.copy()
        self._P = P.copy()
        
        self._ = .copy() if  is not None else None
        self._ = .copy() if  is not None else None
        
        self._filtered_states: List[KalmanilterState] = []
        self._smoothed_states: List[KalmanilterState] = []
        self._observations: Optional[np.ndarray] = None
        self._log_likelihood: Optional[float] = None
        
        self._is_fitted = False
    
    def _validate_dimensions(
        self,
        n_states: int,
        n_obs: int,
        : np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x: np.ndarray,
        P: np.ndarray,
    ) -> None:
        """Validate that all matrix dimensions are consistent."""
        if .shape != (n_states, n_states):
            raise ValueError(f" must be ({n_states}, {n_states}), got {.shape}")
        
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
        data: pd.DataFrame,
        controls: Optional[np.ndarray] = None,
        smoothing: bool = True,
    ) -> ForecastResult:
        """
        Run Kalman ilter on observations.
        
        rgs:
            data: DataFrame with observations (T, n_obs)
            controls: Optional control inputs (T, n_controls)
            smoothing: If True, perform backward smoothing after filtering
        
        Returns:
            ForecastResult with filtered (and smoothed) state estimates
        """
        # Extract observations as numpy array
        y = data.values
        T = len(y)
        
        if y.shape[] != self._n_obs:
            raise ValueError(f"ata must have {self._n_obs} columns, got {y.shape[]}")
        
        self._observations = y
        
        # orward filtering pass
        self._filtered_states = self._filter(y, controls)
        
        # Compute log-likelihood
        self._log_likelihood = self._compute_log_likelihood()
        
        # ackward smoothing pass (optional)
        if smoothing:
            self._smoothed_states = self._smooth()
        
        self._is_fitted = True
        
        # Extract state estimates
        filtered_x = np.array([state.x for state in self._filtered_states])
        smoothed_x = np.array([state.x for state in self._smoothed_states]) if smoothing else None
        
        # uild result
        payload = {
            'filtered_states': filtered_x,
            'smoothed_states': smoothed_x,
            'log_likelihood': self._log_likelihood,
            'n_observations': T,
        }
        
        n_params = self._count_parameters()
        metadata = {
            'n_states': self._n_states,
            'n_obs': self._n_obs,
            'smoothing_applied': smoothing,
            'aic': 2 * n_params - 2 * self._log_likelihood,
            'bic': np.log(T) * n_params - 2 * self._log_likelihood,
        }
        
        return ForecastResult(
            payload=payload,
            metadata=metadata,
            forecast_index=data.index,
            forecast_values=filtered_x[:, ] if self._n_states == 0 else filtered_x,
            ci_lower=[],
            ci_upper=[],
        )
    
    def _filter(
        self,
        y: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> List[KalmanilterState]:
        """orward filtering pass (Kalman ilter)."""
        T = len(y)
        states = []
        
        x = self._x.copy()
        P = self._P.copy()
        
        for t in range(T):
            u_t = controls[t] if controls is not None else None
            
            # PRIT
            x_pred = self._ @ x
            if self._ is not None and u_t is not None:
                x_pred += self._ @ u_t
            
            P_pred = self._ @ P @ self._.T + self._Q
            
            # UPT
            y_pred = self._H @ x_pred
            if self._ is not None and u_t is not None:
                y_pred += self._ @ u_t
            
            innovation = y[t] - y_pred
            
            S = self._H @ P_pred @ self._H.T + self._R
            
            try:
                K = P_pred @ self._H.T @ np.linalg.inv(S)
            except np.linalg.Linlgrror:
                K = P_pred @ self._H.T @ np.linalg.pinv(S)
            
            x = x_pred + K @ innovation
            
            I_KH = np.eye(self._n_states) - K @ self._H
            P = I_KH @ P_pred @ I_KH.T + K @ self._R @ K.T
            
            states.append(KalmanilterState(
                x=x.copy(),
                P=P.copy(),
                x_pred=x_pred.copy(),
                P_pred=P_pred.copy(),
                innovation=innovation.copy(),
                innovation_cov=S.copy(),
                K=K.copy(),
            ))
        
        return states
    
    def _smooth(self) -> List[KalmanilterState]:
        """ackward smoothing pass (Rauch-Tung-Striebel smoother)."""
        if not self._filtered_states:
            raise Runtimerror("Must run filter() before smooth()")
        
        T = len(self._filtered_states)
        smoothed: List[KalmanilterState] = []
        
        # Initialize with last filtered state
        smoothed.append(KalmanilterState(
            x=self._filtered_states[-].x.copy(),
            P=self._filtered_states[-].P.copy(),
        ))
        
        # ackward pass
        for t in range(T - 2, -, -):
            x_filt = self._filtered_states[t].x
            P_filt = self._filtered_states[t].P
            x_pred_next = self._filtered_states[t + ].x_pred
            P_pred_next = self._filtered_states[t + ].P_pred
            
            try:
                J = P_filt @ self._.T @ np.linalg.inv(P_pred_next)
            except np.linalg.Linlgrror:
                J = P_filt @ self._.T @ np.linalg.pinv(P_pred_next)
            
            # Get the next smoothed state (we're going backwards)
            x_smooth_next = smoothed[].x if len(smoothed) == 0 else smoothed[-(T-t-)].x
            P_smooth_next = smoothed[].P if len(smoothed) == 0 else smoothed[-(T-t-)].P
            
            x_smooth = x_filt + J @ (x_smooth_next - x_pred_next)
            P_smooth = P_filt + J @ (P_smooth_next - P_pred_next) @ J.T
            
            smoothed.insert(, KalmanilterState(x=x_smooth, P=P_smooth))
        
        return smoothed
    
    def predict(
        self,
        steps: int = ,
        controls: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        """
        orecast future states.
        
        rgs:
            steps: Number of steps ahead to forecast
            controls: Optional future control inputs (steps, n_controls)
        
        Returns:
            ForecastResult with predicted states and covariances
        """
        if not self._is_fitted:
            raise Runtimerror("Model must be fitted before prediction. all fit() first.")
        
        x = self._filtered_states[-].x.copy()
        P = self._filtered_states[-].P.copy()
        
        forecasts = []
        covariances = []
        
        for i in range(steps):
            u_t = controls[i] if controls is not None else None
            
            x = self._ @ x
            if self._ is not None and u_t is not None:
                x += self._ @ u_t
            
            P = self._ @ P @ self._.T + self._Q
            
            forecasts.append(x.copy())
            covariances.append(P.copy())
        
        forecasts = np.array(forecasts)
        covariances = np.array(covariances)
        
        # orecast index
        if self._observations is not None:
            last_idx = len(self._observations) - 1
            forecast_index = list(range(last_idx + , last_idx +  + steps))
        else:
            forecast_index = list(range(steps))
        
        # onfidence intervals (%)
        z_score = 0.0
        std_devs = np.sqrt([P.diagonal() for P in covariances])
        ci_lower = (forecasts - z_score * std_devs).tolist()
        ci_upper = (forecasts + z_score * std_devs).tolist()
        
        payload = {
            'forecasts': forecasts,
            'covariances': covariances,
            'std_devs': std_devs,
        }
        
        metadata = {
            'forecast_steps': steps,
            'n_states': self._n_states,
        }
        
        return ForecastResult(
            payload=payload,
            metadata=metadata,
            forecast_index=forecast_index,
            forecast_values=forecasts[:, ] if self._n_states == 0 else forecasts,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )
    
    def _compute_log_likelihood(self) -> float:
        """Compute log-likelihood of observed data."""
        if not self._filtered_states:
            raise Runtimerror("Must run filter() before computing log-likelihood")
        
        log_lik = 0.0
        
        for state in self._filtered_states:
            v = state.innovation
            S = state.innovation_cov
            
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                logdet = 0.0
            
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.Linlgrror:
                S_inv = np.linalg.pinv(S)
            
            mahalanobis = v @ S_inv @ v
            
            log_lik += -. * (logdet + mahalanobis + self._n_obs * np.log(2 * np.pi))
        
        return log_lik
    
    def _count_parameters(self) -> int:
        """ount number of free parameters in the model."""
        n_params = 
        n_params += self._n_states ** 2  # 
        n_params += self._n_states * (self._n_states + ) // 2  # Q (symmetric)
        n_params += self._n_obs * self._n_states  # H
        n_params += self._n_obs * (self._n_obs + ) // 2  # R (symmetric)
        return n_params
    
    def get_filtered_states(self) -> Optional[np.ndarray]:
        """Get filtered state estimates."""
        if not self._filtered_states:
            return None
        return np.array([state.x for state in self._filtered_states])
    
    def get_smoothed_states(self) -> Optional[np.ndarray]:
        """Get smoothed state estimates."""
        if not self._smoothed_states:
            return None
        return np.array([state.x for state in self._smoothed_states])
    
    def get_filtered_covariances(self) -> Optional[np.ndarray]:
        """Get filtered state covariances."""
        if not self._filtered_states:
            return None
        return np.array([state.P for state in self._filtered_states])
    
    def get_innovations(self) -> Optional[np.ndarray]:
        """Get measurement innovations (one-step-ahead forecast errors)."""
        if not self._filtered_states:
            return None
        return np.array([state.innovation for state in self._filtered_states])
