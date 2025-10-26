"""
STL Decomposition for Anomaly Detection
========================================

MIT License - Gate 2 Phase 2.5 Anomaly Detection
author: KR Labs
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL as STLDecomposer

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult


class STLAnomalyModel(BaseModel):
    """
    STL (Seasonal-Trend decomposition using Loess) for anomaly detection.
    
    Decomposes time series into trend, seasonal, and residual components.
    Anomalies identified as residuals exceeding threshold (typically ±3σ).
    
    Parameters
    ----------
    seasonal : int, default=7
        Length of seasonal period
    trend : int or None, default=None
        Length of trend smoother (None = auto-computed from seasonal)
    robust : bool, default=True
        Use robust fitting (resistant to outliers)
    threshold : float, default=3.0
        Number of standard deviations for anomaly threshold
    """
    
    def __init__(
        self,
        seasonal: int = 7,
        trend: Optional[int] = None,
        robust: bool = True,
        threshold: float = 3.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seasonal = seasonal
        self.trend = trend
        self.robust = robust
        self.threshold = threshold
        
        self.stl_result_ = None
        self.residual_mean_ = None
        self.residual_std_ = None
        self.anomaly_indices_ = None
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None, **kwargs) -> 'STLAnomalyModel':
        """
        Fit STL decomposition to time series.
        
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Time series data
        X : array-like, optional
            Not used (for interface compatibility)
        **kwargs : dict
            Additional arguments
        
        Returns
        -------
        self : STLAnomalyModel
            Fitted model instance
        """
        # Perform STL decomposition
        stl = STLDecomposer(
            y,
            seasonal=self.seasonal,
            trend=self.trend,
            robust=self.robust
        )
        self.stl_result_ = stl.fit()
        
        # Compute residual statistics
        residuals = self.stl_result_.resid
        self.residual_mean_ = float(np.mean(residuals))
        self.residual_std_ = float(np.std(residuals))
        
        # Identify anomalies (residuals beyond threshold)
        threshold_value = self.threshold * self.residual_std_
        self.anomaly_indices_ = np.where(
            np.abs(residuals - self.residual_mean_) > threshold_value
        )[0]
        
        return self
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """
        Get detected anomalies.
        
        Returns
        -------
        anomalies : dict
            Anomaly indices, scores, and summary statistics
        """
        if self.stl_result_ is None:
            raise ValueError("Model must be fitted before detecting anomalies")
        
        residuals = self.stl_result_.resid
        
        # Anomaly scores (z-scores of residuals)
        anomaly_scores = (residuals - self.residual_mean_) / self.residual_std_
        
        return {
            'anomaly_indices': self.anomaly_indices_.tolist(),
            'anomaly_count': len(self.anomaly_indices_),
            'anomaly_scores': anomaly_scores[self.anomaly_indices_].tolist(),
            'threshold': self.threshold,
            'residual_mean': self.residual_mean_,
            'residual_std': self.residual_std_
        }
    
    def decompose(self) -> Dict[str, np.ndarray]:
        """
        Get STL decomposition components.
        
        Returns
        -------
        components : dict
            Trend, seasonal, and residual components
        """
        if self.stl_result_ is None:
            raise ValueError("Model must be fitted before decomposing")
        
        return {
            'trend': self.stl_result_.trend,
            'seasonal': self.stl_result_.seasonal,
            'residual': self.stl_result_.resid
        }
    
    def forecast(self, steps: int = 1, X_future: Optional[np.ndarray] = None, **kwargs) -> ForecastResult:
        """
        Forecast using trend + seasonal components (naive extrapolation).
        
        Parameters
        ----------
        steps : int, default=1
            Number of steps to forecast
        X_future : array-like, optional
            Not used
        **kwargs : dict
            Additional arguments
        
        Returns
        -------
        result : ForecastResult
            Forecasts (trend + last seasonal cycle)
        """
        if self.stl_result_ is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Naive forecast: last trend value + repeating seasonal pattern
        last_trend = self.stl_result_.trend[-1]
        seasonal = self.stl_result_.seasonal[-self.seasonal:]
        
        # Repeat seasonal pattern for forecast horizon
        seasonal_forecast = np.tile(seasonal, steps // self.seasonal + 1)[:steps]
        
        point_forecast = last_trend + seasonal_forecast
        
        # Intervals based on residual variance
        alpha = kwargs.get('alpha', 0.05)
        z = 1.96  # 95% CI
        margin = z * self.residual_std_
        
        lower = point_forecast - margin
        upper = point_forecast + margin
        
        return ForecastResult(
            point_forecast=point_forecast,
            lower=lower,
            upper=upper,
            alpha=alpha
        )
