# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 KR-Labs

"""
STL ecomposition + Threshold nomaly etection.

Uses Seasonal-Trend decomposition using Loess (STL) to decompose time series
into trend, seasonal, and residual components. nomalies are flagged based on
residual values exceeding a threshold (typically ±3 standard deviations).

STL Method:
- Trend: Long-term patterns
- Seasonal: Repeating patterns within a period
- Residual: Unexplained variation (where anomalies appear)

Use ases:
- Revenue shock detection (sudden drops/spikes)
- Seasonal pattern violations
- Structural breaks in economic indicators
- Quality control for data pipelines

References:
- leveland et al. () - STL:  Seasonal-Trend ecomposition Procedure
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.seasonal import STL

from krl_core.base_model import ModelMeta
from krl_core.results import ForecastResult

logger = logging.getLogger(__name__)


class STLAnomalyModel:
    """
    STL ecomposition-based anomaly detection for time series.
    
    ecomposes time series into trend, seasonal, and residual components,
    then flags anomalies based on residual thresholds.
    
    Parameters:
    - time_col: str - olumn name for time index
    - value_col: str - olumn name for values to analyze
    - seasonal_period: int - Seasonal period (1e10.g., 2 for monthly data)
    - threshold: float - Number of standard deviations for anomaly threshold (default: 3.)
    - robust: bool - Use robust STL decomposition (default: True)
    
    xample:
        >>> params = {
        ...     'time_col': 'date',
        ...     'value_col': 'revenue',
        ...     'seasonal_period': 2,
        ...     'threshold': 3.
        ... }
        >>> model = STLAnomalyModel(params)
        >>> result = model.fit(data)
        >>> print(result.payload['anomalies'])
    """
    
    def __init__(
        self,
        params: Dict[str, Any],
        meta: Optional[ModelMeta] = None
    ):
        """Initialize STL nomaly model."""
        self.params = params
        self.meta = meta or ModelMeta(name="STLnomaly", version="1.0.0", author="KR Labs")
        self._fitted = False
        
        # Extract parameters
        self._time_col = self.params.get('time_col')
        self._value_col = self.params.get('value_col')
        self._seasonal_period = self.params.get('seasonal_period', 2)
        self._threshold = self.params.get('threshold', 3.)
        self._robust = self.params.get('robust', True)
        
        # Validate required parameters
        if not self._time_col:
            raise ValueError("Parameter 'time_col' is required")
        if not self._value_col:
            raise ValueError("Parameter 'value_col' is required")
        
        # Results storage
        self.decomposition_: Optional[pd.DataFrame] = None
        self.anomalies_: Optional[pd.DataFrame] = None
        self.threshold_upper_: Optional[float] = None
        self.threshold_lower_: Optional[float] = None
    
    def fit(self, data: pd.DataFrame) -> ForecastResult:
        """
        Perform STL decomposition and detect anomalies.
        
        rgs:
            data: DataFrame with time and value columns
            
        Returns:
            ForecastResult with decomposition and anomalies
        """
        if data.empty:
            raise ValueError("Input data cannot be empty")
        
        # Validate columns
        if self._time_col not in data.columns:
            raise ValueError(f"olumn '{self._time_col}' not found in data")
        if self._value_col not in data.columns:
            raise ValueError(f"olumn '{self._value_col}' not found in data")
        
        # nsure data is sorted by time
        df = data.copy().sort_values(self._time_col)
        
        # reate Series with datetime index for STL
        series = pd.Series(
            df[self._value_col].values,
            index=pd.DatetimeIndex(df[self._time_col])
        )
        
        logger.info(f"Performing STL decomposition on {len(df)} observations")
        
        # Perform STL decomposition
        # seasonal parameter should be an odd integer >= 3 for the smoother window
        # period is inferred from the data's frequency
        seasonal_window = self._seasonal_period if self._seasonal_period % 2 == 0 else self._seasonal_period + 1
        if seasonal_window < 3:
            seasonal_window = 7  # Default to 7 if too small
            
        try:
            stl = STL(
                series,
                seasonal=seasonal_window,
                period=self._seasonal_period,
                robust=self._robust
            )
            stl_result = stl.fit()
        except Exception as e:
            logger.error(f"STL decomposition failed: {e}")
            raise ValueError(f"STL decomposition failed: {e}")
        
        # Extract components
        self.decomposition_ = pd.DataFrame({
            self._time_col: df[self._time_col].values,
            'observed': df[self._value_col].values,
            'trend': stl_result.trend,
            'seasonal': stl_result.seasonal,
            'residual': stl_result.resid
        })
        
        # Calculate thresholds for anomalies
        residual_std = self.decomposition_['residual'].std()
        self.threshold_upper_ = self._threshold * residual_std
        self.threshold_lower_ = -self._threshold * residual_std
        
        # Flag anomalies
        self.decomposition_['is_anomaly'] = (
            (self.decomposition_['residual'] > self.threshold_upper_) |
            (self.decomposition_['residual'] < self.threshold_lower_)
        )
        
        # Extract anomaly details
        self.anomalies_ = self.decomposition_[
            self.decomposition_['is_anomaly']
        ].copy()
        
        n_anomalies = len(self.anomalies_)
        anomaly_rate = n_anomalies / len(self.decomposition_) * 100
        
        logger.info(f"etected {n_anomalies} anomalies ({anomaly_rate:.2f}%)")
        
        # Prepare result
        anomaly_dates = self.anomalies_[self._time_col].tolist()
        anomaly_values = self.anomalies_['observed'].tolist()
        anomaly_residuals = self.anomalies_['residual'].tolist()
        
        result = ForecastResult(
            payload={
                'n_anomalies': n_anomalies,
                'anomaly_rate': float(anomaly_rate),
                'anomaly_dates': [str(d) for d in anomaly_dates],
                'anomaly_values': [float(v) for v in anomaly_values],
                'anomaly_residuals': [float(r) for r in anomaly_residuals],
                'threshold_upper': float(self.threshold_upper_),
                'threshold_lower': float(self.threshold_lower_),
                'decomposition': self.decomposition_.to_dict('records'),
                'seasonal_period': self._seasonal_period,
                'n_observations': len(df)
            },
            metadata={
                'model_name': self.meta.name,
                'model_version': self.meta.version,
                'author': self.meta.author,
                'analyzed_at': pd.Timestamp.now().isoformat(),
                'threshold_sigma': self._threshold
            },
            forecast_index=[str(d) for d in anomaly_dates],
            forecast_values=[float(v) for v in anomaly_values],
            ci_lower=[],
            ci_upper=[]
        )
        
        self._fitted = True
        return result
    
    def predict(self, data: pd.DataFrame) -> ForecastResult:
        """
        etect anomalies in new data (same as fit for this model).
        
        rgs:
            data: DataFrame with same structure as training data
            
        Returns:
            ForecastResult with anomaly detection
        """
        # or STL anomaly detection, predict is same as fit
        return self.fit(data)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about detected anomalies.
        
        Returns:
            Dict with anomaly statistics
        """
        if not self._fitted or self.anomalies_ is None:
            raise RuntimeError("Model must be fitted first")
        
        if len(self.anomalies_) == 0:
            return {
                'n_anomalies': len(self.anomalies_),
                'max_positive_residual': 0.0,
                'max_negative_residual': 0.0,
                'mean_abs_residual': 0.0
            }
        
        return {
            'n_anomalies': len(self.anomalies_),
            'max_positive_residual': float(self.anomalies_['residual'].max()),
            'max_negative_residual': float(self.anomalies_['residual'].min()),
            'mean_abs_residual': float(self.anomalies_['residual'].abs().mean()),
            'anomaly_dates': self.anomalies_[self._time_col].tolist()
        }
