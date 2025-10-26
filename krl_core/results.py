# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Result Classes for KRL Model Zoo
=================================

Apache 2.0 License - Gate 1 Foundation
Author: KR Labs

Defines standardized result classes for model outputs:
- ForecastResult: Time series forecasts with prediction intervals
- CausalResult: Causal inference results (treatment effects, etc.)
- ClassificationResult: Classification model outputs
- BaseResult: Abstract base for custom result types
"""

from typing import Dict, Any, Optional, List
import numpy as np
from pydantic import BaseModel, Field, field_validator


class BaseResult(BaseModel):
    """
    Abstract base class for model results.
    
    All result classes inherit from this Pydantic BaseModel.
    Provides automatic validation, serialization, and JSON export.
    
    Attributes
    ----------
    model_name : str
        Name of model that generated results
    timestamp : str
        ISO 8601 timestamp of result generation
    metadata : dict
        Additional result metadata
    """
    model_name: str = "Unknown"
    timestamp: str = Field(default_factory=lambda: __import__('datetime').datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class ForecastResult(BaseResult):
    """
    Forecast results with point estimates and prediction intervals.
    
    Used by all forecasting models (ARIMA, GARCH, ML regressors, etc.)
    
    Attributes
    ----------
    point_forecast : array-like
        Point forecast values
    lower : array-like, optional
        Lower bound of prediction interval
    upper : array-like, optional
        Upper bound of prediction interval
    alpha : float
        Significance level for prediction intervals (default 0.05 = 95% CI)
    forecast_horizon : int
        Number of steps forecasted
    
    Examples
    --------
    >>> result = ForecastResult(
    ...     point_forecast=np.array([1.0, 1.1, 1.2]),
    ...     lower=np.array([0.8, 0.9, 1.0]),
    ...     upper=np.array([1.2, 1.3, 1.4]),
    ...     alpha=0.05
    ... )
    >>> print(f"Forecast: {result.point_forecast}")
    >>> print(f"95% CI: [{result.lower[0]:.2f}, {result.upper[0]:.2f}]")
    """
    point_forecast: np.ndarray
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    alpha: float = 0.05
    forecast_horizon: Optional[int] = None
    
    @field_validator('point_forecast', 'lower', 'upper', mode='before')
    @classmethod
    def validate_arrays(cls, v):
        """Convert to numpy array and validate."""
        if v is None:
            return v
        v = np.asarray(v)
        if v.ndim != 1:
            raise ValueError("Forecast arrays must be 1-dimensional")
        return v
    
    @field_validator('alpha')
    @classmethod
    def validate_alpha(cls, v):
        """Validate significance level."""
        if not 0 < v < 1:
            raise ValueError("alpha must be between 0 and 1")
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set forecast_horizon if not provided
        if self.forecast_horizon is None:
            self.forecast_horizon = len(self.point_forecast)
        
        # Validate interval bounds if provided
        if self.lower is not None and self.upper is not None:
            if len(self.lower) != len(self.point_forecast):
                raise ValueError("lower must match point_forecast length")
            if len(self.upper) != len(self.point_forecast):
                raise ValueError("upper must match point_forecast length")
            if np.any(self.lower > self.upper):
                raise ValueError("lower must be <= upper for all points")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns
        -------
        dict
            Dictionary with numpy arrays converted to lists
        """
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'point_forecast': self.point_forecast.tolist(),
            'lower': self.lower.tolist() if self.lower is not None else None,
            'upper': self.upper.tolist() if self.upper is not None else None,
            'alpha': self.alpha,
            'forecast_horizon': self.forecast_horizon,
            'metadata': self.metadata
        }


class CausalResult(BaseResult):
    """
    Causal inference results (treatment effects, counterfactuals, etc.)
    
    Used by causal models (DiD, RDD, Synthetic Control, DML, etc.)
    
    Attributes
    ----------
    treatment_effect : float
        Average treatment effect (ATE)
    std_error : float, optional
        Standard error of treatment effect
    p_value : float, optional
        P-value for treatment effect significance
    confidence_interval : tuple of float, optional
        Confidence interval for treatment effect
    treatment_group_mean : float, optional
        Mean outcome for treatment group
    control_group_mean : float, optional
        Mean outcome for control group
    heterogeneous_effects : dict, optional
        Subgroup-specific treatment effects
    
    Examples
    --------
    >>> result = CausalResult(
    ...     treatment_effect=5.2,
    ...     std_error=1.3,
    ...     p_value=0.0001,
    ...     confidence_interval=(2.6, 7.8)
    ... )
    >>> print(f"ATE: {result.treatment_effect:.2f} (p={result.p_value:.4f})")
    """
    treatment_effect: float
    std_error: Optional[float] = None
    p_value: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    treatment_group_mean: Optional[float] = None
    control_group_mean: Optional[float] = None
    heterogeneous_effects: Optional[Dict[str, float]] = None
    
    @field_validator('p_value')
    @classmethod
    def validate_p_value(cls, v):
        """Validate p-value range."""
        if v is not None and not 0 <= v <= 1:
            raise ValueError("p_value must be between 0 and 1")
        return v
    
    @field_validator('confidence_interval')
    @classmethod
    def validate_ci(cls, v):
        """Validate confidence interval."""
        if v is not None:
            if len(v) != 2:
                raise ValueError("confidence_interval must have 2 elements")
            if v[0] > v[1]:
                raise ValueError("CI lower bound must be <= upper bound")
        return v
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """
        Check if treatment effect is statistically significant.
        
        Parameters
        ----------
        alpha : float, default=0.05
            Significance level
        
        Returns
        -------
        bool
            True if p_value < alpha (significant)
        """
        if self.p_value is None:
            raise ValueError("p_value not available")
        return self.p_value < alpha


class ClassificationResult(BaseResult):
    """
    Classification model results.
    
    Used by classification models (anomaly detection, etc.)
    
    Attributes
    ----------
    predictions : array-like
        Predicted class labels
    probabilities : array-like, optional
        Class probabilities (n_samples, n_classes)
    decision_scores : array-like, optional
        Decision function scores
    class_labels : list of str, optional
        Class label names
    
    Examples
    --------
    >>> result = ClassificationResult(
    ...     predictions=np.array([0, 1, 1, 0]),
    ...     probabilities=np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.8, 0.2]]),
    ...     class_labels=['normal', 'anomaly']
    ... )
    >>> print(f"Predicted: {result.predictions}")
    """
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    decision_scores: Optional[np.ndarray] = None
    class_labels: Optional[List[str]] = None
    
    @field_validator('predictions', 'probabilities', 'decision_scores', mode='before')
    @classmethod
    def validate_arrays(cls, v):
        """Convert to numpy array."""
        if v is None:
            return v
        return np.asarray(v)
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Validate probabilities if provided
        if self.probabilities is not None:
            if self.probabilities.ndim not in [1, 2]:
                raise ValueError("probabilities must be 1D or 2D array")
            if len(self.probabilities) != len(self.predictions):
                raise ValueError("probabilities must match predictions length")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'predictions': self.predictions.tolist(),
            'probabilities': self.probabilities.tolist() if self.probabilities is not None else None,
            'decision_scores': self.decision_scores.tolist() if self.decision_scores is not None else None,
            'class_labels': self.class_labels,
            'metadata': self.metadata
        }


class SpecializationResult(BaseResult):
    """
    Regional specialization analysis results.
    
    Used by location quotient, shift-share, and other regional models.
    
    Attributes
    ----------
    location_quotients : dict, optional
        Industry -> LQ value mapping
    shift_share_decomposition : dict, optional
        NS, IM, CS components
    specialized_industries : list of str, optional
        Industries where region is specialized (LQ > threshold)
    rankings : dict, optional
        Region rankings by various metrics
    
    Examples
    --------
    >>> result = SpecializationResult(
    ...     location_quotients={'tech': 2.5, 'healthcare': 0.8},
    ...     specialized_industries=['tech']
    ... )
    """
    location_quotients: Optional[Dict[str, float]] = None
    shift_share_decomposition: Optional[Dict[str, float]] = None
    specialized_industries: Optional[List[str]] = None
    rankings: Optional[Dict[str, Any]] = None


class AnomalyResult(BaseResult):
    """
    Anomaly detection results.
    
    Used by STL, Isolation Forest, and other anomaly detection models.
    
    Attributes
    ----------
    anomaly_indices : array-like
        Indices of detected anomalies
    anomaly_scores : array-like
        Anomaly scores for all samples
    threshold : float, optional
        Threshold used for anomaly detection
    anomaly_labels : array-like, optional
        Binary labels (1 = normal, -1 = anomaly)
    
    Examples
    --------
    >>> result = AnomalyResult(
    ...     anomaly_indices=np.array([10, 25, 47]),
    ...     anomaly_scores=np.random.randn(100),
    ...     threshold=3.0
    ... )
    """
    anomaly_indices: np.ndarray
    anomaly_scores: np.ndarray
    threshold: Optional[float] = None
    anomaly_labels: Optional[np.ndarray] = None
    
    @field_validator('anomaly_indices', 'anomaly_scores', 'anomaly_labels', mode='before')
    @classmethod
    def validate_arrays(cls, v):
        """Convert to numpy array."""
        if v is None:
            return v
        return np.asarray(v)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'anomaly_indices': self.anomaly_indices.tolist(),
            'anomaly_scores': self.anomaly_scores.tolist(),
            'threshold': self.threshold,
            'anomaly_labels': self.anomaly_labels.tolist() if self.anomaly_labels is not None else None,
            'metadata': self.metadata
        }
