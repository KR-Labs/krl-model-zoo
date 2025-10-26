"""
Isolation Forest for Anomaly Detection
=======================================

MIT License - Gate 2 Phase 2.5 Anomaly Detection
author: KR Labs
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as SklearnIsolationForest

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult


class IsolationForestAnomalyModel(BaseModel):
    """
    Isolation Forest for multivariate anomaly detection.
    
    Isolates observations by randomly selecting features and split values.
    Anomalies require fewer splits (shorter path lengths) to isolate.
    
    Parameters
    ----------
    contamination : float or 'auto', default='auto'
        Expected proportion of anomalies in dataset
    n_estimators : int, default=100
        Number of isolation trees
    max_samples : int or float, default='auto'
        Samples per tree ('auto' = min(256, n_samples))
    max_features : int or float, default=1.0
        Features per tree (1.0 = all features)
    random_state : int or None, default=42
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        contamination: str | float = 'auto',
        n_estimators: int = 100,
        max_samples: str | int | float = 'auto',
        max_features: int | float = 1.0,
        random_state: Optional[int] = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        
        self.model_ = None
        self.feature_names_ = None
        self.anomaly_scores_ = None
        self.anomaly_labels_ = None
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None, **kwargs) -> 'IsolationForestAnomalyModel':
        """
        Fit Isolation Forest to data.
        
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target variable (used as feature if X is None)
        X : array-like, shape (n_samples, n_features), optional
            Feature matrix for multivariate anomaly detection
        **kwargs : dict
            Additional arguments (feature_names)
        
        Returns
        -------
        self : IsolationForestAnomalyModel
            Fitted model instance
        """
        # Use y as single feature if X not provided
        if X is None:
            X = y.reshape(-1, 1)
            self.feature_names_ = ['y']
        else:
            self.feature_names_ = kwargs.get('feature_names',
                                             [f'X{i}' for i in range(X.shape[1])])
        
        # Initialize model
        self.model_ = SklearnIsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
        # Fit model
        self.model_.fit(X)
        
        # Store anomaly predictions
        self.anomaly_labels_ = self.model_.predict(X)  # 1 = normal, -1 = anomaly
        self.anomaly_scores_ = self.model_.decision_function(X)  # Higher = more normal
        
        return self
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """
        Get detected anomalies.
        
        Returns
        -------
        anomalies : dict
            Anomaly indices, labels, scores, and summary statistics
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before detecting anomalies")
        
        # Anomaly indices (where label == -1)
        anomaly_indices = np.where(self.anomaly_labels_ == -1)[0]
        
        return {
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_count': len(anomaly_indices),
            'anomaly_labels': self.anomaly_labels_.tolist(),
            'anomaly_scores': self.anomaly_scores_.tolist(),
            'contamination': self.contamination
        }
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for new samples.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New samples to score
        
        Returns
        -------
        scores : array-like, shape (n_samples,)
            Anomaly scores (higher = more normal)
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before scoring")
        
        return self.model_.decision_function(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels for new samples.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New samples to classify
        
        Returns
        -------
        labels : array-like, shape (n_samples,)
            Anomaly labels (1 = normal, -1 = anomaly)
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before predicting")
        
        return self.model_.predict(X)
    
    def forecast(self, steps: int = 1, X_future: Optional[np.ndarray] = None, **kwargs) -> ForecastResult:
        """
        Not applicable for anomaly detection.
        
        Raises
        ------
        NotImplementedError
            Isolation Forest is for anomaly detection, not forecasting
        """
        raise NotImplementedError("Isolation Forest does not support forecasting")
