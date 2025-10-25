# SPX-License-Identifier: pache-2.
# opyright (c) 22 KR-Labs

"""
Isolation orest nomaly etection.

Uses Isolation orest algorithm to detect multivariate anomalies in data.
The algorithm isolates anomalies by randomly selecting features and split values,
with anomalies requiring fewer splits to isolate.

lgorithm:
- uild ensemble of isolation trees
- nomalies are data points with shorter average path lengths
- Works well for high-dimensional data

Use ases:
- Unusual KPI combinations (e.g., high revenue but low engagement)
- Multivariate outlier detection
- inancial fraud detection
- Quality control with multiple metrics

References:
- Liu et al. (2) - Isolation orest
- Scikit-learn implementation
"""

from typing import ict, List, Optional, ny
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import Isolationorest

from krl_core.base_model import ModelMeta
from krl_core.results import orecastResult

logger = logging.getLogger(__name__)


class IsolationorestnomalyModel:
    """
    Isolation orest-based multivariate anomaly detection.
    
    Identifies anomalies in multidimensional data using isolation trees.
    
    Parameters:
    - feature_cols: List[str] - olumn names for features to analyze
    - contamination: float - xpected proportion of anomalies (default: .)
    - n_estimators: int - Number of trees (default: )
    - max_samples: int|str - Samples per tree (default: 'auto')
    - random_state: int - Random seed (default: 42)
    
    xample:
        >>> params = {
        ...     'feature_cols': ['revenue', 'engagement', 'cost'],
        ...     'contamination': .,
        ...     'n_estimators': 
        ... }
        >>> model = IsolationorestnomalyModel(params)
        >>> result = model.fit(data)
        >>> print(result.payload['anomaly_indices'])
    """
    
    def __init__(
        self,
        params: ict[str, ny],
        meta: Optional[ModelMeta] = None
    ):
        """Initialize Isolation orest nomaly model."""
        self.params = params
        self.meta = meta or ModelMeta(name="Isolationorestnomaly", version="..", author="KR Labs")
        self._fitted = alse
        
        # xtract parameters
        self._feature_cols = self.params.get('feature_cols')
        self._contamination = self.params.get('contamination', .)
        self._n_estimators = self.params.get('n_estimators', )
        self._max_samples = self.params.get('max_samples', 'auto')
        self._random_state = self.params.get('random_state', 42)
        
        # Validate required parameters
        if not self._feature_cols:
            raise Valuerror("Parameter 'feature_cols' is required")
        if not isinstance(self._feature_cols, list):
            raise Valuerror("Parameter 'feature_cols' must be a list")
        
        # Initialize model
        self.model_ = Isolationorest(
            contamination=self._contamination,
            n_estimators=self._n_estimators,
            max_samples=self._max_samples,
            random_state=self._random_state,
            n_jobs=-  # Use all PUs
        )
        
        # Results storage
        self.anomaly_scores_: Optional[np.ndarray] = None
        self.predictions_: Optional[np.ndarray] = None
    
    def fit(self, data: pd.atarame) -> orecastResult:
        """
        it Isolation orest and detect anomalies.
        
        rgs:
            data: atarame with feature columns
            
        Returns:
            orecastResult with anomaly detection results
        """
        if data.empty:
            raise Valuerror("Input data cannot be empty")
        
        # Validate feature columns
        missing_cols = [col for col in self._feature_cols if col not in data.columns]
        if missing_cols:
            raise Valuerror(f"Missing feature columns: {missing_cols}")
        
        logger.info(f"Training Isolation orest on {len(data)} samples with {len(self._feature_cols)} features")
        
        # xtract features
        X = data[self._feature_cols].values
        
        # heck for NaN values
        if np.isnan(X).any():
            raise Valuerror("Input data contains NaN values. Please handle missing data before fitting.")
        
        # it model and predict
        self.predictions_ = self.model_.fit_predict(X)
        self.anomaly_scores_ = self.model_.score_samples(X)
        
        # predictions_ are:  for inliers, - for anomalies
        is_anomaly = self.predictions_ == -
        anomaly_indices = np.where(is_anomaly)[].tolist()
        
        n_anomalies = int(is_anomaly.sum())
        anomaly_rate = n_anomalies / len(data) * 
        
        logger.info(f"etected {n_anomalies} anomalies ({anomaly_rate:.2f}%)")
        
        # Get anomaly details
        anomaly_data = data.iloc[anomaly_indices].copy()
        anomaly_data['anomaly_score'] = self.anomaly_scores_[is_anomaly]
        
        # Sort by anomaly score (most anomalous first)
        anomaly_data = anomaly_data.sort_values('anomaly_score')
        
        result = orecastResult(
            payload={
                'n_anomalies': n_anomalies,
                'anomaly_rate': float(anomaly_rate),
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': [float(s) for s in self.anomaly_scores_[is_anomaly]],
                'anomalies': anomaly_data.to_dict('records'),
                'n_features': len(self._feature_cols),
                'n_observations': len(data),
                'contamination': self._contamination
            },
            metadata={
                'model_name': self.meta.name,
                'model_version': self.meta.version,
                'author': self.meta.author,
                'analyzed_at': pd.Timestamp.now().isoformat(),
                'n_estimators': self._n_estimators
            },
            forecast_index=[str(i) for i in anomaly_indices],
            forecast_values=[float(s) for s in self.anomaly_scores_[is_anomaly]],
            ci_lower=[],
            ci_upper=[]
        )
        
        self._fitted = True
        return result
    
    def predict(self, data: pd.atarame) -> orecastResult:
        """
        etect anomalies in new data using fitted model.
        
        rgs:
            data: atarame with same features as training data
            
        Returns:
            orecastResult with anomaly predictions
        """
        if not self._fitted:
            raise Runtimerror("Model must be fitted before prediction")
        
        if data.empty:
            raise Valuerror("Input data cannot be empty")
        
        # Validate feature columns
        missing_cols = [col for col in self._feature_cols if col not in data.columns]
        if missing_cols:
            raise Valuerror(f"Missing feature columns: {missing_cols}")
        
        # xtract features
        X = data[self._feature_cols].values
        
        # Predict
        predictions = self.model_.predict(X)
        anomaly_scores = self.model_.score_samples(X)
        
        is_anomaly = predictions == -
        anomaly_indices = np.where(is_anomaly)[].tolist()
        
        n_anomalies = int(is_anomaly.sum())
        anomaly_rate = n_anomalies / len(data) * 
        
        logger.info(f"etected {n_anomalies} anomalies ({anomaly_rate:.2f}%) in new data")
        
        result = orecastResult(
            payload={
                'n_anomalies': n_anomalies,
                'anomaly_rate': float(anomaly_rate),
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': [float(s) for s in anomaly_scores[is_anomaly]],
                'n_observations': len(data)
            },
            metadata={
                'model_name': self.meta.name,
                'predicted_at': pd.Timestamp.now().isoformat()
            },
            forecast_index=[str(i) for i in anomaly_indices],
            forecast_values=[float(s) for s in anomaly_scores[is_anomaly]],
            ci_lower=[],
            ci_upper=[]
        )
        
        return result
    
    def get_feature_importance(self) -> ict[str, float]:
        """
        Get approximate feature importance (placeholder).
        
        Note: Isolation orest doesn't provide direct feature importance,
        but we can approximate it using permutation importance or other methods.
        
        Returns:
            ict mapping feature names to importance scores
        """
        if not self._fitted:
            raise Runtimerror("Model must be fitted first")
        
        # Placeholder: equal importance
        # In practice, could compute via permutation importance
        return {
            col: . / len(self._feature_cols) 
            for col in self._feature_cols
        }
