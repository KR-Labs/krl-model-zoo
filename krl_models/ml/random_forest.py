"""
Random Forest Regressor for economic forecasting
================================================

MIT License - Gate 2 Phase 2.3 ML Baseline
author: KR Labs
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult


class RandomForestModel(BaseModel):
    """
    Random Forest regressor with hyperparameter tuning.
    
    Ensemble of decision trees with bootstrap aggregating (bagging).
    Reduces overfitting via averaging multiple trees trained on random subsets.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int or None, default=None
        Maximum depth of trees (None = unlimited)
    min_samples_split : int, default=2
        Minimum samples required to split internal node
    min_samples_leaf : int, default=1
        Minimum samples required at leaf node
    max_features : str or float, default='sqrt'
        Number of features for best split ('sqrt', 'log2', or float)
    bootstrap : bool, default=True
        Use bootstrap sampling when building trees
    random_state : int or None, default=42
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all processors)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        bootstrap: bool = True,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model_ = None
        self.feature_names_ = None
        self.feature_importances_ = None
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None, **kwargs) -> 'RandomForestModel':
        """
        Fit Random Forest model to training data.
        
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target variable
        X : array-like, shape (n_samples, n_features)
            Feature matrix (required)
        **kwargs : dict
            Additional arguments (feature_names for tracking)
        
        Returns
        -------
        self : RandomForestModel
            Fitted model instance
        """
        if X is None:
            raise ValueError("Random Forest requires feature matrix X")
        
        # Store feature names if provided
        self.feature_names_ = kwargs.get('feature_names', 
                                         [f'X{i}' for i in range(X.shape[1])])
        
        # Initialize model
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Fit model
        self.model_.fit(X, y)
        
        # Store feature importances
        self.feature_importances_ = dict(zip(
            self.feature_names_,
            self.model_.feature_importances_
        ))
        
        return self
    
    def forecast(self, steps: int = 1, X_future: Optional[np.ndarray] = None, **kwargs) -> ForecastResult:
        """
        Generate forecasts using fitted model.
        
        Parameters
        ----------
        steps : int, default=1
            Number of steps to forecast (must match X_future rows)
        X_future : array-like, shape (steps, n_features)
            Future feature values (required)
        **kwargs : dict
            Additional arguments (alpha for prediction intervals)
        
        Returns
        -------
        result : ForecastResult
            Forecasts with prediction intervals from tree quantiles
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before forecasting")
        
        if X_future is None:
            raise ValueError("Random Forest requires future features X_future")
        
        if X_future.shape[0] != steps:
            raise ValueError(f"X_future must have {steps} rows (one per forecast step)")
        
        # Point forecasts
        point_forecast = self.model_.predict(X_future)
        
        # Prediction intervals from tree predictions
        alpha = kwargs.get('alpha', 0.05)
        all_predictions = np.array([tree.predict(X_future) for tree in self.model_.estimators_])
        
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower = np.percentile(all_predictions, lower_quantile * 100, axis=0)
        upper = np.percentile(all_predictions, upper_quantile * 100, axis=0)
        
        return ForecastResult(
            point_forecast=point_forecast,
            lower=lower,
            upper=upper,
            alpha=alpha
        )
    
    def cross_validate(self, y: np.ndarray, X: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform k-fold cross-validation.
        
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target variable
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        cv : int, default=5
            Number of folds
        
        Returns
        -------
        scores : dict
            Mean and std of cross-validation scores (R^2)
        """
        if self.model_ is None:
            # Create temporary model for CV
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:
            model = self.model_
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=self.n_jobs)
        
        return {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'scores': scores.tolist()
        }
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importance scores (Gini importance).
        
        Returns
        -------
        importances : dict
            Feature names mapped to importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted to compute feature importances")
        
        return self.feature_importances_
