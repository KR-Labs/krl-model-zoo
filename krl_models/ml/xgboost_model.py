"""
XGBoost Regressor for economic forecasting
===========================================

MIT License - Gate 2 Phase 2.3 ML Baseline
author: KR Labs
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult


class XGBoostModel(BaseModel):
    """
    XGBoost gradient boosting regressor with early stopping.
    
    Sequentially builds decision trees where each tree corrects errors
    of previous trees. Uses second-order gradients for optimization.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of boosting rounds
    max_depth : int, default=6
        Maximum tree depth
    learning_rate : float, default=0.3
        Shrinkage rate for boosting (eta)
    subsample : float, default=1.0
        Fraction of samples for training each tree
    colsample_bytree : float, default=1.0
        Fraction of features for training each tree
    gamma : float, default=0
        Minimum loss reduction for split (regularization)
    reg_alpha : float, default=0
        L1 regularization on weights
    reg_lambda : float, default=1
        L2 regularization on weights
    early_stopping_rounds : int or None, default=10
        Stop if validation metric doesn't improve for this many rounds
    random_state : int or None, default=42
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        early_stopping_rounds: Optional[int] = 10,
        random_state: Optional[int] = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        
        self.model_ = None
        self.feature_names_ = None
        self.feature_importances_ = None
        self.best_iteration_ = None
    
    def fit(
        self, 
        y: np.ndarray, 
        X: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> 'XGBoostModel':
        """
        Fit XGBoost model with optional early stopping.
        
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target variable
        X : array-like, shape (n_samples, n_features)
            Feature matrix (required)
        X_val : array-like, shape (n_val_samples, n_features), optional
            Validation features for early stopping
        y_val : array-like, shape (n_val_samples,), optional
            Validation targets for early stopping
        **kwargs : dict
            Additional arguments (feature_names, verbose)
        
        Returns
        -------
        self : XGBoostModel
            Fitted model instance
        """
        if X is None:
            raise ValueError("XGBoost requires feature matrix X")
        
        # Store feature names
        self.feature_names_ = kwargs.get('feature_names',
                                         [f'X{i}' for i in range(X.shape[1])])
        
        # Prepare training data
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names_)
        
        # Prepare validation data if provided
        eval_set = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names_)
            eval_set = [(dtrain, 'train'), (dval, 'val')]
        else:
            eval_set = [(dtrain, 'train')]
        
        # XGBoost parameters
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'objective': 'reg:squarederror',
            'seed': self.random_state
        }
        
        # Train model
        verbose_eval = kwargs.get('verbose', False)
        
        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=eval_set,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=verbose_eval
        )
        
        # Store best iteration
        if hasattr(self.model_, 'best_iteration'):
            self.best_iteration_ = self.model_.best_iteration
        
        # Store feature importances
        importance_dict = self.model_.get_score(importance_type='gain')
        self.feature_importances_ = {
            feat: importance_dict.get(feat, 0.0) for feat in self.feature_names_
        }
        
        return self
    
    def forecast(self, steps: int = 1, X_future: Optional[np.ndarray] = None, **kwargs) -> ForecastResult:
        """
        Generate forecasts using fitted model.
        
        Parameters
        ----------
        steps : int, default=1
            Number of steps to forecast
        X_future : array-like, shape (steps, n_features)
            Future feature values (required)
        **kwargs : dict
            Additional arguments (alpha for intervals - not supported by XGBoost)
        
        Returns
        -------
        result : ForecastResult
            Point forecasts (no prediction intervals from standard XGBoost)
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before forecasting")
        
        if X_future is None:
            raise ValueError("XGBoost requires future features X_future")
        
        if X_future.shape[0] != steps:
            raise ValueError(f"X_future must have {steps} rows")
        
        # Prepare future data
        dfuture = xgb.DMatrix(X_future, feature_names=self.feature_names_)
        
        # Point forecast
        point_forecast = self.model_.predict(dfuture)
        
        # XGBoost doesn't provide native prediction intervals
        # Return NaN intervals
        alpha = kwargs.get('alpha', 0.05)
        
        return ForecastResult(
            point_forecast=point_forecast,
            lower=np.full(steps, np.nan),
            upper=np.full(steps, np.nan),
            alpha=alpha
        )
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importance scores (gain-based).
        
        Returns
        -------
        importances : dict
            Feature names mapped to importance scores
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be fitted to compute feature importances")
        
        return self.feature_importances_
