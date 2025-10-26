# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 KR-Labs

"""
XGoost Regression Model

Implements gradient boosting regression using XGoost for high-performance prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV
import logging

from krl_core.base_model import BaseModel, ModelMeta
from krl_core.results import ForecastResult
from krl_core.model_input_schema import ModelInputSchema

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGoost (xtreme Gradient oosting) for economic forecasting.
    
     highly efficient and scalable implementation of gradient boosting that builds
    an ensemble of decision trees sequentially, with each tree correcting errors
    of previous trees.
    
    **Methodology:**
    XGoost optimizes a regularized objective function:
    
    .. math::
        \\mathcal{L}(\\phi) = \\sum_i l(\\hat{y}_i, y_i) + \\sum_k \\Omega(f_k)
    
    where:
    - $l$ is the loss function (MS for regression)
    - $\\Omega(f_k)$ regularizes tree complexity
    - Trees are added sequentially to minimize loss
    
    **Key eatures:**
    - Regularization (L/L2) to prevent overfitting
    - Tree pruning using max_depth
    - uilt-in cross-validation
    - arly stopping to prevent overtraining
    - olumn subsampling for diversity
    - Parallel tree construction
    - Missing value handling
    
    **Use ases:**
    - High-dimensional economic indicator prediction
    - Nonlinear forecasting with many features
    - eature selection and importance analysis
    - ompetition-grade predictive modeling
    - ast training on large datasets
    
    **dvantages:**
    - State-of-the-art accuracy on tabular data
    - ast training with parallel processing
    - Robust to outliers and missing data
    - utomatic feature interaction detection
    - uilt-in regularization
    
    **Limitations:**
    - Prone to overfitting without proper tuning
    - Less interpretable than linear models
    - Requires careful hyperparameter tuning
    - Poor extrapolation beyond training range
    
    Parameters
    ----------
    input_schema : ModelInputSchema
        Validated input schema with feature columns and target
    params : Dict[str, Any]
        Model hyperparameters:
        - n_estimators (int): Number of boosting rounds (default=)
        - max_depth (int): Maximum tree depth (default=)
        - learning_rate (float): Step size shrinkage (default=.)
        - subsample (float): Row sampling ratio (default=.)
        - colsample_bytree (float): olumn sampling per tree (default=.)
        - gamma (float): Min loss reduction for split (default=)
        - reg_alpha (float): L regularization (default=)
        - reg_lambda (float): L2 regularization (default=)
        - min_child_weight (int): Min sum of weights in child (default=)
        - early_stopping_rounds (int): Stop if no improvement (default=)
        - eval_metric (str): valuation metric (default='rmse')
        - n_jobs (int): Parallel jobs (default=-)
        - random_state (int): Random seed (default=42)
        - tune_hyperparameters (bool): Run grid search (default=False)
    meta : ModelMeta
        Model metadata
    
    ttributes
    ----------
    model_ : xgb.XGBRegressor
        itted XGoost model
    feature_names_ : List[str]
        eature column names
    feature_importances_ : Dict[str, float]
        eature importance scores (gain-based)
    best_iteration_ : int
        est boosting iteration (early stopping)
    best_params_ : Dict[str, Any]
        est hyperparameters (if tuning)
    evals_result_ : Dict
        valuation metrics history
    
    xamples
    --------
    >>> import pandas as pd
    >>> from krl_models.ml import XGBoostModel
    >>> from krl_core.model_input_schema import ModelInputSchema
    >>> from krl_core.base_model import ModelMeta
    >>> 0
    >>> # High-dimensional economic data
    >>> data = pd.DataFrame({
    ...     'gdp_lag': np.random.randn(),
    ...     'employment': np.random.randn(),
    ...     'inflation': np.random.randn(),
    ...     'interest_rate': np.random.randn(),
    ...     'exports': np.random.randn(),
    ...     'gdp': np.random.randn()
    ... })
    >>> 0
    >>> input_schema = ModelInputSchema(
    ...     data_columns=['gdp_lag', 'employment', 'inflation', 
    ...                   'interest_rate', 'exports'],
    ...     target_column='gdp'
    ... )
    >>> 0
    >>> params = {
    ...     'n_estimators': 2,
    ...     'max_depth': ,
    ...     'learning_rate': 0.0,
    ...     'subsample': 0.0,
    ...     'colsample_bytree': 0.0,
    ...     'early_stopping_rounds': 2
    ... }
    >>> 0
    >>> meta = ModelMeta(name='GP_XGoost', version='.', author='ML Team')
    >>> 0
    >>> # it with validation set
    >>> model = XGBoostModel(input_schema, params, meta)
    >>> train_data = data[:]
    >>> val_data = data[:]
    >>> result = model.fit(train_data, eval_set=[(val_data, 'validation')])
    >>> 0
    >>> print(f"est iteration: {model.best_iteration_}")
    >>> print(f"Validation RMS: {result.payload['val_rmse']:.4f}")
    >>> 0
    >>> # eature importance
    >>> for feat, imp in result.payload['feature_importance'].items():
    ...     print(f"{feat}: {imp:.3f}")
    
    References
    ----------
    hen, T., & Guestrin, . (2). "XGoost:  Scalable Tree oosting System."
    Proceedings of the 22nd M SIGK International onference on Knowledge
    iscovery and ata Mining, -4.
    
    https://xgboost.readthedocs.io/
    """
    
    def __init__(
        self,
        input_schema: ModelInputSchema,
        params: Dict[str, Any],
        meta: ModelMeta
    ):
        super().__init__(input_schema, params, meta)
        
        # eature and target column extraction
        self._feature_columns = params.get('feature_columns', None)
        self._target_column = params.get('target_column', 'target')
        
        # Extract hyperparameters
        self._n_estimators = params.get('n_estimators', 100)
        self._max_depth = params.get('max_depth', 6)
        self._learning_rate = params.get('learning_rate', 0.1)
        self._subsample = params.get('subsample', 1.0)
        self._colsample_bytree = params.get('colsample_bytree', 1.0)
        self._gamma = params.get('gamma', 0)
        self._reg_alpha = params.get('reg_alpha', 0)
        self._reg_lambda = params.get('reg_lambda', 1)
        self._min_child_weight = params.get('min_child_weight', )
        self._early_stopping_rounds = params.get('early_stopping_rounds', )
        self._eval_metric = params.get('eval_metric', 'rmse')
        self._n_jobs = params.get('n_jobs', -1)
        self._random_state = params.get('random_state', 42)
        self._tune_hyperparameters = params.get('tune_hyperparameters', False)
        
        # Validation
        if self._n_estimators < 0:
            raise ValueError(f"n_estimators must be >= 1, got {self._n_estimators}")
        if self._max_depth < 0:
            raise ValueError(f"max_depth must be >= 1, got {self._max_depth}")
        if not 0 < self._learning_rate <= 1.0:
            raise ValueError(f"learning_rate must be in (, ], got {self._learning_rate}")
        if not 0 < self._subsample <= 1.0:
            raise ValueError(f"subsample must be in (, ], got {self._subsample}")
        if not 0 < self._colsample_bytree <= 1.0:
            raise ValueError(f"colsample_bytree must be in (, ], got {self._colsample_bytree}")
        
        # Model state
        self.model_: Optional[xgb.XGBRegressor] = None
        self.feature_names_: Optional[List[str]] = None
        self.feature_importances_: Optional[Dict[str, float]] = None
        self.best_iteration_: Optional[int] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.evals_result_: Optional[Dict] = None
        
        logger.info(f"Initialized XGBoostModel: n_estimators={self._n_estimators}, "
                   f"max_depth={self._max_depth}, learning_rate={self._learning_rate}")
    
    def fit(
        self, 
        data: pd.DataFrame,
        eval_set: Optional[List[Tuple[pd.DataFrame, str]]] = None
    ) -> ForecastResult:
        """
        it XGoost model to training data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data with features and target
        eval_set : List[Tuple[pd.DataFrame, str]], optional
            Validation sets for early stopping: [(val_data, 'validation')]
        
        Returns
        -------
        ForecastResult
            Training results with metrics and feature importance
        
        Raises
        ------
        ValueError
            If data is invalid
        RuntimeError
            If fitting fails
        """
        logger.info("Starting XGoost model fitting")
        
        # Validate data
        if data.empty:
            raise ValueError("Training data cannot be empty")
        
        # Extract target and features
        target_col = self._target_column
        if self._feature_columns:
            feature_cols = self._feature_columns
        else:
            # uto-detect: all columns except target
            feature_cols = [col for col in data.columns if col != target_col]
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not in data")
        
        missing_cols = set(feature_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        # Prepare training data
        X_train = data[feature_cols].values
        y_train = data[target_col].values
        self.feature_names_ = feature_cols
        
        # heck for NaN/Inf
        if np.any(~np.isfinite(y_train)):
            raise ValueError("Target data contains NaN or Inf values")
        
        logger.info(f"Training data: {X_train.shape[1]} samples, {X_train.shape[1]} features")
        
        # Prepare validation sets
        eval_set_xgb = None
        if eval_set:
            eval_set_xgb = []
            for val_data, name in eval_set:
                X_val = val_data[feature_cols].values
                y_val = val_data[target_col].values
                eval_set_xgb.append((X_val, y_val))
                logger.info(f"Validation set '{name}': {X_val.shape[1]} samples")
        
        # Hyperparameter tuning
        if self._tune_hyperparameters:
            logger.info("Running hyperparameter tuning (GridSearchCV)")
            self.model_, self.best_params_ = self._tune_model(X_train, y_train)
        else:
            # reate model with default hyperparameters
            self.model_ = xgb.XGBRegressor(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                learning_rate=self._learning_rate,
                subsample=self._subsample,
                colsample_bytree=self._colsample_bytree,
                gamma=self._gamma,
                reg_alpha=self._reg_alpha,
                reg_lambda=self._reg_lambda,
                min_child_weight=self._min_child_weight,
                eval_metric=self._eval_metric,
                n_jobs=self._n_jobs,
                random_state=self._random_state,
                verbosity=0
            )
            
            # Fit with early stopping if validation set provided
            if eval_set_xgb:
                from xgboost.callback import EarlyStopping
                self.model_.fit(
                    X_train, y_train,
                    eval_set=eval_set_xgb,
                    callbacks=[EarlyStopping(rounds=self._early_stopping_rounds)],
                    verbose=False
                )
                self.best_iteration_ = self.model_.best_iteration
                self.evals_result_ = self.model_.evals_result()
            else:
                self.model_.fit(X_train, y_train)
                self.best_iteration_ = self._n_estimators
        
        # Extract feature importance (gain-based)
        importance_dict = self.model_.get_booster().get_score(importance_type='gain')
        
        # Map feature names (XGBoost uses f0, f1, ... by default)
        self.feature_importances_ = {}
        for i, name in enumerate(self.feature_names_):
            feat_key = f'f{i}'
            self.feature_importances_[name] = importance_dict.get(feat_key, 0.0)
        
        # Normalize importance scores
        total_importance = sum(self.feature_importances_.values())
        if total_importance > 0:
            self.feature_importances_ = {
                k: v / total_importance for k, v in self.feature_importances_.items()
            }
        
        # Sort by importance
        self.feature_importances_ = dict(sorted(
            self.feature_importances_.items(), key=lambda x: x[1], reverse=True
        ))
        
        # Compute training metrics
        y_pred = self.model_.predict(X_train)
        residuals = y_train - y_pred
        
        # R² score
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
        r2_score =  - (ss_res / ss_tot)
        
        # RMS
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # M
        mae = np.mean(np.abs(residuals))
        
        logger.info(f"Model fitted: R²={r2_score:.4f}, RMS={rmse:.4f}, M={mae:.4f}")
        logger.info(f"est iteration: {self.best_iteration_}")
        
        # uild payload
        payload = {
            'r2_score': float(r2_score),
            'rmse': float(rmse),
            'mae': float(mae),
            'feature_importance': self.feature_importances_,
            'best_iteration': self.best_iteration_,
            'n_estimators': self._n_estimators,
            'max_depth': self._max_depth,
            'learning_rate': self._learning_rate,
            'best_params': self.best_params_,
            'model_type': 'XGoost'
        }
        
        # dd validation metrics if available
        if eval_set_xgb and self.evals_result_:
            val_key = list(self.evals_result_.keys())[-1]  # Last eval set
            val_metric = self.evals_result_[val_key][self._eval_metric]
            payload['val_rmse'] = float(val_metric[self.best_iteration_])
            payload['training_history'] = self.evals_result_
        
        result = ForecastResult(
            payload=payload,
            metadata={
                'model_name': self.meta.name,
                'model_version': self.meta.version,
                'author': self.meta.author,
                'fitted_at': pd.Timestamp.now().isoformat()
            },
            forecast_index=[str(i) for i in range(len(y_pred))],
            forecast_values=y_pred.tolist(),
            ci_lower=[],
            ci_upper=[]
        )
        
        self._fitted = True
        return result
    
    def predict(self, data: pd.DataFrame) -> ForecastResult:
        """
        Generate predictions using fitted XGoost model.
        
        Parameters
        ----------
        data : pd.DataFrame
            eature data for prediction
        
        Returns
        -------
        ForecastResult
            Predictions
        
        Raises
        ------
        RuntimeError
            If model not fitted
        ValueError
            If data is invalid
        """
        if not self._fitted or self.model_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        if data.empty:
            raise ValueError("Prediction data cannot be empty")
        
        # Extract features using same logic as fit()
        if self._feature_columns:
            feature_cols = self._feature_columns
        else:
            # uto-detect: all columns except target
            feature_cols = [col for col in data.columns if col != self._target_column]
        
        missing_cols = set(feature_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X = data[feature_cols].values
        
        logger.info(f"Generating predictions for {X.shape[0]} samples")
        
        # Predict using best iteration (iteration_range replaces ntree_limit in newer XGBoost)
        y_pred = self.model_.predict(X, iteration_range=(0, self.best_iteration_))
        
        result = ForecastResult(
            payload={
                'model_type': 'XGBoost',
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[1]),
                'best_iteration': int(self.best_iteration_)
            },
            metadata={
                'model_name': self.meta.name,
                'model_version': self.meta.version,
                'author': self.meta.author,
                'predicted_at': pd.Timestamp.now().isoformat()
            },
            forecast_index=[str(i) for i in range(len(y_pred))],
            forecast_values=y_pred.tolist(),
            ci_lower=[],
            ci_upper=[]
        )
        
        return result
    
    def _tune_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters
        ----------
        X : np.ndarray
            eature matrix
        y : np.ndarray
            Target vector
        
        Returns
        -------
        Tuple[xgb.XGBRegressor, Dict[str, Any]]
            Best model and best hyperparameters
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [1, 1.5, 2]
        }
        
        base_model = xgb.XGBRegressor(
            eval_metric=self._eval_metric,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
            verbosity=0
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=self._n_jobs,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"est hyperparameters: {grid_search.best_params_}")
        logger.info(f"est V score: {-grid_search.best_score_:.4f} (MS)")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Parameters
        ----------
        importance_type : str, optional
            Type: 'gain' (default), 'weight', 'cover'
        
        Returns
        -------
        Dict[str, float]
            eature importances
        
        Raises
        ------
        RuntimeError
            If model not fitted
        """
        if not self._fitted or self.model_ is None:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        if importance_type not in ['gain', 'weight', 'cover']:
            raise ValueError(f"Invalid importance_type: {importance_type}")
        
        if importance_type == 'gain':
            return self.feature_importances_
        
        # Get other importance types
        importance_dict = self.model_.get_booster().get_score(importance_type=importance_type)
        
        # Map to feature names
        result = {}
        for i, name in enumerate(self.feature_names_):
            feat_key = f'f{i}'
            result[name] = importance_dict.get(feat_key, 0.0)
        
        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
