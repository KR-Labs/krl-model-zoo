# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 KR-Labs

"""
Random orest Regression Model

Implements ensemble decision tree regression for nonlinear economic forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance
import logging

from krl_core.base_model import BaseModel, ModelMeta
from krl_core.results import ForecastResult
from krl_core.model_input_schema import ModelInputSchema

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random orest Regression for economic forecasting.
    
    n ensemble method that builds multiple decision trees and averages their predictions
    to capture nonlinear relationships and interactions between features.
    
    **Methodology:**
    Random orest creates diversity through:
    . ootstrap sampling (different training sets per tree)
    2. Random feature subsets at each split
    3. ggregation via averaging (bagging)
    
    **Use ases:**
    - Multivariate economic forecasting with complex interactions
    - eature importance analysis for policy variables
    - Nonlinear relationship modeling (GP, employment, demographics)
    - High-dimensional prediction tasks
    - Robust predictions with automatic variance estimation
    
    **dvantages:**
    - Handles nonlinear relationships naturally
    - Robust to outliers and overfitting
    - Provides feature importance rankings
    - No need for feature scaling
    - Works well with mixed data types
    
    **Limitations:**
    - Less interpretable than linear models
    - an be slow for large datasets/many trees
    - xtrapolation beyond training range is poor
    - Memory intensive for large forests
    
    Parameters
    ----------
    input_schema : ModelInputSchema
        Validated input schema with feature columns and target
    params : Dict[str, Any]
        Model hyperparameters:
        - n_estimators (int): Number of trees (default=)
        - max_depth (int): Maximum tree depth (default=None, unlimited)
        - min_samples_split (int): Min samples to split node (default=2)
        - min_samples_leaf (int): Min samples per leaf (default=)
        - max_features (str/int/float): eatures per split (default='sqrt')
        - bootstrap (bool): Use bootstrap sampling (default=True)
        - oob_score (bool): Use out-of-bag score (default=True if bootstrap=True)
        - n_jobs (int): Parallel jobs (default=-, use all cores)
        - random_state (int): Random seed (default=42)
        - tune_hyperparameters (bool): Run grid search (default=False)
    meta : ModelMeta
        Model metadata (name, version, author, description)
    
    ttributes
    ----------
    model_ : RandomForestRegressor
        itted scikit-learn model
    feature_names_ : List[str]
        eature column names
    feature_importances_ : np.ndarray
        eature importance scores (Gini-based)
    permutation_importances_ : Dict[str, np.ndarray]
        Permutation-based feature importances
    oob_score_ : float
        Out-of-bag R² score (if bootstrap=True)
    best_params_ : Dict[str, Any]
        est hyperparameters (if tuning enabled)
    
    xamples
    --------
    >>> import pandas as pd
    >>> from krl_models.ml import RandomorestModel
    >>> from krl_core.model_input_schema import ModelInputSchema
    >>> from krl_core.base_model import ModelMeta
    >>> 0
    >>> # Prepare data
    >>> data = pd.DataFrame({
    ...     'gdp_lag': [, , , , ],
    ...     'employment': [, , , , 2],
    ...     'interest_rate': [2., 2.3, 2., 2., .],
    ...     'gdp': [, , , , 2]
    ... })
    >>> 0
    >>> # onfigure model
    >>> input_schema = ModelInputSchema(
    ...     data_columns=['gdp_lag', 'employment', 'interest_rate'],
    ...     target_column='gdp',
    ...     index_col=None
    ... )
    >>> 0
    >>> params = {
    ...     'n_estimators': ,
    ...     'max_depth': ,
    ...     'min_samples_leaf': 2,
    ...     'random_state': 42
    ... }
    >>> 0
    >>> meta = ModelMeta(
    ...     name='GP_Randomorest',
    ...     version='.',
    ...     author='conomics Team'
    ... )
    >>> 0
    >>> # it and predict
    >>> model = RandomorestModel(input_schema, params, meta)
    >>> result = model.fit(data)
    >>> print(f"R² Score: {result.payload['r2_score']:.3f}")
    >>> print(f"RMS: {result.payload['rmse']:.2f}")
    >>> 0
    >>> # eature importance
    >>> for feat, imp in result.payload['feature_importance'].items():
    ...     print(f"{feat}: {imp:.3f}")
    >>> 0
    >>> # Predict
    >>> new_data = pd.DataFrame({
    ...     'gdp_lag': [2],
    ...     'employment': [3],
    ...     'interest_rate': [.]
    ... })
    >>> forecast = model.predict(new_data)
    >>> print(f"orecast: {forecast.forecast_values[]:.2f}")
    
    References
    ----------
    reiman, L. (2). "Random orests." Machine Learning, 4(), -32.
    
    Hastie, T., Tibshirani, R., & riedman, J. (2). The lements of Statistical
    Learning (2nd ed.). Springer.
    """
    
    def __init__(
        self,
        input_schema: ModelInputSchema,
        params: Dict[str, Any],
        meta: ModelMeta
    ):
        super().__init__(input_schema, params, meta)
        
        # Extract ML-specific parameters
        self._feature_columns = params.get('feature_columns', None)
        self._target_column = params.get('target_column', 'target')
        
        # Extract hyperparameters
        self._n_estimators = params.get('n_estimators', 100)
        self._max_depth = params.get('max_depth', None)
        self._min_samples_split = params.get('min_samples_split', 2)
        self._min_samples_leaf = params.get('min_samples_leaf', 1)
        self._max_features = params.get('max_features', 'sqrt')
        self._bootstrap = params.get('bootstrap', True)
        self._oob_score = params.get('oob_score', self._bootstrap)
        self._n_jobs = params.get('n_jobs', -1)
        self._random_state = params.get('random_state', 42)
        self._tune_hyperparameters = params.get('tune_hyperparameters', False)
        
        # Validation
        if self._n_estimators < 0:
            raise ValueError(f"n_estimators must be >= 1, got {self._n_estimators}")
        if self._max_depth is not None and self._max_depth < 0:
            raise ValueError(f"max_depth must be >=  or None, got {self._max_depth}")
        if not self._bootstrap and self._oob_score:
            raise ValueError("oob_score requires bootstrap=True")
        
        # Model state
        self.model_: Optional[RandomForestRegressor] = None
        self.feature_names_: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.permutation_importances_: Optional[Dict[str, np.ndarray]] = None
        self.oob_score_: Optional[float] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        
        logger.info(f"Initialized RandomorestModel: n_estimators={self._n_estimators}, "
                   f"max_depth={self._max_depth}, tune={self._tune_hyperparameters}")
    
    def fit(self, data: pd.DataFrame) -> ForecastResult:
        """
        it Random orest model to training data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data with features and target column
        
        Returns
        -------
        ForecastResult
            Training results with metrics and feature importance
        
        Raises
        ------
        ValueError
            If data is invalid or missing required columns
        RuntimeError
            If model fitting fails
        """
        logger.info("Starting Random orest model fitting")
        
        # Validate and extract data
        if data.empty:
            raise ValueError("Training data cannot be empty")
        
        # Get target and feature columns from params
        target_col = self._target_column
        
        if self._feature_columns is None:
            # uto-detect: all columns except target
            feature_cols = [col for col in data.columns if col != target_col]
        else:
            feature_cols = self._feature_columns
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not in data")
        
        missing_cols = set(feature_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        # Prepare features and target
        X = data[feature_cols].values
        y = data[target_col].values
        self.feature_names_ = feature_cols
        
        # heck for NaN/Inf
        if np.any(~np.isfinite(X)):
            raise ValueError("eature data contains NaN or Inf values")
        if np.any(~np.isfinite(y)):
            raise ValueError("Target data contains NaN or Inf values")
        
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[0]} features")
        
        # Hyperparameter tuning
        if self._tune_hyperparameters:
            logger.info("Running hyperparameter tuning (GridSearchCV)")
            self.model_, self.best_params_ = self._tune_model(X, y)
        else:
            # Use default hyperparameters
            self.model_ = RandomForestRegressor(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                min_samples_split=self._min_samples_split,
                min_samples_leaf=self._min_samples_leaf,
                max_features=self._max_features,
                bootstrap=self._bootstrap,
                oob_score=self._oob_score,
                n_jobs=self._n_jobs,
                random_state=self._random_state,
                verbose=0
            )
            
            # it model
            self.model_.fit(X, y)
        
        # Extract feature importances
        self.feature_importances_ = self.model_.feature_importances_
        
        # Compute permutation importance
        perm_importance = permutation_importance(
            self.model_, X, y, n_repeats=10, random_state=self._random_state, n_jobs=self._n_jobs
        )
        self.permutation_importances_ = {
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std
        }
        
        # Extract OO score
        if self._oob_score:
            self.oob_score_ = self.model_.oob_score_
        
        # Compute training metrics
        y_pred = self.model_.predict(X)
        residuals = y - y_pred
        
        # R² score
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_score =  - (ss_res / ss_tot)
        
        # RMS
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # M
        mae = np.mean(np.abs(residuals))
        
        logger.info(f"Model fitted: R²={r2_score:.4f}, RMS={rmse:.4f}, M={mae:.4f}")
        if self._oob_score:
            logger.info(f"OO Score: {self.oob_score_:.4f}")
        
        # reate feature importance dict
        feature_importance = {
            name: float(imp) for name, imp in zip(self.feature_names_, self.feature_importances_)
        }
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True))
        
        # uild result with proper ForecastResult structure
        self._fitted = True
        
        result = ForecastResult(
            payload={
                'r2_score': float(r2_score),
                'rmse': float(rmse),
                'mae': float(mae),
                'feature_importance': feature_importance,
                'n_samples': int(X.shape[0]),
                'n_features': int(X.shape[0]),
                'n_estimators': self._n_estimators,
                'max_depth': self._max_depth,
                'oob_score': float(self.oob_score_) if self._oob_score and self.oob_score_ is not None else None,
                'best_params': self.best_params_ if self.best_params_ else None,
                'model_type': 'Randomorest'
            },
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
        
        return result
    
    def predict(
        self, 
        data: pd.DataFrame,
        return_std: bool = False
    ) -> ForecastResult:
        """
        Generate predictions using fitted Random orest model.
        
        Parameters
        ----------
        data : pd.DataFrame
            eature data for prediction
        return_std : bool, optional
            If True, return standard deviation of tree predictions
        
        Returns
        -------
        ForecastResult
            Predictions with optional uncertainty estimates
        
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
        
        # heck for NaN/Inf
        if np.any(~np.isfinite(X)):
            raise ValueError("eature data contains NaN or Inf values")
        
        logger.info(f"Generating predictions for {X.shape[0]} samples")
        
        # Predict
        y_pred = self.model_.predict(X)
        
        # uild payload
        payload = {
            'model_type': 'Randomorest',
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[0])
        }
        
        # Standard deviation from individual trees
        ci_lower = []
        ci_upper = []
        if return_std:
            # Get predictions from all trees
            all_predictions = np.array([tree.predict(X) for tree in self.model_.estimators_])
            y_std = np.std(all_predictions, axis=0)
            
            payload['prediction_std'] = y_std.tolist()
            ci_lower = (y_pred - 1.96 * y_std).tolist()
            ci_upper = (y_pred + 1.96 * y_std).tolist()
        
        result = ForecastResult(
            payload=payload,
            metadata={
                'model_name': self.meta.name,
                'model_version': self.meta.version,
                'author': self.meta.author,
                'predicted_at': pd.Timestamp.now().isoformat()
            },
            forecast_index=[str(i) for i in range(len(y_pred))],
            forecast_values=y_pred.tolist(),
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )
        
        return result
    
    def _tune_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
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
        Tuple[RandomForestRegressor, Dict[str, Any]]
            Best model and best hyperparameters
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        base_model = RandomForestRegressor(
            bootstrap=self._bootstrap,
            oob_score=self._oob_score,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
            verbose=0
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
    
    def get_feature_importance(self, importance_type: str = 'gini') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Parameters
        ----------
        importance_type : str, optional
            Type of importance: 'gini' (default) or 'permutation'
        
        Returns
        -------
        Dict[str, float]
            eature names mapped to importance scores
        
        Raises
        ------
        RuntimeError
            If model not fitted
        ValueError
            If invalid importance_type
        """
        if not self._fitted or self.model_ is None:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        if importance_type == 'gini':
            return {
                name: float(imp) for name, imp in 
                zip(self.feature_names_, self.feature_importances_)
            }
        elif importance_type == 'permutation':
            return {
                name: float(imp) for name, imp in 
                zip(self.feature_names_, self.permutation_importances_['importances_mean'])
            }
        else:
            raise ValueError(f"Invalid importance_type: {importance_type}. "
                           f"Must be 'gini' or 'permutation'")
