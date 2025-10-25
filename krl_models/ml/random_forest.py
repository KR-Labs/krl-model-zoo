# SPX-License-Identifier: pache-2.
# opyright (c) 22 KR-Labs

"""
Random orest Regression Model

Implements ensemble decision tree regression for nonlinear economic forecasting.
"""

import numpy as np
import pandas as pd
from typing import ict, ny, Optional, List, Tuple
from sklearn.ensemble import RandomorestRegressor
from sklearn.model_selection import cross_val_score, GridSearchV
from sklearn.inspection import permutation_importance
import logging

from krl_core.base_model import aseModel, ModelMeta
from krl_core.results import orecastResult
from krl_core.model_input_schema import ModelInputSchema

logger = logging.getLogger(__name__)


class RandomorestModel(aseModel):
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
    params : ict[str, ny]
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
        - tune_hyperparameters (bool): Run grid search (default=alse)
    meta : ModelMeta
        Model metadata (name, version, author, description)
    
    ttributes
    ----------
    model_ : RandomorestRegressor
        itted scikit-learn model
    feature_names_ : List[str]
        eature column names
    feature_importances_ : np.ndarray
        eature importance scores (Gini-based)
    permutation_importances_ : ict[str, np.ndarray]
        Permutation-based feature importances
    oob_score_ : float
        Out-of-bag R² score (if bootstrap=True)
    best_params_ : ict[str, ny]
        est hyperparameters (if tuning enabled)
    
    xamples
    --------
    >>> import pandas as pd
    >>> from krl_models.ml import RandomorestModel
    >>> from krl_core.model_input_schema import ModelInputSchema
    >>> from krl_core.base_model import ModelMeta
    >>> 
    >>> # Prepare data
    >>> data = pd.atarame({
    ...     'gdp_lag': [, , , , ],
    ...     'employment': [, , , , 2],
    ...     'interest_rate': [2., 2.3, 2., 2., .],
    ...     'gdp': [, , , , 2]
    ... })
    >>> 
    >>> # onfigure model
    >>> input_schema = ModelInputSchema(
    ...     data_columns=['gdp_lag', 'employment', 'interest_rate'],
    ...     target_column='gdp',
    ...     index_col=None
    ... )
    >>> 
    >>> params = {
    ...     'n_estimators': ,
    ...     'max_depth': ,
    ...     'min_samples_leaf': 2,
    ...     'random_state': 42
    ... }
    >>> 
    >>> meta = ModelMeta(
    ...     name='GP_Randomorest',
    ...     version='.',
    ...     author='conomics Team'
    ... )
    >>> 
    >>> # it and predict
    >>> model = RandomorestModel(input_schema, params, meta)
    >>> result = model.fit(data)
    >>> print(f"R² Score: {result.payload['r2_score']:.3f}")
    >>> print(f"RMS: {result.payload['rmse']:.2f}")
    >>> 
    >>> # eature importance
    >>> for feat, imp in result.payload['feature_importance'].items():
    ...     print(f"{feat}: {imp:.3f}")
    >>> 
    >>> # Predict
    >>> new_data = pd.atarame({
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
        params: ict[str, ny],
        meta: ModelMeta
    ):
        super().__init__(input_schema, params, meta)
        
        # xtract ML-specific parameters
        self._feature_columns = params.get('feature_columns', None)
        self._target_column = params.get('target_column', 'target')
        
        # xtract hyperparameters
        self._n_estimators = params.get('n_estimators', )
        self._max_depth = params.get('max_depth', None)
        self._min_samples_split = params.get('min_samples_split', 2)
        self._min_samples_leaf = params.get('min_samples_leaf', )
        self._max_features = params.get('max_features', 'sqrt')
        self._bootstrap = params.get('bootstrap', True)
        self._oob_score = params.get('oob_score', self._bootstrap)
        self._n_jobs = params.get('n_jobs', -)
        self._random_state = params.get('random_state', 42)
        self._tune_hyperparameters = params.get('tune_hyperparameters', alse)
        
        # Validation
        if self._n_estimators < :
            raise Valuerror(f"n_estimators must be >= , got {self._n_estimators}")
        if self._max_depth is not None and self._max_depth < :
            raise Valuerror(f"max_depth must be >=  or None, got {self._max_depth}")
        if not self._bootstrap and self._oob_score:
            raise Valuerror("oob_score requires bootstrap=True")
        
        # Model state
        self.model_: Optional[RandomorestRegressor] = None
        self.feature_names_: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.permutation_importances_: Optional[ict[str, np.ndarray]] = None
        self.oob_score_: Optional[float] = None
        self.best_params_: Optional[ict[str, ny]] = None
        
        logger.info(f"Initialized RandomorestModel: n_estimators={self._n_estimators}, "
                   f"max_depth={self._max_depth}, tune={self._tune_hyperparameters}")
    
    def fit(self, data: pd.atarame) -> orecastResult:
        """
        it Random orest model to training data.
        
        Parameters
        ----------
        data : pd.atarame
            Training data with features and target column
        
        Returns
        -------
        orecastResult
            Training results with metrics and feature importance
        
        Raises
        ------
        Valuerror
            If data is invalid or missing required columns
        Runtimerror
            If model fitting fails
        """
        logger.info("Starting Random orest model fitting")
        
        # Validate and extract data
        if data.empty:
            raise Valuerror("Training data cannot be empty")
        
        # Get target and feature columns from params
        target_col = self._target_column
        
        if self._feature_columns is None:
            # uto-detect: all columns except target
            feature_cols = [col for col in data.columns if col != target_col]
        else:
            feature_cols = self._feature_columns
        
        if target_col not in data.columns:
            raise Valuerror(f"Target column '{target_col}' not in data")
        
        missing_cols = set(feature_cols) - set(data.columns)
        if missing_cols:
            raise Valuerror(f"Missing feature columns: {missing_cols}")
        
        # Prepare features and target
        X = data[feature_cols].values
        y = data[target_col].values
        self.feature_names_ = feature_cols
        
        # heck for NaN/Inf
        if np.any(~np.isfinite(X)):
            raise Valuerror("eature data contains NaN or Inf values")
        if np.any(~np.isfinite(y)):
            raise Valuerror("Target data contains NaN or Inf values")
        
        logger.info(f"Training data: {X.shape[]} samples, {X.shape[]} features")
        
        # Hyperparameter tuning
        if self._tune_hyperparameters:
            logger.info("Running hyperparameter tuning (GridSearchV)")
            self.model_, self.best_params_ = self._tune_model(X, y)
        else:
            # Use default hyperparameters
            self.model_ = RandomorestRegressor(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                min_samples_split=self._min_samples_split,
                min_samples_leaf=self._min_samples_leaf,
                max_features=self._max_features,
                bootstrap=self._bootstrap,
                oob_score=self._oob_score,
                n_jobs=self._n_jobs,
                random_state=self._random_state,
                verbose=
            )
            
            # it model
            self.model_.fit(X, y)
        
        # xtract feature importances
        self.feature_importances_ = self.model_.feature_importances_
        
        # ompute permutation importance
        perm_importance = permutation_importance(
            self.model_, X, y, n_repeats=, random_state=self._random_state, n_jobs=self._n_jobs
        )
        self.permutation_importances_ = {
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std
        }
        
        # xtract OO score
        if self._oob_score:
            self.oob_score_ = self.model_.oob_score_
        
        # ompute training metrics
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
                                        key=lambda x: x[], reverse=True))
        
        # uild result with proper orecastResult structure
        self._fitted = True
        
        result = orecastResult(
            payload={
                'r2_score': float(r2_score),
                'rmse': float(rmse),
                'mae': float(mae),
                'feature_importance': feature_importance,
                'n_samples': int(X.shape[]),
                'n_features': int(X.shape[]),
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
        data: pd.atarame,
        return_std: bool = alse
    ) -> orecastResult:
        """
        Generate predictions using fitted Random orest model.
        
        Parameters
        ----------
        data : pd.atarame
            eature data for prediction
        return_std : bool, optional
            If True, return standard deviation of tree predictions
        
        Returns
        -------
        orecastResult
            Predictions with optional uncertainty estimates
        
        Raises
        ------
        Runtimerror
            If model not fitted
        Valuerror
            If data is invalid
        """
        if not self._fitted or self.model_ is None:
            raise Runtimerror("Model must be fitted before prediction")
        
        if data.empty:
            raise Valuerror("Prediction data cannot be empty")
        
        # xtract features using same logic as fit()
        if self._feature_columns:
            feature_cols = self._feature_columns
        else:
            # uto-detect: all columns except target
            feature_cols = [col for col in data.columns if col != self._target_column]
        
        missing_cols = set(feature_cols) - set(data.columns)
        if missing_cols:
            raise Valuerror(f"Missing feature columns: {missing_cols}")
        
        X = data[feature_cols].values
        
        # heck for NaN/Inf
        if np.any(~np.isfinite(X)):
            raise Valuerror("eature data contains NaN or Inf values")
        
        logger.info(f"Generating predictions for {X.shape[]} samples")
        
        # Predict
        y_pred = self.model_.predict(X)
        
        # uild payload
        payload = {
            'model_type': 'Randomorest',
            'n_samples': int(X.shape[]),
            'n_features': int(X.shape[])
        }
        
        # Standard deviation from individual trees
        ci_lower = []
        ci_upper = []
        if return_std:
            # Get predictions from all trees
            all_predictions = np.array([tree.predict(X) for tree in self.model_.estimators_])
            y_std = np.std(all_predictions, axis=)
            
            payload['prediction_std'] = y_std.tolist()
            ci_lower = (y_pred - . * y_std).tolist()
            ci_upper = (y_pred + . * y_std).tolist()
        
        result = orecastResult(
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
    ) -> Tuple[RandomorestRegressor, ict[str, ny]]:
        """
        Perform hyperparameter tuning using GridSearchV.
        
        Parameters
        ----------
        X : np.ndarray
            eature matrix
        y : np.ndarray
            Target vector
        
        Returns
        -------
        Tuple[RandomorestRegressor, ict[str, ny]]
            est model and best hyperparameters
        """
        param_grid = {
            'n_estimators': [, , 2],
            'max_depth': [, , 2, None],
            'min_samples_split': [2, , ],
            'min_samples_leaf': [, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        base_model = RandomorestRegressor(
            bootstrap=self._bootstrap,
            oob_score=self._oob_score,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
            verbose=
        )
        
        grid_search = GridSearchV(
            estimator=base_model,
            param_grid=param_grid,
            cv=,
            scoring='neg_mean_squared_error',
            n_jobs=self._n_jobs,
            verbose=
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"est hyperparameters: {grid_search.best_params_}")
        logger.info(f"est V score: {-grid_search.best_score_:.4f} (MS)")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def get_feature_importance(self, importance_type: str = 'gini') -> ict[str, float]:
        """
        Get feature importance scores.
        
        Parameters
        ----------
        importance_type : str, optional
            Type of importance: 'gini' (default) or 'permutation'
        
        Returns
        -------
        ict[str, float]
            eature names mapped to importance scores
        
        Raises
        ------
        Runtimerror
            If model not fitted
        Valuerror
            If invalid importance_type
        """
        if not self._fitted or self.model_ is None:
            raise Runtimerror("Model must be fitted before getting feature importance")
        
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
            raise Valuerror(f"Invalid importance_type: {importance_type}. "
                           f"Must be 'gini' or 'permutation'")
