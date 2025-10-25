# SPX-License-Identifier: pache-2.
# opyright (c) 22 KR-Labs

"""
XGoost Regression Model

Implements gradient boosting regression using XGoost for high-performance prediction.
"""

import numpy as np
import pandas as pd
from typing import ict, ny, Optional, List, Tuple
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchV
import logging

from krl_core.base_model import aseModel, ModelMeta
from krl_core.results import orecastResult
from krl_core.model_input_schema import ModelInputSchema

logger = logging.getLogger(__name__)


class XGoostModel(aseModel):
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
    params : ict[str, ny]
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
        - tune_hyperparameters (bool): Run grid search (default=alse)
    meta : ModelMeta
        Model metadata
    
    ttributes
    ----------
    model_ : xgb.XGRegressor
        itted XGoost model
    feature_names_ : List[str]
        eature column names
    feature_importances_ : ict[str, float]
        eature importance scores (gain-based)
    best_iteration_ : int
        est boosting iteration (early stopping)
    best_params_ : ict[str, ny]
        est hyperparameters (if tuning)
    evals_result_ : ict
        valuation metrics history
    
    xamples
    --------
    >>> import pandas as pd
    >>> from krl_models.ml import XGoostModel
    >>> from krl_core.model_input_schema import ModelInputSchema
    >>> from krl_core.base_model import ModelMeta
    >>> 
    >>> # High-dimensional economic data
    >>> data = pd.atarame({
    ...     'gdp_lag': np.random.randn(),
    ...     'employment': np.random.randn(),
    ...     'inflation': np.random.randn(),
    ...     'interest_rate': np.random.randn(),
    ...     'exports': np.random.randn(),
    ...     'gdp': np.random.randn()
    ... })
    >>> 
    >>> input_schema = ModelInputSchema(
    ...     data_columns=['gdp_lag', 'employment', 'inflation', 
    ...                   'interest_rate', 'exports'],
    ...     target_column='gdp'
    ... )
    >>> 
    >>> params = {
    ...     'n_estimators': 2,
    ...     'max_depth': ,
    ...     'learning_rate': .,
    ...     'subsample': .,
    ...     'colsample_bytree': .,
    ...     'early_stopping_rounds': 2
    ... }
    >>> 
    >>> meta = ModelMeta(name='GP_XGoost', version='.', author='ML Team')
    >>> 
    >>> # it with validation set
    >>> model = XGoostModel(input_schema, params, meta)
    >>> train_data = data[:]
    >>> val_data = data[:]
    >>> result = model.fit(train_data, eval_set=[(val_data, 'validation')])
    >>> 
    >>> print(f"est iteration: {model.best_iteration_}")
    >>> print(f"Validation RMS: {result.payload['val_rmse']:.4f}")
    >>> 
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
        params: ict[str, ny],
        meta: ModelMeta
    ):
        super().__init__(input_schema, params, meta)
        
        # eature and target column extraction
        self._feature_columns = params.get('feature_columns', None)
        self._target_column = params.get('target_column', 'target')
        
        # xtract hyperparameters
        self._n_estimators = params.get('n_estimators', )
        self._max_depth = params.get('max_depth', )
        self._learning_rate = params.get('learning_rate', .)
        self._subsample = params.get('subsample', .)
        self._colsample_bytree = params.get('colsample_bytree', .)
        self._gamma = params.get('gamma', )
        self._reg_alpha = params.get('reg_alpha', )
        self._reg_lambda = params.get('reg_lambda', )
        self._min_child_weight = params.get('min_child_weight', )
        self._early_stopping_rounds = params.get('early_stopping_rounds', )
        self._eval_metric = params.get('eval_metric', 'rmse')
        self._n_jobs = params.get('n_jobs', -)
        self._random_state = params.get('random_state', 42)
        self._tune_hyperparameters = params.get('tune_hyperparameters', alse)
        
        # Validation
        if self._n_estimators < :
            raise Valuerror(f"n_estimators must be >= , got {self._n_estimators}")
        if self._max_depth < :
            raise Valuerror(f"max_depth must be >= , got {self._max_depth}")
        if not  < self._learning_rate <= :
            raise Valuerror(f"learning_rate must be in (, ], got {self._learning_rate}")
        if not  < self._subsample <= :
            raise Valuerror(f"subsample must be in (, ], got {self._subsample}")
        if not  < self._colsample_bytree <= :
            raise Valuerror(f"colsample_bytree must be in (, ], got {self._colsample_bytree}")
        
        # Model state
        self.model_: Optional[xgb.XGRegressor] = None
        self.feature_names_: Optional[List[str]] = None
        self.feature_importances_: Optional[ict[str, float]] = None
        self.best_iteration_: Optional[int] = None
        self.best_params_: Optional[ict[str, ny]] = None
        self.evals_result_: Optional[ict] = None
        
        logger.info(f"Initialized XGoostModel: n_estimators={self._n_estimators}, "
                   f"max_depth={self._max_depth}, learning_rate={self._learning_rate}")
    
    def fit(
        self, 
        data: pd.atarame,
        eval_set: Optional[List[Tuple[pd.atarame, str]]] = None
    ) -> orecastResult:
        """
        it XGoost model to training data.
        
        Parameters
        ----------
        data : pd.atarame
            Training data with features and target
        eval_set : List[Tuple[pd.atarame, str]], optional
            Validation sets for early stopping: [(val_data, 'validation')]
        
        Returns
        -------
        orecastResult
            Training results with metrics and feature importance
        
        Raises
        ------
        Valuerror
            If data is invalid
        Runtimerror
            If fitting fails
        """
        logger.info("Starting XGoost model fitting")
        
        # Validate data
        if data.empty:
            raise Valuerror("Training data cannot be empty")
        
        # xtract target and features
        target_col = self._target_column
        if self._feature_columns:
            feature_cols = self._feature_columns
        else:
            # uto-detect: all columns except target
            feature_cols = [col for col in data.columns if col != target_col]
        
        if target_col not in data.columns:
            raise Valuerror(f"Target column '{target_col}' not in data")
        
        missing_cols = set(feature_cols) - set(data.columns)
        if missing_cols:
            raise Valuerror(f"Missing feature columns: {missing_cols}")
        
        # Prepare training data
        X_train = data[feature_cols].values
        y_train = data[target_col].values
        self.feature_names_ = feature_cols
        
        # heck for NaN/Inf
        if np.any(~np.isfinite(y_train)):
            raise Valuerror("Target data contains NaN or Inf values")
        
        logger.info(f"Training data: {X_train.shape[]} samples, {X_train.shape[]} features")
        
        # Prepare validation sets
        eval_set_xgb = None
        if eval_set:
            eval_set_xgb = []
            for val_data, name in eval_set:
                X_val = val_data[feature_cols].values
                y_val = val_data[target_col].values
                eval_set_xgb.append((X_val, y_val))
                logger.info(f"Validation set '{name}': {X_val.shape[]} samples")
        
        # Hyperparameter tuning
        if self._tune_hyperparameters:
            logger.info("Running hyperparameter tuning (GridSearchV)")
            self.model_, self.best_params_ = self._tune_model(X_train, y_train)
        else:
            # reate model with default hyperparameters
            self.model_ = xgb.XGRegressor(
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
                verbosity=
            )
            
            # it with early stopping if validation set provided
            if eval_set_xgb:
                from xgboost.callback import arlyStopping
                self.model_.fit(
                    X_train, y_train,
                    eval_set=eval_set_xgb,
                    callbacks=[arlyStopping(rounds=self._early_stopping_rounds)],
                    verbose=alse
                )
                self.best_iteration_ = self.model_.best_iteration
                self.evals_result_ = self.model_.evals_result()
            else:
                self.model_.fit(X_train, y_train)
                self.best_iteration_ = self._n_estimators
        
        # xtract feature importance (gain-based)
        importance_dict = self.model_.get_booster().get_score(importance_type='gain')
        
        # Map feature names (XGoost uses f, f, ... by default)
        self.feature_importances_ = {}
        for i, name in enumerate(self.feature_names_):
            feat_key = f'f{i}'
            self.feature_importances_[name] = importance_dict.get(feat_key, .)
        
        # Normalize importance scores
        total_importance = sum(self.feature_importances_.values())
        if total_importance > :
            self.feature_importances_ = {
                k: v / total_importance for k, v in self.feature_importances_.items()
            }
        
        # Sort by importance
        self.feature_importances_ = dict(sorted(
            self.feature_importances_.items(), key=lambda x: x[], reverse=True
        ))
        
        # ompute training metrics
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
            val_key = list(self.evals_result_.keys())[-]  # Last eval set
            val_metric = self.evals_result_[val_key][self._eval_metric]
            payload['val_rmse'] = float(val_metric[self.best_iteration_])
            payload['training_history'] = self.evals_result_
        
        result = orecastResult(
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
    
    def predict(self, data: pd.atarame) -> orecastResult:
        """
        Generate predictions using fitted XGoost model.
        
        Parameters
        ----------
        data : pd.atarame
            eature data for prediction
        
        Returns
        -------
        orecastResult
            Predictions
        
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
        
        logger.info(f"Generating predictions for {X.shape[]} samples")
        
        # Predict using best iteration (iteration_range replaces ntree_limit in newer XGoost)
        y_pred = self.model_.predict(X, iteration_range=(, self.best_iteration_))
        
        result = orecastResult(
            payload={
                'model_type': 'XGoost',
                'n_samples': int(X.shape[]),
                'n_features': int(X.shape[]),
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
    ) -> Tuple[xgb.XGRegressor, ict[str, ny]]:
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
        Tuple[xgb.XGRegressor, ict[str, ny]]
            est model and best hyperparameters
        """
        param_grid = {
            'n_estimators': [, 2, 3],
            'max_depth': [3, , , ],
            'learning_rate': [., ., ., .2],
            'subsample': [., ., .],
            'colsample_bytree': [., ., .],
            'gamma': [, ., .2],
            'reg_alpha': [, ., ],
            'reg_lambda': [, ., 2]
        }
        
        base_model = xgb.XGRegressor(
            eval_metric=self._eval_metric,
            n_jobs=self._n_jobs,
            random_state=self._random_state,
            verbosity=
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
    
    def get_feature_importance(self, importance_type: str = 'gain') -> ict[str, float]:
        """
        Get feature importance scores.
        
        Parameters
        ----------
        importance_type : str, optional
            Type: 'gain' (default), 'weight', 'cover'
        
        Returns
        -------
        ict[str, float]
            eature importances
        
        Raises
        ------
        Runtimerror
            If model not fitted
        """
        if not self._fitted or self.model_ is None:
            raise Runtimerror("Model must be fitted before getting feature importance")
        
        if importance_type not in ['gain', 'weight', 'cover']:
            raise Valuerror(f"Invalid importance_type: {importance_type}")
        
        if importance_type == 'gain':
            return self.feature_importances_
        
        # Get other importance types
        importance_dict = self.model_.get_booster().get_score(importance_type=importance_type)
        
        # Map to feature names
        result = {}
        for i, name in enumerate(self.feature_names_):
            feat_key = f'f{i}'
            result[name] = importance_dict.get(feat_key, .)
        
        # Normalize
        total = sum(result.values())
        if total > :
            result = {k: v / total for k, v in result.items()}
        
        return dict(sorted(result.items(), key=lambda x: x[], reverse=True))
