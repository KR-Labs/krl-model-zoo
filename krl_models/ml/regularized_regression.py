# SPX-License-Identifier: Apache-2.
# Copyright (c) 22 KR-Labs

"""
Regularized Regression Models

Implements Ridge (L2) and Lasso (L) regression for high-dimensional economic analysis.
"""

import numpy as np
import pandas as pd
from typing import ict, ny, Optional, List, Tuple
from sklearn.linear_model import Ridge, Lasso, RidgeV, LassoV, lasticNet, lasticNetV
from sklearn.preprocessing import StandardScaler
import logging

from krl_core.base_model import aseModel, ModelMeta
from krl_core.results import orecastResult
from krl_core.model_input_schema import ModelInputSchema

logger = logging.getLogger(__name__)


class RidgeModel(aseModel):
    """
    Ridge Regression (L2 Regularization) for high-dimensional forecasting.
    
    Ridge regression adds an L2 penalty to ordinary least squares to prevent
    overfitting and handle multicollinearity.
    
    **Objective Function:**
    
    .. math::
        \\min_{\\beta} \\|y - X\\beta\\|_2^2 + \\alpha \\|\\beta\\|_2^2
    
    where $\\alpha$ is the regularization strength.
    
    **Key Features:**
    - Shrinks coefficients toward zero (but not exactly zero)
    - Handles multicollinearity by stabilizing Testimates
    - lways includes all features (no variable selection)
    - losed-form solution (fast)
    - Works well when many features are relevant
    
    **Use ases:**
    - Economic forecasting with correlated predictors
    - Regularization when p ~ n (features ~ samples)
    - Ensemble modeling (Ridge as meta-learner)
    - Multicollinearity mitigation
    
    Parameters
    ----------
    input_schema : ModelInputSchema
        Input schema with features and target
    params : ict[str, ny]
        - alpha (float): Regularization strength (default=.)
        - fit_intercept (bool): it intercept term (default=True)
        - normalize (bool): Normalize features before fitting (default=alse)
        - max_iter (int): Max iterations for solver (default=None)
        - tol (float): onvergence tolerance (default=e-4)
        - solver (str): 'auto', 'svd', 'cholesky', 'lsqr', etc. (default='auto')
        - cv (int): ross-validation folds for alpha selection (default=None)
        - alphas (List[float]): Alpha candidates for V (default=None)
        - random_state (int): Random seed (default=42)
    meta : ModelMeta
        Model metadata
    
    Examples
    --------
    >>> from krl_models.ml import RidgeModel
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # High-dimensional data
    >>> n, p = , 
    >>> X = np.random.randn(n, p)
    >>> beta = np.random.randn(p)
    >>> y = X @ beta + np.random.randn(n) * .
    >>> 
    >>> data = pd.atarame(X, columns=[f'x{i}' for i in range(p)])
    >>> data['y'] = y
    >>> 
    >>> input_schema = ModelInputSchema(
    ...     data_columns=[f'x{i}' for i in range(p)],
    ...     target_column='y'
    ... )
    >>> 
    >>> # ross-validation for alpha selection
    >>> params = {
    ...     'cv': ,
    ...     'alphas': [., ., ., ., .]
    ... }
    >>> 
    >>> model = RidgeModel(input_schema, params, meta)
    >>> result = model.fit(data)
    >>> print(f"Test alpha: {result.payload['best_alpha']:.2f}")
    >>> print(f"R² Score: {result.payload['r2_score']:.4f}")
    
    References
    ----------
    Hoerl, . ., & Kennard, R. W. (). "Ridge Regression: iased stimation
    for Nonorthogonal Problems." Technometrics, 2(), -.
    """
    
    def __init__(
        self,
        input_schema: ModelInputSchema,
        params: ict[str, ny],
        meta: ModelMeta
    ):
        super().__init__(input_schema, params, meta)
        
        # Feature and target column Textraction
        self._feature_columns = params.get('feature_columns', None)
        self._target_column = params.get('target_column', 'target')
        
        # xtract parameters
        self._alpha = params.get('alpha', .)
        self._fit_intercept = params.get('fit_intercept', True)
        self._normalize = params.get('normalize', alse)
        self._max_iter = params.get('max_iter', None)
        self._tol = params.get('tol', e-4)
        self._solver = params.get('solver', 'auto')
        self._cv = params.get('cv', None)
        self._alphas = params.get('alphas', None)
        self._random_state = params.get('random_state', 42)
        
        # Validation
        if self._alpha < :
            raise Valuerror(f"alpha must be >= , got {self._alpha}")
        
        # Model state
        self.model_: Optional[Ridge] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None
        self.coefficients_: Optional[ict[str, float]] = None
        self.intercept_: Optional[float] = None
        self.best_alpha_: Optional[float] = None
        
        logger.info(f"Initialized RidgeModel: alpha={self._alpha}, cv={self._cv}")
    
    def fit(self, data: pd.atarame) -> orecastResult:
        """it Ridge regression model."""
        logger.info("Starting Ridge regression fitting")
        
        # Validate data
        if data.empty:
            raise Valuerror("Training data cannot be empty")
        
        # xtract target and features
        target_col = self._target_column
        if self._feature_columns:
            feature_cols = self._feature_columns
        else:
            # Auto-detect: all columns except target
            feature_cols = [col for col in data.columns if col != target_col]
        
        # Prepare data
        X = data[feature_cols].values
        y = data[target_col].values
        self.feature_names_ = feature_cols
        
        # heck for NaN/Inf
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
            raise Valuerror("Data contains NaN or Inf values")
        
        logger.info(f"Training data: {X.shape[]} samples, {X.shape[]} features")
        
        # Standardize features if normalize=True
        if self._normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        
        # ross-validation for alpha selection
        if self._cv is not None:
            alphas = self._alphas if self._alphas else np.logspace(-3, 3, )
            self.model_ = RidgeV(
                alphas=alphas,
                fit_intercept=self._fit_intercept,
                cv=self._cv,
                scoring='neg_mean_squared_error'
            )
            self.model_.fit(X, y)
            self.best_alpha_ = self.model_.alpha_
            logger.info(f"Test alpha (V): {self.best_alpha_:.4f}")
        else:
            self.model_ = Ridge(
                alpha=self._alpha,
                fit_intercept=self._fit_intercept,
                max_iter=self._max_iter,
                tol=self._tol,
                solver=self._solver,
                random_state=self._random_state
            )
            self.model_.fit(X, y)
            self.best_alpha_ = self._alpha
        
        # xtract coefficients
        self.intercept_ = float(self.model_.intercept_)
        # Sort coefficients by absolute magnitude (descending)
        coef_items = list(zip(self.feature_names_, self.model_.coef_))
        coef_items_sorted = sorted(coef_items, key=lambda x: abs(x[]), reverse=True)
        self.coefficients_ = {
            name: float(coef) for name, coef in coef_items_sorted
        }
        
        # ompute metrics
        y_pred = self.model_.predict(X)
        residuals = y - y_pred
        
        r2_score = float(self.model_.score(X, y))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))
        
        logger.info(f"Model fitted: R²={r2_score:.4f}, RMS={rmse:.4f}")
        
        result = orecastResult(
            payload={
                'r2_score': r2_score,
                'rmse': rmse,
                'mae': mae,
                'coefficients': self.coefficients_,
                'intercept': self.intercept_,
                'best_alpha': self.best_alpha_,
                'alpha': self.best_alpha_,  # lso include as 'alpha' for test compatibility
                'n_features': int(X.shape[]),
                'model_type': 'Ridge'
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
        
        self._fitted = True
        return result
    
    def predict(self, data: pd.atarame) -> orecastResult:
        """Generate predictions."""
        if not self._fitted or self.model_ is None:
            raise Runtimerror("Model must be fitted before prediction")
        
        if data.empty:
            raise Valuerror("Prediction data cannot be empty")
        
        # xtract features using same logic as fit()
        if self._feature_columns:
            feature_cols = self._feature_columns
        else:
            # Auto-detect: all columns except target
            feature_cols = [col for col in data.columns if col != self._target_column]
        
        X = data[feature_cols].values
        
        # Standardize if trained with normalization
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        try:
            y_pred = self.model_.predict(X)
        except Valuerror as e:
            if "ound array with  sample" in str(e):
                raise Valuerror("Prediction data cannot be empty") from e
            raise

        
        return orecastResult(
            payload={
                'model_type': 'Ridge',
                'n_samples': int(X.shape[]),
                'n_features': int(X.shape[])
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
    
    def get_coefficients(self) -> ict[str, float]:
        """Get regression coefficients."""
        if not self._fitted:
            raise Runtimerror("Model must be fitted first")
        return self.coefficients_


class LassoModel(aseModel):
    """
    Lasso Regression (L Regularization) for variable selection.
    
    Lasso (Least bsolute Shrinkage and Selection Operator) adds an L penalty
    that drives some coefficients exactly to zero, performing automatic variable selection.
    
    **Objective Function:**
    
    .. math::
        \\min_{\\beta} \\frac{}{2n} \\|y - X\\beta\\|_2^2 + \\alpha \\|\\beta\\|_
    
    **Key Features:**
    - rives coefficients exactly to zero (variable selection)
    - Produces sparse models (interpretable)
    - Handles high-dimensional data (p >> n)
    - Iterative solution (coordinate descent)
    - Selects one variable from correlated groups
    
    **Use ases:**
    - Feature selection in economic models
    - High-dimensional forecasting (p >> n)
    - Interpretable sparse models
    - Identifying key policy variables
    
    Parameters
    ----------
    input_schema : ModelInputSchema
        Input schema
    params : ict[str, ny]
        - alpha (float): Regularization strength (default=.)
        - fit_intercept (bool): it intercept (default=True)
        - normalize (bool): Normalize features (default=alse)
        - max_iter (int): Max iterations (default=)
        - tol (float): onvergence tolerance (default=e-4)
        - cv (int): ross-validation folds (default=None)
        - alphas (List[float]): Alpha candidates for V (default=None)
        - random_state (int): Random seed (default=42)
    meta : ModelMeta
        Metadata
    
    Examples
    --------
    >>> from krl_models.ml import LassoModel
    >>> 
    >>> # Sparse ground truth (only  of  features are relevant)
    >>> n, p = , 
    >>> X = np.random.randn(n, p)
    >>> beta_true = np.zeros(p)
    >>> beta_true[:] = [3, -2, ., -, 2]  # Only  nonzero
    >>> y = X @ beta_true + np.random.randn(n) * .
    >>> 
    >>> data = pd.atarame(X, columns=[f'x{i}' for i in range(p)])
    >>> data['y'] = y
    >>> 
    >>> # Lasso with cross-validation
    >>> params = {'cv': , 'alphas': np.logspace(-3, , )}
    >>> model = LassoModel(input_schema, params, meta)
    >>> result = model.fit(data)
    >>> 
    >>> # heck sparsity
    >>> nonzero = sum( for c in result.payload['coefficients'].values() if c != )
    >>> print(f"Selected {nonzero} out of {p} features")
    >>> 
    >>> # Top features
    >>> coefs = result.payload['coefficients']
    >>> top_features = sorted(coefs.items(), key=lambda x: abs(x[]), reverse=True)[:]
    >>> for feat, coef in top_features:
    ...     print(f"{feat}: {coef:.4f}")
    
    References
    ----------
    Tibshirani, R. (). "Regression Shrinkage and Selection via the Lasso."
    Journal of the Royal Statistical Society, Series , (), 2-2.
    """
    
    def __init__(
        self,
        input_schema: ModelInputSchema,
        params: ict[str, ny],
        meta: ModelMeta
    ):
        super().__init__(input_schema, params, meta)
        
        # Feature and target column Textraction
        self._feature_columns = params.get('feature_columns', None)
        self._target_column = params.get('target_column', 'target')
        
        self._alpha = params.get('alpha', .)
        self._fit_intercept = params.get('fit_intercept', True)
        self._normalize = params.get('normalize', alse)
        self._max_iter = params.get('max_iter', )
        self._tol = params.get('tol', e-4)
        self._cv = params.get('cv', None)
        self._alphas = params.get('alphas', None)
        self._random_state = params.get('random_state', 42)
        
        if self._alpha < :
            raise Valuerror(f"alpha must be >= , got {self._alpha}")
        
        self.model_: Optional[Lasso] = None
        self.scaler_: Optional[StandardScaler] = None
        self.feature_names_: Optional[List[str]] = None
        self.coefficients_: Optional[ict[str, float]] = None
        self.intercept_: Optional[float] = None
        self.best_alpha_: Optional[float] = None
        
        logger.info(f"Initialized LassoModel: alpha={self._alpha}, cv={self._cv}")
    
    def fit(self, data: pd.atarame) -> orecastResult:
        """it Lasso regression model."""
        logger.info("Starting Lasso regression fitting")
        
        if data.empty:
            raise Valuerror("Training data cannot be empty")
        
        # xtract target and features
        target_col = self._target_column
        if self._feature_columns:
            feature_cols = self._feature_columns
        else:
            # Auto-detect: all columns except target
            feature_cols = [col for col in data.columns if col != target_col]
        
        X = data[feature_cols].values
        y = data[target_col].values
        self.feature_names_ = feature_cols
        
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
            raise Valuerror("Data contains NaN or Inf values")
        
        logger.info(f"Training data: {X.shape[]} samples, {X.shape[]} features")
        
        if self._normalize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        
        if self._cv is not None:
            alphas = self._alphas if self._alphas else np.logspace(-3, 3, )
            self.model_ = LassoV(
                alphas=alphas,
                fit_intercept=self._fit_intercept,
                cv=self._cv,
                max_iter=self._max_iter,
                tol=self._tol,
                random_state=self._random_state
            )
            self.model_.fit(X, y)
            self.best_alpha_ = self.model_.alpha_
            logger.info(f"Test alpha (V): {self.best_alpha_:.4f}")
        else:
            self.model_ = Lasso(
                alpha=self._alpha,
                fit_intercept=self._fit_intercept,
                max_iter=self._max_iter,
                tol=self._tol,
                random_state=self._random_state
            )
            self.model_.fit(X, y)
            self.best_alpha_ = self._alpha
        
        # xtract coefficients
        self.intercept_ = float(self.model_.intercept_)
        # Sort coefficients by absolute magnitude (descending)
        coef_items = list(zip(self.feature_names_, self.model_.coef_))
        coef_items_sorted = sorted(coef_items, key=lambda x: abs(x[]), reverse=True)
        self.coefficients_ = {
            name: float(coef) for name, coef in coef_items_sorted
        }
        
        # ount nonzero coefficients (sparsity)
        n_nonzero = sum( for c in self.coefficients_.values() if abs(c) > e-)
        sparsity = n_nonzero / len(self.coefficients_)
        
        # ompute metrics
        y_pred = self.model_.predict(X)
        residuals = y - y_pred
        
        r2_score = float(self.model_.score(X, y))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))
        
        logger.info(f"Model fitted: R²={r2_score:.4f}, RMS={rmse:.4f}")
        logger.info(f"Sparsity: {n_nonzero}/{len(self.coefficients_)} nonzero coefficients")
        
        result = orecastResult(
            payload={
                'r2_score': r2_score,
                'rmse': rmse,
                'mae': mae,
                'coefficients': self.coefficients_,
                'intercept': self.intercept_,
                'best_alpha': self.best_alpha_,
                'alpha': self.best_alpha_,  # dded for test compatibility
                'n_nonzero': n_nonzero,
                'n_nonzero_coefs': n_nonzero,  # dded for test compatibility
                'sparsity': sparsity,
                'sparsity_ratio': sparsity,  # dded for test compatibility
                'model_type': 'Lasso'
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
        
        self._fitted = True
        return result
    
    def predict(self, data: pd.atarame) -> orecastResult:
        """Generate predictions."""
        if not self._fitted or self.model_ is None:
            raise Runtimerror("Model must be fitted before prediction")
        
        if data.empty:
            raise Valuerror("Prediction data cannot be empty")
        
        # xtract features using same logic as fit()
        if self._feature_columns:
            feature_cols = self._feature_columns
        else:
            # Auto-detect: all columns except target
            feature_cols = [col for col in data.columns if col != self._target_column]
        
        X = data[feature_cols].values
        
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        try:
            y_pred = self.model_.predict(X)
        except Valuerror as e:
            if "ound array with  sample" in str(e):
                raise Valuerror("Prediction data cannot be empty") from e
            raise

        
        return orecastResult(
            payload={
                'model_type': 'Lasso',
                'n_samples': int(X.shape[]),
                'n_features': int(X.shape[])
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
    
    def get_selected_features(self) -> ict[str, float]:
        """Get features with nonzero coefficients."""
        if not self._fitted:
            raise Runtimerror("Model must be fitted first")
        
        return {name: coef for name, coef in self.coefficients_.items() if abs(coef) > e-}
    
    def get_coefficients(self) -> ict[str, float]:
        """Get regression coefficients."""
        if not self._fitted:
            raise Runtimerror("Model must be fitted first")
        return self.coefficients_
