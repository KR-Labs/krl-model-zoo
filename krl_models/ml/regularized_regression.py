"""
Regularized Linear Regression (Ridge, Lasso, ElasticNet)
=========================================================

MIT License - Gate 2 Phase 2.3 ML Baseline
author: KR Labs
"""

from typing import Dict, Any, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult


class RegularizedRegression(BaseModel):
    """
    Regularized linear regression with L1/L2 penalties.
    
    Ridge (L2): Shrinks coefficients smoothly, keeps all features
    Lasso (L1): Sparse solutions via feature selection (some coeffs = 0)
    ElasticNet: Combines L1 + L2 penalties
    
    Parameters
    ----------
    regularization : {'ridge', 'lasso', 'elasticnet'}, default='ridge'
        Type of regularization
    alpha : float, default=1.0
        Regularization strength (higher = more penalty)
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (0 = ridge, 1 = lasso)
    fit_intercept : bool, default=True
        Include intercept term
    normalize : bool, default=True
        Standardize features before fitting
    max_iter : int, default=1000
        Maximum iterations for optimization
    random_state : int or None, default=42
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        regularization: Literal['ridge', 'lasso', 'elasticnet'] = 'ridge',
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        normalize: bool = True,
        max_iter: int = 1000,
        random_state: Optional[int] = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.coefficients_ = None
        self.intercept_ = None
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None, **kwargs) -> 'RegularizedRegression':
        """
        Fit regularized regression model.
        
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target variable
        X : array-like, shape (n_samples, n_features)
            Feature matrix (required)
        **kwargs : dict
            Additional arguments (feature_names)
        
        Returns
        -------
        self : RegularizedRegression
            Fitted model instance
        """
        if X is None:
            raise ValueError("Regularized regression requires feature matrix X")
        
        # Store feature names
        self.feature_names_ = kwargs.get('feature_names',
                                         [f'X{i}' for i in range(X.shape[1])])
        
        # Normalize features if requested
        if self.normalize:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X
        
        # Initialize model based on regularization type
        if self.regularization == 'ridge':
            self.model_ = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
        elif self.regularization == 'lasso':
            self.model_ = Lasso(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
        elif self.regularization == 'elasticnet':
            self.model_ = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown regularization: {self.regularization}")
        
        # Fit model
        self.model_.fit(X_scaled, y)
        
        # Store coefficients
        self.coefficients_ = dict(zip(self.feature_names_, self.model_.coef_))
        self.intercept_ = float(self.model_.intercept_)
        
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
            Additional arguments (alpha for intervals - bootstrap-based)
        
        Returns
        -------
        result : ForecastResult
            Point forecasts with residual-based intervals
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before forecasting")
        
        if X_future is None:
            raise ValueError("Regularized regression requires future features X_future")
        
        if X_future.shape[0] != steps:
            raise ValueError(f"X_future must have {steps} rows")
        
        # Scale features if trained with normalization
        if self.normalize and self.scaler_ is not None:
            X_future_scaled = self.scaler_.transform(X_future)
        else:
            X_future_scaled = X_future
        
        # Point forecast
        point_forecast = self.model_.predict(X_future_scaled)
        
        # Simple prediction intervals (assumes constant variance)
        # In practice, should use residual bootstrap or conformal prediction
        alpha = kwargs.get('alpha', 0.05)
        
        return ForecastResult(
            point_forecast=point_forecast,
            lower=np.full(steps, np.nan),  # Placeholder
            upper=np.full(steps, np.nan),  # Placeholder
            alpha=alpha
        )
    
    def get_coefficients(self) -> Dict[str, float]:
        """
        Get regression coefficients.
        
        Returns
        -------
        coefficients : dict
            Feature names mapped to coefficient values
        """
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted to get coefficients")
        
        return self.coefficients_
    
    def get_intercept(self) -> float:
        """
        Get regression intercept.
        
        Returns
        -------
        intercept : float
            Intercept value
        """
        if self.intercept_ is None:
            raise ValueError("Model must be fitted to get intercept")
        
        return self.intercept_
