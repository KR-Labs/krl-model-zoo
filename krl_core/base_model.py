"""
Base Model Abstract Class for KRL Model Zoo
============================================

Apache 2.0 License - Gate 1 Foundation
Author: KR Labs

This module defines the abstract base class that all KRL models inherit from.
Provides standardized interface for fit(), forecast(), and validate() methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import hashlib
import json
from datetime import datetime
import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field


class ModelMetadata(PydanticBaseModel):
    """
    Metadata for model tracking and provenance.
    
    Attributes
    ----------
    name : str
        Human-readable model name
    version : str
        Model version (semantic versioning)
    author : str
        Model author/organization
    created_at : str
        ISO 8601 timestamp of creation
    description : str
        Brief model description
    tags : list of str
        Categorization tags
    """
    name: str
    version: str = "1.0.0"
    author: str = "KR-Labs"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    description: str = ""
    tags: List[str] = Field(default_factory=list)


class BaseModel(ABC):
    """
    Abstract base class for all KRL models.
    
    All models in the KRL Model Zoo must inherit from this class and implement:
    - fit(y, X, **kwargs) → self
    - forecast(steps, **kwargs) → ForecastResult
    
    Provides:
    - Standardized interface across 100+ models
    - Metadata tracking (name, version, author, SHA256 hash)
    - Parameter validation
    - Reproducibility via SHA256 hashing
    
    Parameters
    ----------
    name : str, optional
        Model name (defaults to class name)
    version : str, optional
        Model version (semantic versioning)
    description : str, optional
        Brief model description
    tags : list of str, optional
        Categorization tags (e.g., ['time-series', 'volatility'])
    **kwargs : dict
        Additional model-specific parameters
    
    Attributes
    ----------
    metadata_ : ModelMetadata
        Model metadata (name, version, author, etc.)
    params_ : dict
        Model parameters (hyperparameters + fit results)
    hash_ : str or None
        SHA256 hash of model state (for reproducibility)
    fitted_ : bool
        Whether model has been fitted to data
    
    Examples
    --------
    >>> class MyModel(BaseModel):
    ...     def fit(self, y, X=None, **kwargs):
    ...         # Implementation
    ...         return self
    ...     def forecast(self, steps=1, **kwargs):
    ...         # Implementation
    ...         return ForecastResult(...)
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        version: str = "1.0.0",
        description: str = "",
        tags: Optional[List[str]] = None,
        **kwargs
    ):
        # Metadata
        self.metadata_ = ModelMetadata(
            name=name or self.__class__.__name__,
            version=version,
            author="KR-Labs",
            description=description,
            tags=tags or []
        )
        
        # Parameters (hyperparameters + fitted parameters)
        self.params_ = kwargs
        
        # State tracking
        self.hash_ = None
        self.fitted_ = False
    
    @abstractmethod
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None, **kwargs) -> 'BaseModel':
        """
        Fit model to training data.
        
        Parameters
        ----------
        y : array-like, shape (n_samples,)
            Target variable (endogenous variable for time series)
        X : array-like, shape (n_samples, n_features), optional
            Exogenous variables (features)
        **kwargs : dict
            Additional fit parameters (e.g., validation data, callbacks)
        
        Returns
        -------
        self : BaseModel
            Fitted model instance
        
        Notes
        -----
        Subclasses must implement this method. After fitting:
        - Set self.fitted_ = True
        - Compute self.hash_ via _compute_hash()
        - Store fitted parameters in self.params_
        """
        pass
    
    @abstractmethod
    def forecast(self, steps: int = 1, **kwargs) -> 'ForecastResult':
        """
        Generate forecasts.
        
        Parameters
        ----------
        steps : int, default=1
            Number of steps ahead to forecast
        **kwargs : dict
            Additional forecast parameters (e.g., confidence level, exogenous future values)
        
        Returns
        -------
        result : ForecastResult
            Forecast results with point estimates and prediction intervals
        
        Raises
        ------
        ValueError
            If model has not been fitted (self.fitted_ == False)
        
        Notes
        -----
        Subclasses must implement this method. Should check self.fitted_ before forecasting.
        """
        pass
    
    def validate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute validation metrics.
        
        Parameters
        ----------
        y_true : array-like, shape (n_samples,)
            True values
        y_pred : array-like, shape (n_samples,)
            Predicted values
        
        Returns
        -------
        metrics : dict
            Dictionary of metric name → value
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - mape: Mean Absolute Percentage Error
            - r2: R-squared coefficient
        
        Examples
        --------
        >>> model = MyModel()
        >>> model.fit(y_train)
        >>> y_pred = model.forecast(steps=len(y_test)).point_forecast
        >>> metrics = model.validate(y_test, y_pred)
        >>> print(f"RMSE: {metrics['rmse']:.3f}")
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Mean Absolute Error
        mae = float(np.mean(np.abs(y_true - y_pred)))
        
        # Root Mean Squared Error
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
        # Mean Absolute Percentage Error
        # Avoid division by zero
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if np.any(mask) else np.inf
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns
        -------
        params : dict
            Model parameters (hyperparameters + fitted parameters)
        
        Examples
        --------
        >>> model = MyModel(alpha=0.05)
        >>> model.fit(y)
        >>> params = model.get_params()
        >>> print(params['alpha'])
        0.05
        """
        return self.params_.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Parameters
        ----------
        **params : dict
            Parameters to update
        
        Returns
        -------
        self : BaseModel
            Model instance with updated parameters
        
        Notes
        -----
        If model was previously fitted, this resets fitted_ to False.
        
        Examples
        --------
        >>> model = MyModel()
        >>> model.set_params(alpha=0.01, max_iter=1000)
        >>> model.fit(y)
        """
        for key, value in params.items():
            self.params_[key] = value
        
        # Reset fitted state if parameters changed
        if self.fitted_:
            self.fitted_ = False
            self.hash_ = None
        
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns
        -------
        metadata : dict
            Model metadata (name, version, author, created_at, etc.)
        
        Examples
        --------
        >>> model = MyModel()
        >>> meta = model.get_metadata()
        >>> print(f"Model: {meta['name']} v{meta['version']}")
        """
        return self.metadata_.model_dump()
    
    def _compute_hash(self) -> str:
        """
        Compute SHA256 hash of model state for reproducibility.
        
        Returns
        -------
        hash_str : str
            Hexadecimal SHA256 hash
        
        Notes
        -----
        Hash includes:
        - Model class name
        - Model parameters
        - Fitted parameters (if fitted)
        
        Used for:
        - Reproducibility tracking
        - Model versioning
        - Detecting parameter changes
        """
        # Serialize model state
        state = {
            'class': self.__class__.__name__,
            'params': self.params_,
            'fitted': self.fitted_
        }
        
        # Convert to deterministic JSON string
        state_str = json.dumps(state, sort_keys=True, default=str)
        
        # Compute SHA256 hash
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def __repr__(self) -> str:
        """String representation of model."""
        fitted_status = "fitted" if self.fitted_ else "not fitted"
        return f"{self.metadata_.name}(version={self.metadata_.version}, {fitted_status})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
