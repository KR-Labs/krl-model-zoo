# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Model Input Schema with Pydantic V2 Validation
===============================================

Apache 2.0 License - Gate 1 Foundation
Author: KR Labs

Validates and processes model inputs before fitting.
Ensures type safety and provides feature engineering utilities.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator


class ModelInputSchema(BaseModel):
    """
    Schema for validating model inputs.
    
    Validates y (target), X (features), and additional parameters.
    Provides feature extraction and preprocessing utilities.
    
    Attributes
    ----------
    y : array-like
        Target variable (endogenous)
    X : array-like, optional
        Feature matrix (exogenous)
    params : dict
        Additional model parameters
    feature_names : list of str, optional
        Names of features in X
    
    Examples
    --------
    >>> schema = ModelInputSchema(
    ...     y=np.array([1, 2, 3]),
    ...     X=np.array([[1, 2], [2, 3], [3, 4]]),
    ...     params={'alpha': 0.05},
    ...     feature_names=['x1', 'x2']
    ... )
    >>> print(f"Target shape: {schema.y.shape}")
    >>> print(f"Features shape: {schema.X.shape}")
    """
    y: np.ndarray
    X: Optional[np.ndarray] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    feature_names: Optional[List[str]] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @field_validator('y', mode='before')
    @classmethod
    def validate_y(cls, v):
        """Validate and convert target variable."""
        v = np.asarray(v)
        if v.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        if len(v) == 0:
            raise ValueError("y cannot be empty")
        if not np.isfinite(v).all():
            raise ValueError("y contains NaN or inf values")
        return v
    
    @field_validator('X', mode='before')
    @classmethod
    def validate_X(cls, v):
        """Validate and convert feature matrix."""
        if v is None:
            return v
        v = np.asarray(v)
        if v.ndim not in [1, 2]:
            raise ValueError("X must be 1D or 2D array")
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        if not np.isfinite(v).all():
            raise ValueError("X contains NaN or inf values")
        return v
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Validate X and y shapes match
        if self.X is not None:
            if len(self.X) != len(self.y):
                raise ValueError(f"X and y must have same length (X: {len(self.X)}, y: {len(self.y)})")
        
        # Generate feature names if not provided
        if self.X is not None and self.feature_names is None:
            n_features = self.X.shape[1] if self.X.ndim == 2 else 1
            self.feature_names = [f'x{i}' for i in range(n_features)]
        
        # Validate feature names length
        if self.X is not None and self.feature_names is not None:
            n_features = self.X.shape[1] if self.X.ndim == 2 else 1
            if len(self.feature_names) != n_features:
                raise ValueError(f"feature_names length ({len(self.feature_names)}) must match X features ({n_features})")
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame with named columns.
        
        Returns
        -------
        df : DataFrame
            DataFrame with 'y' column and feature columns
        """
        data = {'y': self.y}
        if self.X is not None and self.feature_names is not None:
            for i, name in enumerate(self.feature_names):
                data[name] = self.X[:, i] if self.X.ndim == 2 else self.X
        return pd.DataFrame(data)
    
    def split(self, train_size: float = 0.8) -> tuple:
        """
        Split into train/test sets.
        
        Parameters
        ----------
        train_size : float, default=0.8
            Proportion of data for training
        
        Returns
        -------
        train_schema, test_schema : tuple of ModelInputSchema
            Train and test splits
        """
        n = len(self.y)
        split_idx = int(n * train_size)
        
        train_data = {
            'y': self.y[:split_idx],
            'X': self.X[:split_idx] if self.X is not None else None,
            'params': self.params.copy(),
            'feature_names': self.feature_names
        }
        
        test_data = {
            'y': self.y[split_idx:],
            'X': self.X[split_idx:] if self.X is not None else None,
            'params': self.params.copy(),
            'feature_names': self.feature_names
        }
        
        return ModelInputSchema(**train_data), ModelInputSchema(**test_data)
