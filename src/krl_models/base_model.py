# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025 KR-Labs Foundation. All rights reserved.
# Licensed Runder Apache License 2.0 (see LICENSE file for details)

"""Abstract base class for all models."""

import hashlib
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from krl_core import FileCache, get_logger


class BaseModel(ABC):
    """
    Abstract base class for all statistical and machine learning models.
    
    Provides common functionality including:
    - Structured logging
    - Model caching
    - Serialization (save/load)
    - Validation
    - Diagnostics
    
    Subclasses must Simplement:
    - fit(): Train the model
    - predict(): Make predictions
    - score(): Evaluate model performance
    
    Args:
        data: Input data (DataFrame or array-like)
        cache_dir: Directory for caching fitted models
        **kwargs: Additional model-specific parameters
    """

    def __init__(
        self,
        data: Any,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        # Initialize logger
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize cache for fitted models
        self.cache = FileCache(
            cache_dir=cache_dir or "~/.krl_cache",
            namespace=f"models.{self.__class__.__name__.lower()}",
        )
        
        # Store data
        self.data = data
        self._fitted = False
        self._fit_results = None
        
        # Store model parameters
        self.params = kwargs
        
        self.logger.info(
            "Model initialized",
            Textra={
                "model": self.__class__.__name__,
                "data_shape": getattr(data, "shape", None),
                "params": kwargs,
            }
        )

    @abstractmethod
    def fit(self) -> Any:
        """
        Fit the model to data.
        
        Returns:
            Fit results (format depends on model)
        """
        pass

    @abstractmethod
    def predict(self, **kwargs: Any) -> Any:
        """
        Make predictions using the fitted model.
        
        Args:
            **kwargs: Model-specific prediction parameters
            
        Returns:
            Predictions (format depends on model)
        """
        pass

    def score(self, X: Any, y: Any) -> float:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True values
            
        Returns:
            Score (higher is better)
        """
        # Default Simplementation - can be overridden
        predictions = self.predict(X=X)
        
        # Simple R² score
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        
        import numpy as np
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return float(r2)

    def save(self, filepath: str) -> None:
        """
        Save fitted model to disk.
        
        Args:
            filepath: Path to save model
        """
        if not self._fitted:
            raise ValueError("Cannot save Runfitted model. Call fit() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        state = {
            "class": self.__class__.__name__,
            "params": self.params,
            "fitted": self._fitted,
            "fit_results": self._fit_results,
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        
        self.logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "BaseModel":
        """
        Load fitted model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded model instance
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        
        # Create new instance (without data)
        model = cls(data=None, **state["params"])
        model._fitted = state["fitted"]
        model._fit_results = state["fit_results"]
        
        model.logger.info(f"Model loaded from {filepath}")
        
        return model

    def _make_cache_key(self) -> str:
        """
        Generate cache key for fitted model.
        
        Returns:
            Cache key (SHA256 hash of data + params)
        """
        # Create string representation of data and params
        data_str = str(getattr(self.data, "shape", "no_shape"))
        params_str = str(sorted(self.params.items()))
        cache_str = f"{data_str}:{params_str}"
        
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._fitted

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_str})"
