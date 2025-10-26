# ----------------------------------------------------------------------
# Copyright (c) 2024 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""Base model abstraction for all KRL models."""

from __future__ import annotations

import abc
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from krl_core.results import BaseResult
from krl_core.utils import compute_dataframe_hash


@dataclass
class ModelMeta:
    """Metadata for model versioning and provenance."""

    name: str
    version: str = "1.0.0"
    author: str = "KR-Labs"
    description: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class BaseModel(abc.ABC):
    """
    Core abstract model class for krl-model-zoo-core.

    Responsibilities:
    - Accept standardized input (ModelInputSchema)
    - Provide fit/predict/persist hooks
    - Produce Result objects (BaseResult subclass)
    - Compute deterministic run-hash and register run metadata

    All models in the KRL ecosystem inherit from this base class to ensure:
    - Consistent interface across econometric, ML, Bayesian, causal models
    - Automatic provenance tracking and reproducibility
    - Serialization and caching support
    - Integration with visualization and dashboard layers

    Example:
        ```python
        from krl_core import BaseModel, ModelInputSchema, ForecastResult

        class MyModel(BaseModel):
            def fit(self) -> BaseResult:
                # Your training logic here
                return ForecastResult(...)

            def predict(self, steps=10) -> BaseResult:
                # Your prediction logic here
                return ForecastResult(...)
        ```
    """

    def __init__(
        self,
        input_schema=None,
        params: Optional[Dict[str, Any]] = None,
        meta: Optional[ModelMeta] = None,
    ):
        """
        Initialize base model.

        Args:
            input_schema: ModelInputSchema instance with validated data (optional for some models)
            params: Model hyperparameters (stored for reproducibility)
            meta: Model metadata (name, version, author)
        """
        self.input_schema = input_schema
        self.params = params or {}
        self.meta = meta or ModelMeta(name=self.__class__.__name__)
        self._is_fitted = False
        self._fit_time: Optional[str] = None
        self._fit_result: Optional[BaseResult] = None

    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> BaseResult:
        """
        Train/fit the model.

        Must be implemented by subclasses. Should set self._is_fitted = True
        upon successful completion.

        Returns:
            BaseResult subclass (ForecastResult, CausalResult, etc.)
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement fit()")

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> BaseResult:
        """
        Generate predictions/forecasts.

        Must be implemented by subclasses. Should check self._is_fitted before
        making predictions.

        Returns:
            BaseResult subclass with predictions
        
        Raises:
            NotImplementedError: If not implemented by subclass
            RuntimeError: If called before fit()
        """
        raise NotImplementedError("Subclasses must implement predict()")

    def is_fitted(self) -> bool:
        """
        Check if model has been fitted.

        Returns:
            True if model is fitted, False otherwise
        """
        return self._is_fitted

    def serialize(self) -> bytes:
        """
        Serialize model to bytes for persistence.

        Returns:
            Pickled model bytes
        
        Raises:
            pickle.PicklingError: If model cannot be pickled
        """
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes) -> BaseModel:
        """
        Deserialize model from bytes.

        Args:
            data: Pickled model bytes

        Returns:
            Restored BaseModel instance
        
        Raises:
            pickle.UnpicklingError: If data cannot be unpickled
        """
        return pickle.loads(data)

    def run_hash(self) -> str:
        """
        Compute deterministic hash of model + input + params.

        This enables exact reproducibility checking: same inputs → same hash.
        
        Hash Components:
            - Model name and version
            - Input data hash (SHA256 of sorted DataFrame)
            - Parameters (JSON-serialized, sorted keys)

        Returns:
            SHA256 hex digest (64 characters)
        """
        # Compute input hash
        if self.input_schema is not None:
            input_hash = compute_dataframe_hash(self.input_schema.to_dataframe())
        else:
            input_hash = "no_input"

        # Build hash components
        components = {
            "model": f"{self.meta.name}@{self.meta.version}",
            "input_hash": input_hash,
            "params": self.params,
        }

        # Compute SHA256 of JSON-serialized components
        hash_str = json.dumps(components, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    @property
    def input_hash(self) -> str:
        """
        Compute hash of input data only.

        Returns:
            SHA256 hex digest of input DataFrame
        """
        if self.input_schema is not None:
            return compute_dataframe_hash(self.input_schema.to_dataframe())
        return "no_input"

    def register_run(self, registry, result: BaseResult) -> None:
        """
        Register model run with ModelRegistry.

        Args:
            registry: ModelRegistry instance
            result: Result from fit() or predict()
        """
        run_hash = self.run_hash()
        
        # Log run metadata
        registry.log_run(
            run_hash=run_hash,
            model_name=self.meta.name,
            version=self.meta.version,
            input_hash=self.input_hash,
            params=self.params,
        )
        
        # Log result
        registry.log_result(
            run_hash=run_hash,
            result_hash=result.result_hash,
            result=result.to_json(),
        )

    def _process_data(self, data):
        """
        Convert input to numpy array or DataFrame.
        
        Args:
            data: Input data (DataFrame, array, etc.)
        
        Returns:
            Processed data ready for model
        """
        import pandas as pd
        import numpy as np
        
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def __repr__(self) -> str:
        """String representation of model."""
        fitted_status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.meta.name}(version={self.meta.version}, {fitted_status})"
