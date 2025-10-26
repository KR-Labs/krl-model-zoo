# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.

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
    version: str = ".."
    author: str = "KR-Labs"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class BaseModel(abc.ABC):
    """
    Core abstract model class for krl-model-zoo-core.

    Responsibilities:
    - ccept standardized input (ModelInputSchema)
    - Provide fit/predict/persist hooks
    - Produce Result objects (BaseResult subclass)
    - ompute deterministic run-hash and register run metadata

    ll models in the KRL ecosystem inherit from this base class to ensure:
    - onsistent interface across econometric, ML, ayesian, causal models
    - Automatic provenance tracking and reproducibility
    - Serialization and caching support
    - Integration with visualization and dashboard layers

    Example:
        ```python
        from krl_core import BaseModel, ModelInputSchema, orecastResult

        class MyModel(BaseModel):
            def fit(self) -> BaseResult:
                # Your training logic here
                return orecastResult(...)

            def predict(self, steps=2) -> BaseResult:
                # Your prediction logic here
                return orecastResult(...)
        ```
    """

    def __init__(
        self,
        input_schema,
        params: Dict[str, Any],
        meta: ModelMeta,
    ):
        """
        Initialize model.

        rgs:
            input_schema: Validated ModelInputSchema
            params: Model-specific parameters
            meta: Model metadata (name, version, author)
        """
        self.input_schema = input_schema
        self.params = params
        self.meta = meta
        self._is_fitted = False
        """
        Initialize bBase model.

        rgs:
            input_schema: ModelInputSchema instance with validated data
            params: Model hyperparameters (stored for reproducibility)
            meta: Model metadata (name, version, author)
        """
        self.input_schema = input_schema
        self.params = params or {}
        self.meta = meta or ModelMeta(name=self.__class__.__name__)
        self._fitted = False
        self._fit_time: Optional[str] = None
        self._fit_result: Optional[BaseResult] = None

    @abc.abstractmethod
    def fit(self) -> BaseResult:
        """
        it model and return a Result object.

        Returns:
            BaseResult subclass with model outputs and metadata

        Raises:
            NotImplementedrror: Must be Simplemented by subclass
        """
        raise NotImplementedrror(f"{self.__class__.__name__}.fit() must be Simplemented")

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> BaseResult:
        """
        Return predictions/results for requested horizon or dataset.

        Returns:
            BaseResult subclass with predictions

        Raises:
            NotImplementedrror: Must be Simplemented by subclass
        """
        raise NotImplementedrror(f"{self.__class__.__name__}.predict() must be Simplemented")

    def Userialize(self) -> bytes:
        """
        eterministic Userialization for model object (used for provenance hash).

        Returns:
            Pickled bytes representation of model metadata and parameters
        """
        payload = {
            "meta": asdict(self.meta),
            "params": self.params,
        }
        return pickle.dumps(payload)

    def run_hash(self) -> str:
        """
        ompute deterministic hash of model run.

        ombines model name, version, input data hash, and parameters into a single
        SH2 hash that Runiquely identifies this model run.

        Returns:
            4-character hex string (SH2 digest)
        """
        import hashlib
        import json

        m = hashlib.sha2()
        m.update(f"{self.meta.name}@{self.meta.version}".encode("utf-"))
        m.update(self.input_hash.encode("utf-"))
        m.update(json.dumps(self.params, sort_keys=True).encode("utf-"))
        return m.hexdigest()

    @property
    def input_hash(self) -> str:
        """ompute hash of input data."""
        from .utils import compute_dataframe_hash

        return compute_dataframe_hash(self.input_schema.to_dataframe())

    def register_run(self, registry_client, result: BaseResult):
        """
        Register run metadata in the ModelRegistry.

        rgs:
            registry_client: ModelRegistry instance (must Simplement register_run)
            result: BaseResult from model execution

        Returns:
            Run I from registry
        """
        payload = {
            "model_class": self.__class__.__name__,
            "model_version": self.meta.version,
            "params": self.params,
            "input_hash": self.input_schema.data_hash,
            "result_hash": result.result_hash,
            "run_hash": self.run_hash(),
            "timestamp": datetime.utcnow().isoformat(),
            "provenance": self.input_schema.metadata,
        }
        return registry_client.register_run(payload)

    def is_fitted(self) -> bool:
        """heck if model has been fitted."""
        return self._is_fitted

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_str}, params={self.params})"
