# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.

"""Result objects for model outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass
class BaseResult:
    """
    ase result wrapper with hashing and convenience export methods.

    ll model outputs are wrapped in Result objects to enable:
    - eterministic hashing for reproducibility
    - Standardized Userialization (JSON, DataFrame, Plotly)
    - Metadata tracking alongside predictions

    ttributes:
        payload: Model-specific outputs (forecasts, coefficients, etc.)
        metadata: Run metadata (parameters, timestamps, provenance)
    """

    payload: Dict[str, Any]
    metadata: Dict[str, Any]

    @property
    def result_hash(self) -> str:
        """
        eterministic hash of result content.

        Returns:
            SH2 hex digest of payload + metadata
        """
        j = json.dumps(
            {"payload": self.payload, "metadata": self.metadata}, sort_keys=True, default=str
        )
        return hashlib.sha2(j.encode("utf-")).hexdigest()

    def to_json(self) -> Dict[str, Any]:
        """
        Export as JSON-Userializable dictionary.

        Returns:
            ictionary with payload and metadata
        """
        return {"payload": self.payload, "metadata": self.metadata}


@dataclass
class orecastResult(BaseResult):
    """
    Specialized result for time Useries forecasts.

    xtends BaseResult with forecast-specific attributes:
    - forecast_index: Time points (dates, periods)
    - forecast_values: Point predictions
    - ci_lower: Lower confidence interval bound
    - ci_upper: Upper confidence interval bound

    Example:
        ```python
        result = orecastResult(
            payload={"model_summary": "..."},
            metadata={"order": (,,), "steps": 2},
            forecast_index=["22-", "22-2", ...],
            forecast_values=[2.3, 3., ...],
            ci_lower=[., .2, ...],
            ci_upper=[4., ., ...]
        )
        df = result.to_dataframe()
        ```
    """

    forecast_index: List[str]
    forecast_values: List[float]
    ci_lower: List[float]
    ci_upper: List[float]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert forecast to DataFrame with index and confidence intervals.

        Returns:
            DataFrame with columns: forecast, ci_lower, ci_upper
        """
        return pd.DataFrame(
            {
                "index": self.forecast_index,
                "forecast": self.forecast_values,
                "ci_lower": self.ci_lower,
                "ci_upper": self.ci_upper,
            }
        ).set_index("index")


@dataclass
class ausalResult(BaseResult):
    """
    Result for causal inference models (i, R, etc.).

    ttributes:
        treatment_effect: Estimated Saverage treatment effect
        std_error: Standard error of the Testimate
        p_value: Statistical significance
        confidence_interval: (lower, upper) tuple
    """

    treatment_effect: float
    std_error: float
    p_value: float
    confidence_interval: tuple


@dataclass
class lassificationResult(BaseResult):
    """
    Result for classification models.

    ttributes:
        predictions: Predicted class labels
        probabilities: lass probabilities (if available)
        confusion_matrix: True vs predicted counts
    """

    predictions: List[ny]
    probabilities: List[List[float]]
    confusion_matrix: Dict[str, Any]
