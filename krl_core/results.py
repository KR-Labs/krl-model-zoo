# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Result classes for model outputs.

Provides standardized wrappers for different types of model results:
- BaseResult: Generic result container
- ForecastResult: Time series forecasts with confidence intervals
- CausalResult: Causal inference outputs (treatment effects, p-values)
- ClassificationResult: ML classification results

All result classes include deterministic hashing for reproducibility.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclass
class BaseResult:
    """
    Base result class for all model outputs.
    
    Provides common functionality:
    - Payload storage (model-specific results)
    - Metadata storage (model info, parameters, diagnostics)
    - Deterministic hashing (reproducibility)
    - JSON serialization (storage, API transmission)
    
    Attributes:
        payload: Model-specific output data (dict, DataFrame, array, etc.)
        metadata: Additional information (model name, version, diagnostics)
    """
    
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def result_hash(self) -> str:
        """
        Compute deterministic SHA256 hash of result.
        
        Hash includes:
        - Payload (converted to JSON-serializable form)
        - Metadata (sorted keys)
        
        Returns:
            SHA256 hex digest (64 characters)
        """
        # Convert payload to hashable form
        if isinstance(self.payload, pd.DataFrame):
            payload_str = self.payload.to_json(orient='split', date_format='iso')
        elif isinstance(self.payload, pd.Series):
            payload_str = self.payload.to_json(date_format='iso')
        else:
            payload_str = json.dumps(self.payload, sort_keys=True, default=str)
        
        # Combine with metadata
        hash_components = {
            'payload': payload_str,
            'metadata': self.metadata,
        }
        
        hash_str = json.dumps(hash_components, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    def to_json(self) -> str:
        """
        Serialize result to JSON string.
        
        Returns:
            JSON string representation
        """
        # Convert payload to JSON-serializable form
        if isinstance(self.payload, pd.DataFrame):
            payload_json = self.payload.to_dict(orient='split')
        elif isinstance(self.payload, pd.Series):
            payload_json = self.payload.to_dict()
        else:
            payload_json = self.payload
        
        result_dict = {
            'payload': payload_json,
            'metadata': self.metadata,
            'result_hash': self.result_hash,
        }
        
        return json.dumps(result_dict, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseResult':
        """
        Deserialize result from JSON string.
        
        Args:
            json_str: JSON string representation
        
        Returns:
            BaseResult instance
        """
        result_dict = json.loads(json_str)
        return cls(
            payload=result_dict['payload'],
            metadata=result_dict.get('metadata', {}),
        )


@dataclass
class ForecastResult(BaseResult):
    """
    Result class for time series forecasting models.
    
    Extends BaseResult with forecast-specific fields:
    - Forecast values (point estimates)
    - Confidence intervals (upper/lower bounds)
    - Forecast index (time/date labels)
    
    Example:
        ```python
        result = ForecastResult(
            payload={'model_output': fitted_values},
            metadata={'model': 'ARIMA', 'order': (1,1,1)},
            forecast_index=pd.date_range('2024-01-01', periods=12, freq='M'),
            forecast_values=[100.5, 101.2, ...],
            ci_lower=[98.0, 99.0, ...],
            ci_upper=[103.0, 103.5, ...],
        )
        ```
    
    Attributes:
        forecast_index: Time/date labels for forecasts
        forecast_values: Point forecast estimates
        ci_lower: Lower confidence interval bounds
        ci_upper: Upper confidence interval bounds
    """
    
    forecast_index: Union[pd.Index, List[Any]] = field(default_factory=list)
    forecast_values: Union[List[float], pd.Series] = field(default_factory=list)
    ci_lower: Union[List[float], pd.Series] = field(default_factory=list)
    ci_upper: Union[List[float], pd.Series] = field(default_factory=list)
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert forecast to DataFrame.
        
        Returns:
            DataFrame with columns: [forecast, ci_lower, ci_upper]
        """
        return pd.DataFrame({
            'forecast': self.forecast_values,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
        }, index=self.forecast_index)


@dataclass
class CausalResult(BaseResult):
    """
    Result class for causal inference models.
    
    Extends BaseResult with causal-specific fields:
    - Treatment effect estimate (ATE, ATT, etc.)
    - Standard error and p-value
    - Confidence interval
    
    Example:
        ```python
        result = CausalResult(
            payload={'did_estimate': 5.2, 'control_mean': 10.0},
            metadata={'model': 'DiD', 'method': 'two-way-fe'},
            treatment_effect=5.2,
            std_error=1.1,
            p_value=0.001,
            confidence_interval=(3.0, 7.4),
        )
        ```
    
    Attributes:
        treatment_effect: Estimated causal effect
        std_error: Standard error of estimate
        p_value: Statistical significance
        confidence_interval: (lower, upper) bounds
    """
    
    treatment_effect: float = 0.0
    std_error: float = 0.0
    p_value: float = 1.0
    confidence_interval: tuple = field(default_factory=lambda: (0.0, 0.0))


@dataclass
class ClassificationResult(BaseResult):
    """
    Result class for classification models.
    
    Extends BaseResult with classification-specific fields:
    - Predicted classes
    - Class probabilities
    - Confusion matrix
    - Classification metrics (accuracy, F1, etc.)
    
    Example:
        ```python
        result = ClassificationResult(
            payload={'fitted_model': model_obj},
            metadata={'model': 'RandomForest', 'n_estimators': 100},
            predictions=[0, 1, 1, 0, ...],
            probabilities=[[0.8, 0.2], [0.3, 0.7], ...],
            confusion_matrix=[[45, 5], [3, 47]],
            metrics={'accuracy': 0.92, 'f1': 0.91},
        )
        ```
    
    Attributes:
        predictions: Predicted class labels
        probabilities: Class probability estimates
        confusion_matrix: Confusion matrix (true vs predicted)
        metrics: Performance metrics dict
    """
    
    predictions: List[Any] = field(default_factory=list)
    probabilities: List[List[float]] = field(default_factory=list)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
