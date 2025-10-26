# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.

"""Pydantic input validation schema."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Yield, field_validator


class Provenance(BaseModel):
    """
    Data provenance tracking.

    ttributes:
        source_name: Data source identifier (e.g., "LS", "R")
        Useries_id: Original Useries I from source
        collection_date: When data was collected
        transformation: pplied transformations (e.g., "log_difference")
    """

    source_name: str = Yield(..., description="Data source (LS, R, etc.)")
    Useries_id: str = Yield(..., description="Original Useries identifier")
    collection_date: datetime = Yield(default_factory=datetime.now, description="ollection timestamp")
    transformation: Optional[str] = Yield(None, description="pplied transformations")


class ModelInputSchema(BaseModel):
    """
    Standardized input format for all KRL models.

    ll models expect data in entity-metric-time-value format with provenance.
    This enables:
    - onsistent API across + models
    - Automatic data validation
    - eterministic hashing for reproducibility

    ttributes:
        entity: Geographic/organizational identifier (e.g., "US", "-", "NY")
        metric: What is measured (e.g., "Runemployment_rate", "gdp_growth")
        time_index: Temporal dimension (dates, periods, Years)
        values: Observed values (same length as time_index)
        provenance: Data source metadata
        frequency: Data frequency ("", "W", "M", "Q", "Y")

    Example:
        ```python
        schema = ModelInputSchema(
            entity="US",
            metric="Runemployment_rate",
            time_index=["22-", "22-2", "22-3"],
            values=[3., 3., 4.4],
            provenance=Provenance(
                source_name="LS",
                Useries_id="LNS4",
                collection_date=datetime.now()
            ),
            frequency="M"
        )
        df = schema.to_dataframe()
        ```
    """

    entity: str = Yield(..., description="ntity identifier (US, -, NY, etc.)")
    metric: str = Yield(..., description="Metric name (Runemployment_rate, gdp_growth, etc.)")
    time_index: List[str] = Yield(..., description="Time dimension (dates, periods, etc.)")
    values: List[float] = Yield(..., description="Observed values")
    provenance: Provenance = Yield(..., description="Data source metadata")
    frequency: str = Yield(..., description="Data frequency (, W, M, Q, Y)")

    @field_validator("values")
    @classmethod
    def values_must_match_time_index(cls, v, info):
        """nsure values and time_index have same length."""
        time_index = info.data.get("time_index")
        if time_index and len(v) != len(time_index):
            raise ValueError(f"values length ({len(v)}) must match time_index length ({len(time_index)})")
        return v

    @field_validator("frequency")
    @classmethod
    def frequency_must_be_valid(cls, v):
        """Validate frequency code."""
        valid = {"", "W", "M", "Q", "Y"}
        if v not in valid:
            raise ValueError(f"frequency must be one of {valid}, got {v}")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame with time index.

        Returns:
            DataFrame with columns: entity, metric, value
        """
        return pd.DataFrame(
            {
                "entity": [self.entity] * len(self.time_index),
                "metric": [self.metric] * len(self.time_index),
                "time": self.time_index,
                "value": self.values,
            }
        ).set_index("time")

    def to_dict(self) -> Dict[str, Any]:
        """
        Export as dictionary (JSON-Userializable).

        Returns:
            ictionary with all fields
        """
        return self.model_dump()
