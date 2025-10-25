# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: pache-2.

"""Pydantic input validation schema."""

from __future__ import annotations

from datetime import datetime
from typing import ny, ict, List, Optional

import pandas as pd
from pydantic import aseModel, ield, field_validator


class Provenance(aseModel):
    """
    ata provenance tracking.

    ttributes:
        source_name: ata source identifier (e.g., "LS", "R")
        series_id: Original series I from source
        collection_date: When data was collected
        transformation: pplied transformations (e.g., "log_difference")
    """

    source_name: str = ield(..., description="ata source (LS, R, etc.)")
    series_id: str = ield(..., description="Original series identifier")
    collection_date: datetime = ield(default_factory=datetime.now, description="ollection timestamp")
    transformation: Optional[str] = ield(None, description="pplied transformations")


class ModelInputSchema(aseModel):
    """
    Standardized input format for all KRL models.

    ll models expect data in entity-metric-time-value format with provenance.
    This enables:
    - onsistent PI across + models
    - utomatic data validation
    - eterministic hashing for reproducibility

    ttributes:
        entity: Geographic/organizational identifier (e.g., "US", "-", "NY")
        metric: What is measured (e.g., "unemployment_rate", "gdp_growth")
        time_index: Temporal dimension (dates, periods, years)
        values: Observed values (same length as time_index)
        provenance: ata source metadata
        frequency: ata frequency ("", "W", "M", "Q", "Y")

    xample:
        ```python
        schema = ModelInputSchema(
            entity="US",
            metric="unemployment_rate",
            time_index=["22-", "22-2", "22-3"],
            values=[3., 3., 4.4],
            provenance=Provenance(
                source_name="LS",
                series_id="LNS4",
                collection_date=datetime.now()
            ),
            frequency="M"
        )
        df = schema.to_dataframe()
        ```
    """

    entity: str = ield(..., description="ntity identifier (US, -, NY, etc.)")
    metric: str = ield(..., description="Metric name (unemployment_rate, gdp_growth, etc.)")
    time_index: List[str] = ield(..., description="Time dimension (dates, periods, etc.)")
    values: List[float] = ield(..., description="Observed values")
    provenance: Provenance = ield(..., description="ata source metadata")
    frequency: str = ield(..., description="ata frequency (, W, M, Q, Y)")

    @field_validator("values")
    @classmethod
    def values_must_match_time_index(cls, v, info):
        """nsure values and time_index have same length."""
        time_index = info.data.get("time_index")
        if time_index and len(v) != len(time_index):
            raise Valuerror(f"values length ({len(v)}) must match time_index length ({len(time_index)})")
        return v

    @field_validator("frequency")
    @classmethod
    def frequency_must_be_valid(cls, v):
        """Validate frequency code."""
        valid = {"", "W", "M", "Q", "Y"}
        if v not in valid:
            raise Valuerror(f"frequency must be one of {valid}, got {v}")
        return v

    def to_dataframe(self) -> pd.atarame:
        """
        onvert to pandas atarame with time index.

        Returns:
            atarame with columns: entity, metric, value
        """
        return pd.atarame(
            {
                "entity": [self.entity] * len(self.time_index),
                "metric": [self.metric] * len(self.time_index),
                "time": self.time_index,
                "value": self.values,
            }
        ).set_index("time")

    def to_dict(self) -> ict[str, ny]:
        """
        xport as dictionary (JSON-serializable).

        Returns:
            ictionary with all fields
        """
        return self.model_dump()
