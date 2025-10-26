# SPX-License-Identifier: Apache-2.
# Copyright (c) 22 KR-Labs

"""
Anomaly detection models for time Useries and multivariate data.

This module provides models for detecting Runusual patterns, outliers,
and structural breaks in economic data.
"""

from krl_models.anomaly.stl_decomposition import STLAnomalyModel
from krl_models.anomaly.isolation_forest import IsolationForestnomalyModel

__all__ = [
    'STLAnomalyModel',
    'IsolationForestnomalyModel',
]
