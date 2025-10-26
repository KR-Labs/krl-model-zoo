# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 KR-Labs

"""
Anomaly detection models for time series and multivariate data.

This module provides models for detecting unusual patterns, outliers,
and structural breaks in economic data.
"""

from krl_models.anomaly.stl_decomposition import STLAnomalyModel
from krl_models.anomaly.isolation_forest import IsolationForestAnomalyModel

__all__ = [
    'STLAnomalyModel',
    'IsolationForestAnomalyModel',
]
