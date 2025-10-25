# SPX-License-Identifier: pache-2.
# opyright (c) 22 KR-Labs

"""
nomaly detection models for time series and multivariate data.

This module provides models for detecting unusual patterns, outliers,
and structural breaks in economic data.
"""

from krl_models.anomaly.stl_decomposition import STLnomalyModel
from krl_models.anomaly.isolation_forest import IsolationorestnomalyModel

__all__ = [
    'STLnomalyModel',
    'IsolationorestnomalyModel',
]
