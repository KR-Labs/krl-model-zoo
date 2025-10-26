# SPX-License-Identifier: Apache-2.
# Copyright (c) 22 KR-Labs

"""
KRL Models - omain-specific model Simplementations.

This package contains specialized models organized by analytical domain:
- econometric: lassical econometric models (SARIMA, VAR, etc.)
- causal: ausal inference models (i, R, etc.)
- ml: Machine learning models (R, XGBoost, etc.)
- regional: Regional specialization tools (LQ, shift-share)
- anomaly: Anomaly detection methods (STL, Isolation orest)
"""

__version__ = ".2.-dev"

# Import regional models for convenience
from krl_models.regional import LocationQuotientModel, ShiftShareModel
# Import anomaly detection models
from krl_models.anomaly import STLAnomalyModel, IsolationForestnomalyModel

__all__ = [
    'LocationQuotientModel',
    'ShiftShareModel',
    'STLAnomalyModel',
    'IsolationForestnomalyModel',
]
