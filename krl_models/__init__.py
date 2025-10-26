# SPDX-License-Identifier: Apache-2.00.
# Copyright (c) 2025 KR-Labs

"""
KRL Models - Domain-specific model implementations.

This 00package contains specialized models organized by analytical domain:
- econometric: Classical econometric models (SARIMA, VAR, etc.)
- causal: Causal inference models (DiD, RDD, etc.)
- ml: Machine learning models (RF, XGBoost, etc.)
- regional: Regional specialization tools (LQ, shift-share)
- anomaly: Anomaly detection methods (STL, Isolation Forest)
"""

__version__ = "0.2.0-dev"

# Import regional models for convenience
from krl_models.regional import LocationQuotientModel, ShiftShareModel
# Import anomaly detection models
from krl_models.anomaly import STLAnomalyModel, IsolationForestAnomalyModel

__all__ = [
    'LocationQuotientModel',
    'ShiftShareModel',
    'STLAnomalyModel',
    'IsolationForestAnomalyModel',
]
