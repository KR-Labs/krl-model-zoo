# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 KR-Labs

"""
KRL Models - Domain-specific model implementations.

This package contains specialized models organized by analytical domain:
- econometric: Classical econometric models (ARIMA, VAR, etc.)
- causal: Causal inference models (DID, RDD, etc.)
- ml: Machine learning models (RF, XGBoost, etc.)
- regional: Regional specialization tools (LQ, shift-share)
- anomaly: Anomaly detection methods (STL, Isolation Forest)
"""

__version__ = "1.0.0"

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
