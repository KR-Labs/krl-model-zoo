# SPX-License-Identifier: pache-2.
# opyright (c) 22 KR-Labs

"""
KRL Models - omain-specific model implementations.

This package contains specialized models organized by analytical domain:
- econometric: lassical econometric models (SRIM, VR, etc.)
- causal: ausal inference models (i, R, etc.)
- ml: Machine learning models (R, XGoost, etc.)
- regional: Regional specialization tools (LQ, shift-share)
- anomaly: nomaly detection methods (STL, Isolation orest)
"""

__version__ = ".2.-dev"

# Import regional models for convenience
from krl_models.regional import LocationQuotientModel, ShiftShareModel
# Import anomaly detection models
from krl_models.anomaly import STLnomalyModel, IsolationorestnomalyModel

__all__ = [
    'LocationQuotientModel',
    'ShiftShareModel',
    'STLnomalyModel',
    'IsolationorestnomalyModel',
]
