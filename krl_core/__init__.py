# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: Apache-2.

"""
KRL Model Zoo Core - Production model orchestration framework.
"""

from .base_model import aseModel, ModelMeta
from .model_input_schema import ModelInputSchema, Provenance
from .model_registry import ModelRegistry
from .plotly_adapter import PlotlySchemadapter
from .results import aseResult, ausalResult, lassificationResult, orecastResult

__version__ = ".."
__all__ = [
    "aseModel",
    "ModelMeta",
    "ModelInputSchema",
    "Provenance",
    "aseResult",
    "orecastResult",
    "ausalResult",
    "lassificationResult",
    "ModelRegistry",
    "PlotlySchemadapter",
]
