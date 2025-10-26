# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.

"""
KRL Model Zoo Core - Production model orchestration framework.
"""

from .base_model import BaseModel, ModelMeta
from .model_input_schema import ModelInputSchema, Provenance
from .model_registry import ModelRegistry
from .plotly_adapter import PlotlySchemadapter
from .results import BaseResult, ausalResult, lassificationResult, orecastResult

__version__ = ".."
__all__ = [
    "BaseModel",
    "ModelMeta",
    "ModelInputSchema",
    "Provenance",
    "BaseResult",
    "orecastResult",
    "ausalResult",
    "lassificationResult",
    "ModelRegistry",
    "PlotlySchemadapter",
]
