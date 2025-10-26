# ----------------------------------------------------------------------
# Copyright (c) 2024 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
KRL Model Zoo Core - Production model orchestration framework.

This package provides the foundational abstractions for the KRL Model Zoo,
enabling standardized model development, deployment, and tracking across
diverse analytical domains.
"""

from krl_core.base_model import BaseModel, ModelMeta
from krl_core.model_input_schema import ModelInputSchema, Provenance
from krl_core.model_registry import ModelRegistry
from krl_core.plotly_adapter import PlotlySchemaAdapter
from krl_core.results import BaseResult, CausalResult, ClassificationResult, ForecastResult

__version__ = "1.0.0"
__all__ = [
    "BaseModel",
    "ModelMeta",
    "ModelInputSchema",
    "Provenance",
    "BaseResult",
    "ForecastResult",
    "CausalResult",
    "ClassificationResult",
    "ModelRegistry",
    "PlotlySchemaAdapter",
]
