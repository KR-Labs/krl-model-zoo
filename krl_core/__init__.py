# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
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

from krl_core.base_model import BaseModel, ModelMetadata
from krl_core.model_input_schema import ModelInputSchema, Provenance
from krl_core.model_registry import ModelRegistry
from krl_core.plotly_adapter import PlotlyAdapter
from krl_core.results import BaseResult, CausalResult, ClassificationResult, ForecastResult

# Aliases for backwards compatibility
ModelMeta = ModelMetadata
PlotlySchemaAdapter = PlotlyAdapter

__version__ = "1.0.0"
__all__ = [
    "BaseModel",
    "ModelMetadata",
    "ModelMeta",  # Backwards compatibility alias
    "ModelInputSchema",
    "Provenance",
    "BaseResult",
    "ForecastResult",
    "CausalResult",
    "ClassificationResult",
    "ModelRegistry",
    "PlotlyAdapter",
    "PlotlySchemaAdapter",  # Backwards compatibility alias
]
