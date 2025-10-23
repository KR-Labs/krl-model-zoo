# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025 KR-Labs Foundation. All rights reserved.
# Licensed under Apache License 2.0 (see LICENSE file for details)

"""
KRL Model Zoo - Production-ready models for causal inference and forecasting.
"""

from .__version__ import __author__, __license__, __version__
from .base_model import BaseModel

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "BaseModel",
]
