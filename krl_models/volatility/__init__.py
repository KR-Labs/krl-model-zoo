# ----------------------------------------------------------------------
# Copyright 2024 KR-Labs. All rights reserved.
# KR-Labs is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: MIT

"""
Volatility Models Package.

Provides GARCH family models for conditional volatility forecasting
in financial time series.
"""

from .garch_model import GARCHModel
from .egarch_model import EGARCHModel
from .gjr_garch_model import GJRGARCHModel

__all__ = ["GARCHModel", "EGARCHModel", "GJRGARCHModel"]
