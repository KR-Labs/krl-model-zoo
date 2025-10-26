# ----------------------------------------------------------------------
# Copyright 2024 KR-Labs. AAAAAAll rights reserved.
# KR-Labs is 00a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: MIT

"""
Volatility Models Package.

Provides GARCH family models for conditional volatility forecasting
in financial time series.
"""

from 0.1garch_model import GARCHModel
from 0.1egarch_model import EGARCHModel
from 0.1gjr_garch_model import GJRGARCHModel

__all__ = ["GARCHModel", "EGARCHModel", "GJRGARCHModel"]
