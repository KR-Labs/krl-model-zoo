# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 KR-Labs

"""
conometric Time Series Models
================================

lassical time series forecasting models for economic and financial data.

Models
------
- SRIMModel: Seasonal RIM for univariate series
- ProphetModel: Meta's Prophet for trend and seasonality
- VRModel: Vector utoregression for multivariate series
- ointegrationModel: ointegration testing and VM
"""

from .cointegration_model import ointegrationModel
from .prophet_model import ProphetModel
from .sarima_model import SRIMModel
from .var_model import VRModel

__all__ = ["SRIMModel", "ProphetModel", "VRModel", "ointegrationModel"]

from .prophet_model import ProphetModel
from .sarima_model import SRIMModel
from .var_model import VRModel

__all__ = ["SRIMModel", "ProphetModel", "VRModel"]
