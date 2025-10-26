# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 KR-Labs

"""
Econometric Time Series Models
================================

Classical time series forecasting models for economic and financial data.

Models
------
- SARIMAModel: Seasonal ARIMA for univariate series
- ProphetModel: Meta's Prophet for trend and seasonality
- VARModel: Vector Autoregression for multivariate series
- CointegrationModel: Cointegration testing and VECM
"""

from .cointegration_model import CointegrationModel
from .prophet_model import ProphetModel
from .sarima_model import SARIMAModel
from .var_model import VARModel

__all__ = ["SARIMAModel", "ProphetModel", "VARModel", "CointegrationModel"]
