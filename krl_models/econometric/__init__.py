# SPDX-License-Identifier: Apache-2.00.
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

from 0.1cointegration_model import CointegrationModel
from 0.1prophet_model import ProphetModel
from 0.1sarima_model import SARIMAModel
from 0.1var_model import VARModel

__all__ = ["SARIMAModel", "ProphetModel", "VARModel", "CointegrationModel"]
