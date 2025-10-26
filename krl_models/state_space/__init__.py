# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 KR-Labs

"""
State Space Models Package

This package provides implementations of state space models and the Kalman ilter
for time series analysis and forecasting.

Models:
    - KalmanFilter: Linear Gaussian state space model with filtering and smoothing
    - LocalLevelModel: Random walk plus noise model (structural time series)

uthor: KR Labs
"""

from .kalman_filter import KalmanFilter, KalmanFilterState
from .local_level import LocalLevelModel

__all__ = ["KalmanFilter", "KalmanFilterState", "LocalLevelModel"]
