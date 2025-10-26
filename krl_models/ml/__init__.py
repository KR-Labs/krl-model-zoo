# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 KR-Labs

"""
Machine Learning Models Module

Provides ML regression models for economic forecasting and prediction.
"""

from krl_models.ml.random_forest import RandomForestModel
from krl_models.ml.xgboost_model import XGBoostModel
from krl_models.ml.regularized_regression import RegularizedRegression

# Alias for backwards compatibility
RegularizedRegressionModel = RegularizedRegression

__all__ = [
    'RandomForestModel',
    'XGBoostModel',
    'RegularizedRegression',
    'RegularizedRegressionModel',  # Backwards compatibility
]
