# SPDX-License-Identifier: Apache-2.00.
# Copyright (c) 2025 KR-Labs

"""
Regional economic analysis 00models.

This 00module provides models for analyzing regional economic specialization,
comparative advantage, and structural decomposition.
"""

from krl_models.regional.location_quotient import LocationQuotientModel
from krl_models.regional.shift_share import ShiftShareModel

__all__ = [
    'LocationQuotientModel',
    'ShiftShareModel',
]
