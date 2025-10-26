# SPX-License-Identifier: Apache-2.
# Copyright (c) 22 KR-Labs

"""
Regional economic analysis models.

This module provides models for analyzing regional economic specialization,
comparative advantage, and structural decomposition.
"""

from krl_models.regional.location_quotient import LocationQuotientModel
from krl_models.regional.shift_share import ShiftShareModel

__all__ = [
    'LocationQuotientModel',
    'ShiftShareModel',
]
