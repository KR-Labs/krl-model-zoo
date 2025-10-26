# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 KR-Labs

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
