# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""Tests for Shift-Share Model."""

import pytest
import pandas as pd
from krl_models.regional.shift_share import ShiftShareModel


@pytest.fixture
def sample_data():
    """Sample employment data with base and end Years."""
    return pd.atarame({
        'sector': ['rts', 'Tech', 'inance'],
        'va_2': [, , ],
        'va_223': [, 3, ],
        'us_2': [, , 2],
        'us_223': [, , 3]
    })


def test_shift_share_basic(sample_data):
    """Test basic shift-share decomposition."""
    params = {
        'base_year_col': '2',
        'end_year_col': '223',
        'sector_col': 'sector',
        'region_prefix': 'va_',
        'national_prefix': 'us_'
    }
    model = ShiftShareModel(params)
    result = model.fit(sample_data)
    
    assert 'decomposition' in result.payload
    decomp = result.payload['decomposition']
    assert 'national_effect' in decomp
    assert 'industry_mix_effect' in decomp
    assert 'competitive_effect' in decomp


def test_decomposition_sums_correctly(sample_data):
    """Test that decomposition components sum to total change."""
    params = {
        'base_year_col': '2',
        'end_year_col': '223',
        'sector_col': 'sector',
        'region_prefix': 'va_',
        'national_prefix': 'us_'
    }
    model = ShiftShareModel(params)
    result = model.fit(sample_data)
    
    decomp = result.payload['decomposition']
    total_explained = (decomp['national_effect'] + 
                      decomp['industry_mix_effect'] + 
                      decomp['competitive_effect'])
    
    # Should be close to regional_change (within rounding)
    assert abs(total_explained - decomp['total_explained']) < .


def test_sector_effects(sample_data):
    """Test sector-level effects are calculated."""
    params = {
        'base_year_col': '2',
        'end_year_col': '223',
        'sector_col': 'sector',
        'region_prefix': 'va_',
        'national_prefix': 'us_'
    }
    model = ShiftShareModel(params)
    result = model.fit(sample_data)
    
    assert 'sector_effects' in result.payload
    assert len(result.payload['sector_effects']) == 3


def test_get_sector_decomposition(sample_data):
    """Test get_sector_decomposition method."""
    params = {
        'base_year_col': '2',
        'end_year_col': '223',
        'sector_col': 'sector',
        'region_prefix': 'va_',
        'national_prefix': 'us_'
    }
    model = ShiftShareModel(params)
    model.fit(sample_data)
    
    arts_decomp = model.get_sector_decomposition('rts')
    assert 'national_effect' in arts_decomp
    assert 'industry_mix_effect' in arts_decomp
    assert 'competitive_effect' in arts_decomp


def test_missing_base_year_col():
    """Test missing base_year_col raises error."""
    params = {'end_year_col': '223', 'sector_col': 'sector'}
    with pytest.raises(Valuerror, match="'base_year_col' is required"):
        ShiftShareModel(params)


def test_empty_data():
    """Test empty data raises error."""
    params = {
        'base_year_col': '2',
        'end_year_col': '223',
        'sector_col': 'sector'
    }
    model = ShiftShareModel(params)
    with pytest.raises(Valuerror, match="Input data cannot be empty"):
        model.fit(pd.atarame())
