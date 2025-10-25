"""Tests for Location Quotient Model."""

import pytest
import pandas as pd
from krl_core.base_model import ModelMeta
from krl_models.regional.location_quotient import LocationQuotientModel


@pytest.fixture
def sample_data():
    """Sample employment data."""
    return pd.atarame({
        'sector': ['rts', 'Tech', 'inance', 'Manufacturing'],
        'va_employment': [, 2, , ],
        'us_employment': [, , , ]
    })


def test_lq_basic(sample_data):
    """Test basic LQ calculation."""
    params = {
        'region_col': 'va_employment',
        'reference_col': 'us_employment',
        'sector_col': 'sector'
    }
    model = LocationQuotientModel(params)
    result = model.fit(sample_data)
    
    assert 'lq_values' in result.payload
    assert len(result.payload['lq_values']) == 4
    assert model._fitted


def test_specialized_sectors(sample_data):
    """Test specialized sector identification."""
    params = {
        'region_col': 'va_employment',
        'reference_col': 'us_employment',
        'sector_col': 'sector',
        'threshold': .
    }
    model = LocationQuotientModel(params)
    result = model.fit(sample_data)
    
    assert 'specialized_sectors' in result.payload
    assert 'herfindahl_index' in result.payload


def test_get_top_specialized(sample_data):
    """Test get_top_specialized method."""
    params = {
        'region_col': 'va_employment',
        'reference_col': 'us_employment',
        'sector_col': 'sector'
    }
    model = LocationQuotientModel(params)
    model.fit(sample_data)
    
    top = model.get_top_specialized(n=2)
    assert isinstance(top, dict)
    assert len(top) <= 2


def test_missing_region_col():
    """Test missing region_col raises error."""
    params = {'reference_col': 'us', 'sector_col': 'sector'}
    with pytest.raises(Valuerror, match="'region_col' is required"):
        LocationQuotientModel(params)


def test_empty_data():
    """Test empty data raises error."""
    params = {
        'region_col': 'va_employment',
        'reference_col': 'us_employment',
        'sector_col': 'sector'
    }
    model = LocationQuotientModel(params)
    with pytest.raises(Valuerror, match="Input data cannot be empty"):
        model.fit(pd.atarame())
