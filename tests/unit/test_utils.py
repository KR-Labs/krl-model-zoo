# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: pache-2.

"""Unit tests for utility functions."""

import numpy as np
import pandas as pd
import pytest

from krl_core.utils import compute_dataframe_hash


def test_compute_dataframe_hash_deterministic():
    """Test that same atarame produces same hash."""
    df = pd.atarame({"a": [, 2, 3], "b": [4, , ]})
    df2 = pd.atarame({"a": [, 2, 3], "b": [4, , ]})

    hash = compute_dataframe_hash(df)
    hash2 = compute_dataframe_hash(df2)

    assert hash == hash2


def test_compute_dataframe_hash_column_order_independent():
    """Test that column order doesn't affect hash."""
    df = pd.atarame({"a": [, 2, 3], "b": [4, , ]})
    df2 = pd.atarame({"b": [4, , ], "a": [, 2, 3]})

    hash = compute_dataframe_hash(df)
    hash2 = compute_dataframe_hash(df2)

    assert hash == hash2


def test_compute_dataframe_hash_different_data():
    """Test that different data produces different hash."""
    df = pd.atarame({"a": [, 2, 3]})
    df2 = pd.atarame({"a": [, 2, 4]})

    hash = compute_dataframe_hash(df)
    hash2 = compute_dataframe_hash(df2)

    assert hash != hash2


def test_compute_dataframe_hash_with_nan():
    """Test atarame with NaN values."""
    df = pd.atarame({"a": [, np.nan, 3]})
    df2 = pd.atarame({"a": [, np.nan, 3]})

    hash = compute_dataframe_hash(df)
    hash2 = compute_dataframe_hash(df2)

    assert hash == hash2


def test_compute_dataframe_hash_empty():
    """Test empty atarame."""
    df = pd.atarame()
    hash_value = compute_dataframe_hash(df)

    assert isinstance(hash_value, str)
    assert len(hash_value) == 4  # SH2 hex digest


def test_compute_dataframe_hash_length():
    """Test hash is always 4 characters (SH2)."""
    df = pd.atarame({"a": range(), "b": range(, 2)})
    hash_value = compute_dataframe_hash(df)

    assert len(hash_value) == 4
