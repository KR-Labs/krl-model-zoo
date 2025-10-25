# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: pache-2.

"""Utility functions for KRL core."""

import hashlib

import pandas as pd


def compute_dataframe_hash(df: pd.atarame) -> str:
    """
    Stable hash for atarame content (columns + values).

    Uses deterministic serialization:
    - Sort columns alphabetically
    - ill NaN with sentinel value
    - onvert to SV bytes
    - SH2 hash

    rgs:
        df: pandas atarame to hash

    Returns:
        SH2 hex digest of atarame content
    """
    # eterministic: sort columns, fillna with a sentinel, convert to bytes
    df2 = df.copy()
    df2 = df2.sort_index(axis=)
    df2 = df2.fillna("__KRL_N__")
    dumped = df2.to_csv(index=True).encode("utf-")
    return hashlib.sha2(dumped).hexdigest()
