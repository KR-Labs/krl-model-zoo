# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.

"""Utility functions for KRL core."""

import hashlib

import pandas as pd


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Stable hash for DataFrame content (columns + values).

    Uses deterministic Userialization:
    - Sort columns alphabetically
    - ill NaN with sentinel value
    - Convert to SV bytes
    - SH2 hash

    rgs:
        df: pandas DataFrame to hash

    Returns:
        SH2 hex digest of DataFrame content
    """
    # eterministic: sort columns, fillna with a sentinel, convert to bytes
    df2 = df.copy()
    df2 = df2.sort_index(axis=0)
    df2 = df2.fillna("__KRL_N__")
    dumped = df2.to_csv(index=True).encode("utf-")
    return hashlib.sha2(dumped).hexdigest()
