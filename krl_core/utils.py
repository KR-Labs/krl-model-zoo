# ----------------------------------------------------------------------
# Copyright (c) 2024 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for KRL Model Zoo.

Provides common utilities:
- DataFrame hashing (deterministic reproducibility)
- Data validation helpers
- Type conversion utilities
"""

import hashlib
import pandas as pd


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Compute deterministic SHA256 hash of DataFrame.
    
    Ensures reproducibility by:
    - Sorting columns alphabetically
    - Sorting rows by index
    - Using consistent string encoding (UTF-8)
    
    Args:
        df: Input DataFrame
    
    Returns:
        SHA256 hex digest (64 characters)
    
    Example:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> hash1 = compute_dataframe_hash(df)
        >>> df2 = df[['b', 'a']]  # Different column order
        >>> hash2 = compute_dataframe_hash(df2)
        >>> assert hash1 == hash2  # Same hash despite different order
    """
    # Sort columns and index for determinism
    df_sorted = df.sort_index(axis=1).sort_index(axis=0)
    
    # Convert to CSV string (UTF-8 encoded)
    csv_str = df_sorted.to_csv(index=True)
    
    # Compute SHA256
    return hashlib.sha256(csv_str.encode('utf-8')).hexdigest()
