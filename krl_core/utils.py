# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Utility Functions for KRL Core
===============================

Apache 2.0 License - Gate 1 Foundation
Author: KR Labs

Common utilities for data validation, preprocessing, and error handling.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_array(
    arr: Union[np.ndarray, list, pd.Series],
    name: str = "array",
    ndim: Optional[int] = None,
    min_length: int = 1
) -> np.ndarray:
    """
    Validate and convert array-like input.
    
    Parameters
    ----------
    arr : array-like
        Input array
    name : str, default='array'
        Name for error messages
    ndim : int, optional
        Required number of dimensions
    min_length : int, default=1
        Minimum length
    
    Returns
    -------
    arr : np.ndarray
        Validated numpy array
    
    Raises
    ------
    ValueError
        If validation fails
    
    Examples
    --------
    >>> arr = validate_array([1, 2, 3], ndim=1, min_length=2)
    >>> print(arr.shape)
    (3,)
    """
    arr = np.asarray(arr)
    
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {arr.ndim}D")
    
    if len(arr) < min_length:
        raise ValueError(f"{name} must have at least {min_length} elements, got {len(arr)}")
    
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf values")
    
    return arr


def handle_missing(
    arr: np.ndarray,
    method: str = 'drop'
) -> np.ndarray:
    """
    Handle missing values in array.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array
    method : str, default='drop'
        Method: 'drop', 'forward', 'backward', 'mean'
    
    Returns
    -------
    arr : np.ndarray
        Array with missing values handled
    """
    if method == 'drop':
        return arr[~np.isnan(arr)]
    elif method == 'forward':
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        np.maximum.accumulate(idx, out=idx)
        return arr[idx]
    elif method == 'backward':
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(len(mask)), len(mask) - 1)
        idx = np.minimum.accumulate(idx[::-1])[::-1]
        return arr[idx]
    elif method == 'mean':
        arr = arr.copy()
        arr[np.isnan(arr)] = np.nanmean(arr)
        return arr
    else:
        raise ValueError(f"Unknown method: {method}")


def train_test_split(
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
    train_size: float = 0.8
) -> Tuple:
    """
    Split data into train/test sets.
    
    Parameters
    ----------
    y : np.ndarray
        Target variable
    X : np.ndarray, optional
        Feature matrix
    train_size : float, default=0.8
        Proportion for training
    
    Returns
    -------
    splits : tuple
        (y_train, y_test) or (y_train, y_test, X_train, X_test)
    """
    n = len(y)
    split_idx = int(n * train_size)
    
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if X is not None:
        X_train, X_test = X[:split_idx], X[split_idx:]
        return y_train, y_test, X_train, X_test
    
    return y_train, y_test


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Compute common forecasting metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    metrics : dict
        Dictionary with mae, rmse, mape, r2
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2)
    }


def safe_log(arr: np.ndarray) -> np.ndarray:
    """
    Safe logarithm (handles zeros and negatives).
    
    Parameters
    ----------
    arr : np.ndarray
        Input array
    
    Returns
    -------
    log_arr : np.ndarray
        Logarithm with safe handling
    """
    arr = np.asarray(arr)
    arr = np.maximum(arr, 1e-10)  # Replace zeros/negatives
    return np.log(arr)


def standardize(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Standardize array to zero mean and unit variance.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array
    
    Returns
    -------
    standardized : np.ndarray
        Standardized array
    mean : float
        Original mean
    std : float
        Original standard deviation
    """
    mean = np.mean(arr)
    std = np.std(arr)
    
    if std == 0:
        logger.warning("Array has zero variance, returning original")
        return arr, mean, std
    
    return (arr - mean) / std, mean, std


def check_stationarity(series: np.ndarray) -> bool:
    """
    Simple stationarity check using rolling statistics.
    
    Parameters
    ----------
    series : np.ndarray
        Time series
    
    Returns
    -------
    is_stationary : bool
        True if likely stationary
    """
    if len(series) < 20:
        logger.warning("Series too short for stationarity check")
        return True
    
    # Check if rolling mean/std are stable
    window = len(series) // 4
    rolling_mean = np.convolve(series, np.ones(window)/window, mode='valid')
    
    # If rolling mean variance is small, likely stationary
    mean_variance = np.var(rolling_mean)
    series_variance = np.var(series)
    
    return mean_variance < 0.1 * series_variance
