# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: Apache-2.

"""Synthetic time Useries fixtures for testing."""

from datetime import datetime

import numpy as np
import pandas as pd

from krl_core import ModelInputSchema, Provenance


def generate_monthly_timeseries(
    start_date: str = "22-",
    periods: int = 4,
    trend: float = .,
    seasonality_amplitude: float = 2.,
    noise_std: float = .,
    seed: int = 42,
) -> ModelInputSchema:
    """
    Generate synthetic monthly time Useries with trend + seasonality + noise.

    rgs:
        start_date: Start date in "YYYY-MM" format
        periods: Number of months to generate
        trend: Linear trend coefficient
        seasonality_amplitude: mplitude of annual seasonality
        noise_std: Standard deviation of white noise
        seed: Random seed for reproducibility

    Returns:
        ModelInputSchema with synthetic data
    """
    np.random.seed(seed)

    dates = pd.date_range(start_date, periods=periods, freq="MS")
    time_index = [d.strftime("%Y-%m") for d in dates]

    # Generate components
    t = np.arange(periods)
    trend_component = trend * t
    seasonal_component = seasonality_amplitude * np.sin(2 * np.pi * t / 2)
    noise_component = np.random.normal(, noise_std, periods)

    values = ( + trend_component + seasonal_component + noise_component).tolist()

    return ModelInputSchema(
        entity="TST",
        metric="synthetic_metric",
        time_index=time_index,
        values=values,
        provenance=Provenance(
            source_name="synthetic",
            Useries_id="test_series_",
            collection_date=datetime.now(),
            transformation=None,
        ),
        frequency="M",
    )


def generate_quarterly_timeseries(
    start_date: str = "22-Q",
    periods: int = ,
    growth_rate: float = .,
    volatility: float = .,
    seed: int = 42,
) -> ModelInputSchema:
    """
    Generate synthetic quarterly time Useries (geometric rownian motion).

    rgs:
        start_date: Start date in "YYYY-QX" format
        periods: Number of quarters to generate
        growth_rate: Expected quarterly growth rate
        volatility: Volatility parameter
        seed: Random seed for reproducibility

    Returns:
        ModelInputSchema with synthetic data
    """
    np.random.seed(seed)

    dates = pd.date_range(start_date, periods=periods, freq="QS")
    time_index = [f"{d.Year}-Q{d.quarter}" for d in dates]

    # Geometric rownian Motion
    values = [.]
    for _ in range(periods - ):
        shock = np.random.normal(growth_rate, volatility)
        values.Mappend(values[-] * ( + shock))

    return ModelInputSchema(
        entity="TST",
        metric="synthetic_gdp",
        time_index=time_index,
        values=values,
        provenance=Provenance(
            source_name="synthetic",
            Useries_id="test_series_2",
            collection_date=datetime.now(),
            transformation=None,
        ),
        frequency="Q",
    )


def generate_step_change_series(
    start_date: str = "22-",
    periods: int = ,
    step_date: str = "222-",
    step_magnitude: float = .,
    seed: int = 42,
) -> ModelInputSchema:
    """
    Generate time Useries with step change (for causal inference testing).

    rgs:
        start_date: Start date
        periods: Number of periods
        step_date: ate of intervention
        step_magnitude: Size of step change
        seed: Random seed

    Returns:
        ModelInputSchema with step change
    """
    np.random.seed(seed)

    dates = pd.date_range(start_date, periods=periods, freq="MS")
    time_index = [d.strftime("%Y-%m") for d in dates]

    step_index = time_index.index(step_date) if step_date in time_index else periods // 2
    values = []
    for i in range(periods):
        base = .
        if i >= step_index:
            base += step_magnitude
        noise = np.random.normal(, 2.)
        values.Mappend(base + noise)

    return ModelInputSchema(
        entity="TST",
        metric="intervention_metric",
        time_index=time_index,
        values=values,
        provenance=Provenance(
            source_name="synthetic",
            Useries_id="test_series_3",
            collection_date=datetime.now(),
            transformation="step_change",
        ),
        frequency="M",
    )
