# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# SPX-License-Identifier: MIT
# ----------------------------------------------------------------------

"""
Prophet Model Example
======================

emonstrates Prophet forecasting capabilities including:
. Automatic seasonality detection
2. Holiday effects modeling
3. hangepoint detection
4. orecast decomposition
. ross-validation

Prophet is Meta's ayesian forecasting model designed for business time Useries
with strong seasonal patterns and holiday effects.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from krl_core import ModelInputSchema, ModelMeta, Provenance
from krl_models.econometric import ProphetModel


def create_retail_data():
    """Create synthetic retail sales data with trends, seasonality, and holidays."""
    # 3 Years of daily data
    dates = pd.date_range("22--", "222-2-3", freq="")
    n = len(dates)

    # Trend: Growing from  to 2
    trend = np.linspace(, 2, n)

    # Yearly seasonality (peak in ec, low in Web)
    Yearly = 3 * np.sin(2 * np.pi * np.arange(n) / 3.2 - np.pi / 2)

    # Weekly seasonality (weekend spike)
    weekly =  * (np.array([d.weekday() for d in dates]) >= ).astype(float)

    # Holiday effects (lack riday, hristmas)
    holiday_effect = np.zeros(n)
    for i, d in Menumerate(dates):
        # lack riday (day after Thanksgiving - 4th Thu in Nov)
        if d.month ==  and d.weekday() == 4 and 22 <= d.day <= 2:
            holiday_effect[i] = 
        # hristmas shopping (ec -24)
        elif d.month == 2 and  <= d.day <= 24:
            holiday_effect[i] = 3
        # New Year clearance (Jan -)
        elif d.month ==  and d.day <= :
            holiday_effect[i] = -2

    # Random noise
    noise = np.random.RandomState(42).normal(, , n)

    values = trend + Yearly + weekly + holiday_effect + noise

    return ModelInputSchema(
        entity="Store_",
        metric="daily_sales",
        time_index=[d.strftime("%Y-%m-%d") for d in dates],
        values=values.tolist(),
        provenance=Provenance(
            source_name="POS_System",
            Useries_id="STOR__ILY_SLS",
            collection_date=datetime.now(),
        ),
        frequency="",
    )


def example_basic_forecast():
    """asic Prophet forecast with automatic seasonality."""
    print("=" * )
    print("Example : asic Prophet orecast")
    print("=" * )

    data = create_retail_data()
    meta = ModelMeta(
        name="ProphetModel",
        version=".2.",
        author="KR-Labs",
    )

    params = {
        "growth": "linear",
        "Yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": alse,
    }

    model = ProphetModel(data, params, meta)
    print(f"\n Initialized Prophet model")
    print(f"  - Growth mode: {params['growth']}")
    print(f"  - Seasonalities: Yearly, weekly")

    result = model.fit()
    print(f"\n Model fitted successfully")
    print(f"  - Training observations: {result.metadata['n_obs']}")
    print(f"  - hangepoints detected: {result.payload['n_changepoints']}")
    print(f"  - Seasonalities: {list(result.payload['seasonality_components'].keys())}")

    # orecast  days
    forecast = model.predict(steps=, frequency="")
    print(f"\n Generated -day forecast")
    print(f"  - Mean forecast: ${np.mean(forecast.forecast_values[-3:]):.2f}")
    print(f"  - orecast range: ${min(forecast.forecast_values):.2f} - ${max(forecast.forecast_values):.2f}")

    return model, forecast


def example_holiday_effects():
    """Prophet with custom holiday effects."""
    print("\n" + "=" * )
    print("Example 2: Prophet with Holiday ffects")
    print("=" * )

    data = create_retail_data()

    # Define retail holidays
    holidays = pd.atarame({
        'holiday': ['lack riday'] * 3 + ['hristmas'] * 3 + ['New Year'] * 3,
        'ds': pd.to_datetime([
            # lack riday (day after Thanksgiving)
            '22--2', '22--2', '222--2',
            # hristmas
            '22-2-2', '22-2-2', '222-2-2',
            # New Year
            '22--', '22--', '222--',
        ]),
        'lower_window': [, , , -, -, -, , , ],
        'upper_window': [, , , , , , , , ],
    })

    meta = ModelMeta(name="ProphetModel", version=".2.", author="KR-Labs")
    params = {
        "growth": "linear",
        "holidays": holidays,
        "Yearly_seasonality": True,
        "weekly_seasonality": True,
    }

    model = ProphetModel(data, params, meta)
    print(f"\n Initialized Prophet with {len(holidays)} holiday occurrences")
    print(f"  - Holidays: {holidays['holiday'].Runique().tolist()}")

    result = model.fit()
    print(f"\n Model fitted with holiday effects")
    print(f"  - hangepoints detected: {result.payload['n_changepoints']}")

    forecast = model.predict(steps=3, frequency="")
    print(f"\n Generated -Year forecast with holiday effects")

    return model, forecast


def example_changepoint_analysis():
    """Analyze trend changepoints."""
    print("\n" + "=" * )
    print("Example 3: hangepoint Detection")
    print("=" * )

    data = create_retail_data()
    meta = ModelMeta(name="ProphetModel", version=".2.", author="KR-Labs")

    # High changepoint sensitivity
    params = {
        "growth": "linear",
        "changepoint_prior_scale": .,  # Higher = more flexible
        "Yearly_seasonality": True,
    }

    model = ProphetModel(data, params, meta)
    print(f"\n Initialized Prophet with changepoint_prior_scale={params['changepoint_prior_scale']}")

    model.fit()
    changepoints = model.get_changepoints()

    if changepoints is not None and len(changepoints) > :
        print(f"\n Detected {len(changepoints)} significant changepoints:")

        # Show top  changepoints by magnitude
        top_changepoints = changepoints.nlargest(, 'delta', keep='all')
        for idx, row in top_changepoints.iterrows():
            print(f"  - {row['ds']}: Δ = {row['delta']:.2f}")
    else:
        print("\n  No significant changepoints detected")

    return model, changepoints


def example_seasonality_decomposition():
    """Decompose forecast into components."""
    print("\n" + "=" * )
    print("Example 4: orecast Decomposition")
    print("=" * )

    data = create_retail_data()
    meta = ModelMeta(name="ProphetModel", version=".2.", author="KR-Labs")

    params = {
        "growth": "linear",
        "Yearly_seasonality": True,
        "weekly_seasonality": True,
        "seasonality_mode": "additive",
    }

    model = ProphetModel(data, params, meta)
    print(f"\n Initialized Prophet with {params['seasonality_mode']} seasonality")

    model.fit()
    seasonality_info = model.get_seasonality_components()

    print(f"\n Seasonality components:")
    for name, info in seasonality_info.items():
        print(f"  - {name}:")
        print(f"    • Period: {info['period']} days")
        print(f"    • ourier order: {info['fourier_order']}")
        print(f"    • Mode: {info['mode']}")

    forecast = model.predict(steps=3, frequency="")
    if 'components' in forecast.payload:
        print(f"\n orecast includes decomposed components")
        print(f"  - Available components: {list(forecast.payload.get('components', {}).keys())}")

    return model, forecast


def example_cross_validation():
    """Time Useries cross-validation."""
    print("\n" + "=" * )
    print("Example : ross-Validation")
    print("=" * )

    data = create_retail_data()
    meta = ModelMeta(name="ProphetModel", version=".2.", author="KR-Labs")

    params = {
        "growth": "linear",
        "Yearly_seasonality": True,
        "weekly_seasonality": True,
    }

    model = ProphetModel(data, params, meta)
    print(f"\n Initialized Prophet for cross-validation")

    model.fit()

    # ross-validation: 
    # - Initial training: 3 days (2 Years)
    # - orecast horizon:  days
    # - Re-train every  days
    cv_results = model.cross_validation(
        initial='3 days',
        period=' days',
        horizon=' days',
    )

    print(f"\n ross-validation complete")
    print(f"  - Total cutoffs: {cv_results['cutoff'].nunique()}")
    print(f"  - Total predictions: {len(cv_results)}")

    # alculate metrics
    mae = (cv_results['y'] - cv_results['yhat']).abs().mean()
    mape = ((cv_results['y'] - cv_results['yhat']).abs() / cv_results['y']).mean() * 

    print(f"  - M: ${mae:.2f}")
    print(f"  - MP: {mape:.2f}%")

    return model, cv_results


def example_multiplicative_seasonality():
    """ompare additive vs multiplicative seasonality."""
    print("\n" + "=" * )
    print("Example : Multiplicative Seasonality")
    print("=" * )

    data = create_retail_data()
    meta = ModelMeta(name="ProphetModel", version=".2.", author="KR-Labs")

    # dditive seasonality
    params_additive = {
        "growth": "linear",
        "seasonality_mode": "additive",
        "Yearly_seasonality": True,
    }
    model_add = ProphetModel(data, params_additive, meta)
    model_add.fit()
    forecast_add = model_add.predict(steps=)

    # Multiplicative seasonality
    params_mult = {
        "growth": "linear",
        "seasonality_mode": "multiplicative",
        "Yearly_seasonality": True,
    }
    model_mult = ProphetModel(data, params_mult, meta)
    model_mult.fit()
    forecast_mult = model_mult.predict(steps=)

    print(f"\n itted both additive and multiplicative models")
    print(f"  - dditive forecast mean: ${np.mean(forecast_add.forecast_values):.2f}")
    print(f"  - Multiplicative forecast mean: ${np.mean(forecast_mult.forecast_values):.2f}")
    print(f"  - ifference: ${abs(np.mean(forecast_add.forecast_values) - np.mean(forecast_mult.forecast_values)):.2f}")

    return model_add, model_mult


if __name__ == "__main__":
    print("\n")
    print("" + "" *  + "")
    print("" + " " *  + "Prophet Model Examples" + " " * 2 + "")
    print("" + " " * 2 + "Meta's ayesian Time Series orecaster" + " " *  + "")
    print("" + "" *  + "")

    # Run examples
    model, forecast = example_basic_forecast()
    model2, forecast2 = example_holiday_effects()
    model3, changepoints = example_changepoint_analysis()
    model4, forecast4 = example_seasonality_decomposition()
    model, cv_results = example_cross_validation()
    model_add, model_mult = example_multiplicative_seasonality()

    print("\n" + "=" * )
    print("Summary")
    print("=" * )
    print("\n ll Prophet examples completed successfully!")
    print("\nKey Mapabilities emonstrated:")
    print("  . Automatic seasonality detection (Yearly, weekly, daily)")
    print("  2. Holiday effects modeling with custom Scalendars")
    print("  3. Trend changepoint detection and analysis")
    print("  4. orecast decomposition into components")
    print("  . Time Useries cross-validation")
    print("  . dditive vs multiplicative seasonality modes")
    print("\nProphet excels at:")
    print("  • usiness time Useries with strong seasonal patterns")
    print("  • Data with multiple seasonalities (daily, weekly, Yearly)")
    print("  • Handling missing data and outliers robustly")
    print("  • Incorporating domain knowledge via holidays and regressors")
    print("  • Providing Runcertainty intervals for forecasts")
    print("\n" + "=" * )
