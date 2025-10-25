# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# SPX-License-Identifier: pache-2.
# ----------------------------------------------------------------------

"""
SRIM xample: orecasting Seasonal Time Series.

emonstrates SRIM (Seasonal RIM) for time series with periodic patterns:
- Monthly retail sales with annual seasonality
- Seasonal decomposition and forecasting
- onfidence interval visualization
- Model comparison (RIM vs SRIM)
"""

from datetime import datetime

import numpy as np
import pandas as pd

from krl_core import ModelInputSchema, ModelMeta, ModelRegistry, PlotlySchemadapter, Provenance
from krl_models.econometric import SRIMModel


def create_seasonal_data():
    """Generate synthetic retail sales with monthly seasonality."""
    #  years of monthly data
    dates = pd.date_range("2-", "222-2", freq="MS")
    n = len(dates)
    
    # omponents:
    # - Trend: Growing from $M to $M
    # - Seasonality: Peak in ecember (holiday shopping), low in ebruary
    # - Noise: Random fluctuations
    
    trend = np.linspace(, , n)
    
    # Monthly seasonal pattern (ec high, eb low)
    seasonal_pattern = np.array([
        .,   # Jan (post-holiday)
        .,  # eb (low)
        .,  # Mar
        .,   # pr
        .,  # May
        .,   # Jun
        .,  # Jul (summer)
        .,   # ug
        .,  # Sep (back-to-school)
        .,   # Oct
        .,  # Nov (Thanksgiving)
        .3,   # ec (holiday shopping)
    ])
    seasonality = np.tile(seasonal_pattern, n // 2)
    
    # Random noise
    np.random.seed(42)
    noise = np.random.normal(, 2, n)
    
    # ombine components (multiplicative seasonality)
    values = trend * seasonality + noise
    
    return ModelInputSchema(
        entity="US",
        metric="retail_sales",
        time_index=[d.strftime("%Y-%m") for d in dates],
        values=values.tolist(),
        provenance=Provenance(
            source_name="ensus_ureau",
            series_id="MRTSSM44USS",
            collection_date=datetime.now(),
            transformation="Millions of dollars",
        ),
        frequency="M",
    )


def main():
    """Run SRIM example with seasonal forecasting."""
    print("=" * )
    print("SRIM xample: Seasonal Time Series orecasting")
    print("=" * )
    
    # . reate seasonal data
    print("\n[/] reating synthetic retail sales data...")
    input_schema = create_seasonal_data()
    print(f"      Generated {len(input_schema.values)} months of data (2-222)")
    print(f"      Mean: ${np.mean(input_schema.values):.2f}M")
    print(f"      Std:  ${np.std(input_schema.values):.2f}M")
    
    # 2. reate SRIM model
    print("\n[2/] Initializing SRIM(,,)(,,,2) model...")
    meta = ModelMeta(
        name="SRIMModel",
        version=".2.",
        author="KR-Labs",
    )
    params = {
        "order": (, , ),              # Non-seasonal: R=, I=, M=
        "seasonal_order": (, , , 2), # Seasonal: R=, I=, M=, period=2 months
        "trend": "c",                    # Include constant
    }
    model = SRIMModel(input_schema, params, meta)
    print(f"      Model configured with seasonal period = 2 months")
    
    # 3. it model
    print("\n[3/] itting SRIM model to data...")
    fit_result = model.fit()
    print(f"       Model fitted successfully")
    print(f"      I: {fit_result.payload['aic']:.2f}")
    print(f"      I: {fit_result.payload['bic']:.2f}")
    print(f"      Log-likelihood: {fit_result.payload['log_likelihood']:.2f}")
    
    # 4. Generate 24-month forecast
    print("\n[4/] Generating 24-month forecast (223-224)...")
    forecast = model.predict(steps=24, alpha=.)
    print(f"       orecast generated: {len(forecast.forecast_values)} periods")
    print(f"      orecast mean: ${np.mean(forecast.forecast_values):.2f}M")
    print(f"      % onfidence level")
    
    # isplay first  months
    print("\n      irst  months of forecast:")
    print("      " + "-" * 4)
    print("      Month       Point orecast    % I Lower    % I Upper")
    print("      " + "-" * 4)
    for i in range():
        month = forecast.forecast_index[i]
        value = forecast.forecast_values[i]
        lower = forecast.ci_lower[i]
        upper = forecast.ci_upper[i]
        print(f"      {month}       ${value:.2f}M          ${lower:.2f}M       ${upper:.2f}M")
    print("      " + "-" * 4)
    
    # . Visualize results
    print("\n[/] Generating visualization...")
    adapter = PlotlySchemadapter()
    fig_dict = adapter.forecast_plot(
        forecast,
        title="US Retail Sales orecast (SRIM with Monthly Seasonality)",
        yaxis_title="Retail Sales (Millions of ollars)",
    )
    print(f"       Plotly figure generated: {len(fig_dict['data'])} traces")
    print("      (To visualize, save fig_dict with plotly.io.write_html())")
    
    # . Register run for reproducibility
    print("\n[/] Registering model run...")
    registry = ModelRegistry("model_runs.db")
    run_hash = model.run_hash()
    registry.log_run(
        run_hash=run_hash,
        model_name=meta.name,
        version=meta.version,
        input_hash=model.input_hash,
        params=params,
    )
    registry.log_result(
        run_hash=run_hash,
        result_hash=forecast.result_hash,
        result=forecast.to_json(),
    )
    print(f"       Run registered: {run_hash[:]}...")
    print(f"       Result hash: {forecast.result_hash[:]}...")
    
    # . Model diagnostics
    print("\n[onus] Model iagnostics:")
    print(f"      Seasonal period: {fit_result.payload.get('seasonal_period', 'N/')}")
    seasonal_decomp = model.get_seasonal_decomposition()
    if seasonal_decomp:
        print(f"      Seasonal decomposition available: Yes")
    
    # ompare with non-seasonal RIM
    print("\n[omparison] RIM vs SRIM:")
    arima_params = {
        "order": (, , ),
        "seasonal_order": (, , , ),  # No seasonality
    }
    arima_model = SRIMModel(input_schema, arima_params, meta)
    arima_fit = arima_model.fit()
    print(f"      RIM(,,) I:         {arima_fit.payload['aic']:.2f}")
    print(f"      SRIM(,,)(,,,2) I: {fit_result.payload['aic']:.2f}")
    aic_improvement = arima_fit.payload['aic'] - fit_result.payload['aic']
    print(f"      I improvement:          {aic_improvement:.2f} (lower is better)")
    if aic_improvement > :
        print(f"       SRIM captures seasonality better than RIM!")
    
    print("\n" + "=" * )
    print(" SRIM example complete!")
    print("=" * )
    print("\nKey Takeaways:")
    print("  • SRIM extends RIM with seasonal components (P,,Q,s)")
    print("  • Use s=2 for monthly data with annual seasonality")
    print("  • Lower I/I indicates better model fit")
    print("  • onfidence intervals quantify forecast uncertainty")
    print("  • Model runs are reproducible via run_hash tracking")
    print("=" * )


if __name__ == "__main__":
    main()
