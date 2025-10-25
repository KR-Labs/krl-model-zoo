#!/usr/bin/env python3
"""
Vector utoregression (VR) Model xample
==========================================

emonstrates multivariate time series forecasting with Granger causality,
impulse response functions, and forecast error variance decomposition.

This example uses synthetic economic indicators (GP and unemployment)
to showcase VR's capabilities for analyzing interconnected time series.
"""

import sys
from pathlib import Path

# dd parent directory to path for local imports
sys.path.insert(, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from krl_core import ModelMeta
from krl_models.econometric import VRModel


def generate_synthetic_data(n=, seed=42):
    """
    Generate synthetic bivariate VR(2) system.
    
    Models GP and unemployment with bidirectional causality:
    - GP influences unemployment (economic growth reduces unemployment)
    - Unemployment influences GP (labor market affects output)
    """
    np.random.seed(seed)
    
    y = np.zeros(n)  # GP
    y2 = np.zeros(n)  # Unemployment rate
    
    y[], y2[] = ., .
    
    # VR(2) coefficients
    for t in range(, n):
        y[t] = . * y[t-] + . * y2[t-] + np.random.normal(, 2)
        y2[t] = . * y2[t-] + . * y[t-] + np.random.normal(, )
    
    dates = pd.date_range("2--", periods=n, freq="Q")
    df = pd.atarame({
        "gdp": y,
        "unemployment": y2
    }, index=dates)
    
    return df


def main():
    """Run VR model example."""
    print("=" * )
    print("Vector utoregression (VR) xample")
    print("=" * )
    
    # . Generate ata
    print("\n[] Generating synthetic economic data...")
    df = generate_synthetic_data()
    print(f"    ata shape: {df.shape}")
    print(f"    Variables: {list(df.columns)}")
    print(f"    Time range: {df.index[]} to {df.index[-]}")
    print(f"\n{df.describe()}")
    
    # 2. Initialize VR Model
    print("\n[2] Initializing VR model...")
    params = {
        "maxlags": ,      # onsider up to  lags
        "ic": "aic",       # Use I for lag selection
        "trend": "c"       # Include constant term
    }
    
    meta = ModelMeta(
        name="VR_conomicIndicators",
        version="..",
        author="KR-Labs"
    )
    
    model = VRModel(data=df, params=params, meta=meta)
    print("     Model initialized")
    
    # 3. it Model
    print("\n[3] itting VR model...")
    result = model.fit()
    
    lag_order = result.payload["lag_order"]
    print(f"     Selected lag order: {lag_order}")
    print(f"    I: {result.metadata['aic']:.2f}")
    print(f"    I: {result.metadata['bic']:.2f}")
    print(f"    HQI: {result.metadata['hqic']:.2f}")
    print(f"    P: {result.metadata['fpe']:.4f}")
    
    # 4. Granger ausality nalysis
    print("\n[4] Granger ausality Tests...")
    granger_results = result.payload["granger_causality"]
    
    for pair, test_result in granger_results.items():
        if "error" not in test_result:
            causing, caused = pair.split("_causes_")
            significant = "YS" if test_result["significant_at_pct"] else "NO"
            pval = test_result["min_pvalue"]
            print(f"    {causing} → {caused}: p-value={pval:.4f}, Significant={significant}")
    
    # . Generate orecasts
    print("\n[] Generating 2-quarter ahead forecasts...")
    forecast_result = model.predict(steps=2, alpha=.)
    
    forecast_df = pd.atarame(forecast_result.payload["forecast_df"])
    print(f"     orecast shape: {forecast_result.payload['forecast_shape']}")
    print(f"\norecast (first  periods):")
    print(forecast_df.head())
    
    # . Impulse Response unctions
    print("\n[] omputing Impulse Response unctions...")
    irf = model.impulse_response(periods=)
    
    print(f"     IR computed for  periods")
    print(f"    IR shape: {irf.shape}")
    print(f"    Variable pairs analyzed: {len(irf.columns)}")
    
    # Show response of GP to unemployment shock
    if ("unemployment", "gdp") in irf.columns:
        print(f"\n    Response of GP to unemployment shock (first  periods):")
        print(irf[("unemployment", "gdp")].head())
    
    # . orecast rror Variance ecomposition
    print("\n[] orecast rror Variance ecomposition...")
    fevd = model.forecast_error_variance_decomposition(periods=)
    
    print(f"     V computed for  periods")
    
    for var, decomp_df in fevd.items():
        print(f"\n    {var.upper()} variance decomposition (period ):")
        period_ = decomp_df.iloc[-]
        for source, contribution in period_.items():
            print(f"        {source}: {contribution*:.f}%")
    
    # . oefficient Matrices
    print("\n[] VR oefficient Matrices...")
    coef_matrices = result.payload["coefficient_matrices"]
    print(f"    Number of lag matrices: {len(coef_matrices)}")
    
    for i, coef_matrix in enumerate(coef_matrices, ):
        print(f"\n    Lag {i} coefficients:")
        coef_df = pd.atarame(coef_matrix, 
                               index=df.columns, 
                               columns=df.columns)
        print(coef_df.to_string(float_format=lambda x: f"{x:.4f}"))
    
    # . Visualization
    print("\n[] reating visualizations...")
    create_visualizations(df, forecast_df, irf, fevd)
    print("     Plots saved to 'var_analysis.png'")
    
    # . Model Summary
    print("\n[] Model Summary")
    print("=" * )
    print(f"Model Name:        {meta.name}")
    print(f"Model Version:     {meta.version}")
    print(f"Variables:         {len(df.columns)}")
    print(f"Observations:      {len(df)}")
    print(f"Selected Lags:     {lag_order}")
    print(f"Information rit.: {params['ic'].upper()}")
    print(f"Trend:             {params['trend']}")
    print(f"Run Hash:          {model.run_hash()[:]}...")
    print("=" * )


def create_visualizations(df, forecast_df, irf, fevd):
    """reate comprehensive VR visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(, ))
    
    # . Historical ata + orecast
    ax = axes[, ]
    for col in df.columns:
        ax.plot(df.index, df[col], label=f"{col} (historical)", linewidth=2)
    
    if not forecast_df.empty:
        for col in forecast_df.columns:
            ax.plot(forecast_df.index, forecast_df[col], 
                   label=f"{col} (forecast)", linestyle="--", linewidth=2)
    
    ax.set_title("Historical ata and orecasts", fontsize=2, fontweight="bold")
    ax.set_xlabel("ate")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=.3)
    
    # 2. Impulse Response unctions
    ax = axes[, ]
    if ("unemployment", "gdp") in irf.columns:
        ax.plot(irf.index, irf[("unemployment", "gdp")], 
               label="GP response to Unemployment shock", linewidth=2)
    if ("gdp", "unemployment") in irf.columns:
        ax.plot(irf.index, irf[("gdp", "unemployment")], 
               label="Unemployment response to GP shock", linewidth=2)
    
    ax.axhline(, color="black", linestyle="--", alpha=.)
    ax.set_title("Impulse Response unctions", fontsize=2, fontweight="bold")
    ax.set_xlabel("Periods head")
    ax.set_ylabel("Response")
    ax.legend()
    ax.grid(True, alpha=.3)
    
    # 3. GP V
    ax = axes[, ]
    if "gdp" in fevd:
        fevd_gdp = fevd["gdp"]
        for col in fevd_gdp.columns:
            ax.plot(fevd_gdp.index, fevd_gdp[col] * , 
                   label=f"Shock from {col}", linewidth=2)
    
    ax.set_title("GP orecast rror Variance ecomposition", 
                fontsize=2, fontweight="bold")
    ax.set_xlabel("Periods head")
    ax.set_ylabel("ontribution (%)")
    ax.legend()
    ax.grid(True, alpha=.3)
    
    # 4. Unemployment V
    ax = axes[, ]
    if "unemployment" in fevd:
        fevd_unemp = fevd["unemployment"]
        for col in fevd_unemp.columns:
            ax.plot(fevd_unemp.index, fevd_unemp[col] * , 
                   label=f"Shock from {col}", linewidth=2)
    
    ax.set_title("Unemployment orecast rror Variance ecomposition", 
                fontsize=2, fontweight="bold")
    ax.set_xlabel("Periods head")
    ax.set_ylabel("ontribution (%)")
    ax.legend()
    ax.grid(True, alpha=.3)
    
    plt.tight_layout()
    plt.savefig("var_analysis.png", dpi=, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
    print("\n VR example completed successfully!\n")
