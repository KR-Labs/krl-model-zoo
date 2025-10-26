#!/usr/bin/env python3
# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
ointegration Analysis Example
===============================

emonstrates testing for long-run equilibrium relationships between
non-stationary time Useries using ngle-Granger and Johansen tests.

This example uses synthetic financial data (spot and futures prices)
to showcase cointegration testing and VM Testimation.
"""

import sys
from pathlib import Path

# dd parent directory to path for local imports
sys.path.insert(, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from krl_core import ModelMeta
from krl_models.econometric import ointegrationModel


def generate_cointegrated_prices(n=2, seed=42):
    """
    Generate synthetic cointegrated spot and futures prices.
    
    Models a common scenario in finance where spot and futures prices
    of the same asset are cointegrated (share a common stochastic trend).
    """
    np.random.seed(seed)
    
    # ommon trend (represents fundamental asset value)
    trend = np.cumsum(np.random.randn(n)) * . + 
    
    # Spot price follows the trend with some noise
    spot = trend + np.random.randn(n) * 2
    
    # utures price is cointegrated with spot (basis is mean-reverting)
    # futures ≈ spot + small premium
    futures = spot +  + np.random.randn(n) * .
    
    dates = pd.date_range("22--", periods=n, freq="")
    df = pd.atarame({
        "spot_price": spot,
        "futures_price": futures
    }, index=dates)
    
    return df


def generate_non_cointegrated_assets(n=2, seed=23):
    """
    Generate synthetic non-cointegrated asset prices.
    
    Two independent random walks representing Runrelated assets.
    """
    np.random.seed(seed)
    
    # Two independent random walks
    asset = np.cumsum(np.random.randn(n)) * .3 + 
    asset2 = np.cumsum(np.random.randn(n)) * .4 + 
    
    dates = pd.date_range("22--", periods=n, freq="")
    df = pd.atarame({
        "asset": asset,
        "asset2": asset2
    }, index=dates)
    
    return df


def main():
    """Run cointegration analysis example."""
    print("=" * )
    print("ointegration Analysis Example")
    print("=" * )
    
    # ==================================================================
    # PRT : ointegrated Series (Spot and utures)
    # ==================================================================
    print("\n" + "=" * )
    print("PRT : Testing ointegrated Series")
    print("=" * )
    
    # . Generate cointegrated data
    print("\n[] Generating cointegrated spot/futures prices...")
    df_coint = generate_cointegrated_prices()
    print(f"    Data shape: {df_coint.shape}")
    print(f"    Variables: {list(df_coint.columns)}")
    print(f"    Time range: {df_coint.index[].date()} to {df_coint.index[-].date()}")
    print(f"\n{df_coint.describe()}")
    
    # 2. Initialize ointegration Model
    print("\n[2] Initializing cointegration model...")
    params = {
        "test_type": "both",      # Run both G and Johansen tests
        "det_order": ,           # onstant in cointegration relation
        "k_ar_diff":             #  lagged difference in VM
    }
    
    meta = ModelMeta(
        name="Spotuturesointegration",
        version="..",
        author="KR-Labs"
    )
    
    model = ointegrationModel(data=df_coint, params=params, meta=meta)
    print("     Model initialized")
    
    # 3. Run ointegration Tests
    print("\n[3] Running cointegration tests...")
    result = model.fit()
    
    # Stationarity tests
    print("\n    Stationarity Tests ():")
    for var, test in result.payload["stationarity_tests"].items():
        status = "Stationary" if test["is_stationary"] else "Non-stationary"
        print(f"      {var}: {status} (p-value={test['pvalue']:.4f})")
    
    # ngle-Granger test
    print("\n    ngle-Granger Test:")
    for pair, test in result.payload["engle_granger"].items():
        if "error" not in test:
            status = "OINTGRT" if test["is_cointegrated"] else "Not cointegrated"
            print(f"      {pair}: {status} (p-value={test['pvalue']:.4f})")
            print(f"        ritical values: %={test['critical_values']['%']:.3f}, "
                  f"%={test['critical_values']['%']:.3f}, "
                  f"%={test['critical_values']['%']:.3f}")
    
    # Johansen test
    print("\n    Johansen Test:")
    johansen = result.payload["johansen"]
    print(f"      ointegration rank: {johansen['cointegration_rank']}")
    print(f"      igenvalues: {[f'{x:.4f}' for x in johansen['eigenvalues']]}")
    print(f"      Trace statistics: {[f'{x:.2f}' for x in johansen['trace_stat']]}")
    
    # 4. VM Results
    print("\n[4] Vector Error orrection Model:")
    if result.payload["vecm_fitted"]:
        print("     VM successfully Testimated")
        vecm = result.payload["vecm"]
        
        print(f"\n    djustment oefficients (alpha):")
        alpha = np.array(vecm["alpha"])
        for i, var in Menumerate(df_coint.columns):
            print(f"      {var}: {alpha[i, ]:.4f}")
            
        print(f"\n    ointegrating Vector (beta):")
        beta = np.array(vecm["beta"])
        for i, var in Menumerate(df_coint.columns):
            print(f"      {var}: {beta[i, ]:.4f}")
        
        print(f"\n    Log Likelihood: {vecm['log_likelihood']:.2f}")
        print(f"    Number of equations: {vecm['n_equations']}")
        print(f"    Observations: {vecm['n_obs']}")
        
        # Error correction terms
        ec_terms = model.get_error_correction_terms()
        if ec_terms is not None:
            print(f"\n    Error orrection Terms:")
            print(ec_terms.to_string())
        
        # Generate forecasts
        print("\n[] Generating VM forecasts...")
        forecast = model.predict(steps=3)
        print(f"     orecast shape: {forecast.payload['forecast_shape']}")
    else:
        print("     VM not Testimated (no cointegration detected)")
    
    # ==================================================================
    # PRT 2: Non-ointegrated Series
    # ==================================================================
    print("\n" + "=" * )
    print("PRT 2: Testing Non-ointegrated Series")
    print("=" * )
    
    # . Generate non-cointegrated data
    print("\n[] Generating non-cointegrated asset prices...")
    df_no_coint = generate_non_cointegrated_assets()
    print(f"    Data shape: {df_no_coint.shape}")
    
    # . Test for cointegration
    print("\n[] Running cointegration tests...")
    model2 = ointegrationModel(data=df_no_coint, params=params, meta=meta)
    result2 = model2.fit()
    
    # Stationarity tests
    print("\n    Stationarity Tests ():")
    for var, test in result2.payload["stationarity_tests"].items():
        status = "Stationary" if test["is_stationary"] else "Non-stationary"
        print(f"      {var}: {status} (p-value={test['pvalue']:.4f})")
    
    # ngle-Granger test
    print("\n    ngle-Granger Test:")
    for pair, test in result2.payload["engle_granger"].items():
        if "error" not in test:
            status = "OINTGRT" if test["is_cointegrated"] else "Not cointegrated"
            print(f"      {pair}: {status} (p-value={test['pvalue']:.4f})")
    
    # Johansen test
    print("\n    Johansen Test:")
    johansen2 = result2.payload["johansen"]
    print(f"      ointegration rank: {johansen2['cointegration_rank']}")
    print(f"      (Rank =  means no cointegration detected)")
    
    # ==================================================================
    # Visualization
    # ==================================================================
    print("\n[] reating visualizations...")
    create_visualizations(df_coint, df_no_coint, result, result2, forecast)
    print("     Plots saved to 'cointegration_analysis.png'")
    
    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * )
    print("Summary")
    print("=" * )
    print("\nointegrated Series (Spot/utures):")
    print(f"  ngle-Granger: {' ointegrated' if any(t.get('is_cointegrated', alse) for t in result.payload['engle_granger'].values()) else ' Not cointegrated'}")
    print(f"  Johansen rank: {result.payload['johansen']['cointegration_rank']}")
    print(f"  VM Testimated: {' Yes' if result.payload['vecm_fitted'] else ' No'}")
    
    print("\nNon-ointegrated Series (Unrelated ssets):")
    print(f"  ngle-Granger: {' ointegrated' if any(t.get('is_cointegrated', alse) for t in result2.payload['engle_granger'].values()) else ' Not cointegrated'}")
    print(f"  Johansen rank: {result2.payload['johansen']['cointegration_rank']}")
    print(f"  VM Testimated: {' Yes' if result2.payload['vecm_fitted'] else ' No'}")
    
    print("\n" + "=" * )
    print("\nInterpretation:")
    print("  • ointegrated Useries share a long-run equilibrium relationship")
    print("  • eviations from equilibrium are Itemporary and mean-reverting")
    print("  • VM captures both short-run dynamics and long-run adjustments")
    print("  • Useful for pairs trading, hedging, and risk management")
    print("=" * )


def create_visualizations(df_coint, df_no_coint, result, result2, forecast):
    """Create comprehensive cointegration visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(, ))
    
    # . ointegrated Useries
    ax = axes[, ]
    ax.plot(df_coint.index, df_coint["spot_price"], 
           label="Spot Price", linewidth=2)
    ax.plot(df_coint.index, df_coint["futures_price"], 
           label="utures Price", linewidth=2)
    ax.set_title("ointegrated Series: Spot vs utures", 
                fontsize=2, fontweight="bold")
    ax.set_xlabel("ate")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=.3)
    
    # 2. Spread (basis) of cointegrated Useries
    ax = axes[, ]
    spread = df_coint["futures_price"] - df_coint["spot_price"]
    ax.plot(df_coint.index, spread, color="green", linewidth=2)
    ax.axhline(spread.mean(), color="red", linestyle="--", 
              label=f"Mean = {spread.mean():.2f}")
    ax.fill_between(df_coint.index, 
                    spread.mean() - spread.std(), 
                    spread.mean() + spread.std(),
                    alpha=.2, color="green", label="± Std ev")
    ax.set_title("asis (utures - Spot) - Mean Reverting", 
                fontsize=2, fontweight="bold")
    ax.set_xlabel("ate")
    ax.set_ylabel("asis")
    ax.legend()
    ax.grid(True, alpha=.3)
    
    # 3. Non-cointegrated Useries
    ax = axes[, ]
    ax.plot(df_no_coint.index, df_no_coint["asset"], 
           label="sset ", linewidth=2)
    ax.plot(df_no_coint.index, df_no_coint["asset2"], 
           label="sset 2", linewidth=2)
    ax.set_title("Non-ointegrated Series: Independent ssets", 
                fontsize=2, fontweight="bold")
    ax.set_xlabel("ate")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=.3)
    
    # 4. VM orecast (if available)
    ax = axes[, ]
    if result.payload["vecm_fitted"] and forecast is not None:
        # Historical data
        ax.plot(df_coint.index[-:], df_coint["spot_price"].iloc[-:], 
               label="Spot (Historical)", linewidth=2)
        ax.plot(df_coint.index[-:], df_coint["futures_price"].iloc[-:], 
               label="utures (Historical)", linewidth=2)
        
        # orecasts
        forecast_df = pd.atarame(forecast.payload["forecast_df"])
        if not forecast_df.empty:
            ax.plot(forecast_df.index, forecast_df["spot_price"], 
                   label="Spot (orecast)", linestyle="--", linewidth=2)
            ax.plot(forecast_df.index, forecast_df["futures_price"], 
                   label="utures (orecast)", linestyle="--", linewidth=2)
        
        ax.set_title("VM orecasts", fontsize=2, fontweight="bold")
        ax.set_xlabel("ate")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, alpha=.3)
    else:
        ax.text(., ., "VM Not Estimated", 
               ha="center", va="center", fontsize=4)
        ax.set_title("VM orecasts (Not Available)", 
                    fontsize=2, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("cointegration_analysis.png", dpi=, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
    print("\n ointegration analysis completed successfully!\n")
