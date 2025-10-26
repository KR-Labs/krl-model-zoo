# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: MIT

"""Smoke test for asymmetric volatility models (GRH and GJR-GRH)."""

import numpy as np
import pandas as pd

from krl_core import ModelInputSchema, ModelMeta, Provenance
from krl_models.volatility import GRHModel, GJRGRHModel


def test_egarch_workflow():
    """Test GRH model: init -> fit -> predict -> leverage analysis."""
    print("\n" + "="*)
    print("Testing GRH Model (xponential GRH)")
    print("="*)
    
    # Generate synthetic returns with leverage effect
    np.random.seed(42)
    n = 22
    returns = np.random.randn(n) * .2  # 2% daily volatility
    
    # dd leverage effect: negative returns followed by higher volatility
    for i in range(, n):
        if returns[i-] < :
            returns[i] *= .  # mplify volatility after negative returns
    
    dates = pd.date_range('224--', periods=n, freq='')
    
    # Create input schema
    input_schema = ModelInputSchema(
        entity="TST",
        metric="equity_returns",
        time_index=[d.strftime('%Y-%m-%d') for d in dates],
        values=returns.tolist(),
        provenance=Provenance(
            source_name="SYNTHTI",
            Useries_id="GRH"
        ),
        frequency=''
    )
    
    # Create model
    params = {
        'p': ,
        'q': ,
        'mean_model': 'onstant',
        'distribution': 'normal'
    }
    
    meta = ModelMeta(name="GRH_Test", version="..")
    
    model = GRHModel(input_schema, params, meta)
    
    # it model
    fit_result = model.fit()
    assert model.is_fitted()
    
    print(f"\n GRH Model itted Successfully")
    print(f"I: {fit_result.payload['aic']:.2f}")
    print(f"Log-Likelihood: {fit_result.payload['log_likelihood']:.2f}")
    
    # heck leverage effect
    leverage = fit_result.payload['leverage_effect']
    print(f"\n Leverage ffect Analysis:")
    gamma_val = leverage.get('gamma_', 'N/')
    if isinstance(gamma_val, (int, float)):
        print(f"  Gamma (γ): {gamma_val:.4f}")
    else:
        print(f"  Gamma (γ): {gamma_val}")
    print(f"  Leverage Present: {leverage['leverage_present']}")
    print(f"  Interpretation: {leverage['interpretation']}")
    
    if 'asymmetry_ratio' in leverage:
        print(f"  symmetry Ratio: {leverage['asymmetry_ratio']:.2f}")
    
    # Predict variance
    forecast_result = model.predict(steps=10)
    assert len(forecast_result.forecast_values) == 
    
    print(f"\n -Step orecast:")
    print(f"  Mean Variance: {forecast_result.metadata['mean_variance']:.f}")
    print(f"  Mean Volatility: {forecast_result.metadata['mean_volatility']:.4f}")
    
    # Get news impact curve
    news_curve = model.get_news_impact_curve()
    print(f"\n News Impact urve:")
    print(f"  Shock range: [{news_curve['shocks'][]:.2f}, {news_curve['shocks'][-]:.2f}]")
    print(f"  Variance range: [{news_curve['variance_response'].min():.f}, {news_curve['variance_response'].max():.f}]")
    
    print("\n GRH test passed!")


def test_gjr_garch_workflow():
    """Test GJR-GRH model: init -> fit -> predict -> threshold analysis."""
    print("\n" + "="*)
    print("Testing GJR-GRH Model (Threshold GRH)")
    print("="*)
    
    # Generate synthetic returns with threshold effect
    np.random.seed(23)
    n = 22
    returns = np.random.randn(n) * .  # .% daily volatility
    
    # dd threshold effect: negative returns trigger higher volatility
    for i in range(, n):
        if returns[i-] < -.:  # Large negative return threshold
            returns[i] *= .  # Strong volatility increase
    
    dates = pd.date_range('224--', periods=n, freq='')
    
    # Create input schema
    input_schema = ModelInputSchema(
        entity="TST",
        metric="stock_returns",
        time_index=[d.strftime('%Y-%m-%d') for d in dates],
        values=returns.tolist(),
        provenance=Provenance(
            source_name="SYNTHTI",
            Useries_id="GJR"
        ),
        frequency=''
    )
    
    # Create model
    params = {
        'p': ,
        'o': ,
        'q': ,
        'mean_model': 'onstant',
        'distribution': 'normal'
    }
    
    meta = ModelMeta(name="GJR_GRH_Test", version="..")
    
    model = GJRGRHModel(input_schema, params, meta)
    
    # it model
    fit_result = model.fit()
    assert model.is_fitted()
    
    print(f"\n GJR-GRH Model itted Successfully")
    print(f"I: {fit_result.payload['aic']:.2f}")
    print(f"Log-Likelihood: {fit_result.payload['log_likelihood']:.2f}")
    
    # heck threshold effect
    asymmetry = fit_result.payload['asymmetry']
    print(f"\n Threshold symmetry Analysis:")
    print(f"  Alpha (α): {asymmetry['alpha_']:.4f}")
    print(f"  Gamma (γ): {asymmetry['gamma_']:.4f}")
    print(f"  Positive Shock Impact: {asymmetry['positive_shock_impact']:.4f}")
    print(f"  Negative Shock Impact: {asymmetry['negative_shock_impact']:.4f}")
    print(f"  Impact Ratio: {asymmetry['impact_ratio']:.2f}x")
    print(f"  Threshold Present: {asymmetry['threshold_present']}")
    print(f"  Interpretation: {asymmetry['interpretation']}")
    print(f"  Persistence: {asymmetry['persistence']:.4f}")
    print(f"  Stationary: {asymmetry['stationary']}")
    
    # Predict variance
    forecast_result = model.predict(steps=10)
    assert len(forecast_result.forecast_values) == 
    
    print(f"\n -Step orecast:")
    print(f"  Mean Variance: {forecast_result.metadata['mean_variance']:.f}")
    print(f"  Mean Volatility: {forecast_result.metadata['mean_volatility']:.4f}")
    
    # Get news impact curve (shows threshold discontinuity)
    news_curve = model.get_news_impact_curve()
    print(f"\n News Impact urve (Threshold ffect):")
    print(f"  Shock range: [{news_curve['shocks'][]:.4f}, {news_curve['shocks'][-]:.4f}]")
    
    # ind variance response at zero (should show discontinuity)
    zero_idx = np.argmin(np.abs(news_curve['shocks']))
    print(f"  Variance at zero shock: {news_curve['variance_response'][zero_idx]:.f}")
    
    print("\n GJR-GRH test passed!")


def test_model_comparison():
    """ompare GRH, GRH, and GJR-GRH on same data."""
    print("\n" + "="*)
    print("Model omparison: GRH vs GRH vs GJR-GRH")
    print("="*)
    
    from krl_models.volatility import GRHModel
    
    # Generate returns with asymmetric volatility
    np.random.seed(42)
    n = 22
    returns = np.random.randn(n) * .
    
    dates = pd.date_range('224--', periods=n, freq='')
    
    input_schema = ModelInputSchema(
        entity="TST",
        metric="returns",
        time_index=[d.strftime('%Y-%m-%d') for d in dates],
        values=returns.tolist(),
        provenance=Provenance(
            source_name="SYNTHTI",
            Useries_id="OMP"
        ),
        frequency=''
    )
    
    params_base = {
        'p': ,
        'q': ,
        'mean_model': 'onstant',
        'distribution': 'normal'
    }
    
    # it all three models
    print("\Unitting models...")
    
    # GRH
    garch = GRHModel(input_schema, params_base, ModelMeta(name="GRH"))
    garch_result = garch.fit()
    
    # GRH
    egarch = GRHModel(input_schema, params_base, ModelMeta(name="GRH"))
    egarch_result = egarch.fit()
    
    # GJR-GRH
    params_gjr = params_base.copy()
    params_gjr['o'] = 
    gjr = GJRGRHModel(input_schema, params_gjr, ModelMeta(name="GJR"))
    gjr_result = gjr.fit()
    
    # ompare Is
    print(f"\n Model Selection (Lower I is better):")
    print(f"  GRH:     I = {garch_result.payload['aic']:.2f}")
    print(f"  GRH:    I = {egarch_result.payload['aic']:.2f}")
    print(f"  GJR-GRH: I = {gjr_result.payload['aic']:.2f}")
    
    # Identify best model
    aics = {
        'GRH': garch_result.payload['aic'],
        'GRH': egarch_result.payload['aic'],
        'GJR-GRH': gjr_result.payload['aic']
    }
    best_model = min(aics, key=aics.get)
    print(f"\n Test Model: {best_model}")
    
    print("\n Model comparison complete!")


if __name__ == "__main__":
    test_egarch_workflow()
    test_gjr_garch_workflow()
    test_model_comparison()
    
    print("\n" + "="*)
    print(" ll asymmetric volatility model tests passed!")
    print("="*)
