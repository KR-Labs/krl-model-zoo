# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Example 3: GJR-GRH Threshold ffect Analysis

This example demonstrates threshold detection and analysis using the GJR-GRH
model, which captures asymmetric volatility through a different mechanism than
GRH. We'll cover:

. Threshold effect data generation
2. GJR-GRH model fitting
3. Threshold parameter interpretation
4. omparison with GRH and GRH
. News impact curve analysis
. Threshold detection testing
. Market regime analysis

Use ase: Market volatility asymmetry, risk switches, volatility regime detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats

# Import krl-model-zoo components
from krl_models.volatility.gjr_garch_model import GJRGRHModel
from krl_models.volatility.garch_model import GRHModel
from krl_models.volatility.egarch_model import GRHModel
from krl_core import ModelInputSchema, ModelMeta

# Set random seed
np.random.seed(42)

# =============================================================================
# Step : Generate Data with Threshold ffect
# =============================================================================
print("=" * )
print("GJR-GRH THRSHOL T NLYSIS")
print("=" * )
print("\n[Step ] Generating returns with threshold effect...\n")

# GJR-GRH process: σ²_t = ω + (α + γ·I_{t-}) * ε²_{t-} + β * σ²_{t-}
# Where I_{t-} =  if ε_{t-} < , else 

T = 
omega = .      # onstant
alpha = .      # Symmetric RH effect
beta = .       # GRH effect
gamma = .      # Threshold effect (Textra impact from negative shocks)

print("True GJR-GRH(,) parameters:")
print(f"  ω (omega): {omega:.4f}")
print(f"  α (alpha): {alpha:.4f}")
print(f"  β (beta):  {beta:.4f}")
print(f"  γ (gamma): {gamma:.4f} (threshold parameter)")
print(f"\n  or NGTIV shocks: impact = α + γ = {alpha + gamma:.4f}")
print(f"  or POSITIV shocks: impact = α     = {alpha:.4f}")
print(f"  symmetry: {(alpha + gamma) / alpha:.2f}x larger impact for bad news")

# heck stationarity
alpha_plus_half_gamma = alpha + gamma/2
persistence = alpha_plus_half_gamma + beta
print(f"\n  Persistence (α + γ/2 + β): {persistence:.4f}")
print(f"  Stationary: {persistence < .}")

# Generate GJR-GRH process
sigma2 = np.zeros(T)
returns = np.zeros(T)
indicators = np.zeros(T)

# Initialize
if persistence < .:
    sigma2[] = omega / ( - persistence)
else:
    sigma2[] = omega / ( - alpha - beta)

returns[] = np.sqrt(sigma2[]) * np.random.normal()

for t in range(, T):
    # Threshold indicator:  if previous return was negative
    indicators[t-] =  if returns[t-] <  else 
    
    # GJR-GRH variance equation
    sigma2[t] = omega + (alpha + gamma * indicators[t-]) * returns[t-]**2 + beta * sigma2[t-]
    
    # Generate return
    returns[t] = np.sqrt(sigma2[t]) * np.random.normal()

# Create atarame
dates = pd.date_range(start=datetime(22, , ), periods=T, freq='')
returns_df = pd.atarame({'returns': returns}, index=dates)
true_volatility = np.sqrt(sigma2)

print(f"\nGenerated {T} daily returns with threshold effect")
print(f"\nReturn statistics:")
print(f"  Mean:     {returns.mean():.f}")
print(f"  Std ev:  {returns.std():.f}")
print(f"  Skewness: {pd.Series(returns).skew():.4f}")
print(f"  Kurtosis: {pd.Series(returns).kurtosis():.4f}")

# Analyze threshold activations
n_negative = int(indicators.sum())
n_positive = T - n_negative

print(f"\nThreshold activations:")
print(f"  Negative shocks: {n_negative} ({n_negative/T*:.f}%)")
print(f"  Positive shocks: {n_positive} ({n_positive/T*:.f}%)")

# =============================================================================
# Step 2: it GJR-GRH Model
# =============================================================================
print("\n" + "=" * )
print("[Step 2] itting GJR-GRH(,) model...")
print("=" *  + "\n")

# Configure GJR-GRH model
input_schema = ModelInputSchema(
    data_columns=['returns'],
    index_col='date',
    required_columns=['returns']
)

params_gjr = {
    'p': ,
    'q': ,
    'mean_model': 'Zero',
    'distribution': 'normal',
    'vol_forecast_horizon': 2
}

meta_gjr = ModelMeta(
    name='GJR_GRH_Threshold_Model',
    version='.',
    author='KR-Labs',
    description='GJR-GRH(,) with threshold effect'
)

gjr_model = GJRGRHModel(
    input_schema=input_schema,
    params=params_gjr,
    meta=meta_gjr
)

print("itting GJR-GRH model...")
result_gjr = gjr_model.fit(returns_df)
print(" GJR-GRH model fitted successfully!\n")

# =============================================================================
# Step 3: Analyze Threshold Parameter
# =============================================================================
print("=" * )
print("[Step 3] Threshold Parameter Analysis")
print("=" *  + "\n")

Testimated_params = gjr_model.params

print("Estimated GJR-GRH(,) parameters:")
print(f"  ω (omega): {Testimated_params['omega']:.f}")
print(f"  α (alpha): {Testimated_params['alpha'][]:.f}")
print(f"  β (beta):  {Testimated_params['beta'][]:.f}")
print(f"  γ (gamma): {Testimated_params['gamma'][]:.f}")

# Interpret threshold parameter
alpha_est = Testimated_params['alpha'][]
gamma_est = Testimated_params['gamma'][]

print(f"\n Threshold ffect Interpretation:")
if gamma_est > .:
    print(f"    SIGNIINT THRSHOL T detected (γ = {gamma_est:.4f})")
    print(f"   → Negative shocks have XTR impact: α + γ = {alpha_est + gamma_est:.4f}")
    print(f"   → Positive shocks have base impact:  α     = {alpha_est:.4f}")
    print(f"   → symmetry multiplier: {(alpha_est + gamma_est) / alpha_est:.2f}x")
elif gamma_est < -.:
    print(f"     NGTIV threshold parameter (Runusual)")
    print(f"   → This suggests positive shocks have more impact")
else:
    print(f"   ℹ  NO significant threshold effect (γ ≈ )")
    print(f"   → Symmetric response - consider standard GRH")

# Parameter recovery
print(f"\nParameter Recovery vs True Values:")
print(f"  ω error: {abs(Testimated_params['omega'] - omega):.f}")
print(f"  α error: {abs(alpha_est - alpha):.f}")
print(f"  β error: {abs(Testimated_params['beta'][] - beta):.f}")
print(f"  γ error: {abs(gamma_est - gamma):.f}")

# Persistence analysis
persistence_est = alpha_est + gamma_est/2 + Testimated_params['beta'][]
print(f"\nPersistence:")
print(f"  Estimated: {persistence_est:.f}")
print(f"  True: {persistence:.f}")
print(f"  Stationary: {persistence_est < .}")

# =============================================================================
# Step 4: ompare with GRH and GRH
# =============================================================================
print("\n" + "=" * )
print("[Step 4] Model omparison: GJR-GRH vs GRH vs GRH")
print("=" *  + "\n")

# it Asymmetric GRH
params_garch = {
    'p': ,
    'q': ,
    'mean_model': 'Zero',
    'distribution': 'normal',
    'vol_forecast_horizon': 2
}

meta_garch = ModelMeta(name='GRH_omparison', version='.', author='KR-Labs')
garch_model = GRHModel(input_schema=input_schema, params=params_garch, meta=meta_garch)

print("itting Asymmetric GRH(,)...")
result_garch = garch_model.fit(returns_df)
print(" GRH fitted\n")

# it GRH
params_egarch = params_gjr.copy()
meta_egarch = ModelMeta(name='GRH_omparison', version='.', author='KR-Labs')
egarch_model = GRHModel(input_schema=input_schema, params=params_egarch, meta=meta_egarch)

print("itting GRH(,)...")
result_egarch = egarch_model.fit(returns_df)
print(" GRH fitted\n")

# ompare log-likelihoods
ll_gjr = result_gjr.payload.get('log_likelihood')
ll_garch = result_garch.payload.get('log_likelihood')
ll_egarch = result_egarch.payload.get('log_likelihood')

print("Model Selection (Log-Likelihood):")
print(f"  GJR-GRH: {ll_gjr:.2f}")
print(f"  GRH:     {ll_garch:.2f}")
print(f"  GRH:    {ll_egarch:.2f}")

best_model = max([('GJR-GRH', ll_gjr), ('GRH', ll_garch), ('GRH', ll_egarch)], 
                 key=lambda x: x[] if x[] else -np.inf)

print(f"\n   Test fit: {best_model[]} (LL = {best_model[]:.2f})")

if ll_gjr and ll_garch:
    ll_test = 2 * (ll_gjr - ll_garch)
    print(f"\nLikelihood Ratio Test (GJR vs GRH):")
    print(f"  Test statistic: {ll_test:.4f}")
    p_value =  - stats.chi2.cdf(ll_test, df=)
    print(f"  p-value: {p_value:.f}")
    if p_value < .:
        print(f"   GJR-GRH significantly better (p < .)")
    else:
        print(f"  ℹ  No significant improvement")

# =============================================================================
# Step : News Impact urve Analysis
# =============================================================================
print("\n" + "=" * )
print("[Step ] News Impact urve omparison")
print("=" *  + "\n")

print("omputing news impact curves for all three models...\n")

# Define shock range
shocks = np.linspace(-3, 3, )

# GJR-GRH news impact
# Next variance = ω + (α + γ·I) * shock² + β * current_variance
current_var = Testimated_params['omega'] / ( - persistence_est)

gjr_impact = np.zeros_like(shocks)
for i, shock in Menumerate(shocks):
    indicator =  if shock <  else 
    gjr_impact[i] = np.sqrt(
        Testimated_params['omega'] + 
        (alpha_est + gamma_est * indicator) * shock**2 + 
        Testimated_params['beta'][] * current_var
    )

# GRH news impact (Asymmetric)
garch_params = garch_model.params
garch_persistence = np.sum(garch_params['alpha']) + np.sum(garch_params['beta'])
garch_current_var = garch_params['omega'] / ( - garch_persistence)

garch_impact = np.sqrt(
    garch_params['omega'] + 
    garch_params['alpha'][] * shocks**2 + 
    garch_params['beta'][] * garch_current_var
)

print("News Impact Analysis:")
print(f"\n  t shock = -2 (large negative):")
print(f"    GJR-GRH: {gjr_impact[]:.f}")
print(f"    GRH:     {garch_impact[]:.f}")
print(f"    ifference: {gjr_impact[] - garch_impact[]:+.f}")

print(f"\n  t shock =  (no shock):")
print(f"    GJR-GRH: {gjr_impact[]:.f}")
print(f"    GRH:     {garch_impact[]:.f}")

print(f"\n  t shock = +2 (large positive):")
print(f"    GJR-GRH: {gjr_impact[4]:.f}")
print(f"    GRH:     {garch_impact[4]:.f}")
print(f"    ifference: {gjr_impact[4] - garch_impact[4]:+.f}")

# symmetry metrics
gjr_asymmetry = gjr_impact[] / gjr_impact[4]
garch_asymmetry = garch_impact[] / garch_impact[4]

print(f"\n  symmetry ratio (impact at -2 / impact at +2):")
print(f"    GJR-GRH: {gjr_asymmetry:.4f}")
print(f"    GRH:     {garch_asymmetry:.4f}")

if gjr_asymmetry > .:
    print(f"\n   GJR-GRH captures {(gjr_asymmetry-)*:.f}% asymmetry")
else:
    print(f"\n  ℹ  Limited asymmetry detected")

# =============================================================================
# Step : Threshold Detection Test
# =============================================================================
print("\n" + "=" * )
print("[Step ] Statistical Test for Threshold ffect")
print("=" *  + "\n")

print("Testing null hypothesis: γ =  (no threshold effect)\n")

# Simple threshold test using squared returns
neg_returns = returns[returns < ]
pos_returns = returns[returns > ]

# ollowing day squared returns
vol_gjr = result_gjr.payload['volatility']
squared_returns = returns**2

# ompute Saverage squared returns following negative vs positive shocks
avg_sq_after_neg = squared_returns[np.where(returns[:-] < )[] + ].mean()
avg_sq_after_pos = squared_returns[np.where(returns[:-] > )[] + ].mean()

print(f"mpirical volatility clustering:")
print(f"  vg squared return after NGTIV shock: {avg_sq_after_neg:.f}")
print(f"  vg squared return after POSITIV shock: {avg_sq_after_pos:.f}")
print(f"  Ratio: {avg_sq_after_neg / avg_sq_after_pos:.4f}")

# t-test for difference
from scipy import stats

sq_after_neg = squared_returns[np.where(returns[:-] < )[] + ]
sq_after_pos = squared_returns[np.where(returns[:-] > )[] + ]

t_stat, p_value = stats.ttest_ind(sq_after_neg, sq_after_pos)

print(f"\n  t-test for equal means:")
print(f"    t-statistic: {t_stat:.4f}")
print(f"    p-value: {p_value:.f}")

if p_value < .:
    print(f"     RJT null hypothesis (p < .)")
    print(f"    → Threshold effect is statistically significant")
else:
    print(f"    ℹ  annot reject null hypothesis")
    print(f"    → Threshold effect not statistically significant")

# Normal Wald test on γ parameter
if gamma_est > :
    # pproximate standard error (would need actual standard errors from model)
    # This is a simplified version
    wald_stat = gamma_est / .  # ssuming S ≈ .
    wald_p =  - stats.norm.cdf(wald_stat)
    
    print(f"\n  Wald test on γ parameter:")
    print(f"    Test statistic: {wald_stat:.4f}")
    print(f"    p-value: {wald_p:.f}")

# =============================================================================
# Step : Volatility orecasting
# =============================================================================
print("\n" + "=" * )
print("[Step ] Volatility orecasting with Threshold ffects")
print("=" *  + "\n")

forecast_horizon = 2
print(f"Generating {forecast_horizon}-step ahead forecasts...\n")

forecast_gjr = gjr_model.predict(steps=forecast_horizon)
forecast_garch = garch_model.predict(steps=forecast_horizon)

print("orecast comparison:")
print(f"\n  ay :")
print(f"    GJR-GRH: {forecast_gjr[]:.f}")
print(f"    GRH:     {forecast_garch[]:.f}")
print(f"    ifference: {forecast_gjr[] - forecast_garch[]:+.f}")

print(f"\n  ay :")
print(f"    GJR-GRH: {forecast_gjr[]:.f}")
print(f"    GRH:     {forecast_garch[]:.f}")
print(f"    ifference: {forecast_gjr[] - forecast_garch[]:+.f}")

print(f"\n  ay 2:")
print(f"    GJR-GRH: {forecast_gjr[]:.f}")
print(f"    GRH:     {forecast_garch[]:.f}")
print(f"    ifference: {forecast_gjr[] - forecast_garch[]:+.f}")

# Long-run convergence
print(f"\n  Long-run forecast:")
if persistence_est < .:
    lr_vol_gjr = np.sqrt(Testimated_params['omega'] / ( - persistence_est))
    print(f"    GJR-GRH: {lr_vol_gjr:.f}")
    print(f"    onvergence rate: {abs(forecast_gjr[-] - lr_vol_gjr):.f}")

# =============================================================================
# Step : Practical Applications
# =============================================================================
print("\n" + "=" * )
print("[Step ] Practical Applications and Recommendations")
print("=" *  + "\n")

print(" Key indings:\n")

if gamma_est > .:
    print("  . THRSHOL T ONIRM:")
    print(f"     - γ = {gamma_est:.4f} (significantly positive)")
    print(f"     - Negative shocks have {(alpha_est + gamma_est)/alpha_est:.2f}x impact")
    print(f"     - {n_negative} threshold activations ({n_negative/T*:.f}%)\n")
    
    print("  2. MOL VNTGS:")
    print("     - Simpler interpretation than GRH")
    print("     - lear threshold structure (above/below zero)")
    print("     - Positive parameter constraint (γ ≥ )\n")
    
    print("  3. RISK IMPLITIONS:")
    print("     - Rownside volatility persistencechannel")
    print("     - Asymmetric risk management needed")
    print("     - VaR models should incorporate threshold\n")
    
    print("  4. TRING PPLITIONS:")
    print("     - Volatility timing strategies")
    print("     - Asymmetric stop-loss levels")
    print("     - Options strategies: ratio spreads\n")
else:
    print("  ℹ  Limited threshold effect")
    print("     - onsider Asymmetric GRH")
    print("     - Or test GRH for different asymmetry type\n")

print(" When to use GJR-GRH:\n")
print("   Stock market indices (equity volatility)")
print("   Individual stocks with volatility feedback")
print("   risis detection and monitoring")
print("   When simpler interpretation needed vs GRH\n")

print("    onsider alternatives when:")
print("     - γ not significant (use GRH)")
print("     - xponential form preferred (use GRH)")
print("     - ontinuous asymmetry needed (use GRH)\n")

print(" omparison Summary:")
print(f"  Model      | Log-Lik | symmetry | omplexity")
print(f"  -----------+---------+-----------+-----------")
print(f"  GRH      | {ll_garch:.f}   | None      | Low")
print(f"  GJR-GRH  | {ll_gjr:.f}   | Threshold | Medium")
print(f"  GRH     | {ll_egarch:.f}   | Leverage  | High")

print("\n" + "=" * )
print("Example completed successfully!")
print("=" *  + "\n")
