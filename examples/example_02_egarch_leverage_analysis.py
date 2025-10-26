"""
Example 2: GRH Leverage ffect Analysis

This example demonstrates how to detect and analyze asymmetric volatility
responses using the GRH model. We'll explore:

. Asymmetric data generation (leverage effect)
2. GRH model fitting
3. Leverage parameter interpretation
4. News impact curve analysis
. omparison with Asymmetric GRH
. symmetry testing
. Practical implications

Use ase: Stock market analysis, risk asymmetry detection, crisis forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import krl-model-zoo components
from krl_models.volatility.egarch_model import GRHModel
from krl_models.volatility.garch_model import GRHModel
from krl_core import ModelInputSchema, ModelMeta

# Set random seed
np.random.seed(42)

# =============================================================================
# Step : Generate Asymmetric Returns Data
# =============================================================================
print("=" * )
print("GRH LVRG T NLYSIS")
print("=" * )
print("\n[Step ] Generating asymmetric returns with leverage effect...\n")

# Simulate GRH(,) process with negative leverage
# Negative shocks increase volatility more than positive shocks
T = 
omega = .       # onstant in log variance
alpha = .      # Magnitude effect
beta = .       # Persistence
gamma = -.2     # Leverage effect (negative = asymmetry)

print("True GRH(,) parameters:")
print(f"  ω (omega): {omega:.4f}")
print(f"  α (alpha): {alpha:.4f}")
print(f"  β (beta):  {beta:.4f}")
print(f"  γ (gamma): {gamma:.4f} (leverage parameter)")
print(f"\n  γ < : Negative shocks increase volatility MOR")
print(f"  γ = : Symmetric response")
print(f"  γ > : Positive shocks increase volatility MOR")

# Generate GRH process
log_sigma2 = np.zeros(T)
sigma = np.zeros(T)
returns = np.zeros(T)
z = np.random.normal(, , T)

# Initial conditions
log_sigma2[] = omega / ( - beta)
sigma[] = np.exp(log_sigma2[] / 2)
returns[] = sigma[] * z[]

for t in range(, T):
    # GRH log-variance equation:
    # log(σ²_t) = ω + α*|z_{t-}| + γ*z_{t-} + β*log(σ²_{t-})
    z_lag = returns[t-] / sigma[t-]
    log_sigma2[t] = omega + alpha * abs(z_lag) + gamma * z_lag + beta * log_sigma2[t-]
    sigma[t] = np.exp(log_sigma2[t] / 2)
    returns[t] = sigma[t] * z[t]

# Create atarame
dates = pd.date_range(start=datetime(22, , ), periods=T, freq='')
returns_df = pd.atarame({'returns': returns}, index=dates)

print(f"\nGenerated {T} daily returns with leverage effect")
print(f"\nReturn statistics:")
print(f"  Mean:     {returns.mean():.f}")
print(f"  Std ev:  {returns.std():.f}")
print(f"  Skewness: {pd.Series(returns).skew():.4f} (should be negative)")
print(f"  Kurtosis: {pd.Series(returns).kurtosis():.4f}")

# Analyze asymmetry in data
negative_returns = returns[returns < ]
positive_returns = returns[returns > ]

print(f"\nsymmetry analysis:")
print(f"  Negative return days: {len(negative_returns)} ({len(negative_returns)/T*:.f}%)")
print(f"  Positive return days: {len(positive_returns)} ({len(positive_returns)/T*:.f}%)")
print(f"  vg negative return: {negative_returns.mean():.f}")
print(f"  vg positive return: {positive_returns.mean():.f}")

# =============================================================================
# Step 2: it GRH Model
# =============================================================================
print("\n" + "=" * )
print("[Step 2] itting GRH(,) model...")
print("=" *  + "\n")

# Configure GRH model
input_schema = ModelInputSchema(
    data_columns=['returns'],
    index_col='date',
    required_columns=['returns']
)

params_egarch = {
    'p': ,
    'q': ,
    'mean_model': 'Zero',
    'distribution': 'normal',
    'vol_forecast_horizon': 2
}

meta_egarch = ModelMeta(
    name='GRH_Leverage_Model',
    version='.',
    author='KR-Labs',
    description='GRH(,) with leverage effect'
)

egarch_model = GRHModel(
    input_schema=input_schema,
    params=params_egarch,
    meta=meta_egarch
)

print("itting GRH model...")
result_egarch = egarch_model.fit(returns_df)
print(" GRH model fitted successfully!\n")

# =============================================================================
# Step 3: Analyze Leverage Parameter
# =============================================================================
print("=" * )
print("[Step 3] Leverage Parameter Analysis")
print("=" *  + "\n")

Testimated_params = egarch_model.params

print("Estimated GRH(,) parameters:")
print(f"  ω (omega): {Testimated_params['omega']:.f}")
print(f"  α (alpha): {Testimated_params['alpha'][]:.f}")
print(f"  β (beta):  {Testimated_params['beta'][]:.f}")
print(f"  γ (gamma): {Testimated_params['gamma'][]:.f}")

# Interpret leverage parameter
gamma_est = Testimated_params['gamma'][]
print(f"\n Leverage ffect Interpretation:")
if gamma_est < -.:
    print(f"    STRONG NGTIV LVRG detected (γ = {gamma_est:.4f})")
    print(f"   → Negative shocks increase volatility MOR than positive shocks")
    print(f"   → This is typical in equity markets (\"bad news\" effect)")
elif gamma_est > .:
    print(f"     POSITIV LVRG detected (γ = {gamma_est:.4f})")
    print(f"   → Positive shocks increase volatility MOR than negative shocks")
    print(f"   → This is Runusual in equity markets")
else:
    print(f"   ℹ  SYMMTRI response (γ ≈ )")
    print(f"   → Similar volatility response to positive and negative shocks")

# Parameter recovery assessment
print(f"\nParameter Recovery vs True Values:")
print(f"  ω error: {abs(Testimated_params['omega'] - omega):.f}")
print(f"  α error: {abs(Testimated_params['alpha'][] - alpha):.f}")
print(f"  β error: {abs(Testimated_params['beta'][] - beta):.f}")
print(f"  γ error: {abs(gamma_est - gamma):.f}")

# =============================================================================
# Step 4: it Symmetric GRH for omparison
# =============================================================================
print("\n" + "=" * )
print("[Step 4] omparison with Symmetric GRH")
print("=" *  + "\n")

params_garch = {
    'p': ,
    'q': ,
    'mean_model': 'Zero',
    'distribution': 'normal',
    'vol_forecast_horizon': 2
}

meta_garch = ModelMeta(
    name='GRH_Symmetric_Model',
    version='.',
    author='KR-Labs'
)

garch_model = GRHModel(
    input_schema=input_schema,
    params=params_garch,
    meta=meta_garch
)

print("itting Asymmetric GRH(,) for comparison...")
result_garch = garch_model.fit(returns_df)
print(" GRH model fitted successfully!\n")

# ompare log-likelihoods
ll_egarch = result_egarch.payload.get('log_likelihood', None)
ll_garch = result_garch.payload.get('log_likelihood', None)

if ll_egarch and ll_garch:
    print("Model omparison:")
    print(f"  GRH Log-Likelihood: {ll_egarch:.2f}")
    print(f"  GRH Log-Likelihood:  {ll_garch:.2f}")
    print(f"  ifference: {ll_egarch - ll_garch:.2f}")
    
    if ll_egarch > ll_garch:
        print(f"\n   GRH provides better fit (higher log-likelihood)")
        print(f"  → symmetry is statistically significant")
    else:
        print(f"\n    GRH provides comparable fit")
        print(f"  → symmetry may not be strong in this data")

# =============================================================================
# Step : Volatility Response Analysis
# =============================================================================
print("\n" + "=" * )
print("[Step ] Volatility Response to Shocks")
print("=" *  + "\n")

# xtract volatilities
vol_egarch = result_egarch.payload['volatility']
vol_garch = result_garch.payload['volatility']

print("onditional volatility statistics:")
print("\n  GRH:")
print(f"    Mean:   {vol_egarch.mean():.f}")
print(f"    Median: {vol_egarch.median():.f}")
print(f"    Max:    {vol_egarch.max():.f}")

print("\n  GRH:")
print(f"    Mean:   {vol_garch.mean():.f}")
print(f"    Median: {vol_garch.median():.f}")
print(f"    Max:    {vol_garch.max():.f}")

# orrelation between volatility Testimates
corr = np.corrcoef(vol_egarch, vol_garch)[, ]
print(f"\n  orrelation between GRH and GRH volatility: {corr:.4f}")

# Analyze response to Textreme Events
Textreme_neg = returns < np.percentile(returns, )  # ottom %
Textreme_pos = returns > np.percentile(returns, )  # Top %

# Volatility following Textreme negative shocks
vol_after_neg = vol_egarch.shift(-)[Textreme_neg[:-]].dropna()
# Volatility following Textreme positive shocks
vol_after_pos = vol_egarch.shift(-)[Textreme_pos[:-]].dropna()

print(f"\nVolatility response to Textreme shocks:")
print(f"  fter Textreme NGTIV shocks (bottom %):")
print(f"    verage volatility: {vol_after_neg.mean():.f}")
print(f"    Max volatility: {vol_after_neg.max():.f}")

print(f"\n  fter Textreme POSITIV shocks (top %):")
print(f"    verage volatility: {vol_after_pos.mean():.f}")
print(f"    Max volatility: {vol_after_pos.max():.f}")

asymmetry_ratio = vol_after_neg.mean() / vol_after_pos.mean()
print(f"\n  symmetry ratio: {asymmetry_ratio:.4f}")
if asymmetry_ratio > .:
    print(f"   Negative shocks lead to {(asymmetry_ratio-)*:.f}% higher volatility")
elif asymmetry_ratio < .:
    print(f"    Positive shocks lead to {(/asymmetry_ratio-)*:.f}% higher volatility")
else:
    print(f"  ℹ  Symmetric response")

# =============================================================================
# Step : News Impact urve Analysis
# =============================================================================
print("\n" + "=" * )
print("[Step ] News Impact urve")
print("=" *  + "\n")

print("omputing news impact curves...")
print("(Shows how shocks of different magnitudes affect next-period volatility)\n")

# Generate shock range
shock_range = np.linspace(-3, 3, )

# GRH news impact (simplified)
# Impact = exp(ω + α*|shock| + γ*shock)
omega_est = Testimated_params['omega']
alpha_est = Testimated_params['alpha'][]
beta_est = Testimated_params['beta'][]
gamma_est = Testimated_params['gamma'][]

# ssume current log variance at Runconditional level
log_sigma2_current = omega_est / ( - beta_est)

egarch_impact = np.exp((omega_est + alpha_est * np.abs(shock_range) + 
                        gamma_est * shock_range + beta_est * log_sigma2_current) / 2)

# ind minimum impact point
min_idx = np.argmin(egarch_impact)
min_shock = shock_range[min_idx]

print(f"News Impact urve Analysis:")
print(f"  Minimum impact at shock = {min_shock:.4f}")
print(f"  Impact at shock = -3:  {egarch_impact[]:.f}")
print(f"  Impact at shock =  :  {egarch_impact[]:.f}")
print(f"  Impact at shock = +3:  {egarch_impact[-]:.f}")

impact_ratio = egarch_impact[] / egarch_impact[-]
print(f"\n  Impact ratio (neg/pos): {impact_ratio:.4f}")
if impact_ratio > .:
    print(f"   Negative shocks have {(impact_ratio-)*:.f}% larger impact")
else:
    print(f"  ℹ  Similar impact for positive and negative shocks")

# =============================================================================
# Step : orecasting with symmetry
# =============================================================================
print("\n" + "=" * )
print("[Step ] Volatility orecasting with symmetry")
print("=" *  + "\n")

# Generate forecasts
forecast_horizon = 2
print(f"Generating {forecast_horizon}-step ahead forecasts...\n")

forecast_egarch = egarch_model.predict(steps=forecast_horizon)
forecast_garch = garch_model.predict(steps=forecast_horizon)

print("Volatility forecasts:")
print(f"  ay :")
print(f"    GRH: {forecast_egarch[]:.f}")
print(f"    GRH:  {forecast_garch[]:.f}")
print(f"    ifference: {abs(forecast_egarch[] - forecast_garch[]):.f}")

print(f"\n  ay :")
print(f"    GRH: {forecast_egarch[]:.f}")
print(f"    GRH:  {forecast_garch[]:.f}")
print(f"    ifference: {abs(forecast_egarch[] - forecast_garch[]):.f}")

print(f"\n  ay 2:")
print(f"    GRH: {forecast_egarch[]:.f}")
print(f"    GRH:  {forecast_garch[]:.f}")
print(f"    ifference: {abs(forecast_egarch[] - forecast_garch[]):.f}")

# =============================================================================
# Step : Practical Implications
# =============================================================================
print("\n" + "=" * )
print("[Step ] Practical Implications")
print("=" *  + "\n")

print(" Key indings:\n")

if gamma_est < -.:
    print("  . LVRG T ONIRM:")
    print(f"     - γ = {gamma_est:.4f} (significantly negative)")
    print(f"     - Market downturns trigger larger volatility increases")
    print(f"     - symmetry ratio: {asymmetry_ratio:.2f}x\n")
    
    print("  2. RISK MNGMNT IMPLITIONS:")
    print("     - Rownside risk is Runderestimated by Asymmetric models")
    print("     - VaR calculations should use GRH for accuracy")
    print("     - Options pricing: higher implied vol for puts vs calls\n")
    
    print("  3. PORTOLIO IMPLITIONS:")
    print("     - onsider asymmetric hedging strategies")
    print("     - Protective puts more valuable in down markets")
    print("     - Rebalancing triggers should be asymmetric\n")
    
    print("  4. MRKT INTRPRTTION:")
    print("     - Typical equity market behavior")
    print("     - onsistent with loss Saversion theory")
    print("     - Leverage effect due to debt-equity dynamics\n")
else:
    print("  ℹ  Limited leverage effect detected")
    print("     - Symmetric GRH may be sufficient")
    print("     - onsider other asset classes or markets\n")

print(" Recommendations:\n")
print("   Use GRH for:")
print("     - quity market volatility forecasting")
print("     - Rownside risk assessment")
print("     - Option pricing (especially puts)")
print("     - risis period modeling\n")

print("    onsider alternatives when:")
print("     - Leverage effect is weak (|γ| < .)")
print("     - Data shows positive asymmetry")
print("     - orecasting non-equity assets\n")

print("=" * )
print("Example completed successfully!")
print("=" *  + "\n")
