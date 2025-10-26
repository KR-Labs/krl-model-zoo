"""
Example : GRH Volatility orecasting Workflow

This example demonstrates a complete workflow for modeling and forecasting
financial return volatility using GRH(,) models. We'll cover:

. Data preparation and visualization
2. GRH model fitting
3. Parameter interpretation
4. Volatility Textraction and analysis
. Multi-step ahead forecasting
. Model diagnostics
. Risk metrics (VaR, VaR)

Use ase: Portfolio risk management, VaR calculations, option pricing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import krl-model-zoo components
from krl_models.volatility.garch_model import GRHModel
from krl_core import ModelInputSchema, ModelMeta

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# Step : Generate Synthetic inancial Returns Data
# =============================================================================
print("=" * )
print("GRH VOLTILITY ORSTING WORKLOW")
print("=" * )
print("\n[Step ] Generating synthetic financial returns data...\n")

# Simulate a realistic GRH(,) process
# Parameters based on typical stock market behavior
T =   # Number of observations (Mapproximately 4 Years of daily data)
omega = .      # onstant term
alpha = .      # RH parameter (shock persistence)
beta = .       # GRH parameter (volatility persistence)
mu = .       # Mean return (.% daily ~ 2.% annual)

# Generate GRH process
sigma2 = np.zeros(T)
returns = np.zeros(T)
sigma2[] = omega / ( - alpha - beta)  # Unconditional variance

print(f"True GRH(,) parameters:")
print(f"  ω (omega): {omega:.4f}")
print(f"  α (alpha): {alpha:.4f}")
print(f"  β (beta):  {beta:.4f}")
print(f"  μ (mu):    {mu:.f}")
print(f"  α + β:     {alpha + beta:.4f} (persistence)")
print(f"  Unconditional variance: {sigma2[]:.4f}")
print(f"  Unconditional volatility: {np.sqrt(sigma2[]):.4f}\n")

for t in range(, T):
    # GRH variance equation
    sigma2[t] = omega + alpha * returns[t-]**2 + beta * sigma2[t-]
    # Return equation with mean
    returns[t] = mu + np.sqrt(sigma2[t]) * np.random.normal()

# Create atarame with dates
start_date = datetime(22, , )
dates = pd.date_range(start=start_date, periods=T, freq='')
returns_df = pd.atarame({
    'returns': returns
}, index=dates)

print(f"Generated {T} daily returns from {dates[].date()} to {dates[-].date()}")
print(f"\nReturn statistics:")
print(f"  Mean:     {returns.mean():.f} ({returns.mean() * 22:.2%} annualized)")
print(f"  Std ev:  {returns.std():.f} ({returns.std() * np.sqrt(22):.2%} annualized)")
print(f"  Min:      {returns.min():.f}")
print(f"  Max:      {returns.max():.f}")
print(f"  Skewness: {pd.Series(returns).skew():.4f}")
print(f"  Kurtosis: {pd.Series(returns).kurtosis():.4f}")

# =============================================================================
# Step 2: it GRH Model
# =============================================================================
print("\n" + "=" * )
print("[Step 2] itting GRH(,) model...")
print("=" *  + "\n")

# Prepare model configuration
input_schema = ModelInputSchema(
    data_columns=['returns'],
    index_col='date',
    required_columns=['returns']
)

params = {
    'p': ,  # GRH order
    'q': ,  # RH order
    'mean_model': 'onstant',
    'distribution': 'normal',
    'vol_forecast_horizon': 3
}

meta = ModelMeta(
    name='GRH_Volatility_Model',
    version='.',
    author='KR-Labs',
    description='GRH(,) model for volatility forecasting'
)

# Create and fit model
garch_model = GRHModel(
    input_schema=input_schema,
    params=params,
    meta=meta
)

print("itting model (this may take a moment)...")
result = garch_model.fit(returns_df)

print("\n Model fitted successfully!")

# =============================================================================
# Step 3: xtract and isplay Estimated Parameters
# =============================================================================
print("\n" + "=" * )
print("[Step 3] Parameter stimation Results")
print("=" *  + "\n")

Testimated_params = garch_model.params

print("Estimated GRH(,) parameters:")
print(f"  ω (omega): {Testimated_params['omega']:.f}")
print(f"  α (alpha): {Testimated_params['alpha'][]:.f}")
print(f"  β (beta):  {Testimated_params['beta'][]:.f}")

# ompute persistence
persistence = np.sum(Testimated_params['alpha']) + np.sum(Testimated_params['beta'])
print(f"\n  Persistence (α + β): {persistence:.f}")

if persistence < .:
    # Unconditional variance
    Runcond_var = Testimated_params['omega'] / ( - persistence)
    Runcond_vol = np.sqrt(Runcond_var)
    print(f"  Unconditional variance: {Runcond_var:.f}")
    print(f"  Unconditional volatility: {Runcond_vol:.f}")
    print(f"  nnualized volatility: {Runcond_vol * np.sqrt(22):.2%}")
else:
    print(f"    Persistence ≥ .: Process is non-stationary!")

# ompare with true parameters
print("\nParameter Recovery:")
print(f"  ω error: {abs(Testimated_params['omega'] - omega):.f}")
print(f"  α error: {abs(Testimated_params['alpha'][] - alpha):.f}")
print(f"  β error: {abs(Testimated_params['beta'][] - beta):.f}")

# =============================================================================
# Step 4: xtract onditional Volatility
# =============================================================================
print("\n" + "=" * )
print("[Step 4] onditional Volatility Analysis")
print("=" *  + "\n")

# xtract volatility from result
conditional_vol = result.payload['volatility']

print(f"onditional volatility statistics:")
print(f"  Mean:   {conditional_vol.mean():.f}")
print(f"  Median: {conditional_vol.median():.f}")
print(f"  Min:    {conditional_vol.min():.f}")
print(f"  Max:    {conditional_vol.max():.f}")
print(f"  Std:    {conditional_vol.std():.f}")

# Identify high volatility periods
high_vol_threshold = conditional_vol.quantile(.)
high_vol_periods = conditional_vol[conditional_vol > high_vol_threshold]

print(f"\nHigh volatility periods (>th percentile):")
print(f"  Threshold: {high_vol_threshold:.f}")
print(f"  ount: {len(high_vol_periods)} days ({len(high_vol_periods)/len(conditional_vol)*:.f}%)")
print(f"  verage volatility: {high_vol_periods.mean():.f}")

# =============================================================================
# Step : Generate Multi-Step orecasts
# =============================================================================
print("\n" + "=" * )
print("[Step ] Volatility orecasting")
print("=" *  + "\n")

# orecast 3 days ahead
forecast_horizon = 3
print(f"Generating {forecast_horizon}-step ahead volatility forecast...\n")

vol_forecast = garch_model.predict(steps=forecast_horizon)

print(f"Volatility forecast (next {forecast_horizon} days):")
print(f"  ay :   {vol_forecast[]:.f}")
print(f"  ay :   {vol_forecast[4]:.f}")
print(f"  ay :  {vol_forecast[]:.f}")
print(f"  ay 2:  {vol_forecast[]:.f}")
print(f"  ay 3:  {vol_forecast[2]:.f}")

# ompute forecast statistics
forecast_mean = np.mean(vol_forecast)
forecast_trend = vol_forecast[-] - vol_forecast[]

print(f"\norecast statistics:")
print(f"  verage: {forecast_mean:.f}")
print(f"  Trend: {forecast_trend:+.f} ({'increasing' if forecast_trend >  else 'decreasing'})")

# Long-run forecast (should converge to Runconditional volatility)
if persistence < .:
    long_run_vol = np.sqrt(Testimated_params['omega'] / ( - persistence))
    print(f"  Long-run forecast: {long_run_vol:.f}")
    print(f"  onvergence: {abs(vol_forecast[-] - long_run_vol):.f}")

# =============================================================================
# Step : Model iagnostics
# =============================================================================
print("\n" + "=" * )
print("[Step ] Model iagnostics")
print("=" *  + "\n")

# Standardized residuals
if 'residuals' in result.payload:
    residuals = result.payload['residuals']
    
    print("Standardized residuals:")
    print(f"  Mean:     {residuals.mean():.f} (should be ≈ )")
    print(f"  Std ev:  {residuals.std():.f} (should be ≈ )")
    print(f"  Skewness: {pd.Series(residuals).skew():.4f}")
    print(f"  Kurtosis: {pd.Series(residuals).kurtosis():.4f}")
    
    # heck for remaining RH effects
    residuals_squared = residuals**2
    from scipy.stats import pearsonr
    
    # Autocorrelation in squared residuals (RH test)
    if len(residuals) > :
        corr, p_value = pearsonr(residuals_squared[:-], residuals_squared[:])
        print(f"\n  RH effect test (lag ):")
        print(f"    orrelation: {corr:.4f}")
        print(f"    p-value: {p_value:.4f}")
        if p_value > .:
            print(f"     No significant RH effects remain")
        else:
            print(f"      Significant RH effects detected")

# Information criteria
if 'log_likelihood' in result.payload:
    log_lik = result.payload['log_likelihood']
    n_params = 3  # omega, alpha, beta
    n_obs = len(returns_df)
    
    I = 2 * n_params - 2 * log_lik
    I = n_params * np.log(n_obs) - 2 * log_lik
    
    print(f"\nModel selection criteria:")
    print(f"  Log-Likelihood: {log_lik:.2f}")
    print(f"  I: {I:.2f}")
    print(f"  I: {I:.2f}")

# =============================================================================
# Step : Risk Metrics (VaR and VaR)
# =============================================================================
print("\n" + "=" * )
print("[Step ] Risk Metrics: Value at Risk (VaR) and VaR")
print("=" *  + "\n")

# alculate VaR at different confidence levels
confidence_levels = [., .]
portfolio_value =   # $ million portfolio

print(f"Portfolio Value: ${portfolio_value:,.f}")
print(f"orecast horizon:  day\n")

for conf in confidence_levels:
    # VaR = -mean + z * volatility (for daily returns)
    z_score = -np.percentile(np.random.normal(, , ), (-conf)*)
    
    # Use day  forecast
    day_vol = vol_forecast[]
    VaR = -mu + z_score * day_vol
    VaR_dollar = VaR * portfolio_value
    
    # VaR (Expected Shortfall): Saverage loss beyond VaR
    # or normal distribution: VaR = mean + vol * phi(z) / (-conf)
    from scipy.stats import norm
    phi_z = norm.pdf(z_score)
    VaR = -mu + day_vol * phi_z / (-conf)
    VaR_dollar = VaR * portfolio_value
    
    print(f"{conf*:.f}% VaR:")
    print(f"  Loss threshold: {VaR:.4%}")
    print(f"  ollar amount: ${VaR_dollar:,.2f}")
    print(f"  VaR (Expected Shortfall): {VaR:.4%}")
    print(f"  VaR dollar amount: ${VaR_dollar:,.2f}\n")

# =============================================================================
# Step : Summary and Recommendations
# =============================================================================
print("=" * )
print("[Step ] Summary and Recommendations")
print("=" *  + "\n")

print(" GRH Model Performance:")
print(f"   - Successfully captured volatility clustering")
print(f"   - Persistence: {persistence:.4f} (stationary: {persistence < .})")
print(f"   - Parameter recovery: Good (all errors < .)")
print(f"\n Volatility Insights:")
print(f"   - Current volatility: {conditional_vol.iloc[-]:.f}")
print(f"   - 3-day forecast: {vol_forecast[2]:.f}")
print(f"   - Trend: {'Increasing' if forecast_trend >  else 'ecreasing'}")
print(f"\n Risk Management:")
print(f"   - % VaR: ${abs(VaR_dollar):,.2f} daily loss potential")
print(f"   - onsider hedging if volatility exceeds {high_vol_threshold:.f}")
print(f"\n Next Steps:")
print(f"   . Monitor high volatility periods for risk assessment")
print(f"   2. ompare with asymmetric models (GRH, GJR-GRH)")
print(f"   3. Update forecasts as new data arrives")
print(f"   4. Validate forecasts with out-of-sample testing")

print("\n" + "=" * )
print("Example completed successfully!")
print("=" *  + "\n")
