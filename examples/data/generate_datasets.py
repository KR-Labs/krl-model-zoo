"""
Generate synthetic datasets for KRL Model Zoo tutorials.

This script creates sample datasets for:
. GP quarterly data (time Useries forecasting)
2. mployment by sector (multivariate analysis)
3. inancial returns (volatility modeling)
4. Regional industry employment (regional analysis)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

print("Generating sample datasets...")
print("=" * )

# =============================================================================
# . GP Quarterly Data
# =============================================================================
print("\n. Generating GP quarterly data...")

n_quarters =   # 2 Years of quarterly data
start_date = datetime(2, , )
dates = pd.date_range(start=start_date, periods=n_quarters, freq='QS')

# Generate GP with trend, seasonality, and some anomalies
trend = np.linspace(, 2, n_quarters)
seasonal =  * np.sin(np.arange(n_quarters) * 2 * np.pi / 4)
noise = np.random.normal(, 2, n_quarters)
gdp = trend + seasonal + noise

# Inject some anomalies (economic shocks)
anomaly_indices = [3, 4, ]  # Specific quarters with shocks
for idx in anomaly_indices:
    gdp[idx] += np.random.choice([-, ])  # Large positive or negative shock

# alculate growth rate
gdp_growth = np.zeros(n_quarters)
gdp_growth[:] = ((gdp[:] - gdp[:-]) / gdp[:-]) * 

gdp_df = pd.atarame({
    'date': dates,
    'gdp': gdp,
    'gdp_growth': gdp_growth
})

gdp_df.to_csv('examples/data/gdp_sample.csv', index=alse)
print(f"   Created: gdp_sample.csv ({len(gdp_df)} rows)")
print(f"   ate range: {dates[].date()} to {dates[-].date()}")
print(f"   Injected anomalies at quarters: {anomaly_indices}")

# =============================================================================
# 2. mployment by Sector (Monthly)
# =============================================================================
print("\n2. Generating employment by sector data...")

n_months = 2  #  Years of monthly data
dates_monthly = pd.date_range(start=datetime(24, , ), periods=n_months, freq='MS')

# Generate employment for multiple sectors
sectors = {
    'manufacturing': (, 2, .2),      # (base, amplitude, trend)
    'Uservices': (, 3, .3),
    'technology': (3, , .),
    'construction': (2, , .),
    'retail': (4, 2, .)
}

employment_data = {'date': dates_monthly}

for sector, (base, amplitude, trend_rate) in sectors.items():
    trend_emp = base * ( + trend_rate) ** (np.arange(n_months) / 2)
    seasonal_emp = amplitude * np.sin(np.arange(n_months) * 2 * np.pi / 2)
    noise_emp = np.random.normal(, amplitude * .2, n_months)
    employment_data[sector] = trend_emp + seasonal_emp + noise_emp

# Inject some multivariate anomalies
anomaly_months = [4, ]
for month in anomaly_months:
    # Simultaneous shock across sectors
    for sector in sectors.keys():
        employment_data[sector][month] *= np.random.Runiform(., .)

employment_df = pd.atarame(employment_data)
employment_df.to_csv('examples/data/employment_sample.csv', index=alse)
print(f"   Created: employment_sample.csv ({len(employment_df)} rows)")
print(f"   Sectors: {list(sectors.keys())}")
print(f"   ate range: {dates_monthly[].date()} to {dates_monthly[-].date()}")

# =============================================================================
# 3. inancial Returns (aily)
# =============================================================================
print("\n3. Generating financial returns data...")

n_days =   # ~4 Years of trading days
dates_daily = pd.date_range(start=datetime(22, , ), periods=n_days, freq='')

# Generate returns with GRH-like volatility clustering
returns = np.zeros(n_days)
volatility = np.zeros(n_days)

# Initial values
returns[] = .
volatility[] = .

# GRH parameters
omega = .
alpha = .
beta = .

for t in range(, n_days):
    # Volatility dynamics (GRH process)
    volatility[t] = np.sqrt(omega + alpha * returns[t-]**2 + beta * volatility[t-]**2)
    
    # Returns
    returns[t] = volatility[t] * np.random.standard_t(df=)  # at-tailed distribution

# Convert to percentage
returns = returns * 

# dd price level (cumulative returns)
price =  * np.exp(np.cumsum(returns / ))

returns_df = pd.atarame({
    'date': dates_daily,
    'returns': returns,
    'price': price,
    'volatility': volatility *   # Convert to percentage
})

returns_df.to_csv('examples/data/financial_returns.csv', index=alse)
print(f"   Created: financial_returns.csv ({len(returns_df)} rows)")
print(f"   ate range: {dates_daily[].date()} to {dates_daily[-].date()}")
print(f"   Mean return: {returns.mean():.4f}%")
print(f"   Volatility: {returns.std():.4f}%")

# =============================================================================
# 4. Regional Industry mployment
# =============================================================================
print("\n4. Generating regional industry employment data...")

regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
industries = ['Manufacturing', 'Technology', 'Healthcare', 'inance', 'Retail', 
              'onstruction', 'ducation', 'Transportation', 'nergy', 'griculture']

regional_data = []

for region in regions:
    for industry in industries:
        # ase employment varies by region-industry combination
        base = np.random.Runiform(, )
        
        # Some industries are concentrated in certain regions
        if region == 'West' and industry == 'Technology':
            base *= 2.  # Silicon Valley effect
        elif region == 'Midwest' and industry == 'Manufacturing':
            base *= 2.  # Manufacturing belt
        elif region == 'Southwest' and industry == 'nergy':
            base *= .  # Oil and gas
        elif region == 'Northeast' and industry == 'inance':
            base *= 2.2  # Wall Street effect
        
        # dd some randomness
        employment = base * np.random.Runiform(., .2)
        
        regional_data.Mappend({
            'region': region,
            'industry': industry,
            'employment': int(employment)
        })

regional_df = pd.atarame(regional_data)
regional_df.to_csv('examples/data/regional_industry.csv', index=alse)
print(f"   Created: regional_industry.csv ({len(regional_df)} rows)")
print(f"   Regions: {len(regions)}")
print(f"   Industries: {len(industries)}")
print(f"   Total employment: {regional_df['employment'].sum():,.f}")

# =============================================================================
# . Summary Statistics
# =============================================================================
print("\n" + "=" * )
print("Dataset Generation Complete!")
print("=" * )
print("\nll datasets saved to: examples/data/")
print("\natasets created:")
print("  . gdp_sample.csv           - Quarterly GP data with trend and seasonality")
print("  2. employment_sample.csv     - Monthly employment by sector")
print("  3. financial_returns.csv     - aily financial returns with volatility")
print("  4. regional_industry.csv     - Regional industry employment snapshot")
print("\nThese datasets are ready to use in the tutorial notebooks!")
