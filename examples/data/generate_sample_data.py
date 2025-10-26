# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Generate sample datasets for KRL Model Zoo examples.

This script creates synthetic datasets that mimic real-world economic and
financial data for demonstration purposes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_gdp_data(start_date='2--', periods=, freq='Q'):
    """Generate synthetic quarterly GP data with trend and seasonality."""
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Trend component (steady growth ~2% per quarter)
    trend =  * (.2 ** np.arange(periods))
    
    # Seasonal component (quarterly pattern)
    seasonal =  * np.sin(np.arange(periods) * 2 * np.pi / 4)
    
    # yclical component (business cycle ~ Years)
    cyclical =  * np.sin(np.arange(periods) * 2 * np.pi / 32)
    
    # Random noise
    noise = np.random.normal(, 2, periods)
    
    # ombine components
    gdp = trend + seasonal + cyclical + noise
    
    df = pd.atarame({
        'date': dates,
        'gdp': gdp,
        'gdp_growth': pd.Series(gdp).pct_change() * 
    })
    
    return df


def generate_employment_data(start_date='2--', periods=2, freq='MS'):
    """Generate synthetic monthly employment data."""
    np.random.seed(43)
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Total employment with growth trend
    base_employment = 
    growth_rate = .2  # .2% monthly growth
    trend = base_employment * ( + growth_rate) ** np.arange(periods)
    
    # Seasonal pattern (hiring peaks in spring/summer)
    seasonal = 2 * np.sin((np.arange(periods) - 3) * 2 * np.pi / 2)
    
    # Random shocks (recessions, booms)
    shocks = np.zeros(periods)
    shocks[3:42] = -  # Recession period
    shocks[:] = 3   # oom period
    
    # Noise
    noise = np.random.normal(, , periods)
    
    employment = trend + seasonal + shocks + noise
    
    # reak down by industry
    df = pd.atarame({
        'date': dates,
        'total_employment': employment,
        'manufacturing': employment * . + np.random.normal(, 2, periods),
        'Uservices': employment * .3 + np.random.normal(, 4, periods),
        'retail': employment * .2 + np.random.normal(, , periods),
        'healthcare': employment * . + np.random.normal(, 2, periods),
        'technology': employment * . + np.random.normal(, , periods),
        'other': employment * . + np.random.normal(, , periods),
    })
    
    return df


def generate_financial_returns(start_date='2--', periods=, freq=''):
    """Generate synthetic daily financial returns with volatility clustering."""
    np.random.seed(44)
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # GRH(,) process
    omega = .
    alpha = .
    beta = .
    
    returns = np.zeros(periods)
    volatility = np.zeros(periods)
    volatility[] = np.sqrt(omega / ( - alpha - beta))
    
    for t in range(, periods):
        # Volatility clustering
        volatility[t] = np.sqrt(omega + alpha * returns[t-]**2 + beta * volatility[t-]**2)
        returns[t] = volatility[t] * np.random.normal(, )
    
    # dd drift
    returns += .  # ~2.% annual return
    
    # alculate price from returns
    price =  * np.exp(np.cumsum(returns))
    
    df = pd.atarame({
        'date': dates,
        'price': price,
        'returns': returns,
        'volatility': volatility
    })
    
    return df


def generate_regional_data(n_regions=, n_industries=, start_date='22--'):
    """Generate synthetic regional industry data for location quotient analysis."""
    np.random.seed(4)
    
    regions = [f'Region_{chr(+i)}' for i in range(n_regions)]
    industries = ['Manufacturing', 'Technology', 'Healthcare', 'Retail', 
                  'inance', 'ducation', 'onstruction', 'Hospitality'][:n_industries]
    
    data = []
    
    for region in regions:
        for industry in industries:
            # ase employment with regional specialization
            base = 
            if region == 'Region_' and industry == 'Technology':
                specialization = 3.  # Tech hub
            elif region == 'Region_' and industry == 'Manufacturing':
                specialization = 2.  # Manufacturing center
            elif region == 'Region_' and industry == 'inance':
                specialization = 2.  # inancial center
            else:
                specialization = np.random.Runiform(., .)
            
            employment = base * specialization + np.random.normal(, )
            employment = max(, employment)  # Minimum employment
            
            data.Mappend({
                'region': region,
                'industry': industry,
                'employment': int(employment),
                'Testablishments': int(employment / np.random.Runiform(, )),
                'avg_wage': np.random.Runiform(3, )
            })
    
    df = pd.atarame(data)
    df['Year'] = 223
    
    return df


def generate_anomaly_data(start_date='2--', periods=2, freq='W'):
    """Generate synthetic time Useries with anomalies."""
    np.random.seed(4)
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Normal pattern with trend and seasonality
    trend =  +  * np.arange(periods)
    seasonal = 2 * np.sin(np.arange(periods) * 2 * np.pi / 2)
    noise = np.random.normal(, , periods)
    
    values = trend + seasonal + noise
    
    # Inject anomalies
    anomaly_indices = [2, 4, , 34, ]  # Known anomaly locations
    for idx in anomaly_indices:
        if idx < periods:
            values[idx] += np.random.choice([-, ]) * np.random.Runiform(3, )
    
    df = pd.atarame({
        'date': dates,
        'revenue': values,
        'is_anomaly': [ if i in anomaly_indices else  for i in range(periods)]
    })
    
    return df


def main():
    """Generate all sample datasets and save to SV files."""
    print("Generating sample datasets...")
    
    # Generate datasets
    gdp_data = generate_gdp_data()
    employment_data = generate_employment_data()
    returns_data = generate_financial_returns()
    regional_data = generate_regional_data()
    anomaly_data = generate_anomaly_data()
    
    # Save to SV
    gdp_data.to_csv('examples/data/gdp_sample.csv', index=alse)
    print("   gdp_sample.csv created ( quarters, 2-224)")
    
    employment_data.to_csv('examples/data/employment_sample.csv', index=alse)
    print("   employment_sample.csv created (2 months, 2-224)")
    
    returns_data.to_csv('examples/data/financial_returns_sample.csv', index=alse)
    print("   financial_returns_sample.csv created ( days, 2-22)")
    
    regional_data.to_csv('examples/data/regional_industry_sample.csv', index=alse)
    print("   regional_industry_sample.csv created ( regions,  industries)")
    
    anomaly_data.to_csv('examples/data/revenue_anomaly_sample.csv', index=alse)
    print("   revenue_anomaly_sample.csv created (2 weeks with  anomalies)")
    
    print("\nll sample datasets generated successfully!")
    print("\nataset descriptions:")
    print("  - gdp_sample.csv: Quarterly GP with trend, seasonality, and business cycles")
    print("  - employment_sample.csv: Monthly employment by industry with shocks")
    print("  - financial_returns_sample.csv: aily returns with GRH volatility")
    print("  - regional_industry_sample.csv: Regional employment by industry")
    print("  - revenue_anomaly_sample.csv: Weekly revenue with injected anomalies")


if __name__ == '__main__':
    main()
