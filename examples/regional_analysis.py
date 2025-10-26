# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Regional Analysis Example: Location Quotient and Shift-Share
=============================================================

This example demonstrates regional economic analysis using Location Quotient
and Shift-Share models to understand regional industry specialization and
employment growth patterns.

Author: KR-Labs
License: Apache 2.0
"""

import pandas as pd
import numpy as np
from krl_models.regional import LocationQuotientModel, ShiftShareModel
import matplotlib.pyplot as plt
import seaborn as sns


def generate_regional_data():
    """Generate sample regional employment data.
    
    Returns:
        DataFrame with regional employment by industry
    """
    np.random.seed(42)
    
    regions = ['California', 'Texas', 'New York', 'Florida', 'Illinois']
    industries = ['Technology', 'Healthcare', 'Manufacturing', 'Finance', 'Retail']
    years = [2015, 2020]
    
    data = []
    
    for year in years:
        for region in regions:
            for industry in industries:
                # Base employment
                base = {
                    'Technology': {'California': 500000, 'Texas': 200000, 'New York': 300000, 
                                 'Florida': 150000, 'Illinois': 100000},
                    'Healthcare': {'California': 400000, 'Texas': 350000, 'New York': 380000, 
                                 'Florida': 320000, 'Illinois': 280000},
                    'Manufacturing': {'California': 300000, 'Texas': 400000, 'New York': 200000, 
                                    'Florida': 150000, 'Illinois': 350000},
                    'Finance': {'California': 250000, 'Texas': 180000, 'New York': 450000, 
                              'Florida': 200000, 'Illinois': 220000},
                    'Retail': {'California': 380000, 'Texas': 320000, 'New York': 350000, 
                             'Florida': 300000, 'Illinois': 280000}
                }
                
                employment = base[industry][region]
                
                # Growth from 2015 to 2020
                if year == 2020:
                    if industry == 'Technology':
                        growth_rate = 0.25 if region == 'California' else 0.15
                    elif industry == 'Manufacturing':
                        growth_rate = -0.05
                    else:
                        growth_rate = 0.08
                    
                    employment = employment * (1 + growth_rate)
                
                # Add noise
                employment += np.random.normal(0, employment * 0.02)
                
                data.append({
                    'year': year,
                    'region': region,
                    'industry': industry,
                    'employment': int(employment)
                })
    
    return pd.DataFrame(data)


def main():
    """Main execution function."""
    
    print("=" * 75)
    print("Regional Economic Analysis Example")
    print("Location Quotient and Shift-Share Analysis")
    print("=" * 75)
    print()
    
    # 1. Generate data
    print("Step 1: Generating regional employment data...")
    data = generate_regional_data()
    print(f"   Generated data for {data['region'].nunique()} regions")
    print(f"   Covering {data['industry'].nunique()} industries")
    print(f"   Time period: {data['year'].min()} to {data['year'].max()}")
    print()
    
    # Display sample data
    print("   Sample data:")
    print(data.head(10))
    print()
    
    # 2. Location Quotient Analysis
    print("Step 2: Calculating Location Quotients...")
    print("=" * 75)
    
    lq_data = data[data['year'] == 2020]  # Most recent year
    
    lq_model = LocationQuotientModel(
        time_col='year',
        region_col='region',
        industry_col='industry',
        value_col='employment'
    )
    
    lq_results = lq_model.fit(lq_data)
    print("   Location Quotients calculated successfully!")
    print()
    
    # Display LQ results
    print("   Location Quotients by Region and Industry:")
    print("   (LQ > 1.25 indicates significant specialization)")
    print()
    
    lq_matrix = lq_results.location_quotients.pivot(
        index='region', 
        columns='industry', 
        values='lq'
    )
    print(lq_matrix.round(2))
    print()
    
    # Identify specializations
    specializations = lq_results.location_quotients[
        lq_results.location_quotients['lq'] > 1.25
    ].sort_values('lq', ascending=False)
    
    print("   Regional Specializations (LQ > 1.25):")
    for _, row in specializations.iterrows():
        print(f"   {row['region']:12s} - {row['industry']:15s}: LQ = {row['lq']:.2f}")
    print()
    
    # 3. Shift-Share Analysis
    print("Step 3: Performing Shift-Share Analysis...")
    print("=" * 75)
    
    ss_model = ShiftShareModel(
        time_col='year',
        region_col='region',
        industry_col='industry',
        value_col='employment',
        base_period=2015,
        comparison_period=2020
    )
    
    ss_results = ss_model.fit(data)
    print("   Shift-Share analysis completed!")
    print()
    
    # Display shift-share components
    print("   Employment Change Decomposition by Region:")
    print("   Components: National Share | Industry Mix | Regional Shift")
    print()
    
    for region in data['region'].unique():
        region_data = ss_results.growth_components[
            ss_results.growth_components['region'] == region
        ]
        
        total_change = region_data['total_change'].sum()
        national_share = region_data['national_share'].sum()
        industry_mix = region_data['industry_mix'].sum()
        regional_shift = region_data['regional_shift'].sum()
        
        print(f"   {region}:")
        print(f"      Total Change: {total_change:10,.0f}")
        print(f"      National Share: {national_share:10,.0f}")
        print(f"      Industry Mix: {industry_mix:10,.0f}")
        print(f"      Regional Shift: {regional_shift:10,.0f}")
        print()
    
    # 4. Visualizations
    print("Step 4: Creating visualizations...")
    create_visualizations(lq_matrix, ss_results)
    print("   Visualizations saved!")
    print()
    
    # 5. Regional insights
    print("Step 5: Key Insights")
    print("=" * 75)
    generate_insights(lq_results, ss_results, data)
    print()
    
    print("=" * 75)
    print("Example completed successfully!")
    print("=" * 75)


def create_visualizations(lq_matrix, ss_results):
    """Create visualizations for regional analysis.
    
    Args:
        lq_matrix: Location quotient matrix
        ss_results: Shift-share results
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Location Quotient Heatmap
    ax1 = axes[0]
    sns.heatmap(lq_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=1.0, vmin=0.5, vmax=1.5, ax=ax1, cbar_kws={'label': 'Location Quotient'})
    ax1.set_title('Location Quotients by Region and Industry (2020)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Industry', fontsize=12)
    ax1.set_ylabel('Region', fontsize=12)
    
    # Plot 2: Shift-Share Components
    ax2 = axes[1]
    
    # Aggregate by region
    ss_summary = ss_results.growth_components.groupby('region').agg({
        'national_share': 'sum',
        'industry_mix': 'sum',
        'regional_shift': 'sum'
    }).reset_index()
    
    x = np.arange(len(ss_summary))
    width = 0.25
    
    ax2.bar(x - width, ss_summary['national_share'], width, label='National Share', color='steelblue')
    ax2.bar(x, ss_summary['industry_mix'], width, label='Industry Mix', color='coral')
    ax2.bar(x + width, ss_summary['regional_shift'], width, label='Regional Shift', color='seagreen')
    
    ax2.set_xlabel('Region', fontsize=12)
    ax2.set_ylabel('Employment Change', fontsize=12)
    ax2.set_title('Shift-Share Decomposition by Region (2015-2020)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ss_summary['region'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('regional_analysis_results.png', dpi=300, bbox_inches='tight')
    print("   Saved: regional_analysis_results.png")


def generate_insights(lq_results, ss_results, data):
    """Generate key insights from the analysis.
    
    Args:
        lq_results: Location quotient results
        ss_results: Shift-share results
        data: Original data
    """
    print("   Key Findings:")
    print()
    
    # Finding 1: Top specializations
    top_spec = lq_results.location_quotients.nlargest(3, 'lq')
    print("   1. Strongest Regional Specializations:")
    for _, row in top_spec.iterrows():
        print(f"      - {row['region']} in {row['industry']} (LQ: {row['lq']:.2f})")
    print()
    
    # Finding 2: Growth leaders
    growth_by_region = data.groupby('region').apply(
        lambda x: (x[x['year'] == 2020]['employment'].sum() / 
                  x[x['year'] == 2015]['employment'].sum() - 1) * 100
    ).sort_values(ascending=False)
    
    print("   2. Fastest Growing Regions (2015-2020):")
    for region, growth in growth_by_region.head(3).items():
        print(f"      - {region}: {growth:.1f}% growth")
    print()
    
    # Finding 3: Industry trends
    growth_by_industry = data.groupby('industry').apply(
        lambda x: (x[x['year'] == 2020]['employment'].sum() / 
                  x[x['year'] == 2015]['employment'].sum() - 1) * 100
    ).sort_values(ascending=False)
    
    print("   3. Industry Performance:")
    for industry, growth in growth_by_industry.items():
        trend = "Growing" if growth > 0 else "Declining"
        print(f"      - {industry}: {growth:.1f}% ({trend})")
    print()
    
    # Finding 4: Competitive advantages
    positive_shifts = ss_results.growth_components[
        ss_results.growth_components['regional_shift'] > 0
    ].groupby('region')['regional_shift'].sum().sort_values(ascending=False)
    
    print("   4. Regions with Competitive Advantages:")
    print("      (Positive Regional Shift Component)")
    for region, shift in positive_shifts.head(3).items():
        print(f"      - {region}: +{shift:,.0f} jobs from competitive advantage")


if __name__ == '__main__':
    main()
