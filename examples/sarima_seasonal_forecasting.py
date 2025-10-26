# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
SARIMA Seasonal Forecasting Example
====================================

This example demonstrates seasonal time series forecasting using SARIMA
(Seasonal ARIMA) for monthly retail sales data.

Author: KR-Labs
License: Apache 2.0
"""

import pandas as pd
import numpy as np
from krl_models.econometric import SARIMAModel
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def generate_seasonal_data(n_periods=144):
    """Generate sample seasonal time series data.
    
    Args:
        n_periods: Number of time periods to generate (months)
        
    Returns:
        DataFrame with date and sales columns
    """
    np.random.seed(42)
    
    # Time index
    t = np.arange(n_periods)
    
    # Components
    trend = t * 0.3  # Linear trend
    seasonal = 15 * np.sin(2 * np.pi * t / 12)  # Yearly seasonality
    noise = np.random.normal(0, 5, n_periods)
    
    # Combine components
    sales = 100 + trend + seasonal + noise
    
    dates = pd.date_range('2012-01-01', periods=n_periods, freq='M')
    
    return pd.DataFrame({
        'date': dates,
        'sales': sales
    })


def plot_seasonal_decomposition(data):
    """Plot ACF and PACF to examine seasonal patterns.
    
    Args:
        data: Time series data
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF plot
    plot_acf(data['sales'], lags=36, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
    
    # PACF plot
    plot_pacf(data['sales'], lags=36, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('seasonal_acf_pacf.png', dpi=300, bbox_inches='tight')
    print("   ACF/PACF plots saved as: seasonal_acf_pacf.png")


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("SARIMA Seasonal Forecasting Example")
    print("=" * 70)
    print()
    
    # 1. Generate seasonal data
    print("Step 1: Generating seasonal time series data (monthly sales)...")
    data = generate_seasonal_data(n_periods=144)
    print(f"   Generated {len(data)} months of data")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"   Mean sales: {data['sales'].mean():.2f}")
    print(f"   Std sales: {data['sales'].std():.2f}")
    print()
    
    # 2. Visualize ACF/PACF
    print("Step 2: Analyzing seasonal patterns...")
    plot_seasonal_decomposition(data)
    print()
    
    # 3. Split data
    print("Step 3: Splitting data into train/test sets...")
    train_size = 132  # 11 years
    train_data = data[:train_size]
    test_data = data[train_size:]
    print(f"   Training set: {len(train_data)} months")
    print(f"   Test set: {len(test_data)} months")
    print()
    
    # 4. Initialize SARIMA model
    print("Step 4: Initializing SARIMA model...")
    print("   Model specification:")
    print("   - Non-seasonal: ARIMA(1,1,1)")
    print("   - Seasonal: (1,1,1,12) - monthly seasonality")
    print()
    
    model = SARIMAModel(
        time_col='date',
        target_col='sales',
        order=(1, 1, 1),              # Non-seasonal (p,d,q)
        seasonal_order=(1, 1, 1, 12)  # Seasonal (P,D,Q,s)
    )
    print("   Model initialized successfully!")
    print()
    
    # 5. Fit model
    print("Step 5: Fitting SARIMA model to training data...")
    results = model.fit(train_data)
    print("   Model fitted successfully!")
    print()
    
    # 6. Model diagnostics
    print("Step 6: Running model diagnostics...")
    try:
        diagnostics = results.diagnostics()
        print("   Diagnostics completed")
    except:
        print("   Using standard diagnostic methods")
    print()
    
    # 7. Generate forecast
    print("Step 7: Generating 12-month forecast...")
    forecast = model.predict(steps=12)
    print("   Forecast generated!")
    print()
    
    print("   Monthly Forecast:")
    for idx, row in forecast.iterrows():
        print(f"   Month {idx + 1}: {row['forecast']:.2f}")
    print()
    
    # 8. Calculate accuracy
    print("Step 8: Evaluating forecast accuracy...")
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    actual = test_data['sales'].values
    predicted = forecast['forecast'].values
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print(f"   Mean Absolute Error (MAE): {mae:.2f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"   Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print()
    
    # 9. Visualize results
    print("Step 9: Creating visualizations...")
    visualize_results(data, train_data, test_data, forecast, results)
    print("   Visualizations complete!")
    print()
    
    # 10. Seasonal insights
    print("Step 10: Analyzing seasonal patterns...")
    analyze_seasonality(data, forecast)
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


def visualize_results(data, train_data, test_data, forecast, results):
    """Create comprehensive visualization of results.
    
    Args:
        data: Full dataset
        train_data: Training data
        test_data: Test data
        forecast: Forecast results
        results: Model results
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Full time series
    ax1 = axes[0]
    ax1.plot(data['date'], data['sales'], label='Actual Sales', 
             color='blue', linewidth=1.5, alpha=0.7)
    split_date = train_data['date'].iloc[-1]
    ax1.axvline(x=split_date, color='red', linestyle='--', 
                label='Train/Test Split', linewidth=2)
    ax1.set_title('Complete Sales Time Series', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training data with fitted values
    ax2 = axes[1]
    ax2.plot(train_data['date'], train_data['sales'], 
             label='Actual', color='blue', linewidth=2)
    ax2.plot(train_data['date'], results.fitted_values, 
             label='Fitted', color='orange', linestyle='--', linewidth=1.5)
    ax2.set_title('Training Period: Actual vs Fitted', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sales')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Forecast vs actual
    ax3 = axes[2]
    
    # Recent history
    recent = train_data.tail(24)
    ax3.plot(recent['date'], recent['sales'], 
             label='Recent History', color='blue', linewidth=2)
    
    # Actual test data
    ax3.plot(test_data['date'], test_data['sales'], 
             label='Actual Test', color='green', linewidth=2, marker='o', markersize=6)
    
    # Forecast
    forecast_dates = test_data['date'].values
    ax3.plot(forecast_dates, forecast['forecast'].values, 
             label='SARIMA Forecast', color='red', linestyle='--', linewidth=2, marker='s', markersize=6)
    
    ax3.set_title('12-Month Forecast vs Actual', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Sales')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sarima_forecast_results.png', dpi=300, bbox_inches='tight')
    print("   Results saved as: sarima_forecast_results.png")


def analyze_seasonality(data, forecast):
    """Analyze and display seasonal patterns.
    
    Args:
        data: Historical data
        forecast: Forecast data
    """
    # Calculate average sales by month
    data['month'] = pd.to_datetime(data['date']).dt.month
    monthly_avg = data.groupby('month')['sales'].mean()
    
    print("   Average sales by month:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month, avg in monthly_avg.items():
        print(f"   {month_names[month-1]}: {avg:.2f}")
    
    # Identify peak and low months
    peak_month = monthly_avg.idxmax()
    low_month = monthly_avg.idxmin()
    
    print()
    print(f"   Peak sales month: {month_names[peak_month-1]} ({monthly_avg[peak_month]:.2f})")
    print(f"   Lowest sales month: {month_names[low_month-1]} ({monthly_avg[low_month]:.2f})")
    print(f"   Seasonal range: {monthly_avg[peak_month] - monthly_avg[low_month]:.2f}")


if __name__ == '__main__':
    main()
