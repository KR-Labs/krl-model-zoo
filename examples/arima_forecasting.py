"""
ARIMA Time Series Forecasting Example
======================================

This example demonstrates basic time series forecasting using the ARIMA model
from KRL Model Zoo.

Author: KR-Labs
License: Apache 2.0
"""

import pandas as pd
import numpy as np
from krl_models.econometric import ARIMAModel
import matplotlib.pyplot as plt


def generate_sample_data(n_periods=120):
    """Generate sample time series data with trend and noise.
    
    Args:
        n_periods: Number of time periods to generate
        
    Returns:
        DataFrame with date and value columns
    """
    np.random.seed(42)
    
    # Generate time series with trend and noise
    dates = pd.date_range('2015-01-01', periods=n_periods, freq='M')
    trend = np.arange(n_periods) * 0.5
    noise = np.random.normal(0, 5, n_periods)
    values = 100 + trend + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    })


def main():
    """Main execution function."""
    
    print("=" * 60)
    print("ARIMA Time Series Forecasting Example")
    print("=" * 60)
    print()
    
    # 1. Generate sample data
    print("Step 1: Generating sample time series data...")
    data = generate_sample_data(n_periods=120)
    print(f"   Generated {len(data)} monthly observations")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    print()
    
    # 2. Split into train and test
    print("Step 2: Splitting data into train/test sets...")
    train_size = 108  # 9 years training
    train_data = data[:train_size]
    test_data = data[train_size:]
    print(f"   Training set: {len(train_data)} observations")
    print(f"   Test set: {len(test_data)} observations")
    print()
    
    # 3. Initialize ARIMA model
    print("Step 3: Initializing ARIMA(1,1,1) model...")
    model = ARIMAModel(
        time_col='date',
        target_col='value',
        order=(1, 1, 1)  # (p, d, q)
    )
    print("   Model parameters:")
    print(f"   - AR order (p): 1")
    print(f"   - Differencing order (d): 1")
    print(f"   - MA order (q): 1")
    print()
    
    # 4. Fit model
    print("Step 4: Fitting model to training data...")
    results = model.fit(train_data)
    print("   Model fitted successfully!")
    print()
    
    # 5. Display model summary
    print("Step 5: Model Summary")
    print("-" * 60)
    try:
        print(results.summary())
    except:
        print("   Model summary available through results object")
    print()
    
    # 6. Generate forecast
    print("Step 6: Generating 12-month forecast...")
    forecast = model.predict(steps=12)
    print("   Forecast generated successfully!")
    print()
    
    print("   Forecast values:")
    print(forecast)
    print()
    
    # 7. Calculate forecast accuracy
    print("Step 7: Evaluating forecast accuracy...")
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    actual = test_data['value'].values
    predicted = forecast['forecast'].values
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    print(f"   Mean Absolute Error (MAE): {mae:.2f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"   Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print()
    
    # 8. Visualize results
    print("Step 8: Visualizing results...")
    visualize_results(train_data, test_data, forecast, results)
    print("   Visualization complete!")
    print()
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


def visualize_results(train_data, test_data, forecast, results):
    """Create visualization of model results.
    
    Args:
        train_data: Training dataset
        test_data: Test dataset
        forecast: Forecast results
        results: Model results object
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Training data with fitted values
    ax1 = axes[0]
    ax1.plot(train_data['date'], train_data['value'], 
             label='Actual', color='blue', linewidth=2)
    ax1.plot(train_data['date'], results.fitted_values, 
             label='Fitted', color='red', linestyle='--', linewidth=1.5)
    ax1.set_title('Training Data: Actual vs Fitted Values', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Forecast vs actual test data
    ax2 = axes[1]
    
    # Plot training data (last portion)
    train_tail = train_data.tail(24)
    ax2.plot(train_tail['date'], train_tail['value'], 
             label='Historical', color='blue', linewidth=2)
    
    # Plot test data
    ax2.plot(test_data['date'], test_data['value'], 
             label='Actual Test', color='green', linewidth=2, marker='o')
    
    # Plot forecast
    forecast_dates = test_data['date'].values
    ax2.plot(forecast_dates, forecast['forecast'].values, 
             label='Forecast', color='red', linestyle='--', linewidth=2, marker='s')
    
    ax2.set_title('Forecast vs Actual Test Data', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'arima_forecast_example.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Figure saved as: {output_path}")
    
    # Display (comment out if running non-interactively)
    # plt.show()


if __name__ == '__main__':
    main()
