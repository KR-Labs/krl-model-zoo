User Guide
==========

Comprehensive guides for using KRL Model Zoo models in your econometric analysis.

Overview
--------

The KRL Model Zoo provides production-ready econometric and machine learning models for economic analysis and forecasting. This user guide covers:

* **Model Selection**: Choosing the right model for your problem
* **Data Preparation**: Preparing data for modeling
* **Model Fitting**: Training models on your data
* **Forecasting**: Generating predictions
* **Diagnostics**: Validating model assumptions
* **Interpretation**: Understanding results

Getting Started
---------------

If you're new to KRL Model Zoo, start with:

1. :doc:`../installation` - Install the package
2. :doc:`../quickstart` - Quick start tutorial
3. Choose a model family guide below based on your needs

Model Family Guides
-------------------

.. toctree::
   :maxdepth: 2

   econometric
   regional
   anomaly
   volatility
   ml
   state_space

Choosing the Right Model
-------------------------

Time Series Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~

For univariate time series forecasting:

* **ARIMA/SARIMA**: Classical approach, good for stationary or seasonally differenced data
* **Prophet**: Robust to outliers, handles holidays and missing data automatically
* **XGBoost**: Non-linear relationships, feature-rich problems
* **Kalman Filter**: Handle missing data, extract underlying trends

For multivariate time series:

* **VAR**: Multiple related time series, study dynamic interactions
* **Cointegration**: Long-run equilibrium relationships

Regional Analysis
~~~~~~~~~~~~~~~~~

For spatial economic analysis:

* **Location Quotient**: Identify regional industry specializations
* **Shift-Share**: Decompose regional growth into structural components

Anomaly Detection
~~~~~~~~~~~~~~~~~

For outlier identification:

* **STL Anomaly**: Seasonal time series with trend
* **Isolation Forest**: Multivariate anomalies, non-linear patterns

Volatility Modeling
~~~~~~~~~~~~~~~~~~~

For variance forecasting:

* **GARCH**: Basic volatility clustering
* **EGARCH**: Asymmetric effects (leverage)
* **GJR-GARCH**: Threshold effects

Common Workflows
----------------

Basic Forecasting Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_models.econometric import SARIMAModel
   import pandas as pd
   
   # 1. Load and prepare data
   data = pd.read_csv('economic_data.csv')
   data['date'] = pd.to_datetime(data['date'])
   
   # 2. Initialize model
   model = SARIMAModel(
       time_col='date',
       target_col='unemployment_rate',
       order=(1, 1, 1),
       seasonal_order=(1, 1, 1, 12)
   )
   
   # 3. Fit model
   results = model.fit(data)
   
   # 4. Generate forecast
   forecast = model.predict(steps=12)
   
   # 5. Visualize
   results.plot()
   
   # 6. Export
   results.export_to_csv('forecast_results.csv')

Model Comparison Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_models.econometric import ARIMAModel, ProphetModel
   from krl_models.ml import XGBoostModel
   from sklearn.metrics import mean_squared_error
   
   # Split data
   train = data[:-12]
   test = data[-12:]
   
   # Fit multiple models
   models = {
       'ARIMA': ARIMAModel(time_col='date', target_col='value', order=(1,1,1)),
       'Prophet': ProphetModel(time_col='date', target_col='value'),
       'XGBoost': XGBoostModel(time_col='date', target_col='value', 
                               feature_cols=feature_list)
   }
   
   results = {}
   for name, model in models.items():
       model.fit(train)
       forecast = model.predict(steps=12)
       mse = mean_squared_error(test['value'], forecast['forecast'])
       results[name] = {'mse': mse, 'forecast': forecast}
   
   # Select best model
   best_model = min(results, key=lambda x: results[x]['mse'])
   print(f"Best model: {best_model} (MSE: {results[best_model]['mse']:.4f})")

Data Requirements
-----------------

Time Series Data Format
~~~~~~~~~~~~~~~~~~~~~~~

All models expect data in pandas DataFrame format with:

* **Time column**: datetime64 dtype with regular frequency
* **Target column**: numeric dtype (float or int)
* **Exogenous columns** (optional): numeric dtypes

Example:

.. code-block:: python

   import pandas as pd
   
   data = pd.DataFrame({
       'date': pd.date_range('2020-01-01', periods=100, freq='M'),
       'unemployment_rate': [5.0, 5.1, 5.2, ...],
       'gdp_growth': [2.5, 2.6, 2.4, ...],
       'inflation': [2.0, 2.1, 2.0, ...]
   })
   
   # Ensure datetime type
   data['date'] = pd.to_datetime(data['date'])
   
   # Check for missing values
   print(data.isnull().sum())
   
   # Check frequency
   print(pd.infer_freq(data['date']))

Handling Missing Data
~~~~~~~~~~~~~~~~~~~~~

Different models handle missing data differently:

* **ARIMA/SARIMA**: Requires complete data, use interpolation first
* **Prophet**: Handles missing data automatically
* **Kalman Filter**: Naturally accommodates missing observations
* **ML Models**: Drop or impute missing values

.. code-block:: python

   # Interpolation
   data['value'] = data['value'].interpolate(method='linear')
   
   # Forward fill
   data['value'] = data['value'].fillna(method='ffill')
   
   # Drop missing
   data = data.dropna()

Data Transformations
~~~~~~~~~~~~~~~~~~~~

Common transformations:

.. code-block:: python

   import numpy as np
   
   # Log transformation (for exponential growth)
   data['log_value'] = np.log(data['value'])
   
   # Differencing (for non-stationary data)
   data['diff_value'] = data['value'].diff()
   
   # Percentage change
   data['pct_change'] = data['value'].pct_change() * 100
   
   # Seasonal differencing
   data['seasonal_diff'] = data['value'].diff(12)  # For monthly data

Validation and Testing
----------------------

Train-Test Split
~~~~~~~~~~~~~~~~

Use temporal splits for time series:

.. code-block:: python

   # Simple split
   train_size = int(len(data) * 0.8)
   train = data[:train_size]
   test = data[train_size:]
   
   # Fixed window
   train = data[data['date'] < '2022-01-01']
   test = data[data['date'] >= '2022-01-01']

Cross-Validation
~~~~~~~~~~~~~~~~

Use time series cross-validation:

.. code-block:: python

   from sklearn.model_selection import TimeSeriesSplit
   
   tscv = TimeSeriesSplit(n_splits=5)
   
   scores = []
   for train_idx, test_idx in tscv.split(data):
       train = data.iloc[train_idx]
       test = data.iloc[test_idx]
       
       model = SomeModel(...)
       results = model.fit(train)
       forecast = model.predict(steps=len(test))
       
       mse = mean_squared_error(test['value'], forecast['forecast'])
       scores.append(mse)
   
   print(f"Average MSE: {np.mean(scores):.4f}")

Performance Metrics
~~~~~~~~~~~~~~~~~~~

Common metrics for model evaluation:

.. code-block:: python

   from sklearn.metrics import (
       mean_squared_error,
       mean_absolute_error,
       mean_absolute_percentage_error
   )
   
   # Calculate metrics
   mse = mean_squared_error(actual, forecast)
   rmse = np.sqrt(mse)
   mae = mean_absolute_error(actual, forecast)
   mape = mean_absolute_percentage_error(actual, forecast)
   
   print(f"RMSE: {rmse:.4f}")
   print(f"MAE: {mae:.4f}")
   print(f"MAPE: {mape:.2f}%")

Best Practices
--------------

Model Development
~~~~~~~~~~~~~~~~~

1. **Start simple**: Begin with basic models before trying complex ones
2. **Visualize data**: Plot time series to understand patterns
3. **Check stationarity**: Test for unit roots before ARIMA modeling
4. **Use multiple models**: Compare several approaches
5. **Validate thoroughly**: Use proper train-test splits and cross-validation

Production Deployment
~~~~~~~~~~~~~~~~~~~~~

1. **Version models**: Track model versions and parameters
2. **Monitor performance**: Regularly evaluate forecast accuracy
3. **Retrain periodically**: Update models with new data
4. **Document assumptions**: Record data preprocessing and transformations
5. **Handle errors**: Implement robust error handling

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Model won't converge:**

* Check data quality and missing values
* Try different starting values
* Reduce model complexity
* Increase maximum iterations

**Poor forecast accuracy:**

* Add more training data
* Include relevant exogenous variables
* Try different model specifications
* Check for structural breaks

**Computational performance:**

* Reduce data size for initial testing
* Use parallel processing where available
* Consider simpler models for large datasets
* Cache fitted models for reuse

See Also
--------

* :doc:`../api/index` - Complete API reference
* :doc:`../examples` - Detailed examples
* :doc:`../contributing` - Contributing guide
