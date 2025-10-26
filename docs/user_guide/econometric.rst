Econometric Models Guide
========================

Detailed guide for using classical and modern econometric models for time series analysis.

Introduction
------------

Econometric models are the foundation of time series forecasting in economics. This guide covers:

* When to use each model type
* How to specify and estimate models
* Diagnostic testing and validation
* Practical examples with real data

ARIMA Models
------------

AutoRegressive Integrated Moving Average (ARIMA) models are versatile for stationary or differenced time series.

.. note::
   **Implementation Note**: In KRL Model Zoo, ARIMA is implemented in two ways:
   
   1. **Reference Implementation**: ``examples/example_arima_run.py`` - Standalone ARIMAModel 
      class from Gate 1, demonstrating full KRL patterns and interfaces
   2. **Production Use**: ``SARIMAModel`` with ``seasonal_order=(0,0,0,0)`` - SARIMA reduces 
      to ARIMA when no seasonal components are specified (default behavior)
   
   Both implementations wrap statsmodels and provide identical functionality. Use SARIMAModel 
   for production code as it handles both ARIMA and seasonal cases.

Model Specification
~~~~~~~~~~~~~~~~~~~

An ARIMA(p, d, q) model has:

* **p**: AR order (number of lagged values)
* **d**: Differencing order (to achieve stationarity)
* **q**: MA order (number of lagged forecast errors)

When to Use ARIMA
~~~~~~~~~~~~~~~~~

Use ARIMA when:

* Data shows autocorrelation
* Series is stationary or can be made stationary by differencing
* No strong seasonal pattern (use SARIMA instead)
* Need interpretable parameters

Basic Example
~~~~~~~~~~~~~

**Recommended Approach**: Use KRL Data Connectors + SARIMAModel

.. code-block:: python

   from krl_data_connectors import BLSConnector
   from krl_models.econometric import SARIMAModel
   from krl_core import ModelInputSchema, ModelMeta
   
   # Fetch BLS unemployment data with automatic provenance
   bls = BLSConnector()
   unemployment_df = bls.get_series(
       series_id='LNS14000000',  # U.S. unemployment rate
       start_year=2010,
       end_year=2024
   )
   
   # Create input schema
   input_schema = ModelInputSchema(
       entity="US",
       metric="unemployment_rate",
       time_index=unemployment_df['date'].tolist(),
       values=unemployment_df['value'].tolist(),
       provenance=bls.get_provenance('LNS14000000'),
       frequency='M'
   )
   
   # ARIMA(1,1,1) via SARIMA with no seasonal components
   model = SARIMAModel(
       input_schema=input_schema,
       params={
           'order': (1, 1, 1),
           'seasonal_order': (0, 0, 0, 0),  # No seasonality = ARIMA
           'trend': 'c'  # Include constant
       },
       meta=ModelMeta(
           name="UnemploymentARIMA",
           version="1.0.0",
           author="YourName"
       )
   )
   
   # Fit model
   results = model.fit()
   
   # Check diagnostics
   print(f"AIC: {results.payload['aic']:.2f}")
   print(f"BIC: {results.payload['bic']:.2f}")
   
   # Generate 12-month forecast with 95% confidence intervals
   forecast = model.predict(steps=12, alpha=0.05)
   print(f"Forecast: {forecast.forecast_values}")
   print(f"95% CI: {list(zip(forecast.ci_lower, forecast.ci_upper))}")

**Alternative**: Import your own data (for custom datasets)

.. code-block:: python

   import pandas as pd
   from krl_core import ModelInputSchema, Provenance
   from datetime import datetime
   
   # Load your CSV
   data = pd.read_csv('custom_unemployment.csv')
   data['date'] = pd.to_datetime(data['date'])
   
   input_schema = ModelInputSchema(
       entity="CustomRegion",
       metric="unemployment_rate",
       time_index=data['date'].dt.strftime('%Y-%m').tolist(),
       values=data['unemployment_rate'].tolist(),
       provenance=Provenance(
           source_name="CustomSource",
           series_id="unemp_001",
           collection_date=datetime.now(),
           transformation="raw"
       ),
       frequency='M'
   )
   
   # Continue with SARIMAModel as above...

Order Selection
~~~~~~~~~~~~~~~

Use ACF and PACF plots to guide order selection:

.. code-block:: python

   from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
   import matplotlib.pyplot as plt
   
   # Plot ACF and PACF
   fig, axes = plt.subplots(2, 1, figsize=(12, 8))
   plot_acf(data['unemployment_rate'], lags=24, ax=axes[0])
   plot_pacf(data['unemployment_rate'], lags=24, ax=axes[1])
   plt.tight_layout()
   plt.show()
   
   # ACF cuts off at lag q → suggests MA(q)
   # PACF cuts off at lag p → suggests AR(p)

Automated selection:

.. code-block:: python

   from pmdarima import auto_arima
   
   # Auto-select order using AIC
   auto_model = auto_arima(
       data['unemployment_rate'],
       seasonal=False,
       stepwise=True,
       information_criterion='aic',
       trace=True
   )
   
   print(f"Selected order: {auto_model.order}")

SARIMA Models
-------------

Seasonal ARIMA extends ARIMA with seasonal components.

Model Specification
~~~~~~~~~~~~~~~~~~~

SARIMA(p,d,q)(P,D,Q,s) adds seasonal terms:

* **P**: Seasonal AR order
* **D**: Seasonal differencing order
* **Q**: Seasonal MA order
* **s**: Seasonal period (12 for monthly, 4 for quarterly)

When to Use SARIMA
~~~~~~~~~~~~~~~~~~

Use SARIMA when:

* Clear seasonal pattern in data
* Seasonality is regular and predictable
* Monthly, quarterly, or other regular frequency data

Example: Monthly Sales
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_models.econometric import SARIMAModel
   
   # Monthly retail sales with seasonality
   model = SARIMAModel(
       time_col='date',
       target_col='retail_sales',
       order=(1, 1, 1),           # Non-seasonal
       seasonal_order=(1, 1, 1, 12),  # Seasonal (monthly)
       trend='c'
   )
   
   results = model.fit(sales_data)
   
   # 24-month forecast
   forecast = model.predict(steps=24)
   
   # Plot results
   results.plot()

Handling Exogenous Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Include external predictors:

.. code-block:: python

   # SARIMAX with exogenous variables
   model = SARIMAModel(
       time_col='date',
       target_col='sales',
       exog_cols=['gdp_growth', 'consumer_sentiment'],
       order=(1, 1, 1),
       seasonal_order=(1, 1, 1, 12)
   )
   
   results = model.fit(data)
   
   # Forecast requires future exogenous values
   future_exog = pd.DataFrame({
       'gdp_growth': [2.5, 2.6, 2.7],
       'consumer_sentiment': [95, 96, 97]
   })
   
   forecast = model.predict(steps=3, exog=future_exog)

VAR Models
----------

Vector AutoRegression for multivariate time series.

When to Use VAR
~~~~~~~~~~~~~~~

Use VAR when:

* Multiple related time series
* Interest in dynamic interactions
* All variables endogenous
* Granger causality testing

Example: Macro Variables
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_models.econometric import VARModel
   
   # Multiple economic indicators
   model = VARModel(
       time_col='date',
       target_cols=['gdp', 'unemployment', 'inflation'],
       maxlags=12,
       ic='aic'  # Information criterion for lag selection
   )
   
   results = model.fit(macro_data)
   
   # Forecast all variables
   forecast = model.predict(steps=8)
   print(forecast)

Impulse Response Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Study how shocks propagate:

.. code-block:: python

   # Impulse response functions
   irf = model.impulse_response(periods=20)
   
   # Plot IRFs
   irf.plot()
   
   # Cumulative effects
   cumulative_irf = irf.cum_effects
   print(cumulative_irf)

Granger Causality
~~~~~~~~~~~~~~~~~

Test if one variable helps predict another:

.. code-block:: python

   # Test if inflation Granger-causes unemployment
   causality_results = model.granger_causality(
       causing='inflation',
       caused='unemployment',
       maxlag=12
   )
   
   print(f"F-statistic: {causality_results.fvalue}")
   print(f"P-value: {causality_results.pvalue}")
   
   if causality_results.pvalue < 0.05:
       print("Inflation Granger-causes unemployment")

Cointegration
-------------

Test for long-run equilibrium relationships.

When to Use
~~~~~~~~~~~

Use cointegration when:

* Variables are non-stationary (unit roots)
* Suspect long-run relationship exists
* Want to model both short-run and long-run dynamics

Engle-Granger Method
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_models.econometric import CointegrationModel
   
   # Test housing price and income relationship
   model = CointegrationModel(
       time_col='date',
       target_cols=['housing_price', 'income'],
       method='engle-granger'
   )
   
   results = model.fit(data)
   
   print(f"Cointegration test statistic: {results.test_stat}")
   print(f"P-value: {results.pvalue}")
   print(f"Cointegrated: {results.is_cointegrated}")

Johansen Method
~~~~~~~~~~~~~~~

.. code-block:: python

   # Multiple variables
   model = CointegrationModel(
       time_col='date',
       target_cols=['housing_price', 'income', 'interest_rate'],
       method='johansen',
       det_order=1  # Constant in cointegrating relation
   )
   
   results = model.fit(data)
   
   print(f"Cointegration rank: {results.coint_rank}")
   print("Test statistics:")
   print(results.test_stats)

Vector Error Correction Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Estimate VECM given cointegration rank
   vecm_results = model.estimate_vecm(coint_rank=1)
   
   # Forecast with error correction
   forecast = vecm_results.predict(steps=12)
   
   # Error correction term
   ec_term = vecm_results.ec_term
   print("Error correction coefficient:", ec_term)

Prophet
-------

Facebook's Prophet for robust forecasting.

When to Use Prophet
~~~~~~~~~~~~~~~~~~~

Use Prophet when:

* Strong trend and seasonality
* Multiple seasonal patterns (yearly, weekly, daily)
* Many holidays or special events
* Missing data or outliers
* Need automatic forecasting with minimal tuning

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from krl_models.econometric import ProphetModel
   
   # Prophet uses 'ds' and 'y' conventions
   data_prophet = data.rename(columns={'date': 'ds', 'value': 'y'})
   
   model = ProphetModel(
       time_col='ds',
       target_col='y',
       growth='linear',  # or 'logistic' for saturating growth
       seasonality_mode='multiplicative',
       yearly_seasonality=True,
       weekly_seasonality=False,
       daily_seasonality=False
   )
   
   results = model.fit(data_prophet)
   forecast = model.predict(steps=365)  # One year daily forecast

Adding Custom Seasonality
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Add quarterly seasonality
   model = ProphetModel(time_col='ds', target_col='y')
   model.add_seasonality(
       name='quarterly',
       period=91.25,  # days
       fourier_order=5
   )
   
   # Add custom regressor
   model.add_regressor('marketing_spend')
   
   results = model.fit(data)

Handling Holidays
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   # Define holidays
   holidays = pd.DataFrame({
       'holiday': 'christmas',
       'ds': pd.to_datetime(['2020-12-25', '2021-12-25', '2022-12-25']),
       'lower_window': -2,  # 2 days before
       'upper_window': 2    # 2 days after
   })
   
   model = ProphetModel(
       time_col='ds',
       target_col='y',
       holidays=holidays
   )
   
   results = model.fit(data)

Diagnostics
-----------

Residual Analysis
~~~~~~~~~~~~~~~~~

Check model adequacy:

.. code-block:: python

   # Get residuals
   residuals = results.residuals
   
   # Plot residuals
   import matplotlib.pyplot as plt
   
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))
   
   # Time series plot
   axes[0, 0].plot(residuals)
   axes[0, 0].set_title('Residuals over time')
   
   # Histogram
   axes[0, 1].hist(residuals, bins=30)
   axes[0, 1].set_title('Residual distribution')
   
   # ACF
   from statsmodels.graphics.tsaplots import plot_acf
   plot_acf(residuals, lags=24, ax=axes[1, 0])
   
   # Q-Q plot
   from scipy import stats
   stats.probplot(residuals, dist="norm", plot=axes[1, 1])
   
   plt.tight_layout()
   plt.show()

Statistical Tests
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from statsmodels.stats.diagnostic import acorr_ljungbox
   from scipy.stats import jarque_bera
   from arch.univariate import arch_model
   
   # Ljung-Box test for autocorrelation
   lb_test = acorr_ljungbox(residuals, lags=20)
   print("Ljung-Box test:")
   print(lb_test)
   
   # Jarque-Bera test for normality
   jb_stat, jb_pvalue = jarque_bera(residuals)
   print(f"\nJarque-Bera test: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}")
   
   # ARCH test for heteroskedasticity
   arch_test = arch_model(residuals).fit()
   print("\nARCH test:")
   print(arch_test.summary())

Out-of-Sample Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.metrics import mean_squared_error, mean_absolute_error
   import numpy as np
   
   # Split data
   train = data[:-24]
   test = data[-24:]
   
   # Fit and forecast
   model = SARIMAModel(
       time_col='date',
       target_col='value',
       order=(1, 1, 1),
       seasonal_order=(1, 1, 1, 12)
   )
   
   results = model.fit(train)
   forecast = model.predict(steps=24)
   
   # Calculate metrics
   actual = test['value'].values
   predicted = forecast['forecast'].values
   
   mse = mean_squared_error(actual, predicted)
   rmse = np.sqrt(mse)
   mae = mean_absolute_error(actual, predicted)
   mape = np.mean(np.abs((actual - predicted) / actual)) * 100
   
   print(f"RMSE: {rmse:.4f}")
   print(f"MAE: {mae:.4f}")
   print(f"MAPE: {mape:.2f}%")

Advanced Topics
---------------

Structural Breaks
~~~~~~~~~~~~~~~~~

Test for and handle structural breaks:

.. code-block:: python

   from statsmodels.stats.diagnostic import breaks_cusumolsresid
   
   # CUSUM test for parameter stability
   cusum_test = breaks_cusumolsresid(results.resid)
   print(f"CUSUM test statistic: {cusum_test}")
   
   # Fit separate models for different regimes
   pre_break = data[data['date'] < '2020-03-01']
   post_break = data[data['date'] >= '2020-03-01']
   
   model1 = ARIMAModel(time_col='date', target_col='value', order=(1,1,1))
   results1 = model1.fit(pre_break)
   
   model2 = ARIMAModel(time_col='date', target_col='value', order=(1,1,1))
   results2 = model2.fit(post_break)

Intervention Analysis
~~~~~~~~~~~~~~~~~~~~~

Model known interventions:

.. code-block:: python

   # Create intervention dummy variable
   data['covid_impact'] = 0
   data.loc[data['date'] >= '2020-03-01', 'covid_impact'] = 1
   
   # Include as exogenous variable
   model = SARIMAModel(
       time_col='date',
       target_col='unemployment_rate',
       exog_cols=['covid_impact'],
       order=(1, 1, 1),
       seasonal_order=(1, 1, 1, 12)
   )
   
   results = model.fit(data)

See Also
--------

* :doc:`../api/econometric` - Complete API reference
* :doc:`volatility` - Volatility modeling
* :doc:`state_space` - State space methods
