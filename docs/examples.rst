.. Copyright (c) 2024 Sudiata Giddasira, Inc. d/b/a Quipu Research Labs, LLC d/b/a KR-Labs™
.. SPDX-License-Identifier: Apache-2.0

========
Examples
========

This page provides practical examples demonstrating how to use KRL Model Zoo 
for real-world analysis. For working Jupyter notebooks, see the ``examples/`` 
directory in the repository.

Basic Workflow
==============

Every model follows the same pattern:

1. **Prepare data** using ModelInputSchema (preferably via KRL Data Connectors)
2. **Initialize model** with parameters and metadata
3. **Fit model** to data
4. **Generate predictions** or analysis
5. **Visualize** and export results

Example 1: Unemployment Forecasting
====================================

Forecast U.S. unemployment using SARIMA with BLS data.

.. code-block:: python

   from krl_data_connectors import BLSConnector
   from krl_models.econometric import SARIMAModel
   from krl_core import ModelInputSchema, ModelMeta
   
   # Step 1: Fetch unemployment data
   bls = BLSConnector()
   unemployment = bls.get_series(
       series_id='LNS14000000',  # Civilian unemployment rate
       start_year=2010,
       end_year=2024
   )
   
   # Step 2: Create input schema with automatic provenance
   input_schema = ModelInputSchema(
       entity="US",
       metric="unemployment_rate",
       time_index=unemployment['date'].tolist(),
       values=unemployment['value'].tolist(),
       provenance=bls.get_provenance('LNS14000000'),
       frequency='M'
   )
   
   # Step 3: Initialize and fit SARIMA model
   model = SARIMAModel(
       input_schema=input_schema,
       params={
           'order': (2, 1, 2),           # AR, I, MA
           'seasonal_order': (1, 1, 1, 12)  # Seasonal ARIMA with 12-month cycle
       },
       meta=ModelMeta(
           name="US_Unemployment_Forecast",
           version="1.0.0",
           description="12-month ahead unemployment forecast"
       )
   )
   
   results = model.fit()
   print(f"Model fit - AIC: {results.metrics['aic']:.2f}")
   
   # Step 4: Generate forecast
   forecast = model.predict(steps=12, alpha=0.05)  # 12 months with 95% CI
   
   # Step 5: Visualize
   from krl_core import PlotlySchemaAdapter
   
   adapter = PlotlySchemaAdapter()
   fig = adapter.forecast_plot(
       forecast,
       title="U.S. Unemployment Rate Forecast",
       show_provenance=True
   )
   fig.show()
   
   # Export results
   forecast.export_to_csv('unemployment_forecast.csv')
   forecast.save_metadata('unemployment_metadata.json')

Example 2: Housing Price Volatility
====================================

Analyze housing price volatility using GARCH models.

.. code-block:: python

   from krl_data_connectors import FREDConnector
   from krl_models.volatility import GARCHModel, EGARCHModel
   from krl_core import ModelInputSchema, ModelMeta
   import pandas as pd
   
   # Fetch Case-Shiller home price index
   fred = FREDConnector(api_key='your_fred_api_key')
   prices = fred.get_series(
       series_id='CSUSHPISA',  # Case-Shiller U.S. National Home Price Index
       start_date='2000-01-01',
       end_date='2024-10-01'
   )
   
   # Calculate returns (GARCH models price changes, not levels)
   prices['returns'] = prices['value'].pct_change() * 100  # Percentage returns
   prices = prices.dropna()
   
   # Create input schema
   input_schema = ModelInputSchema(
       entity="US_Housing",
       metric="price_returns",
       time_index=prices['date'].tolist(),
       values=prices['returns'].tolist(),
       provenance=fred.get_provenance('CSUSHPISA'),
       frequency='M'
   )
   
   # Compare standard GARCH vs EGARCH (captures asymmetry)
   models = {
       'GARCH': GARCHModel(
           input_schema=input_schema,
           params={'p': 1, 'q': 1},
           meta=ModelMeta(name="Housing_GARCH")
       ),
       'EGARCH': EGARCHModel(
           input_schema=input_schema,
           params={'p': 1, 'q': 1},
           meta=ModelMeta(name="Housing_EGARCH")
       )
   }
   
   # Fit and compare
   for name, model in models.items():
       results = model.fit()
       print(f"{name} - AIC: {results.metrics['aic']:.2f}, "
             f"Log-Likelihood: {results.metrics['log_likelihood']:.2f}")
   
   # EGARCH captures leverage effect (bad news increases volatility more)
   egarch_results = models['EGARCH'].fit()
   print(f"Leverage parameter (gamma): {egarch_results.params.get('gamma', 'N/A')}")

Example 3: Regional Economic Specialization
============================================

Identify specialized industries using Location Quotient.

.. code-block:: python

   from krl_data_connectors import BLSConnector
   from krl_models.regional import LocationQuotientModel
   from krl_core import ModelInputSchema, ModelMeta
   import pandas as pd
   
   # Fetch employment data for region and nation
   bls = BLSConnector()
   
   # Regional employment (e.g., California tech sector)
   regional = bls.get_series('SMS06000005054200001', 2020, 2024)  # CA tech jobs
   
   # National employment (same sector)
   national = bls.get_series('SMS00000005054200001', 2020, 2024)  # US tech jobs
   
   # Combine datasets
   data = pd.DataFrame({
       'date': regional['date'],
       'regional_employment': regional['value'],
       'national_employment': national['value']
   })
   
   # Create input schema
   input_schema = ModelInputSchema(
       entity="California_Tech",
       metric="employment_specialization",
       time_index=data['date'].tolist(),
       values=data[['regional_employment', 'national_employment']].values.tolist(),
       provenance=bls.get_provenance('SMS06000005054200001'),
       frequency='M'
   )
   
   # Calculate Location Quotient
   model = LocationQuotientModel(
       input_schema=input_schema,
       params={},
       meta=ModelMeta(
           name="CA_Tech_Specialization",
           description="Tech sector specialization in California"
       )
   )
   
   results = model.fit()
   lq = results.metrics['location_quotient']
   
   # Interpretation:
   # LQ > 1: Region more specialized than nation
   # LQ = 1: Same as national average
   # LQ < 1: Region less specialized
   
   print(f"Location Quotient: {lq:.2f}")
   if lq > 1.25:
       print("California is significantly specialized in tech employment")

Example 4: Anomaly Detection in Crime Rates
============================================

Detect unusual spikes in crime data using STL decomposition.

.. code-block:: python

   from krl_models.anomaly import STLAnomalyModel
   from krl_core import ModelInputSchema, ModelMeta, Provenance
   import pandas as pd
   from datetime import datetime
   
   # Load crime data (example with custom data)
   crime_data = pd.read_csv('city_crime_monthly.csv')
   
   input_schema = ModelInputSchema(
       entity="City_X",
       metric="violent_crime_rate",
       time_index=crime_data['date'].tolist(),
       values=crime_data['crime_rate'].tolist(),
       provenance=Provenance(
           source_name="City Police Department",
           series_id="crime_monthly_001",
           collection_date=datetime.now(),
           transformation="per_100k_population"
       ),
       frequency='M'
   )
   
   # STL decomposes into trend, seasonal, and residual components
   model = STLAnomalyModel(
       input_schema=input_schema,
       params={
           'period': 12,  # Annual seasonality
           'threshold': 3.0  # Flag outliers > 3 std deviations
       },
       meta=ModelMeta(name="Crime_Anomaly_Detection")
   )
   
   results = model.fit()
   
   # Identify anomalies
   anomalies = results.metrics.get('anomalies', [])
   print(f"Detected {len(anomalies)} anomalous periods:")
   for anomaly in anomalies:
       print(f"  - {anomaly['date']}: {anomaly['value']:.2f} "
             f"(expected: {anomaly['expected']:.2f})")

Example 5: Multivariate Economic Forecasting
=============================================

Forecast multiple related economic indicators using VAR.

.. code-block:: python

   from krl_data_connectors import FREDConnector
   from krl_models.econometric import VARModel
   from krl_core import ModelInputSchema, ModelMeta
   import pandas as pd
   
   fred = FREDConnector(api_key='your_fred_api_key')
   
   # Fetch related economic indicators
   gdp = fred.get_series('GDP', '2000-01-01', '2024-01-01')
   unemployment = fred.get_series('UNRATE', '2000-01-01', '2024-01-01')
   inflation = fred.get_series('CPIAUCSL', '2000-01-01', '2024-01-01')
   
   # Align on common time index
   data = gdp.merge(unemployment, on='date', suffixes=('_gdp', '_unemp'))
   data = data.merge(inflation, on='date')
   data = data.rename(columns={'value': 'inflation'})
   
   # VAR expects multivariate time series
   values = data[['value_gdp', 'value_unemp', 'inflation']].values
   
   input_schema = ModelInputSchema(
       entity="US_Economy",
       metric="gdp_unemployment_inflation",
       time_index=data['date'].tolist(),
       values=values.tolist(),
       provenance=fred.get_provenance('GDP'),
       frequency='Q'
   )
   
   # Fit VAR model
   model = VARModel(
       input_schema=input_schema,
       params={'maxlags': 4},  # Automatic lag selection up to 4 quarters
       meta=ModelMeta(name="Economic_Indicators_VAR")
   )
   
   results = model.fit()
   print(f"Selected lag order: {results.metrics.get('lag_order', 'N/A')}")
   
   # Generate forecast
   forecast = model.predict(steps=8)  # 2 years ahead (quarterly)
   
   # Impulse response functions (how GDP shock affects unemployment)
   irf = results.metrics.get('impulse_response_functions', {})
   print("GDP shock impact on unemployment:", irf)

Example 6: Machine Learning for Poverty Prediction
===================================================

Predict poverty rates using machine learning with demographic features.

.. code-block:: python

   from krl_models.ml import XGBoostModel
   from krl_core import ModelInputSchema, ModelMeta, Provenance
   import pandas as pd
   from datetime import datetime
   
   # Load census tract data with features
   data = pd.read_csv('census_tracts.csv')
   
   # Features: median_income, education_rate, unemployment, housing_cost_burden
   # Target: poverty_rate
   
   features = ['median_income', 'education_rate', 'unemployment', 'housing_cost_burden']
   target = 'poverty_rate'
   
   X = data[features].values
   y = data[target].values
   
   input_schema = ModelInputSchema(
       entity="Census_Tracts",
       metric="poverty_prediction",
       time_index=None,  # Cross-sectional, not time series
       values=X.tolist(),
       target=y.tolist(),
       provenance=Provenance(
           source_name="U.S. Census Bureau ACS",
           series_id="census_acs_5yr",
           collection_date=datetime.now(),
           transformation="tract_level_aggregation"
       ),
       frequency=None
   )
   
   # Train XGBoost model
   model = XGBoostModel(
       input_schema=input_schema,
       params={
           'n_estimators': 100,
           'max_depth': 6,
           'learning_rate': 0.1,
           'objective': 'reg:squarederror'
       },
       meta=ModelMeta(name="Poverty_Prediction_XGB")
   )
   
   results = model.fit()
   print(f"R-squared: {results.metrics['r_squared']:.3f}")
   print(f"RMSE: {results.metrics['rmse']:.3f}")
   
   # Feature importance
   importance = results.metrics.get('feature_importance', {})
   print("\nTop predictors of poverty:")
   for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
       print(f"  {feature}: {score:.3f}")

Additional Examples
===================

For more examples including:

* Cointegration analysis for housing affordability
* Kalman filtering for real-time economic nowcasting
* Shift-share decomposition for regional job growth
* Prophet for seasonal tourism forecasting

See the ``examples/`` directory in the GitHub repository:

https://github.com/KR-Labs/krl-model-zoo/tree/main/examples

Each example includes:

* Complete working code
* Data sources and provenance
* Interpretation guidance
* Visualization examples
* Export formats

Next Steps
==========

* Read the :doc:`user_guide/index` for detailed model documentation
* Review :doc:`api/index` for complete API reference
* Check :doc:`contributing` to add your own examples
