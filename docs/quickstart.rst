Quick Start Guide
=================

Get started with KRL Model Zoo in under 5 minutes.

.. important::
   **KRL Model Zoo integrates seamlessly with KRL Data Connectors** for automatic 
   provenance tracking and validated access to federal datasets (BLS, FRED, Census, 
   CDC, HUD). This is the **recommended approach** for all analyses using public data.

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

Install KRL Model Zoo and the data connector ecosystem:

.. code-block:: bash

   # Install model zoo
   pip install krl-model-zoo
   
   # Install data connectors (HIGHLY RECOMMENDED)
   pip install krl-data-connectors

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/KR-Labs/krl-model-zoo.git
   cd krl-model-zoo
   pip install -e ".[dev,test,docs]"

Verify Installation
~~~~~~~~~~~~~~~~~~~

Verify your installation:

.. code-block:: python

   import krl_models
   import krl_data_connectors
   print(krl_models.__version__)  # Should print: 1.0.0

KRL Ecosystem Architecture
---------------------------

The KRL ecosystem provides an integrated, reproducible analytics pipeline:

.. code-block:: text

   Federal Data (BLS/FRED/Census/CDC/HUD)
              ↓
   KRL Data Connectors (automatic provenance)
              ↓
   KRL Core (BaseModel, ModelInputSchema)
              ↓
   KRL Model Zoo (18 production models)
              ↓
   Results (with full provenance chain)

Your First Model
----------------

**Recommended Workflow**: KRL Data Connectors + KRL Model Zoo

.. note::
   **ARIMA Implementation**: KRL Model Zoo implements ARIMA through ``SARIMAModel`` 
   with ``seasonal_order=(0,0,0,0)``. When seasonal components are zero, SARIMA 
   reduces to ARIMA. A standalone ARIMAModel reference implementation exists in 
   ``examples/example_arima_run.py`` from Gate 1.

Step 1: Fetch Data with KRL Data Connectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_data_connectors import BLSConnector
   
   # Fetch BLS unemployment data with automatic provenance
   bls = BLSConnector()
   unemployment_df = bls.get_series(
       series_id='LNS14000000',  # U.S. unemployment rate
       start_year=2010,
       end_year=2024
   )
   
   # Data includes: date, value, and metadata
   print(unemployment_df.head())
   
   # Get provenance automatically
   provenance = bls.get_provenance('LNS14000000')
   print(f"Source: {provenance.source_name}")

Step 2: Create Input Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_core import ModelInputSchema, ModelMeta
   
   # Create input schema with provenance tracking
   input_schema = ModelInputSchema(
       entity="US",
       metric="unemployment_rate",
       time_index=unemployment_df['date'].tolist(),
       values=unemployment_df['value'].tolist(),
       provenance=provenance,  # Automatic from connector
       frequency='M'
   )

Step 3: Initialize and Fit Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_models.econometric import SARIMAModel
   
   # ARIMA(1,1,1) via SARIMA with no seasonal components
   model = SARIMAModel(
       input_schema=input_schema,
       params={
           'order': (1, 1, 1),
           'seasonal_order': (0, 0, 0, 0),  # No seasonality = ARIMA
           'trend': 'c'
       },
       meta=ModelMeta(
           name="UnemploymentForecast",
           version="1.0.0",
           author="YourName"
       )
   )
   
   # Fit the model
   results = model.fit()
   print(f"AIC: {results.payload['aic']:.2f}")
   print(f"BIC: {results.payload['bic']:.2f}")

Step 4: Generate Forecasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Forecast 12 months ahead with 95% confidence intervals
   forecast = model.predict(steps=12, alpha=0.05)
   
   print("Forecast values:", forecast.forecast_values)
   print("95% CI lower:", forecast.ci_lower)
   print("95% CI upper:", forecast.ci_upper)

Step 5: Visualize Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_core import PlotlySchemaAdapter
   
   # Create interactive Plotly visualization
   adapter = PlotlySchemaAdapter()
   fig = adapter.forecast_plot(
       forecast,
       title="U.S. Unemployment Rate Forecast",
       show_provenance=True  # Displays data source
   )
   
   fig.show()  # Opens in browser
   
   # Or export static image
   fig.write_image('unemployment_forecast.png')

Alternative: Custom Data Import
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For proprietary or non-federal data sources:

.. code-block:: python

   import pandas as pd
   from krl_core import ModelInputSchema, Provenance
   from datetime import datetime
   
   # Load your CSV or database
   df = pd.read_csv('my_custom_data.csv')
   df['date'] = pd.to_datetime(df['date'])
   
   # Create input schema manually
   input_schema = ModelInputSchema(
       entity="MyRegion",
       metric="custom_metric",
       time_index=df['date'].dt.strftime('%Y-%m').tolist(),
       values=df['value'].tolist(),
       provenance=Provenance(
           source_name="CustomDatabase",
           series_id="custom_001",
           collection_date=datetime.now(),
           transformation="cleaned and interpolated"
       ),
       frequency='M'
   )
   
   # Use with any model as shown above...

Complete Workflow Example
--------------------------

Here's a complete multivariate analysis using multiple data sources:

.. code-block:: python

   # 1. Fetch data from multiple sources using KRL Data Connectors
   from krl_data_connectors import BLSConnector, FREDConnector
   from krl_models.econometric import VARModel
   from krl_core import ModelInputSchema, ModelMeta
   
   # Fetch unemployment from BLS
   bls = BLSConnector()
   unemployment = bls.get_series('LNS14000000', 2010, 2024)
   
   # Fetch GDP growth from FRED
   fred = FREDConnector(api_key='your_fred_api_key')
   gdp = fred.get_series('GDP', '2010-01-01', '2024-12-31')
   
   # 2. Merge data on time index
   import pandas as pd
   merged = unemployment.merge(gdp, on='date', suffixes=('_unemp', '_gdp'))
   
   # 3. Create multivariate input schema
   input_schema = ModelInputSchema(
       entity="US_Economy",
       metric="unemployment_gdp",
       time_index=merged['date'].tolist(),
       values=merged[['value_unemp', 'value_gdp']].values.tolist(),
       provenance=bls.get_provenance('LNS14000000'),
       frequency='Q'
   )
   
   # 4. Fit Vector Autoregression (VAR) model
   var_model = VARModel(
       input_schema=input_schema,
       params={'maxlags': 4},
       meta=ModelMeta(name="EconomicVAR", version="1.0.0")
   )
   
   results = var_model.fit()
   
   # 5. Generate forecasts
   forecast = var_model.predict(steps=8)  # 2 years quarterly
   
   # 6. Analyze Granger causality
   print("Granger Causality Tests:")
   print(results.payload['granger_causality'])
   
   # 7. Visualize and export with provenance
   from krl_core import PlotlySchemaAdapter
   
   adapter = PlotlySchemaAdapter()
   fig = adapter.forecast_plot(
       forecast,
       title="U.S. Economic Indicators Forecast",
       show_provenance=True
   )
   
   fig.show()
   
   # Export maintains full provenance chain
   results.export_to_csv('economic_forecast.csv')
   results.save_metadata('economic_forecast_meta.json')

Why Use KRL Data Connectors?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**You could** manually download CSVs from BLS, FRED, Census websites. However, 
KRL Data Connectors provide:

✅ **Automatic Provenance**: Every data point tracks its source  
✅ **Validated APIs**: Pre-configured for federal data sources  
✅ **Rate Limiting**: Respects API limits, prevents blocking  
✅ **Reproducibility**: SHA256 hashing ensures identical results  
✅ **Error Handling**: Graceful retries and clear error messages  
✅ **Time Savings**: No manual parsing or API debugging  

This is especially critical for policy-relevant research requiring audit trails.

Common Use Cases
----------------

Forecasting
~~~~~~~~~~~

Use ARIMA, SARIMA, or Prophet for time series forecasting:

.. code-block:: python

   from krl_models.econometric import ProphetModel
   
   model = ProphetModel(time_col='date', target_col='value')
   results = model.fit(data)
   forecast = model.predict(steps=30)

Anomaly Detection
~~~~~~~~~~~~~~~~~

Detect unusual patterns in your data:

.. code-block:: python

   from krl_models.anomaly import STLAnomalyModel
   
   model = STLAnomalyModel(
       time_col='date',
       value_col='value',
       seasonal_period=12,
       threshold=3.0
   )
   
   results = model.fit(data)
   anomalies = model.get_anomaly_summary()
   print(anomalies)

Regional Analysis
~~~~~~~~~~~~~~~~~

Analyze regional economic specialization:

.. code-block:: python

   from krl_models.regional import LocationQuotientModel
   
   model = LocationQuotientModel(
       region_col='county',
       industry_col='naics_code',
       employment_col='employment'
   )
   
   results = model.fit(regional_data)
   specialized_industries = results.get_specializations(threshold=1.25)

Volatility Modeling
~~~~~~~~~~~~~~~~~~~

Model conditional heteroskedasticity:

.. code-block:: python

   from krl_models.volatility import GARCHModel
   
   model = GARCHModel(
       time_col='date',
       returns_col='returns',
       p=1, q=1  # GARCH(1,1)
   )
   
   results = model.fit(financial_data)
   volatility_forecast = model.predict(steps=10)

Next Steps
----------

* Read the :doc:`installation` guide for advanced setup options
* Explore :doc:`examples` for more detailed use cases
* Check the :doc:`api/index` for complete API documentation
* Learn about specific model families in the User Guide
* Join our community and :doc:`contributing` to the project

Need Help?
----------

* **Documentation**: https://krl-model-zoo.readthedocs.io
* **GitHub Issues**: https://github.com/KR-Labs/krl-model-zoo/issues
* **Email**: info@krlabs.dev

Happy modeling! 🚀
