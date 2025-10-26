.. Copyright (c) 2024 Sudiata Giddasira, Inc. d/b/a Quipu Research Labs, LLC d/b/a KR-Labs™
.. SPDX-License-Identifier: Apache-2.0

KRL Model Zoo Documentation
============================

Welcome to the **KRL Model Zoo** documentation!

The KRL Model Zoo is an open-source library of production-ready socioeconomic 
and econometric models designed for researchers, policymakers, analysts, and 
community organizations seeking to turn data into actionable intelligence.

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

.. image:: https://img.shields.io/badge/version-1.0.0-green.svg
   :target: https://github.com/KR-Labs/krl-model-zoo/releases
   :alt: Version

.. image:: https://readthedocs.org/projects/krl-model-zoo/badge/?version=latest
   :target: https://krl-model-zoo.readthedocs.io/en/latest/
   :alt: Documentation Status

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation
   examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/econometric
   user_guide/regional
   user_guide/anomaly
   user_guide/volatility
   user_guide/machine_learning
   user_guide/state_space

API Reference
-------------

Complete API documentation for all models and utilities.

.. toctree::
   :maxdepth: 2

   api/index
   api/econometric
   api/regional
   api/anomaly
   api/volatility
   api/ml
   api/state_space
   api/core

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   contributing
   development
   testing

.. toctree::
   :maxdepth: 1
   :caption: About

   changelog
   license
   citations

Overview
--------

The **KRL Model Zoo** provides modular, production-grade tools for:

- **Time Series Forecasting**: ARIMA, SARIMA, Prophet
- **Econometric Analysis**: VAR, Cointegration, Structural Breaks
- **Regional Analysis**: Location Quotient, Shift-Share
- **Anomaly Detection**: STL Decomposition, Isolation Forest
- **Volatility Modeling**: GARCH family models
- **Machine Learning**: Random Forest, XGBoost, Regularized Regression
- **State Space Models**: Kalman Filters, Local Level Models

All engineered with transparency, reproducibility, and accessibility in mind.

Key Features
------------

🎯 **18 Production-Ready Models**
   Covering time series, econometrics, ML, regional analysis, and anomaly detection

📊 **455+ Tests with 90% Coverage**
   Rigorous testing ensures reliability and correctness

🔗 **Seamless Data Integration**
   Native compatibility with KRL Data Connectors for federal datasets (BLS, Census, FRED, CDC, HUD)

📖 **Comprehensive Documentation**
   Mathematical formulations, usage guides, and working examples

🧪 **Research-Grade Quality**
   Field-tested for policy analysis, academic research, and community development

🌐 **Open Source**
   Apache 2.0 licensed for maximum accessibility

Quick Start
-----------

Installation
~~~~~~~~~~~~

Install via pip:

.. code-block:: bash

   pip install krl-model-zoo

Or install from source:

.. code-block:: bash

   git clone https://github.com/KR-Labs/krl-model-zoo.git
   cd krl-model-zoo
   pip install -e .

Basic Usage
~~~~~~~~~~~

**Recommended**: Use KRL Data Connectors for data access:

.. code-block:: python

   # Recommended: Use KRL Data Connectors for federal datasets
   from krl_data_connectors import BLSConnector
   from krl_models.econometric import SARIMAModel
   from krl_core import ModelInputSchema, ModelMeta
   
   # Fetch BLS unemployment data
   bls = BLSConnector()
   unemployment_df = bls.get_series(
       series_id='LNS14000000',
       start_year=2020,
       end_year=2024
   )
   
   # Create input schema with provenance
   input_schema = ModelInputSchema(
       entity="US",
       metric="unemployment_rate",
       time_index=unemployment_df['date'].tolist(),
       values=unemployment_df['value'].tolist(),
       provenance=bls.get_provenance(series_id='LNS14000000'),
       frequency='M'
   )
   
   # Note: SARIMA with seasonal_order=(0,0,0,0) is equivalent to ARIMA
   model = SARIMAModel(
       input_schema=input_schema,
       params={'order': (1, 1, 1)},
       meta=ModelMeta(name="UnemploymentForecast", version="1.0.0")
   )
   
   results = model.fit()
   forecast = model.predict(steps=12, alpha=0.05)

**Alternative**: Import your own data (less recommended for federal datasets):

.. code-block:: python

   # For custom/proprietary data only
   import pandas as pd
   from krl_core import ModelInputSchema, Provenance
   from datetime import datetime
   
   # Load your CSV or database
   df = pd.read_csv('my_data.csv')
   
   input_schema = ModelInputSchema(
       entity="MyRegion",
       metric="custom_metric",
       time_index=df['date'].tolist(),
       values=df['value'].tolist(),
       provenance=Provenance(
           source_name="CustomSource",
           series_id="custom_001",
           collection_date=datetime.now(),
           transformation="raw"
       ),
       frequency='M'
   )
   
   # Continue with model as above...

Complete Workflow with Data Connectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**KRL Model Zoo integrates seamlessly with KRL Data Connectors**, providing 
automatic provenance tracking, data validation, and standardized access to 
federal datasets (BLS, Census, FRED, CDC, HUD).

.. code-block:: python

   # Step 1: Fetch data using KRL Data Connectors (RECOMMENDED)
   from krl_data_connectors import BLSConnector, FREDConnector
   from krl_models.econometric import VARModel
   from krl_core import ModelInputSchema, ModelMeta
   
   # Connect to multiple data sources
   bls = BLSConnector()
   fred = FREDConnector(api_key='your_fred_api_key')
   
   # Fetch related economic indicators
   unemployment = bls.get_series('LNS14000000', 2010, 2024)
   gdp = fred.get_series('GDP', '2010-01-01', '2024-12-31')
   
   # Step 2: Apply multivariate model from KRL Model Zoo
   # VAR requires multiple time series aligned on time index
   combined_df = unemployment.merge(gdp, on='date')
   
   model = VARModel(
       input_schema=ModelInputSchema(
           entity="US_Economy",
           metric="unemployment_gdp",
           time_index=combined_df['date'].tolist(),
           values=combined_df[['unemployment', 'gdp']].values.tolist(),
           provenance=bls.get_provenance('LNS14000000'),
           frequency='Q'
       ),
       params={'maxlags': 4},
       meta=ModelMeta(name="EconomicVAR", version="1.0.0")
   )
   
   results = model.fit()
   forecast = model.predict(steps=8)  # 2 years quarterly
   
   # Step 3: Visualize and export with provenance preserved
   from krl_core import PlotlySchemaAdapter
   
   adapter = PlotlySchemaAdapter()
   fig = adapter.forecast_plot(
       forecast,
       title="U.S. Economic Indicators Forecast",
       show_provenance=True  # Displays data source in chart
   )
   
   # Export maintains full provenance chain
   results.export_to_csv('economic_forecast.csv')
   results.save_metadata('economic_forecast_meta.json')

**Why Use KRL Data Connectors?**

✅ **Automatic Provenance**: Every data point tracks its source, collection date, and transformations  
✅ **Validated APIs**: Pre-configured for BLS, FRED, Census Bureau, CDC, HUD  
✅ **Consistent Schema**: All connectors output ModelInputSchema-compatible data  
✅ **Rate Limiting**: Built-in throttling respects API limits  
✅ **Error Handling**: Graceful retries and informative error messages  
✅ **Reproducibility**: SHA256 hashing ensures identical data for identical queries  

For custom data sources, see the User Guide section on data preparation.

Model Families
--------------

Econometric Models
~~~~~~~~~~~~~~~~~~

- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMA**: Seasonal ARIMA
- **VAR**: Vector Autoregression
- **Cointegration**: Long-run equilibrium relationships
- **Prophet**: Facebook's forecasting tool

Regional Analysis
~~~~~~~~~~~~~~~~~

- **Location Quotient**: Measure economic specialization
- **Shift-Share Analysis**: Decompose regional growth

Anomaly Detection
~~~~~~~~~~~~~~~~~

- **STL Decomposition**: Seasonal-Trend decomposition with anomaly flagging
- **Isolation Forest**: Multivariate outlier detection

Volatility Models
~~~~~~~~~~~~~~~~~

- **GARCH**: Generalized Autoregressive Conditional Heteroskedasticity
- **EGARCH**: Exponential GARCH (leverage effects)
- **GJR-GARCH**: Threshold GARCH

Machine Learning
~~~~~~~~~~~~~~~~

- **Random Forest**: Ensemble decision trees
- **XGBoost**: Gradient boosting
- **Ridge/Lasso**: Regularized regression

State Space Models
~~~~~~~~~~~~~~~~~~

- **Kalman Filter**: Optimal state estimation
- **Local Level Model**: Trend extraction with uncertainty

Practical Applications
----------------------

The Model Zoo powers work that matters:

📈 **Labor & Employment**
   Forecasting job trends, analyzing workforce shifts, tracking equity gaps

🏘️ **Housing & Urban Development**
   Modeling affordability, detecting displacement, identifying price volatility

💰 **Income & Inequality**
   Measuring economic disparity, mobility, and opportunity over time

🏥 **Public Health**
   Linking health indicators with economic and environmental conditions

🌆 **Regional Development**
   Assessing industrial strengths, resilience, and competitiveness

Each model is field-tested, policy-relevant, and community-accessible.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
=======

This project is licensed under the **Apache License 2.0**.

Copyright © 2024-2025 Quipu Research Labs, LLC  
A wholly-owned subsidiary of Sudiata Giddasira, Inc.

Trademark Notice
================

**KR-Labs™** and **KRL Model Zoo™** are trademarks of Quipu Research Labs, LLC.  
All rights reserved.

See the :doc:`license` page for full legal information.

