API Reference
=============

Complete API documentation for all KRL Model Zoo modules.

Overview
--------

The KRL Model Zoo provides a consistent API across all model families:

* **Econometric Models**: Time series and econometric analysis
* **Regional Models**: Spatial and regional economic analysis
* **Anomaly Detection**: Outlier and anomaly identification
* **Volatility Models**: Conditional heteroskedasticity modeling
* **Machine Learning**: ML-based forecasting and prediction
* **State Space Models**: Kalman filtering and state estimation
* **Core Utilities**: Base classes and shared functionality

Model Families
--------------

.. toctree::
   :maxdepth: 2

   econometric
   regional
   anomaly
   volatility
   ml
   state_space
   core

Quick Reference
---------------

Econometric Models
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :recursive:

   krl_models.econometric.ARIMAModel
   krl_models.econometric.SARIMAModel
   krl_models.econometric.VARModel
   krl_models.econometric.CointegrationModel
   krl_models.econometric.ProphetModel

Regional Models
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :recursive:

   krl_models.regional.LocationQuotientModel
   krl_models.regional.ShiftShareModel

Anomaly Detection
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :recursive:

   krl_models.anomaly.STLAnomalyModel
   krl_models.anomaly.IsolationForestAnomalyModel

Volatility Models
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :recursive:

   krl_models.volatility.GARCHModel
   krl_models.volatility.EGARCHModel
   krl_models.volatility.GJRGARCHModel

Machine Learning
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :recursive:

   krl_models.ml.RandomForestModel
   krl_models.ml.XGBoostModel
   krl_models.ml.RegularizedRegressionModel

State Space Models
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :recursive:

   krl_models.state_space.KalmanFilterModel
   krl_models.state_space.LocalLevelModel

Core Utilities
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :recursive:

   krl_core.BaseModel
   krl_core.ModelResults
   krl_core.ModelRegistry
   krl_core.ModelInputSchema
   krl_core.PlotlyAdapter

Common Patterns
---------------

All Models Follow This Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_models.{family} import {ModelName}
   
   # Initialize
   model = ModelName(
       time_col='date',
       target_col='value',
       **model_specific_params
   )
   
   # Fit
   results = model.fit(data)
   
   # Predict
   forecast = model.predict(steps=10)
   
   # Visualize
   results.plot()
   
   # Export
   results.export_to_csv('output.csv')

Model Initialization Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common parameters across all models:

* **time_col** (str): Name of the datetime column
* **target_col** (str): Name of the target variable column
* **exog_cols** (list[str], optional): Exogenous variable columns
* **model_id** (str, optional): Unique identifier for the model

Model Results Methods
~~~~~~~~~~~~~~~~~~~~~

All model results objects provide:

* **plot()**: Visualize results
* **export_to_csv()**: Export to CSV format
* **export_to_json()**: Export to JSON format
* **summary()**: Display statistical summary
* **diagnostics()**: Run diagnostic tests

Detailed Documentation
----------------------

For detailed documentation of each model family, see:

* :doc:`econometric` - Time series and econometric models
* :doc:`regional` - Regional analysis models
* :doc:`anomaly` - Anomaly detection models
* :doc:`volatility` - Volatility modeling
* :doc:`ml` - Machine learning models
* :doc:`state_space` - State space models
* :doc:`core` - Core utilities and base classes

Examples
--------

Basic Usage Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_models.econometric import ARIMAModel
   import pandas as pd
   
   # Prepare data
   data = pd.DataFrame({
       'date': pd.date_range('2020-01-01', periods=100, freq='M'),
       'value': range(100)
   })
   
   # Initialize and fit
   model = ARIMAModel(
       time_col='date',
       target_col='value',
       order=(1, 1, 1)
   )
   
   results = model.fit(data)
   forecast = model.predict(steps=12)

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_models.econometric import SARIMAModel
   from krl_core import ModelInputSchema
   
   # Configure with validation
   schema = ModelInputSchema(
       time_col='date',
       target_col='value',
       freq='M',
       min_periods=24
   )
   
   model = SARIMAModel(
       time_col='date',
       target_col='value',
       order=(1, 1, 1),
       seasonal_order=(1, 1, 1, 12),
       input_schema=schema
   )
   
   results = model.fit(data)

Type Hints and Validation
--------------------------

All models use Pydantic for input validation:

.. code-block:: python

   from krl_models.econometric import ARIMAModel
   from pydantic import ValidationError
   
   try:
       model = ARIMAModel(
           time_col='date',
           target_col='value',
           order=(1, 1, 1)  # Valid tuple
       )
   except ValidationError as e:
       print(f"Validation error: {e}")

See Also
--------

* :doc:`../quickstart` - Getting started guide
* :doc:`../examples` - Complete examples
* :doc:`../user_guide/index` - User guide by model family
