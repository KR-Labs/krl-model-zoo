Core Utilities
==============

Base classes and shared functionality for all KRL Model Zoo models.

Overview
--------

The core module provides foundational classes and utilities:

* **BaseModel**: Abstract base class for all models
* **ModelResults**: Container for model results
* **ModelRegistry**: Model registration and metadata
* **ModelInputSchema**: Input validation
* **PlotlyAdapter**: Visualization utilities

Module Contents
---------------

.. automodule:: krl_core
   :members:
   :undoc-members:
   :show-inheritance:

BaseModel
---------

.. autoclass:: krl_core.BaseModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Abstract Methods**

All models must implement:

.. automethod:: krl_core.BaseModel.fit
.. automethod:: krl_core.BaseModel.predict

**Common Methods**

Inherited by all models:

.. automethod:: krl_core.BaseModel.validate_input
.. automethod:: krl_core.BaseModel.export_results
.. automethod:: krl_core.BaseModel.get_metadata

**Example: Creating a Custom Model**

.. code-block:: python

   from krl_core import BaseModel, ModelResults
   import pandas as pd
   
   class MyCustomModel(BaseModel):
       """Custom model implementation."""
       
       def __init__(self, time_col: str, target_col: str, **kwargs):
           super().__init__(
               time_col=time_col,
               target_col=target_col,
               model_type='custom',
               **kwargs
           )
           self.custom_param = kwargs.get('custom_param', 1.0)
       
       def fit(self, data: pd.DataFrame) -> ModelResults:
           """Fit the model."""
           # Validate input
           self.validate_input(data)
           
           # Your fitting logic here
           fitted_values = self._fit_logic(data)
           
           # Return results
           return ModelResults(
               model=self,
               fitted_values=fitted_values,
               residuals=data[self.target_col] - fitted_values,
               metadata=self.get_metadata()
           )
       
       def predict(self, steps: int = 1, **kwargs) -> pd.DataFrame:
           """Generate predictions."""
           # Your prediction logic here
           predictions = self._predict_logic(steps)
           
           return pd.DataFrame({
               'forecast': predictions
           })

ModelResults
------------

.. autoclass:: krl_core.ModelResults
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Attributes**

* **model**: Reference to fitted model
* **fitted_values**: In-sample fitted values
* **residuals**: Residual errors
* **forecast**: Out-of-sample predictions
* **metadata**: Model metadata dictionary

**Methods**

.. automethod:: krl_core.ModelResults.plot
.. automethod:: krl_core.ModelResults.summary
.. automethod:: krl_core.ModelResults.export_to_csv
.. automethod:: krl_core.ModelResults.export_to_json
.. automethod:: krl_core.ModelResults.diagnostics

**Example**

.. code-block:: python

   from krl_models.econometric import ARIMAModel
   
   model = ARIMAModel(time_col='date', target_col='value', order=(1,1,1))
   results = model.fit(data)
   
   # Access results
   print(f"Fitted values: {results.fitted_values}")
   print(f"Residuals: {results.residuals}")
   print(f"AIC: {results.aic}")
   print(f"BIC: {results.bic}")
   
   # Visualize
   results.plot()
   
   # Export
   results.export_to_csv('model_results.csv')
   results.export_to_json('model_results.json')
   
   # Statistical summary
   print(results.summary())

ModelRegistry
-------------

.. autoclass:: krl_core.ModelRegistry
   :members:
   :undoc-members:
   :show-inheritance:

**Methods**

.. automethod:: krl_core.ModelRegistry.register
.. automethod:: krl_core.ModelRegistry.get_model
.. automethod:: krl_core.ModelRegistry.list_models
.. automethod:: krl_core.ModelRegistry.get_metadata

**Example**

.. code-block:: python

   from krl_core import ModelRegistry
   from krl_models.econometric import ARIMAModel, SARIMAModel
   
   # Create registry
   registry = ModelRegistry()
   
   # Register models
   registry.register('arima', ARIMAModel, {
       'description': 'AutoRegressive Integrated Moving Average',
       'category': 'econometric',
       'version': '1.0.0'
   })
   
   registry.register('sarima', SARIMAModel, {
       'description': 'Seasonal ARIMA',
       'category': 'econometric',
       'version': '1.0.0'
   })
   
   # List available models
   print(registry.list_models())
   
   # Get specific model
   ModelClass = registry.get_model('arima')
   model = ModelClass(time_col='date', target_col='value', order=(1,1,1))
   
   # Get metadata
   metadata = registry.get_metadata('sarima')
   print(metadata)

ModelInputSchema
----------------

.. autoclass:: krl_core.ModelInputSchema
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_col** (str): Name of target variable column
* **exog_cols** (list[str], optional): Exogenous variable columns
* **freq** (str, optional): Expected frequency ('D', 'W', 'M', 'Q', 'Y')
* **min_periods** (int, optional): Minimum required observations
* **allow_missing** (bool, optional): Allow missing values

**Methods**

.. automethod:: krl_core.ModelInputSchema.validate
.. automethod:: krl_core.ModelInputSchema.check_frequency
.. automethod:: krl_core.ModelInputSchema.check_stationarity

**Example**

.. code-block:: python

   from krl_core import ModelInputSchema
   from pydantic import ValidationError
   
   # Define schema
   schema = ModelInputSchema(
       time_col='date',
       target_col='unemployment_rate',
       exog_cols=['gdp_growth', 'inflation'],
       freq='M',
       min_periods=24,
       allow_missing=False
   )
   
   try:
       # Validate data
       schema.validate(data)
       print("Data validation passed")
   except ValidationError as e:
       print(f"Validation errors: {e}")
   
   # Check frequency
   detected_freq = schema.check_frequency(data)
   print(f"Detected frequency: {detected_freq}")
   
   # Check stationarity
   is_stationary = schema.check_stationarity(data[schema.target_col])
   print(f"Series is stationary: {is_stationary}")

PlotlyAdapter
-------------

.. autoclass:: krl_core.PlotlyAdapter
   :members:
   :undoc-members:
   :show-inheritance:

**Methods**

.. automethod:: krl_core.PlotlyAdapter.plot_time_series
.. automethod:: krl_core.PlotlyAdapter.plot_forecast
.. automethod:: krl_core.PlotlyAdapter.plot_residuals
.. automethod:: krl_core.PlotlyAdapter.plot_decomposition
.. automethod:: krl_core.PlotlyAdapter.plot_diagnostics

**Example**

.. code-block:: python

   from krl_core import PlotlyAdapter
   import plotly.io as pio
   
   # Initialize adapter
   plotter = PlotlyAdapter(theme='plotly_white')
   
   # Plot time series
   fig = plotter.plot_time_series(
       data=data,
       time_col='date',
       value_col='unemployment_rate',
       title='US Unemployment Rate'
   )
   fig.show()
   
   # Plot forecast with confidence intervals
   fig = plotter.plot_forecast(
       actual=historical_data,
       forecast=forecast_data,
       time_col='date',
       value_col='value',
       conf_int_lower=forecast_lower,
       conf_int_upper=forecast_upper,
       title='GDP Growth Forecast'
   )
   fig.show()
   
   # Plot residual diagnostics
   fig = plotter.plot_diagnostics(
       residuals=results.residuals,
       fitted=results.fitted_values
   )
   fig.show()

Design Patterns
---------------

Model Lifecycle
~~~~~~~~~~~~~~~

All models follow this lifecycle:

1. **Initialization**: Create model with parameters
2. **Validation**: Validate input data
3. **Fitting**: Estimate model parameters
4. **Prediction**: Generate forecasts
5. **Diagnostics**: Check model adequacy
6. **Export**: Save results

.. code-block:: python

   # 1. Initialize
   model = SomeModel(time_col='date', target_col='value', **params)
   
   # 2. Validate (automatic in fit)
   model.validate_input(data)
   
   # 3. Fit
   results = model.fit(data)
   
   # 4. Predict
   forecast = model.predict(steps=12)
   
   # 5. Diagnostics
   diagnostics = results.diagnostics()
   
   # 6. Export
   results.export_to_csv('output.csv')

Consistent Interface
~~~~~~~~~~~~~~~~~~~~

All models provide:

* **fit(data)**: Fit model to data
* **predict(steps)**: Generate forecasts
* **validate_input(data)**: Validate data
* **get_metadata()**: Return model metadata

Type Hints
~~~~~~~~~~

All code uses type hints:

.. code-block:: python

   from typing import List, Optional, Dict, Any
   import pandas as pd
   
   def fit(
       self,
       data: pd.DataFrame,
       exog: Optional[pd.DataFrame] = None
   ) -> ModelResults:
       """Fit model with type hints."""
       pass

Pydantic Validation
~~~~~~~~~~~~~~~~~~~

Input validation using Pydantic:

.. code-block:: python

   from pydantic import BaseModel, validator
   
   class ModelConfig(BaseModel):
       """Model configuration with validation."""
       
       time_col: str
       target_col: str
       order: tuple[int, int, int]
       
       @validator('order')
       def validate_order(cls, v):
           if any(x < 0 for x in v):
               raise ValueError("Order must be non-negative")
           return v

Testing Utilities
-----------------

Model Testing
~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from krl_core import BaseModel
   
   def test_model_interface():
       """Test model implements required interface."""
       model = SomeModel(time_col='date', target_col='value')
       
       # Check methods exist
       assert hasattr(model, 'fit')
       assert hasattr(model, 'predict')
       assert hasattr(model, 'validate_input')
       
       # Check fit returns ModelResults
       results = model.fit(test_data)
       assert isinstance(results, ModelResults)

Mock Data Generation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from krl_core.testing import generate_time_series
   
   # Generate synthetic test data
   data = generate_time_series(
       start='2020-01-01',
       periods=100,
       freq='M',
       trend=0.5,
       seasonal_amplitude=2.0,
       noise_std=1.0
   )

Best Practices
--------------

Extending BaseModel
~~~~~~~~~~~~~~~~~~~

1. **Call super().__init__()**: Always initialize parent class
2. **Implement abstract methods**: fit() and predict() required
3. **Use type hints**: Improve code clarity
4. **Validate inputs**: Use validate_input() or schemas
5. **Document thoroughly**: Docstrings with examples

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   from krl_core.exceptions import ModelError, ValidationError
   
   class MyModel(BaseModel):
       def fit(self, data: pd.DataFrame) -> ModelResults:
           try:
               self.validate_input(data)
               # Fitting logic
           except ValidationError as e:
               raise ValidationError(f"Invalid input: {e}")
           except Exception as e:
               raise ModelError(f"Model fitting failed: {e}")

Logging
~~~~~~~

.. code-block:: python

   import logging
   
   logger = logging.getLogger(__name__)
   
   class MyModel(BaseModel):
       def fit(self, data: pd.DataFrame) -> ModelResults:
           logger.info(f"Fitting {self.__class__.__name__}")
           logger.debug(f"Data shape: {data.shape}")
           # Fitting logic
           logger.info("Fitting complete")
           return results

See Also
--------

* :doc:`../user_guide/extending` - Extending models guide
* :doc:`../user_guide/testing` - Testing guide
* :doc:`../api/index` - All model APIs
