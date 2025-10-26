Econometric Models
==================

Time series and econometric analysis models for forecasting and structural analysis.

.. important::
   **Data Integration**: All econometric models are designed to work with 
   **KRL Data Connectors** for automatic provenance tracking. Use BLSConnector, 
   FREDConnector, or CensusConnector to fetch federal datasets with validated 
   schemas. See :doc:`../quickstart` for examples.

Overview
--------

The econometric module provides implementations of classical and modern time series models:

* **SARIMA**: Seasonal ARIMA (includes ARIMA as special case when seasonal_order=(0,0,0,0))
* **VAR**: Vector AutoRegression for multivariate analysis
* **Cointegration**: Engle-Granger and Johansen cointegration tests
* **Prophet**: Facebook's Prophet for robust trend and seasonality

.. note::
   **ARIMA Implementation**: ARIMA functionality is provided through ``SARIMAModel`` 
   with ``seasonal_order=(0,0,0,0)``. A standalone reference implementation exists 
   in ``examples/example_arima_run.py`` demonstrating the full KRL pattern.

Module Contents
---------------

.. automodule:: krl_models.econometric
   :members:
   :undoc-members:
   :show-inheritance:

SARIMA Model
------------

Seasonal ARIMA model for time series with periodic patterns. **Note**: When 
``seasonal_order=(0,0,0,0)`` (default), this becomes a standard ARIMA model.

.. autoclass:: krl_models.econometric.SARIMAModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **input_schema** (ModelInputSchema): Time series data with provenance
* **params** (dict): Model parameters

  * **order** (tuple): (p, d, q) non-seasonal order
  * **seasonal_order** (tuple, optional): (P, D, Q, s) seasonal order. Default: (0,0,0,0) for ARIMA
  * **trend** (str, optional): Trend specification ('n', 'c', 't', 'ct'). Default: 'c'

* **meta** (ModelMeta): Model metadata (name, version, author)

**Methods**

.. automethod:: krl_models.econometric.SARIMAModel.fit
.. automethod:: krl_models.econometric.SARIMAModel.predict
.. automethod:: krl_models.econometric.SARIMAModel.get_seasonal_factors

**Example (ARIMA via SARIMA)**

.. code-block:: python

   from krl_data_connectors import BLSConnector
   from krl_models.econometric import SARIMAModel
   from krl_core import ModelInputSchema, ModelMeta
   
   # Fetch data with KRL Data Connectors
   bls = BLSConnector()
   data = bls.get_series('LNS14000000', 2010, 2024)
   
   # Create input schema with provenance
   input_schema = ModelInputSchema(
       entity="US",
       metric="unemployment_rate",
       time_index=data['date'].tolist(),
       values=data['value'].tolist(),
       provenance=bls.get_provenance('LNS14000000'),
       frequency='M'
   )
   
   # ARIMA(1,1,1) - no seasonal components
   model = SARIMAModel(
       input_schema=input_schema,
       params={
           'order': (1, 1, 1),
           'seasonal_order': (0, 0, 0, 0),  # ARIMA mode
           'trend': 'c'
       },
       meta=ModelMeta(name="UnemploymentARIMA", version="1.0.0")
   )
   
   results = model.fit()
   forecast = model.predict(steps=12, alpha=0.05)
   
   print(f"AIC: {results.payload['aic']:.2f}")
   print(f"Forecast: {forecast.forecast_values}")

**Example (Seasonal ARIMA)**

.. code-block:: python

   # SARIMA(1,1,1)(1,1,1,12) - monthly seasonality
   model = SARIMAModel(
       input_schema=input_schema,
       params={
           'order': (1, 1, 1),
           'seasonal_order': (1, 1, 1, 12),  # Seasonal mode
           'trend': 'c'
       },
       meta=ModelMeta(name="SeasonalUnemployment", version="1.0.0")
   )
   
   results = model.fit()
   forecast = model.predict(steps=24, alpha=0.05)
   
   # Extract seasonal factors
   seasonal_factors = model.get_seasonal_factors()
   print(f"Seasonal pattern: {seasonal_factors}")

ARIMA Model (Reference)
~~~~~~~~~~~~~~~~~~~~~~~~

A standalone ARIMA implementation exists in ``examples/example_arima_run.py`` 
as the Gate 1 reference implementation. It demonstrates the full KRL pattern 
for wrapping external libraries. For production use, prefer ``SARIMAModel`` 
with ``seasonal_order=(0,0,0,0)`` as it provides identical functionality with 
additional flexibility.

VAR Model
---------

.. autoclass:: krl_models.econometric.VARModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_cols** (list[str]): List of endogenous variable columns
* **maxlags** (int, optional): Maximum number of lags to test
* **ic** (str, optional): Information criterion ('aic', 'bic', 'hqic')
* **trend** (str, optional): Trend specification

**Methods**

.. automethod:: krl_models.econometric.VARModel.fit
.. automethod:: krl_models.econometric.VARModel.predict
.. automethod:: krl_models.econometric.VARModel.impulse_response
.. automethod:: krl_models.econometric.VARModel.granger_causality

**Example**

.. code-block:: python

   from krl_models.econometric import VARModel
   
   model = VARModel(
       time_col='date',
       target_cols=['gdp', 'unemployment', 'inflation'],
       maxlags=12,
       ic='aic'
   )
   
   results = model.fit(macro_data)
   forecast = model.predict(steps=8)
   
   # Impulse response analysis
   irf = model.impulse_response(periods=10)
   irf.plot()

Cointegration Model
-------------------

.. autoclass:: krl_models.econometric.CointegrationModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_cols** (list[str]): Variables to test for cointegration
* **method** (str): Test method ('engle-granger', 'johansen')
* **trend** (str, optional): Trend assumption in cointegrating relation

**Methods**

.. automethod:: krl_models.econometric.CointegrationModel.fit
.. automethod:: krl_models.econometric.CointegrationModel.test_cointegration
.. automethod:: krl_models.econometric.CointegrationModel.estimate_vecm

**Example**

.. code-block:: python

   from krl_models.econometric import CointegrationModel
   
   model = CointegrationModel(
       time_col='date',
       target_cols=['housing_price', 'income', 'interest_rate'],
       method='johansen'
   )
   
   results = model.fit(housing_data)
   print(f"Cointegration rank: {results.coint_rank}")
   print(f"Test statistics: {results.test_stats}")

Prophet Model
-------------

.. autoclass:: krl_models.econometric.ProphetModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_col** (str): Name of target variable column
* **growth** (str, optional): Growth model ('linear', 'logistic')
* **seasonality_mode** (str, optional): 'additive' or 'multiplicative'
* **yearly_seasonality** (bool/int, optional): Include yearly seasonality
* **weekly_seasonality** (bool/int, optional): Include weekly seasonality
* **daily_seasonality** (bool/int, optional): Include daily seasonality

**Methods**

.. automethod:: krl_models.econometric.ProphetModel.fit
.. automethod:: krl_models.econometric.ProphetModel.predict
.. automethod:: krl_models.econometric.ProphetModel.add_regressor
.. automethod:: krl_models.econometric.ProphetModel.add_seasonality

**Example**

.. code-block:: python

   from krl_models.econometric import ProphetModel
   
   model = ProphetModel(
       time_col='ds',  # Prophet uses 'ds' convention
       target_col='y',  # Prophet uses 'y' convention
       growth='linear',
       seasonality_mode='multiplicative',
       yearly_seasonality=True
   )
   
   # Add custom regressor
   model.add_regressor('marketing_spend')
   
   results = model.fit(sales_data)
   forecast = model.predict(steps=90)
   
   # Plot components
   results.plot_components()

Mathematical Background
-----------------------

ARIMA Specification
~~~~~~~~~~~~~~~~~~~

An ARIMA(p, d, q) model is specified as:

.. math::

   \phi(B)(1-B)^d y_t = \theta(B)\epsilon_t

where:

* :math:`\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - ... - \phi_p B^p` is the AR polynomial
* :math:`\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + ... + \theta_q B^q` is the MA polynomial
* :math:`B` is the backshift operator: :math:`B y_t = y_{t-1}`
* :math:`d` is the differencing order
* :math:`\epsilon_t` is white noise

SARIMA Specification
~~~~~~~~~~~~~~~~~~~~~

SARIMA(p,d,q)(P,D,Q,s) extends ARIMA with seasonal components:

.. math::

   \phi(B)\Phi(B^s)(1-B)^d(1-B^s)^D y_t = \theta(B)\Theta(B^s)\epsilon_t

where :math:`\Phi(B^s)` and :math:`\Theta(B^s)` are seasonal AR and MA polynomials.

VAR Specification
~~~~~~~~~~~~~~~~~

A VAR(p) model for k variables is:

.. math::

   \mathbf{y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{y}_{t-1} + ... + \mathbf{A}_p \mathbf{y}_{t-p} + \mathbf{e}_t

where :math:`\mathbf{y}_t` is a k-dimensional vector and :math:`\mathbf{A}_i` are coefficient matrices.

Best Practices
--------------

Model Selection
~~~~~~~~~~~~~~~

1. **Check stationarity** using ADF or KPSS tests
2. **Difference if needed** to achieve stationarity
3. **Examine ACF/PACF** to guide order selection
4. **Use information criteria** (AIC, BIC) for final selection
5. **Validate on hold-out sample**

Diagnostics
~~~~~~~~~~~

Always check:

* Residual autocorrelation (Ljung-Box test)
* Residual normality (Jarque-Bera test)
* Heteroskedasticity (ARCH test)
* Parameter stability (recursive residuals)

Seasonal Data
~~~~~~~~~~~~~

For seasonal data:

* Use SARIMA instead of ARIMA
* Set seasonal order based on data frequency (12 for monthly, 4 for quarterly)
* Consider seasonal differencing if seasonal unit roots present

See Also
--------

* :doc:`../user_guide/econometric` - Detailed user guide
* :doc:`../examples/time_series_analysis` - Complete examples
* :doc:`volatility` - Volatility modeling
* :doc:`state_space` - State space methods
