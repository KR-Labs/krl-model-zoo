State Space Models
==================

Kalman filtering and state space methods for time series analysis and forecasting.

Overview
--------

The state space module provides implementations of state space models:

* **Kalman Filter**: General state space filtering and smoothing
* **Local Level Model**: Simple state space model with level component

Module Contents
---------------

.. automodule:: krl_models.state_space
   :members:
   :undoc-members:
   :show-inheritance:

Kalman Filter Model
--------------------

.. autoclass:: krl_models.state_space.KalmanFilterModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_col** (str): Name of observed variable column
* **transition_matrix** (array, optional): State transition matrix (F)
* **observation_matrix** (array, optional): Observation matrix (H)
* **process_noise** (array, optional): Process noise covariance (Q)
* **observation_noise** (float, optional): Observation noise variance (R)
* **initial_state** (array, optional): Initial state estimate
* **initial_covariance** (array, optional): Initial state covariance

**Methods**

.. automethod:: krl_models.state_space.KalmanFilterModel.fit
.. automethod:: krl_models.state_space.KalmanFilterModel.filter
.. automethod:: krl_models.state_space.KalmanFilterModel.smooth
.. automethod:: krl_models.state_space.KalmanFilterModel.predict
.. automethod:: krl_models.state_space.KalmanFilterModel.estimate_parameters

**Example**

.. code-block:: python

   from krl_models.state_space import KalmanFilterModel
   import numpy as np
   import pandas as pd
   
   # Noisy observations of a random walk
   true_state = np.cumsum(np.random.normal(0, 1, 100))
   observations = true_state + np.random.normal(0, 0.5, 100)
   
   data = pd.DataFrame({
       'date': pd.date_range('2020-01-01', periods=100, freq='D'),
       'value': observations
   })
   
   # Define state space model for random walk
   model = KalmanFilterModel(
       time_col='date',
       target_col='value',
       transition_matrix=np.array([[1.0]]),  # Random walk
       observation_matrix=np.array([[1.0]]),
       process_noise=np.array([[1.0]]),
       observation_noise=0.25
   )
   
   results = model.fit(data)
   
   # Filtered estimates
   filtered = model.filter(data)
   
   # Smoothed estimates (using all data)
   smoothed = model.smooth(data)
   
   # Forecast
   forecast = model.predict(steps=10)

Local Level Model
-----------------

.. autoclass:: krl_models.state_space.LocalLevelModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_col** (str): Name of observed variable column
* **level_variance** (float, optional): Level innovation variance
* **observation_variance** (float, optional): Observation noise variance

**Methods**

.. automethod:: krl_models.state_space.LocalLevelModel.fit
.. automethod:: krl_models.state_space.LocalLevelModel.predict
.. automethod:: krl_models.state_space.LocalLevelModel.smooth
.. automethod:: krl_models.state_space.LocalLevelModel.decompose

**Example**

.. code-block:: python

   from krl_models.state_space import LocalLevelModel
   
   # Economic indicator with level and noise
   data = pd.DataFrame({
       'date': pd.date_range('2015-01-01', periods=120, freq='M'),
       'gdp': [100 + i*0.5 + np.random.normal(0, 2) for i in range(120)]
   })
   
   model = LocalLevelModel(
       time_col='date',
       target_col='gdp',
       level_variance=0.1,    # Small level changes
       observation_variance=4.0  # Moderate observation noise
   )
   
   results = model.fit(data)
   
   # Decompose into level and noise
   decomposition = model.decompose()
   print("Level component:", decomposition['level'])
   print("Noise component:", decomposition['noise'])
   
   # Forecast with uncertainty
   forecast = model.predict(steps=12, return_conf_int=True)
   print("Forecast:", forecast['mean'])
   print("95% CI:", forecast['conf_int'])

Mathematical Background
-----------------------

State Space Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~

General state space model:

**State Equation:**

.. math::

   \mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim N(0, \mathbf{Q})

**Observation Equation:**

.. math::

   y_t = \mathbf{H} \mathbf{x}_t + v_t, \quad v_t \sim N(0, R)

where:

* :math:`\mathbf{x}_t` = state vector (unobserved)
* :math:`y_t` = observation (observed)
* :math:`\mathbf{F}` = transition matrix
* :math:`\mathbf{H}` = observation matrix
* :math:`\mathbf{Q}` = process noise covariance
* :math:`R` = observation noise variance

Kalman Filter Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~

**Prediction Step:**

.. math::

   \hat{\mathbf{x}}_{t|t-1} = \mathbf{F} \hat{\mathbf{x}}_{t-1|t-1}

.. math::

   \mathbf{P}_{t|t-1} = \mathbf{F} \mathbf{P}_{t-1|t-1} \mathbf{F}^T + \mathbf{Q}

**Update Step:**

.. math::

   \mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}^T (\mathbf{H} \mathbf{P}_{t|t-1} \mathbf{H}^T + R)^{-1}

.. math::

   \hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t (y_t - \mathbf{H} \hat{\mathbf{x}}_{t|t-1})

.. math::

   \mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}) \mathbf{P}_{t|t-1}

where:

* :math:`\mathbf{K}_t` = Kalman gain
* :math:`\hat{\mathbf{x}}_{t|t-1}` = predicted state
* :math:`\hat{\mathbf{x}}_{t|t}` = filtered state
* :math:`\mathbf{P}_{t|t}` = state covariance

Kalman Smoother
~~~~~~~~~~~~~~~

Backward pass for smoothing:

.. math::

   \hat{\mathbf{x}}_{t|T} = \hat{\mathbf{x}}_{t|t} + \mathbf{J}_t (\hat{\mathbf{x}}_{t+1|T} - \hat{\mathbf{x}}_{t+1|t})

.. math::

   \mathbf{J}_t = \mathbf{P}_{t|t} \mathbf{F}^T \mathbf{P}_{t+1|t}^{-1}

Local Level Model
~~~~~~~~~~~~~~~~~

Simplest structural model:

**State:**

.. math::

   \mu_t = \mu_{t-1} + \eta_t, \quad \eta_t \sim N(0, \sigma_\eta^2)

**Observation:**

.. math::

   y_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma_\epsilon^2)

This is a special case with:

* :math:`\mathbf{F} = 1`, :math:`\mathbf{H} = 1`
* :math:`\mathbf{Q} = \sigma_\eta^2`, :math:`R = \sigma_\epsilon^2`

Best Practices
--------------

Model Specification
~~~~~~~~~~~~~~~~~~~

1. **Define state clearly**: What unobserved processes drive observations?
2. **Choose appropriate dimensions**: State vector size
3. **Set initial values**: Use sensible starting values
4. **Estimate parameters**: Use MLE or EM algorithm
5. **Validate assumptions**: Check Gaussian residuals

Numerical Stability
~~~~~~~~~~~~~~~~~~~

1. **Use square root filter**: Joseph form for covariance update
2. **Check conditioning**: Monitor covariance matrix condition numbers
3. **Scale data**: Normalize observations if needed
4. **Handle missing data**: Kalman filter naturally handles gaps
5. **Regularization**: Add small values to prevent singularities

Parameter Estimation
~~~~~~~~~~~~~~~~~~~~

Maximum likelihood estimation:

.. code-block:: python

   from scipy.optimize import minimize
   
   def neg_log_likelihood(params, model, data):
       """Negative log-likelihood for parameter estimation."""
       model.set_parameters(params)
       results = model.fit(data)
       return -results.log_likelihood
   
   # Optimize
   initial_params = [1.0, 0.25]  # Initial guess
   result = minimize(neg_log_likelihood, initial_params, args=(model, data))
   optimal_params = result.x

Use Cases
---------

Filtering and Smoothing
~~~~~~~~~~~~~~~~~~~~~~~~

* Remove measurement noise
* Estimate unobserved states
* Interpolate missing values
* Real-time state estimation

Forecasting
~~~~~~~~~~~

* Generate multi-step forecasts
* Compute prediction intervals
* Handle structural changes
* Incorporate external information

Signal Extraction
~~~~~~~~~~~~~~~~~

* Separate trend from noise
* Decompose seasonal components
* Extract business cycles
* Identify turning points

Missing Data
~~~~~~~~~~~~

Kalman filter naturally handles missing observations:

.. code-block:: python

   # Data with missing values
   data_with_gaps = pd.DataFrame({
       'date': pd.date_range('2020-01-01', periods=100, freq='D'),
       'value': [np.nan if i % 10 == 0 else v for i, v in enumerate(observations)]
   })
   
   # Kalman filter automatically skips missing observations
   model = KalmanFilterModel(...)
   results = model.fit(data_with_gaps)
   smoothed = model.smooth(data_with_gaps)  # Interpolates gaps

Advanced Examples
-----------------

Local Linear Trend Model
~~~~~~~~~~~~~~~~~~~~~~~~

Model with level and slope:

.. code-block:: python

   import numpy as np
   
   # State: [level, slope]
   F = np.array([
       [1, 1],  # level_{t} = level_{t-1} + slope_{t-1}
       [0, 1]   # slope_{t} = slope_{t-1}
   ])
   
   H = np.array([[1, 0]])  # Observe level only
   
   Q = np.array([
       [0.1, 0],
       [0, 0.01]
   ])  # Level and slope variances
   
   model = KalmanFilterModel(
       time_col='date',
       target_col='value',
       transition_matrix=F,
       observation_matrix=H,
       process_noise=Q,
       observation_noise=0.5
   )

Seasonal Model
~~~~~~~~~~~~~~

Include seasonal component:

.. code-block:: python

   # For quarterly data (s=4)
   # State: [level, seasonal_1, seasonal_2, seasonal_3]
   
   F = np.array([
       [1, 0, 0, 0],
       [0, -1, -1, -1],
       [0, 1, 0, 0],
       [0, 0, 1, 0]
   ])
   
   H = np.array([[1, 1, 0, 0]])  # Observe level + seasonal
   
   model = KalmanFilterModel(
       time_col='date',
       target_col='value',
       transition_matrix=F,
       observation_matrix=H,
       process_noise=Q,
       observation_noise=R
   )

Regression with Time-Varying Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # y_t = beta_t * x_t + error
   # beta_t = beta_{t-1} + innovation
   
   data = pd.DataFrame({
       'date': pd.date_range('2020-01-01', periods=100),
       'y': observations,
       'x': explanatory_var
   })
   
   # State is time-varying coefficient
   model = KalmanFilterModel(
       time_col='date',
       target_col='y',
       transition_matrix=np.array([[1.0]]),  # Random walk coefficient
       observation_matrix=data['x'].values.reshape(-1, 1),  # Multiply by x
       process_noise=np.array([[0.01]]),
       observation_noise=1.0
   )

Diagnostics
-----------

Residual Analysis
~~~~~~~~~~~~~~~~~

Check innovation residuals:

.. code-block:: python

   results = model.fit(data)
   innovations = results.innovations
   
   # Should be white noise
   from statsmodels.stats.diagnostic import acorr_ljungbox
   lb_test = acorr_ljungbox(innovations, lags=20)
   
   # Should be normal
   from scipy.stats import normaltest
   norm_test = normaltest(innovations)

State Diagnostics
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check state covariance stability
   state_cov = results.filtered_state_covariance
   
   # Check for divergence
   import matplotlib.pyplot as plt
   plt.plot(np.diagonal(state_cov, axis1=1, axis2=2))
   plt.title('State Variance Over Time')
   plt.show()

Model Comparison
~~~~~~~~~~~~~~~~

Compare different specifications:

.. code-block:: python

   from krl_models.state_space import LocalLevelModel
   
   # Fit multiple models
   models = {
       'Local Level': LocalLevelModel(...),
       'Local Linear Trend': KalmanFilterModel(...),
       'Seasonal': KalmanFilterModel(...)
   }
   
   for name, model in models.items():
       results = model.fit(data)
       print(f"{name} - AIC: {results.aic}, BIC: {results.bic}")

See Also
--------

* :doc:`../user_guide/state_space` - Detailed user guide
* :doc:`../examples/kalman_filtering` - Complete examples
* :doc:`econometric` - ARIMA and structural time series
* :doc:`volatility` - Stochastic volatility models
