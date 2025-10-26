Volatility Models
=================

Conditional heteroskedasticity models for analyzing and forecasting variance in economic time series.

Overview
--------

The volatility module provides GARCH family models for conditional variance:

* **GARCH**: Generalized AutoRegressive Conditional Heteroskedasticity
* **EGARCH**: Exponential GARCH with asymmetric effects
* **GJR-GARCH**: Glosten-Jagannathan-Runkle GARCH with threshold effects

Module Contents
---------------

.. automodule:: krl_models.volatility
   :members:
   :undoc-members:
   :show-inheritance:

GARCH Model
-----------

.. autoclass:: krl_models.volatility.GARCHModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_col** (str): Name of target variable column (returns)
* **p** (int, optional): GARCH lag order (default: 1)
* **q** (int, optional): ARCH lag order (default: 1)
* **mean** (str, optional): Mean model ('Zero', 'Constant', 'AR', 'ARX')
* **vol** (str, optional): Volatility process ('GARCH', 'EGARCH', 'GJR-GARCH')
* **dist** (str, optional): Error distribution ('normal', 't', 'skewt')

**Methods**

.. automethod:: krl_models.volatility.GARCHModel.fit
.. automethod:: krl_models.volatility.GARCHModel.forecast_variance
.. automethod:: krl_models.volatility.GARCHModel.conditional_volatility

**Example**

.. code-block:: python

   from krl_models.volatility import GARCHModel
   import pandas as pd
   import numpy as np
   
   # Calculate returns from price data
   prices = pd.DataFrame({
       'date': pd.date_range('2020-01-01', periods=500, freq='D'),
       'price': np.cumsum(np.random.normal(0.001, 0.02, 500)) + 100
   })
   
   prices['returns'] = prices['price'].pct_change() * 100
   
   model = GARCHModel(
       time_col='date',
       target_col='returns',
       p=1,
       q=1,
       mean='Constant',
       dist='normal'
   )
   
   results = model.fit(prices.dropna())
   
   # Forecast variance
   variance_forecast = model.forecast_variance(horizon=10)
   print(f"Volatility forecast: {np.sqrt(variance_forecast)}")
   
   # Plot conditional volatility
   cond_vol = model.conditional_volatility()
   cond_vol.plot()

EGARCH Model
------------

.. autoclass:: krl_models.volatility.EGARCHModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_col** (str): Name of target variable column
* **p** (int, optional): GARCH lag order
* **q** (int, optional): ARCH lag order
* **o** (int, optional): Asymmetry lag order
* **mean** (str, optional): Mean model specification
* **dist** (str, optional): Error distribution

**Methods**

.. automethod:: krl_models.volatility.EGARCHModel.fit
.. automethod:: krl_models.volatility.EGARCHModel.forecast_variance
.. automethod:: krl_models.volatility.EGARCHModel.leverage_effect

**Example**

.. code-block:: python

   from krl_models.volatility import EGARCHModel
   
   # Stock returns often exhibit leverage effect
   model = EGARCHModel(
       time_col='date',
       target_col='returns',
       p=1,
       q=1,
       o=1,  # Asymmetry term
       mean='AR',
       dist='skewt'
   )
   
   results = model.fit(stock_returns)
   
   # Test for leverage effect
   leverage = model.leverage_effect()
   print(f"Leverage coefficient: {leverage}")
   print("Negative shocks increase volatility more" if leverage < 0 else "Symmetric")

GJR-GARCH Model
---------------

.. autoclass:: krl_models.volatility.GJRGARCHModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_col** (str): Name of target variable column
* **p** (int, optional): GARCH lag order
* **q** (int, optional): ARCH lag order
* **mean** (str, optional): Mean model specification
* **dist** (str, optional): Error distribution

**Methods**

.. automethod:: krl_models.volatility.GJRGARCHModel.fit
.. automethod:: krl_models.volatility.GJRGARCHModel.forecast_variance
.. automethod:: krl_models.volatility.GJRGARCHModel.threshold_effect

**Example**

.. code-block:: python

   from krl_models.volatility import GJRGARCHModel
   
   model = GJRGARCHModel(
       time_col='date',
       target_col='returns',
       p=1,
       q=1,
       mean='Constant'
   )
   
   results = model.fit(market_returns)
   
   # Estimate threshold effect
   threshold = model.threshold_effect()
   print(f"Threshold coefficient: {threshold}")

Mathematical Background
-----------------------

GARCH(p,q) Specification
~~~~~~~~~~~~~~~~~~~~~~~~

The GARCH(p,q) model specifies conditional variance:

.. math::

   r_t = \mu + \epsilon_t

.. math::

   \epsilon_t = \sigma_t z_t, \quad z_t \sim N(0,1)

.. math::

   \sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2

where:

* :math:`r_t` = returns at time t
* :math:`\sigma_t^2` = conditional variance
* :math:`\omega > 0` = constant term
* :math:`\alpha_i \geq 0` = ARCH coefficients
* :math:`\beta_j \geq 0` = GARCH coefficients

**Stationarity condition:**

.. math::

   \sum_{i=1}^q \alpha_i + \sum_{j=1}^p \beta_j < 1

EGARCH Specification
~~~~~~~~~~~~~~~~~~~~

EGARCH allows asymmetric effects:

.. math::

   \log(\sigma_t^2) = \omega + \sum_{i=1}^q \alpha_i g(z_{t-i}) + \sum_{j=1}^p \beta_j \log(\sigma_{t-j}^2)

where:

.. math::

   g(z_t) = \theta z_t + \gamma[|z_t| - E|z_t|]

**Leverage effect:**

* :math:`\theta < 0`: Negative shocks increase volatility more
* No coefficient restrictions needed for stationarity

GJR-GARCH Specification
~~~~~~~~~~~~~~~~~~~~~~~

GJR-GARCH includes threshold term:

.. math::

   \sigma_t^2 = \omega + \sum_{i=1}^q (\alpha_i + \gamma_i I_{t-i}^-)\epsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2

where:

* :math:`I_{t-i}^- = 1` if :math:`\epsilon_{t-i} < 0`, else 0
* :math:`\gamma_i > 0`: Negative shocks have larger impact

Best Practices
--------------

Data Preparation
~~~~~~~~~~~~~~~~

1. **Use returns**: Work with percentage returns, not levels
2. **Adequate sample**: Need 500+ observations for reliable estimates
3. **Stationarity**: Ensure mean stationarity
4. **Frequency**: Daily or higher frequency recommended
5. **Outliers**: Consider impact on estimates

Model Selection
~~~~~~~~~~~~~~~

1. **Start simple**: Begin with GARCH(1,1)
2. **Test asymmetry**: Use EGARCH/GJR-GARCH if leverage suspected
3. **Information criteria**: Compare AIC/BIC across specifications
4. **Residual tests**: Check standardized residuals
5. **Forecast evaluation**: Validate on hold-out sample

Estimation
~~~~~~~~~~

1. **Distribution**: Student-t or skewed-t for fat tails
2. **Optimization**: May need multiple starting values
3. **Convergence**: Check optimizer convergence
4. **Parameter constraints**: Verify stationarity
5. **Standard errors**: Use robust standard errors

Use Cases
---------

Risk Management
~~~~~~~~~~~~~~~

* Value-at-Risk (VaR) calculation
* Portfolio risk assessment
* Capital requirement estimation
* Stress testing

Asset Pricing
~~~~~~~~~~~~~

* Option pricing inputs
* Risk premium estimation
* Portfolio optimization
* Hedge ratio calculation

Economic Analysis
~~~~~~~~~~~~~~~~~

* Policy uncertainty measurement
* Market stress indicators
* Crisis detection
* Regime identification

Forecasting
~~~~~~~~~~~

* Volatility forecasting
* Prediction intervals
* Scenario analysis
* Risk projections

Diagnostic Tests
----------------

Standardized Residuals
~~~~~~~~~~~~~~~~~~~~~~

Check that standardized residuals :math:`z_t = \epsilon_t / \sigma_t` are:

* Uncorrelated (Ljung-Box test)
* No remaining ARCH effects
* Approximately normal (or specified distribution)

Model Adequacy
~~~~~~~~~~~~~~

.. code-block:: python

   from krl_models.volatility import GARCHModel
   
   model = GARCHModel(time_col='date', target_col='returns', p=1, q=1)
   results = model.fit(data)
   
   # Check standardized residuals
   std_resid = results.standardized_residuals
   
   # Ljung-Box test on standardized residuals
   from statsmodels.stats.diagnostic import acorr_ljungbox
   lb_test = acorr_ljungbox(std_resid, lags=10)
   
   # ARCH test on standardized residuals
   from arch.univariate import arch_model
   arch_test = arch_model(std_resid).fit()

Forecasting Performance
~~~~~~~~~~~~~~~~~~~~~~~

Evaluate using:

* Mean Squared Error (MSE) of variance forecasts
* Mincer-Zarnowitz regression
* Diebold-Mariano test
* VaR backtesting

Advanced Topics
---------------

Multivariate GARCH
~~~~~~~~~~~~~~~~~~

For multiple series, consider:

* DCC-GARCH (Dynamic Conditional Correlation)
* BEKK models
* Available in specialized packages

Long Memory
~~~~~~~~~~~

For persistent volatility:

* FIGARCH models
* HYGARCH specifications
* Fractional integration

Realized Volatility
~~~~~~~~~~~~~~~~~~~

For high-frequency data:

* HAR models
* Realized GARCH
* Combine with intraday measures

See Also
--------

* :doc:`../user_guide/volatility` - Detailed user guide
* :doc:`../examples/volatility_modeling` - Complete examples
* :doc:`econometric` - Time series methods
* :doc:`state_space` - State space models
