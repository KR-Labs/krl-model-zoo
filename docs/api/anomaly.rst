Anomaly Detection
=================

Outlier and anomaly identification models for detecting unusual patterns in economic data.

Overview
--------

The anomaly detection module provides methods for identifying outliers:

* **STL Anomaly**: Seasonal-Trend decomposition based anomaly detection
* **Isolation Forest**: ML-based multivariate anomaly detection

Module Contents
---------------

.. automodule:: krl_models.anomaly
   :members:
   :undoc-members:
   :show-inheritance:

STL Anomaly Model
-----------------

.. autoclass:: krl_models.anomaly.STLAnomalyModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_col** (str): Name of target variable column
* **period** (int): Seasonal period length
* **seasonal** (int, optional): Length of seasonal smoother
* **trend** (int, optional): Length of trend smoother
* **threshold** (float, optional): Anomaly threshold in standard deviations (default: 3.0)

**Methods**

.. automethod:: krl_models.anomaly.STLAnomalyModel.fit
.. automethod:: krl_models.anomaly.STLAnomalyModel.detect_anomalies
.. automethod:: krl_models.anomaly.STLAnomalyModel.plot_decomposition

**Example**

.. code-block:: python

   from krl_models.anomaly import STLAnomalyModel
   import pandas as pd
   
   # Monthly unemployment data
   data = pd.DataFrame({
       'date': pd.date_range('2015-01-01', periods=100, freq='M'),
       'unemployment_rate': [5.0 + np.random.normal(0, 0.5) for _ in range(100)]
   })
   
   # Add some anomalies
   data.loc[50, 'unemployment_rate'] = 12.0  # Spike
   data.loc[75, 'unemployment_rate'] = 2.0   # Drop
   
   model = STLAnomalyModel(
       time_col='date',
       target_col='unemployment_rate',
       period=12,  # Monthly seasonality
       threshold=3.0
   )
   
   results = model.fit(data)
   anomalies = model.detect_anomalies()
   
   print(f"Found {len(anomalies)} anomalies")
   print(anomalies[['date', 'unemployment_rate', 'residual', 'is_anomaly']])
   
   # Visualize
   results.plot_decomposition()

Isolation Forest Anomaly Model
-------------------------------

.. autoclass:: krl_models.anomaly.IsolationForestAnomalyModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **feature_cols** (list[str]): Feature columns for anomaly detection
* **contamination** (float, optional): Expected proportion of outliers (default: 0.1)
* **n_estimators** (int, optional): Number of trees (default: 100)
* **max_samples** (int/float, optional): Samples to draw for training
* **random_state** (int, optional): Random seed

**Methods**

.. automethod:: krl_models.anomaly.IsolationForestAnomalyModel.fit
.. automethod:: krl_models.anomaly.IsolationForestAnomalyModel.detect_anomalies
.. automethod:: krl_models.anomaly.IsolationForestAnomalyModel.score_samples

**Example**

.. code-block:: python

   from krl_models.anomaly import IsolationForestAnomalyModel
   
   # Multivariate economic data
   data = pd.DataFrame({
       'date': pd.date_range('2020-01-01', periods=200, freq='D'),
       'gdp_growth': np.random.normal(2.5, 1.0, 200),
       'unemployment': np.random.normal(5.0, 0.5, 200),
       'inflation': np.random.normal(2.0, 0.3, 200)
   })
   
   # Add multivariate anomaly
   data.loc[100, ['gdp_growth', 'unemployment', 'inflation']] = [-5.0, 15.0, 8.0]
   
   model = IsolationForestAnomalyModel(
       time_col='date',
       feature_cols=['gdp_growth', 'unemployment', 'inflation'],
       contamination=0.05,  # Expect 5% outliers
       n_estimators=100
   )
   
   results = model.fit(data)
   anomalies = model.detect_anomalies()
   
   # Get anomaly scores (lower = more anomalous)
   scores = model.score_samples(data)
   
   print(f"Detected {anomalies['is_anomaly'].sum()} anomalies")
   print(anomalies[anomalies['is_anomaly']])

Mathematical Background
-----------------------

STL Decomposition
~~~~~~~~~~~~~~~~~

STL (Seasonal and Trend decomposition using Loess) decomposes a time series:

.. math::

   Y_t = T_t + S_t + R_t

where:

* :math:`T_t` = trend component
* :math:`S_t` = seasonal component
* :math:`R_t` = remainder (residual)

**Anomaly Detection:**

A point is anomalous if:

.. math::

   |R_t| > k \cdot \sigma_R

where :math:`k` is the threshold (typically 3) and :math:`\sigma_R` is the standard deviation of residuals.

Isolation Forest
~~~~~~~~~~~~~~~~

Isolation Forest detects anomalies by isolating observations:

**Algorithm:**

1. Randomly select a feature
2. Randomly select a split value between min and max
3. Recursively partition data
4. Anomalies require fewer splits to isolate

**Anomaly Score:**

.. math::

   s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}

where:

* :math:`E(h(x))` = average path length to isolate x
* :math:`c(n)` = average path length for n samples
* :math:`s < 0.5`: normal
* :math:`s > 0.5`: anomalous

Best Practices
--------------

STL Anomaly Detection
~~~~~~~~~~~~~~~~~~~~~

1. **Set period correctly**: 12 for monthly, 4 for quarterly, 7 for daily
2. **Tune threshold**: Start with 3.0, adjust for domain
3. **Validate results**: Manually inspect detected anomalies
4. **Consider context**: Economic shocks may not be errors
5. **Multiple series**: Run on related indicators

Isolation Forest
~~~~~~~~~~~~~~~~

1. **Feature engineering**: Include relevant features
2. **Contamination parameter**: Set based on expected outlier rate
3. **Sufficient data**: Need enough for robust training
4. **Standardization**: Consider scaling features
5. **Ensemble size**: More trees = more stable but slower

Anomaly Types
~~~~~~~~~~~~~

**Point Anomalies:**

* Single unusual observation
* Example: GDP spike due to data error

**Contextual Anomalies:**

* Unusual in specific context
* Example: High unemployment in expansion

**Collective Anomalies:**

* Sequence of unusual observations
* Example: Prolonged negative growth

Use Cases
---------

Data Quality
~~~~~~~~~~~~

* Identify measurement errors
* Detect data entry mistakes
* Flag suspicious values
* Validate data pipelines

Economic Monitoring
~~~~~~~~~~~~~~~~~~~

* Detect structural breaks
* Identify crisis events
* Monitor policy changes
* Track unusual patterns

Forecasting Preparation
~~~~~~~~~~~~~~~~~~~~~~~

* Clean outliers before modeling
* Adjust intervention effects
* Handle one-time events
* Improve forecast accuracy

Policy Analysis
~~~~~~~~~~~~~~~

* Evaluate policy impacts
* Identify regime changes
* Assess intervention effects
* Monitor economic shocks

Interpreting Results
---------------------

True vs False Positives
~~~~~~~~~~~~~~~~~~~~~~~

Not all flagged anomalies are errors:

* **True Positive**: Data error, measurement problem
* **True Negative**: Valid but unusual economic event (recession, policy change)

Always investigate context before discarding anomalies.

Threshold Tuning
~~~~~~~~~~~~~~~~

Adjust thresholds based on:

* Industry standards
* Historical patterns
* Domain knowledge
* Cost of false positives/negatives

Validation Strategies
~~~~~~~~~~~~~~~~~~~~~

1. **Manual review**: Inspect top anomalies
2. **Cross-validation**: Compare with alternative methods
3. **Domain experts**: Consult economists
4. **Related series**: Check correlated indicators
5. **News/events**: Match to known events

See Also
--------

* :doc:`../user_guide/anomaly` - Detailed user guide
* :doc:`../examples/anomaly_detection` - Complete examples
* :doc:`econometric` - Time series methods
* :doc:`ml` - Machine learning models
