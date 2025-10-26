Machine Learning Models
=======================

ML-based forecasting and prediction models for economic time series.

Overview
--------

The machine learning module provides modern ML approaches:

* **Random Forest**: Ensemble tree-based regression
* **XGBoost**: Gradient boosting with optimizations
* **Regularized Regression**: Lasso, Ridge, and Elastic Net

Module Contents
---------------

.. automodule:: krl_models.ml
   :members:
   :undoc-members:
   :show-inheritance:

Random Forest Model
-------------------

.. autoclass:: krl_models.ml.RandomForestModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_col** (str): Name of target variable column
* **feature_cols** (list[str]): Feature columns for prediction
* **n_estimators** (int, optional): Number of trees (default: 100)
* **max_depth** (int, optional): Maximum tree depth
* **min_samples_split** (int, optional): Minimum samples to split node
* **min_samples_leaf** (int, optional): Minimum samples in leaf
* **max_features** (str/int, optional): Features per split
* **random_state** (int, optional): Random seed

**Methods**

.. automethod:: krl_models.ml.RandomForestModel.fit
.. automethod:: krl_models.ml.RandomForestModel.predict
.. automethod:: krl_models.ml.RandomForestModel.feature_importance

**Example**

.. code-block:: python

   from krl_models.ml import RandomForestModel
   import pandas as pd
   
   # Economic indicators data
   data = pd.DataFrame({
       'date': pd.date_range('2015-01-01', periods=100, freq='M'),
       'gdp': np.random.normal(100, 10, 100),
       'unemployment': np.random.normal(5, 1, 100),
       'inflation': np.random.normal(2, 0.5, 100),
       'interest_rate': np.random.normal(3, 0.8, 100)
   })
   
   model = RandomForestModel(
       time_col='date',
       target_col='gdp',
       feature_cols=['unemployment', 'inflation', 'interest_rate'],
       n_estimators=200,
       max_depth=10,
       random_state=42
   )
   
   results = model.fit(data)
   forecast = model.predict(new_data)
   
   # Feature importance
   importance = model.feature_importance()
   print(importance.sort_values(ascending=False))

XGBoost Model
-------------

.. autoclass:: krl_models.ml.XGBoostModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_col** (str): Name of target variable column
* **feature_cols** (list[str]): Feature columns for prediction
* **n_estimators** (int, optional): Number of boosting rounds
* **max_depth** (int, optional): Maximum tree depth
* **learning_rate** (float, optional): Boosting learning rate
* **subsample** (float, optional): Subsample ratio
* **colsample_bytree** (float, optional): Feature subsample ratio
* **reg_alpha** (float, optional): L1 regularization
* **reg_lambda** (float, optional): L2 regularization
* **random_state** (int, optional): Random seed

**Methods**

.. automethod:: krl_models.ml.XGBoostModel.fit
.. automethod:: krl_models.ml.XGBoostModel.predict
.. automethod:: krl_models.ml.XGBoostModel.feature_importance
.. automethod:: krl_models.ml.XGBoostModel.plot_importance

**Example**

.. code-block:: python

   from krl_models.ml import XGBoostModel
   
   model = XGBoostModel(
       time_col='date',
       target_col='housing_price',
       feature_cols=['income', 'population', 'employment', 'interest_rate'],
       n_estimators=500,
       max_depth=6,
       learning_rate=0.01,
       subsample=0.8,
       colsample_bytree=0.8,
       reg_alpha=0.1,
       reg_lambda=1.0
   )
   
   results = model.fit(training_data)
   forecast = model.predict(test_data)
   
   # Plot feature importance
   model.plot_importance()
   
   # Get SHAP values for interpretation
   shap_values = model.get_shap_values(test_data)

Regularized Regression Model
-----------------------------

.. autoclass:: krl_models.ml.RegularizedRegressionModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **target_col** (str): Name of target variable column
* **feature_cols** (list[str]): Feature columns for prediction
* **alpha** (float, optional): Regularization strength
* **l1_ratio** (float, optional): L1/L2 mix (0=Ridge, 1=Lasso, 0-1=Elastic Net)
* **fit_intercept** (bool, optional): Include intercept
* **normalize** (bool, optional): Normalize features
* **max_iter** (int, optional): Maximum iterations

**Methods**

.. automethod:: krl_models.ml.RegularizedRegressionModel.fit
.. automethod:: krl_models.ml.RegularizedRegressionModel.predict
.. automethod:: krl_models.ml.RegularizedRegressionModel.get_coefficients
.. automethod:: krl_models.ml.RegularizedRegressionModel.select_features

**Example**

.. code-block:: python

   from krl_models.ml import RegularizedRegressionModel
   
   # Lasso for feature selection
   model = RegularizedRegressionModel(
       time_col='date',
       target_col='consumption',
       feature_cols=['income', 'wealth', 'debt', 'rate', 'sentiment'],
       alpha=0.1,
       l1_ratio=1.0,  # Pure Lasso
       fit_intercept=True
   )
   
   results = model.fit(data)
   
   # Selected features (non-zero coefficients)
   selected = model.select_features()
   print(f"Selected features: {selected}")
   
   # Get coefficients
   coefs = model.get_coefficients()
   print(coefs[coefs != 0])

Mathematical Background
-----------------------

Random Forest
~~~~~~~~~~~~~

Random Forest builds multiple decision trees:

.. math::

   \hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(x)

where:

* :math:`B` = number of trees
* :math:`T_b(x)` = prediction from tree b
* Each tree trained on bootstrap sample
* Each split considers random feature subset

**Advantages:**

* Handles non-linear relationships
* Robust to outliers
* Automatic feature interaction
* Built-in feature importance

XGBoost
~~~~~~~

XGBoost uses gradient boosting:

.. math::

   \hat{y}^{(t)} = \hat{y}^{(t-1)} + \eta f_t(x)

where:

* :math:`\eta` = learning rate
* :math:`f_t` = new tree minimizing loss

**Objective:**

.. math::

   \mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^t \Omega(f_k)

**Regularization:**

.. math::

   \Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2

Regularized Regression
~~~~~~~~~~~~~~~~~~~~~~

**Ridge (L2):**

.. math::

   \min_{\beta} ||y - X\beta||_2^2 + \lambda ||\beta||_2^2

**Lasso (L1):**

.. math::

   \min_{\beta} ||y - X\beta||_2^2 + \lambda ||\beta||_1

**Elastic Net:**

.. math::

   \min_{\beta} ||y - X\beta||_2^2 + \lambda_1 ||\beta||_1 + \lambda_2 ||\beta||_2^2

Best Practices
--------------

Random Forest
~~~~~~~~~~~~~

1. **Tune n_estimators**: More trees = more stable, diminishing returns after 200-500
2. **Control max_depth**: Prevent overfitting, especially with limited data
3. **Feature engineering**: Create lagged features for time series
4. **Out-of-bag error**: Use for validation without separate test set
5. **Parallel processing**: Set n_jobs=-1 for speed

XGBoost
~~~~~~~

1. **Start conservative**: Low learning_rate (0.01-0.1), high n_estimators
2. **Early stopping**: Monitor validation set, stop when no improvement
3. **Regularization**: Use reg_alpha and reg_lambda
4. **Cross-validation**: Use XGBoost's built-in CV
5. **Feature importance**: Check multiple importance types

Regularized Regression
~~~~~~~~~~~~~~~~~~~~~~~

1. **Standardize features**: Critical for regularization
2. **Cross-validation**: Use CV to select alpha
3. **Lasso for selection**: Use l1_ratio=1.0 for feature selection
4. **Ridge for stability**: Use l1_ratio=0.0 when all features relevant
5. **Elastic Net**: Compromise with l1_ratio=0.5

Time Series Considerations
--------------------------

Feature Engineering
~~~~~~~~~~~~~~~~~~~

Create time series features:

.. code-block:: python

   def create_ts_features(df, target_col, lags=[1, 2, 3, 6, 12]):
       """Create lagged features for time series."""
       for lag in lags:
           df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
       
       # Rolling statistics
       df[f'{target_col}_rolling_mean_3'] = df[target_col].rolling(3).mean()
       df[f'{target_col}_rolling_std_3'] = df[target_col].rolling(3).std()
       
       # Date features
       df['month'] = df['date'].dt.month
       df['quarter'] = df['date'].dt.quarter
       
       return df.dropna()

Walk-Forward Validation
~~~~~~~~~~~~~~~~~~~~~~~

For time series, use chronological splits:

.. code-block:: python

   from sklearn.model_selection import TimeSeriesSplit
   
   tscv = TimeSeriesSplit(n_splits=5)
   
   for train_idx, test_idx in tscv.split(data):
       train = data.iloc[train_idx]
       test = data.iloc[test_idx]
       
       model.fit(train)
       predictions = model.predict(test)

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~

Use grid search with time series CV:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
   
   param_grid = {
       'n_estimators': [100, 200, 500],
       'max_depth': [5, 10, 15],
       'learning_rate': [0.01, 0.05, 0.1]
   }
   
   tscv = TimeSeriesSplit(n_splits=5)
   grid = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
   grid.fit(X, y)

Use Cases
---------

Economic Forecasting
~~~~~~~~~~~~~~~~~~~~

* GDP growth prediction
* Inflation forecasting
* Employment projections
* Demand estimation

Non-Linear Relationships
~~~~~~~~~~~~~~~~~~~~~~~~

* Regime-dependent effects
* Threshold models
* Interaction effects
* Complex dynamics

High-Dimensional Data
~~~~~~~~~~~~~~~~~~~~~

* Many predictors
* Feature selection needed
* Multicollinearity present
* Mixed variable types

Model Comparison
~~~~~~~~~~~~~~~~

Compare ML vs traditional:

.. code-block:: python

   from krl_models.econometric import ARIMAModel
   from krl_models.ml import XGBoostModel
   
   # Traditional
   arima = ARIMAModel(time_col='date', target_col='value', order=(1,1,1))
   arima_results = arima.fit(data)
   arima_forecast = arima.predict(steps=12)
   
   # Machine Learning
   xgb = XGBoostModel(time_col='date', target_col='value', feature_cols=features)
   xgb_results = xgb.fit(data)
   xgb_forecast = xgb.predict(future_data)
   
   # Compare
   from sklearn.metrics import mean_squared_error
   print(f"ARIMA MSE: {mean_squared_error(actual, arima_forecast)}")
   print(f"XGBoost MSE: {mean_squared_error(actual, xgb_forecast)}")

See Also
--------

* :doc:`../user_guide/ml` - Detailed user guide
* :doc:`../examples/ml_forecasting` - Complete examples
* :doc:`econometric` - Traditional time series methods
* :doc:`anomaly` - Isolation Forest for anomaly detection
