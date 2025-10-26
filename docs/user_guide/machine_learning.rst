.. Copyright (c) 2024 Sudiata Giddasira, Inc. d/b/a Quipu Research Labs, LLC d/b/a KR-Labs™
.. SPDX-License-Identifier: Apache-2.0

===================
Machine Learning
===================

Machine learning models provide predictive power for complex, nonlinear 
relationships in socioeconomic data.

Available Models
================

* :class:`~krl_models.ml.RandomForestModel` - Ensemble decision trees
* :class:`~krl_models.ml.XGBoostModel` - Gradient boosting
* :class:`~krl_models.ml.RegularizedRegressionModel` - Ridge/Lasso/Elastic Net

Random Forest
=============

Random forests build multiple decision trees and aggregate their predictions,
providing robust performance and feature importance metrics.

**When to Use**

* Poverty prediction from demographic features
* Educational outcomes modeling
* Health risk assessment
* Feature importance discovery

**Key Parameters**

* ``n_estimators``: Number of trees
* ``max_depth``: Tree depth limit
* ``min_samples_split``: Minimum samples per split
* ``max_features``: Features per split

XGBoost
=======

XGBoost implements gradient boosting with regularization, often achieving 
state-of-the-art performance on structured data.

**When to Use**

* High-stakes predictions requiring maximum accuracy
* Competition-grade benchmarks
* Feature engineering experiments
* Handling missing data

See the :doc:`../api/ml` for complete API documentation.
