.. Copyright (c) 2024 Sudiata Giddasira, Inc. d/b/a Quipu Research Labs, LLC d/b/a KR-Labs™
.. SPDX-License-Identifier: Apache-2.0

=========
Changelog
=========

All notable changes to the KRL Model Zoo will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[1.0.0] - 2024-10-26
====================

Initial Release
---------------

Added
~~~~~

**Gate 1: Core Foundation**

* BaseModel and ModelMeta - Base classes for all models
* ModelInputSchema with Provenance tracking
* Result classes: BaseResult, ForecastResult, CausalResult, ClassificationResult
* ModelRegistry for model management
* PlotlySchemaAdapter for visualization
* Comprehensive utility functions

**Gate 2: Tier 1 Models**

Phase 2.2: Volatility & State Space Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* GARCHModel - Generalized Autoregressive Conditional Heteroskedasticity
* EGARCHModel - Exponential GARCH with leverage effects
* GJRGARCHModel - Threshold GARCH for asymmetric volatility
* KalmanFilter - Optimal state estimation
* LocalLevelModel - Trend extraction with uncertainty

Phase 2.3: Machine Learning Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* RandomForestModel - Ensemble decision trees
* XGBoostModel - Gradient boosting framework
* RegularizedRegressionModel - Ridge, Lasso, and Elastic Net

Phase 2.4: Regional Economic Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* LocationQuotientModel - Economic specialization analysis
* ShiftShareModel - Regional growth decomposition

Phase 2.5: Anomaly Detection Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* STLAnomalyModel - Seasonal-Trend decomposition with anomaly detection
* IsolationForestAnomalyModel - Multivariate outlier detection

**Testing & Quality**

* 455+ comprehensive tests
* 90% code coverage
* Unit, integration, and smoke tests
* Mathematical validation tests

**Documentation**

* Complete API reference
* User guides for all model families
* Mathematical formulations
* Installation and quickstart guides
* Contributing guidelines

Changed
~~~~~~~

* Rebuilt all corrupted files from emoji removal script
* Fixed 40+ syntax errors and typos
* Updated class names for consistency
* Improved import statements

Fixed
~~~~~

* Empty numeric literals and missing digits
* Class name typos (GRH→GARCH, Kalmanilter→KalmanFilter)
* Import typos (GridSearchV→GridSearchCV, RidgeV→RidgeCV)
* Broken f-strings and Exception handlers
* Empty brackets and ranges

[0.1.0] - 2024-10-01
====================

Pre-release
-----------

Initial development versions with experimental features.
