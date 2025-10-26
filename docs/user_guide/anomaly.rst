.. Copyright (c) 2024 Sudiata Giddasira, Inc. d/b/a Quipu Research Labs, LLC d/b/a KR-Labs™
.. SPDX-License-Identifier: Apache-2.0

==================
Anomaly Detection
==================

Anomaly detection models identify unusual patterns, outliers, and structural 
breaks in data - critical for policy monitoring and early warning systems.

Available Models
================

* :class:`~krl_models.anomaly.STLAnomalyModel` - Time series decomposition
* :class:`~krl_models.anomaly.IsolationForestAnomalyModel` - Multivariate detection

STL Anomaly Detection
======================

STL (Seasonal-Trend Decomposition using Loess) separates time series into 
trend, seasonal, and residual components, flagging anomalies in the residuals.

**When to Use**

* Detect unusual spikes in crime, health, or economic data
* Monitor policy interventions for unexpected effects
* Identify data quality issues
* Flag structural breaks requiring investigation

**Key Parameters**

* ``period``: Seasonal cycle length (e.g., 12 for monthly data)
* ``threshold``: Number of standard deviations for anomaly flagging
* ``seasonal``: Seasonality strength (odd integer)

See the :doc:`../api/anomaly` for complete API documentation.
