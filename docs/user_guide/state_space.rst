.. Copyright (c) 2024 Sudiata Giddasira, Inc. d/b/a Quipu Research Labs, LLC d/b/a KR-Labs™
.. SPDX-License-Identifier: Apache-2.0

===================
State Space Models
===================

State space models provide a unified framework for time series analysis,
optimal filtering, and uncertainty quantification.

Available Models
================

* :class:`~krl_models.state_space.KalmanFilter` - Optimal state estimation
* :class:`~krl_models.state_space.LocalLevelModel` - Trend extraction

Kalman Filter
=============

The Kalman filter provides optimal estimates of unobserved states from noisy
observations, widely used in nowcasting and signal extraction.

**When to Use**

* Real-time economic nowcasting
* Extracting trends from noisy data
* Combining multiple data sources
* Uncertainty quantification

**Key Parameters**

* ``state_dim``: Number of hidden states
* ``obs_dim``: Number of observations
* ``transition_matrix``: State evolution
* ``observation_matrix``: State-to-observation mapping
* ``process_noise``: System uncertainty
* ``measurement_noise``: Observation uncertainty

Local Level Model
=================

A special case of state space models focusing on trend extraction with
time-varying variance.

**When to Use**

* Extracting long-term trends
* Smoothing volatile series
* Detecting structural breaks
* Forward-looking indicators

See the :doc:`../api/state_space` for complete API documentation.
