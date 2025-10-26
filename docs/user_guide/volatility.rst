.. Copyright (c) 2024 Sudiata Giddasira, Inc. d/b/a Quipu Research Labs, LLC d/b/a KR-Labs™
.. SPDX-License-Identifier: Apache-2.0

==================
Volatility Models
==================

Volatility models capture time-varying variance in financial and economic data,
essential for risk assessment and policy uncertainty analysis.

Available Models
================

* :class:`~krl_models.volatility.GARCHModel` - Standard GARCH
* :class:`~krl_models.volatility.EGARCHModel` - Exponential GARCH (leverage)
* :class:`~krl_models.volatility.GJRGARCHModel` - Threshold GARCH (asymmetry)

GARCH Models
============

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models 
capture volatility clustering - periods of high volatility tend to persist.

**When to Use**

* Housing price volatility and affordability risk
* Economic policy uncertainty
* Labor market instability
* Financial market stress affecting communities

**Key Parameters**

* ``p``: GARCH order (volatility lags)
* ``q``: ARCH order (squared return lags)
* ``mean``: Mean model specification
* ``dist``: Error distribution

See the :doc:`../api/volatility` for complete API documentation.
