.. Copyright (c) 2024 Sudiata Giddasira, Inc. d/b/a Quipu Research Labs, LLC d/b/a KR-Labs™
.. SPDX-License-Identifier: Apache-2.0

========================
Regional Economic Models
========================

Regional economic models analyze spatial economic patterns, specialization, 
and growth dynamics across geographic areas.

Available Models
================

* :class:`~krl_models.regional.LocationQuotientModel` - Economic specialization
* :class:`~krl_models.regional.ShiftShareModel` - Growth decomposition

Location Quotient
=================

The Location Quotient (LQ) measures the concentration of an economic activity 
in a region relative to a larger reference area (typically the nation).

**When to Use**

* Identify specialized industries in a region
* Assess competitive advantages
* Compare regional vs national economic structure
* Target economic development strategies

**Mathematical Formula**

.. math::

   LQ = \\frac{e_i / e}{E_i / E}

Where:

* :math:`e_i` = Regional employment in industry i
* :math:`e` = Total regional employment
* :math:`E_i` = National employment in industry i
* :math:`E` = Total national employment

**Interpretation**

* LQ > 1: Region more specialized than nation (export base)
* LQ = 1: Same as national average
* LQ < 1: Region less specialized (import base)
* LQ > 1.25: Generally considered significant specialization

See the :doc:`../api/regional` for complete API documentation.
