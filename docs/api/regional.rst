Regional Models
===============

Spatial and regional economic analysis models for understanding geographic patterns and disparities.

Overview
--------

The regional module provides tools for analyzing spatial economic patterns:

* **Location Quotient**: Measure regional industry concentration
* **Shift-Share Analysis**: Decompose regional growth into structural components

Module Contents
---------------

.. automodule:: krl_models.regional
   :members:
   :undoc-members:
   :show-inheritance:

Location Quotient Model
------------------------

.. autoclass:: krl_models.regional.LocationQuotientModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **region_col** (str): Name of region identifier column
* **industry_col** (str): Name of industry identifier column
* **value_col** (str): Name of employment/output value column
* **reference_region** (str, optional): Reference region (default: national total)

**Methods**

.. automethod:: krl_models.regional.LocationQuotientModel.fit
.. automethod:: krl_models.regional.LocationQuotientModel.calculate_lq
.. automethod:: krl_models.regional.LocationQuotientModel.identify_specializations

**Example**

.. code-block:: python

   from krl_models.regional import LocationQuotientModel
   import pandas as pd
   
   # Employment data by region and industry
   data = pd.DataFrame({
       'year': [2020] * 20,
       'region': ['CA', 'CA', 'TX', 'TX', 'NY', 'NY'] * 3 + ['CA', 'TX'],
       'industry': ['Tech', 'Health'] * 10,
       'employment': [50000, 30000, 40000, 35000, 45000, 38000] * 3 + [52000, 41000]
   })
   
   model = LocationQuotientModel(
       time_col='year',
       region_col='region',
       industry_col='industry',
       value_col='employment'
   )
   
   results = model.fit(data)
   
   # Get location quotients
   lq = results.location_quotients
   print(lq[lq > 1.25])  # Industries with significant specialization

Shift-Share Model
------------------

.. autoclass:: krl_models.regional.ShiftShareModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Parameters**

* **time_col** (str): Name of datetime column
* **region_col** (str): Name of region identifier column
* **industry_col** (str): Name of industry identifier column
* **value_col** (str): Name of employment/output value column
* **base_period** (str/int): Base period for comparison
* **comparison_period** (str/int): Comparison period

**Methods**

.. automethod:: krl_models.regional.ShiftShareModel.fit
.. automethod:: krl_models.regional.ShiftShareModel.decompose_growth
.. automethod:: krl_models.regional.ShiftShareModel.calculate_components

**Example**

.. code-block:: python

   from krl_models.regional import ShiftShareModel
   
   # Employment data across multiple periods
   data = pd.DataFrame({
       'year': [2010, 2010, 2010, 2020, 2020, 2020],
       'region': ['CA', 'CA', 'CA', 'CA', 'CA', 'CA'],
       'industry': ['Tech', 'Manufacturing', 'Services', 'Tech', 'Manufacturing', 'Services'],
       'employment': [100000, 50000, 80000, 150000, 45000, 95000]
   })
   
   model = ShiftShareModel(
       time_col='year',
       region_col='region',
       industry_col='industry',
       value_col='employment',
       base_period=2010,
       comparison_period=2020
   )
   
   results = model.fit(data)
   
   # Decompose growth
   components = results.growth_components
   print(f"National Share: {components['national_share']}")
   print(f"Industry Mix: {components['industry_mix']}")
   print(f"Regional Shift: {components['regional_shift']}")

Mathematical Background
-----------------------

Location Quotient
~~~~~~~~~~~~~~~~~

The location quotient (LQ) measures regional industry concentration:

.. math::

   LQ_{ir} = \frac{e_{ir} / e_r}{e_i / e}

where:

* :math:`e_{ir}` = employment in industry i, region r
* :math:`e_r` = total regional employment
* :math:`e_i` = national employment in industry i
* :math:`e` = total national employment

**Interpretation:**

* :math:`LQ > 1`: Region specializes in the industry (exports)
* :math:`LQ = 1`: Regional concentration equals national average
* :math:`LQ < 1`: Region under-represents the industry (imports)
* :math:`LQ > 1.25`: Typically indicates significant specialization

Shift-Share Analysis
~~~~~~~~~~~~~~~~~~~~~

Shift-share decomposes regional growth into three components:

.. math::

   \Delta E_r = NS + IM + RS

where:

**National Share (NS):**

.. math::

   NS = E_{r,0} \cdot g_n

Growth if region grew at national rate.

**Industry Mix (IM):**

.. math::

   IM = \sum_i E_{ir,0} \cdot (g_{in} - g_n)

Growth due to regional industry composition.

**Regional Shift (RS):**

.. math::

   RS = \sum_i E_{ir,0} \cdot (g_{ir} - g_{in})

Growth due to regional competitive advantage.

where:

* :math:`E_{r,0}` = base period regional employment
* :math:`g_n` = national growth rate
* :math:`g_{in}` = national industry i growth rate
* :math:`g_{ir}` = regional industry i growth rate

Best Practices
--------------

Location Quotient Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use appropriate aggregation**: County, MSA, state level
2. **Consider threshold**: LQ > 1.25 for strong specialization
3. **Multiple time periods**: Track specialization changes
4. **Cross-reference**: Validate with other regional data
5. **Industry definitions**: Use consistent NAICS codes

Shift-Share Analysis
~~~~~~~~~~~~~~~~~~~~~

1. **Select meaningful periods**: 5-10 year spans typical
2. **Consistent definitions**: Use same industry classifications
3. **Interpret carefully**: Components not independent
4. **Compare regions**: Benchmark against peers
5. **Validate data**: Check for breaks, revisions

Data Requirements
~~~~~~~~~~~~~~~~~

Both models require:

* **Consistent geography**: Stable regional boundaries
* **Industry classification**: NAICS or similar standard codes
* **Complete coverage**: All industries and regions
* **Temporal consistency**: Same measurement periods

Use Cases
---------

Economic Development
~~~~~~~~~~~~~~~~~~~~

* Identify industry clusters and specializations
* Target industries for recruitment
* Assess competitive position
* Plan workforce development

Regional Planning
~~~~~~~~~~~~~~~~~

* Understand structural change
* Evaluate growth strategies
* Benchmark peer regions
* Allocate resources

Policy Analysis
~~~~~~~~~~~~~~~

* Measure policy effectiveness
* Identify declining industries
* Support industrial policy
* Evaluate incentive programs

See Also
--------

* :doc:`../user_guide/regional` - Detailed user guide
* :doc:`../examples/regional_analysis` - Complete examples
* :doc:`econometric` - Time series methods
