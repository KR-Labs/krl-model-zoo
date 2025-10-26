Installation Guide
==================

This guide covers different ways to install the KRL Model Zoo.

Requirements
------------

System Requirements
~~~~~~~~~~~~~~~~~~~

* Python 3.9 or higher
* 4GB RAM minimum (8GB recommended)
* 1GB free disk space

Dependencies
~~~~~~~~~~~~

Core dependencies are automatically installed:

* pandas >= 1.5.0
* numpy >= 1.23.0
* scipy >= 1.10.0
* statsmodels >= 0.14.0
* scikit-learn >= 1.2.0
* xgboost >= 1.7.0
* prophet >= 1.1.0
* plotly >= 5.14.0
* pydantic >= 2.0.0

Installation Methods
--------------------

Method 1: Install from PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install the KRL Model Zoo:

.. code-block:: bash

   pip install krl-model-zoo

To install with optional dependencies:

.. code-block:: bash

   # For development
   pip install krl-model-zoo[dev]
   
   # For testing
   pip install krl-model-zoo[test]
   
   # For documentation building
   pip install krl-model-zoo[docs]
   
   # Install everything
   pip install krl-model-zoo[all]

Method 2: Install from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the latest development version:

.. code-block:: bash

   git clone https://github.com/KR-Labs/krl-model-zoo.git
   cd krl-model-zoo
   pip install -e .

For development with all dependencies:

.. code-block:: bash

   pip install -e ".[all]"

Method 3: Using Poetry
~~~~~~~~~~~~~~~~~~~~~~

If you use Poetry for dependency management:

.. code-block:: bash

   poetry add krl-model-zoo

Method 4: Using Conda
~~~~~~~~~~~~~~~~~~~~~

Create a conda environment:

.. code-block:: bash

   conda create -n krl-env python=3.11
   conda activate krl-env
   pip install krl-model-zoo

Optional Dependencies
---------------------

Data Connectors Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For seamless integration with KRL Data Connectors:

.. code-block:: bash

   pip install krl-data-connectors

This allows you to fetch data from BLS, Census, FRED, CDC, and other federal sources.

Jupyter Support
~~~~~~~~~~~~~~~

For interactive analysis in Jupyter notebooks:

.. code-block:: bash

   pip install jupyter ipython notebook

Visualization Enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced plotting capabilities:

.. code-block:: bash

   pip install matplotlib seaborn

Verification
------------

Verify Installation
~~~~~~~~~~~~~~~~~~~

Test your installation:

.. code-block:: python

   import krl_models
   from krl_models.econometric import ARIMAModel
   
   print(f"KRL Model Zoo version: {krl_models.__version__}")
   print("Installation successful!")

Run Tests
~~~~~~~~~

If you installed from source, run the test suite:

.. code-block:: bash

   pytest tests/

Check Available Models
~~~~~~~~~~~~~~~~~~~~~~

List all available models:

.. code-block:: python

   from krl_core import ModelRegistry
   
   registry = ModelRegistry()
   available_models = registry.list_models()
   
   for family, models in available_models.items():
       print(f"\n{family}:")
       for model in models:
           print(f"  - {model}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue: numpy version conflict**

.. code-block:: bash

   # Ensure numpy < 2.0.0
   pip install "numpy>=1.23.0,<2.0.0"

**Issue: Prophet installation fails**

On macOS, you may need:

.. code-block:: bash

   conda install -c conda-forge prophet

On Linux:

.. code-block:: bash

   pip install pystan
   pip install prophet

**Issue: XGBoost compilation errors**

Install pre-built wheels:

.. code-block:: bash

   pip install xgboost --no-build-isolation

Platform-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~~

**macOS**

For Apple Silicon (M1/M2), use:

.. code-block:: bash

   arch -arm64 pip install krl-model-zoo

**Windows**

Use Anaconda for easier dependency management:

.. code-block:: bash

   conda create -n krl python=3.11
   conda activate krl
   pip install krl-model-zoo

**Linux**

Ensure gcc and g++ are installed:

.. code-block:: bash

   sudo apt-get install build-essential
   pip install krl-model-zoo

Upgrading
---------

Upgrade to Latest Version
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install --upgrade krl-model-zoo

Check for Updates
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip list --outdated | grep krl-model-zoo

Uninstallation
--------------

To uninstall:

.. code-block:: bash

   pip uninstall krl-model-zoo

Next Steps
----------

* Continue to the :doc:`quickstart` guide
* Explore :doc:`examples`
* Read the :doc:`api/index` documentation
* Join the community: https://github.com/KR-Labs/krl-model-zoo

Getting Help
------------

If you encounter issues:

1. Check the GitHub issues: https://github.com/KR-Labs/krl-model-zoo/issues
2. Read the FAQ: https://krl-model-zoo.readthedocs.io/en/latest/faq.html
3. Contact us: info@krlabs.dev
