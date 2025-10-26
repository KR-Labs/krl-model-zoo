Development Guide
=================

Comprehensive guide for developers working on KRL Model Zoo.

Development Setup
-----------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.9 or higher
* Git
* Virtual environment tool (venv, conda, poetry)
* Make (optional but recommended)

Initial Setup
~~~~~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone https://github.com/KR-Labs/krl-model-zoo.git
   cd krl-model-zoo
   
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install in editable mode with all dependencies
   pip install -e ".[dev,test,docs,all]"
   
   # Install pre-commit hooks
   pre-commit install
   
   # Verify installation
   python -c "import krl_models; print(krl_models.__version__)"
   pytest --version

Project Structure
-----------------

Directory Layout
~~~~~~~~~~~~~~~~

.. code-block:: text

   krl-model-zoo/
   ├── krl_models/          # Main package
   │   ├── econometric/     # Econometric models
   │   ├── regional/        # Regional analysis
   │   ├── anomaly/         # Anomaly detection
   │   ├── volatility/      # Volatility models
   │   ├── ml/              # Machine learning
   │   └── state_space/     # State space models
   ├── krl_core/            # Core utilities
   │   ├── base.py          # BaseModel
   │   ├── results.py       # ModelResults
   │   ├── registry.py      # ModelRegistry
   │   ├── schema.py        # Input validation
   │   └── plotting.py      # Visualization
   ├── tests/               # Test suite
   │   ├── unit/            # Unit tests
   │   ├── integration/     # Integration tests
   │   └── fixtures/        # Test fixtures
   ├── docs/                # Documentation
   │   ├── api/             # API reference
   │   ├── user_guide/      # User guides
   │   └── examples/        # Examples
   ├── examples/            # Example scripts
   ├── pyproject.toml       # Package configuration
   ├── Makefile             # Common commands
   └── README.md            # Project README

Module Organization
~~~~~~~~~~~~~~~~~~~

Each model family follows this structure:

.. code-block:: text

   krl_models/econometric/
   ├── __init__.py          # Exports
   ├── arima.py             # ARIMA implementation
   ├── sarima.py            # SARIMA implementation
   ├── var.py               # VAR implementation
   └── utils.py             # Shared utilities

Development Workflow
--------------------

Day-to-Day Development
~~~~~~~~~~~~~~~~~~~~~~

1. **Start with an issue**: Create or pick an existing issue
2. **Create branch**: Branch from main
3. **Implement changes**: Write code, tests, docs
4. **Run checks**: Linting, testing, type checking
5. **Commit changes**: Use conventional commits
6. **Push and PR**: Submit pull request

Example workflow:

.. code-block:: bash

   # Update main branch
   git checkout main
   git pull origin main
   
   # Create feature branch
   git checkout -b feature/new-garch-variant
   
   # Make changes
   vim krl_models/volatility/my_garch.py
   vim tests/test_my_garch.py
   vim docs/api/volatility.rst
   
   # Run checks
   make lint
   make test
   make docs
   
   # Commit with conventional commit message
   git add .
   git commit -m "feat(volatility): add T-GARCH model"
   
   # Push and create PR
   git push origin feature/new-garch-variant

Conventional Commits
~~~~~~~~~~~~~~~~~~~~

We use conventional commits:

.. code-block:: text

   <type>(<scope>): <subject>
   
   <body>
   
   <footer>

Types:

* **feat**: New feature
* **fix**: Bug fix
* **docs**: Documentation
* **style**: Formatting
* **refactor**: Code restructuring
* **test**: Testing
* **chore**: Maintenance

Examples:

.. code-block:: text

   feat(econometric): add Prophet model integration
   
   fix(arima): handle edge case with zero variance
   
   docs(api): update SARIMA examples
   
   test(volatility): increase GARCH test coverage

Code Quality
------------

Linting and Formatting
~~~~~~~~~~~~~~~~~~~~~~~

Run all checks:

.. code-block:: bash

   # All checks at once
   make lint
   
   # Individual tools
   black krl_models/ krl_core/ tests/
   isort krl_models/ krl_core/ tests/
   flake8 krl_models/ krl_core/ tests/
   mypy krl_models/ krl_core/

Configuration files:

* **.flake8**: Flake8 config
* **pyproject.toml**: Black, isort, mypy config
* **.pre-commit-config.yaml**: Pre-commit hooks

Type Checking
~~~~~~~~~~~~~

We use mypy for static type checking:

.. code-block:: python

   from typing import Optional, List, Dict, Any, Union
   import pandas as pd
   import numpy as np
   
   def process_data(
       data: pd.DataFrame,
       columns: Optional[List[str]] = None,
       config: Dict[str, Any] = None
   ) -> pd.DataFrame:
       """Process data with proper type hints."""
       ...

Run type checks:

.. code-block:: bash

   mypy krl_models/ krl_core/

Testing Strategy
----------------

Test Organization
~~~~~~~~~~~~~~~~~

Tests are organized by type:

* **Unit tests**: Test individual functions/methods
* **Integration tests**: Test component interactions
* **End-to-end tests**: Test complete workflows

Test Fixtures
~~~~~~~~~~~~~

Use pytest fixtures for reusable test data:

.. code-block:: python

   import pytest
   import pandas as pd
   import numpy as np
   
   
   @pytest.fixture
   def time_series_data():
       """Generate standard time series test data."""
       return pd.DataFrame({
           'date': pd.date_range('2020-01-01', periods=100, freq='M'),
           'value': np.cumsum(np.random.normal(0, 1, 100)) + 100
       })
   
   
   @pytest.fixture
   def seasonal_data():
       """Generate seasonal time series."""
       t = np.arange(120)
       trend = t * 0.5
       seasonal = 10 * np.sin(2 * np.pi * t / 12)
       noise = np.random.normal(0, 2, 120)
       
       return pd.DataFrame({
           'date': pd.date_range('2010-01-01', periods=120, freq='M'),
           'value': trend + seasonal + noise + 100
       })

Parameterized Tests
~~~~~~~~~~~~~~~~~~~

Use parametrize for multiple test cases:

.. code-block:: python

   import pytest
   
   
   @pytest.mark.parametrize("order,expected", [
       ((1, 0, 0), "AR(1)"),
       ((0, 0, 1), "MA(1)"),
       ((1, 1, 1), "ARIMA(1,1,1)"),
       ((2, 1, 2), "ARIMA(2,1,2)"),
   ])
   def test_model_names(order, expected):
       """Test model naming for different orders."""
       model = ARIMAModel(time_col='date', target_col='value', order=order)
       assert model.model_name == expected

Test Coverage
~~~~~~~~~~~~~

Monitor coverage:

.. code-block:: bash

   # Generate coverage report
   pytest --cov=krl_models --cov=krl_core --cov-report=html
   
   # View report
   open htmlcov/index.html

Target 90%+ coverage for all new code.

Continuous Integration
----------------------

GitHub Actions
~~~~~~~~~~~~~~

CI pipeline runs on every push and PR:

1. **Linting**: Black, isort, flake8, mypy
2. **Testing**: pytest on Python 3.9, 3.10, 3.11
3. **Coverage**: Upload to Codecov
4. **Docs**: Build documentation
5. **Package**: Test package building

CI configuration in `.github/workflows/`:

.. code-block:: yaml

   name: CI
   
   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]
   
   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: ["3.9", "3.10", "3.11"]
       
       steps:
         - uses: actions/checkout@v3
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: ${{ matrix.python-version }}
         - name: Install dependencies
           run: |
             pip install -e ".[dev,test]"
         - name: Lint
           run: make lint
         - name: Test
           run: make test-cov
         - name: Upload coverage
           uses: codecov/codecov-action@v3

Pre-commit Hooks
~~~~~~~~~~~~~~~~

Automated checks before commit:

.. code-block:: yaml

   # .pre-commit-config.yaml
   repos:
     - repo: https://github.com/psf/black
       rev: 23.3.0
       hooks:
         - id: black
     
     - repo: https://github.com/pycqa/isort
       rev: 5.12.0
       hooks:
         - id: isort
     
     - repo: https://github.com/pycqa/flake8
       rev: 6.0.0
       hooks:
         - id: flake8
     
     - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.4.0
       hooks:
         - id: trailing-whitespace
         - id: end-of-file-fixer
         - id: check-yaml
         - id: check-added-large-files

Documentation
-------------

Building Docs
~~~~~~~~~~~~~

.. code-block:: bash

   # Build HTML documentation
   cd docs
   make html
   
   # View locally
   open _build/html/index.html
   
   # Clean build
   make clean html
   
   # From root directory
   make docs

Documentation Style
~~~~~~~~~~~~~~~~~~~

Follow these conventions:

* Use reStructuredText (RST) format
* Include code examples for all APIs
* Add mathematical notation with LaTeX
* Cross-reference related content
* Keep examples practical and runnable

Example documentation:

.. code-block:: rst

   GARCH Model
   -----------
   
   .. autoclass:: krl_models.volatility.GARCHModel
      :members:
      :undoc-members:
      :show-inheritance:
   
   The GARCH(p,q) model specifies conditional variance:
   
   .. math::
   
      \sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \epsilon_{t-i}^2 
                  + \sum_{j=1}^p \beta_j \sigma_{t-j}^2
   
   **Example:**
   
   .. code-block:: python
   
      from krl_models.volatility import GARCHModel
      
      model = GARCHModel(
          time_col='date',
          target_col='returns',
          p=1,
          q=1
      )
      
      results = model.fit(data)

Performance Optimization
------------------------

Profiling
~~~~~~~~~

Use cProfile for performance analysis:

.. code-block:: python

   import cProfile
   import pstats
   
   # Profile code
   profiler = cProfile.Profile()
   profiler.enable()
   
   # Your code here
   model.fit(large_dataset)
   
   profiler.disable()
   
   # Print stats
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)

Benchmarking
~~~~~~~~~~~~

Use pytest-benchmark:

.. code-block:: python

   import pytest
   
   def test_arima_performance(benchmark, sample_data):
       """Benchmark ARIMA fitting."""
       model = ARIMAModel(time_col='date', target_col='value', order=(1,1,1))
       
       result = benchmark(model.fit, sample_data)
       
       assert result is not None

Run benchmarks:

.. code-block:: bash

   pytest tests/benchmarks/ --benchmark-only

Optimization Tips
~~~~~~~~~~~~~~~~~

1. **Vectorize operations**: Use NumPy/Pandas operations
2. **Cache results**: Use functools.lru_cache
3. **Parallel processing**: Use joblib for embarrassingly parallel tasks
4. **Efficient data structures**: Choose appropriate dtypes
5. **Profile before optimizing**: Measure, don't guess

Debugging
---------

Debug Mode
~~~~~~~~~~

Enable verbose logging:

.. code-block:: python

   import logging
   
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger('krl_models')
   
   # Now see detailed logs
   model.fit(data)

Interactive Debugging
~~~~~~~~~~~~~~~~~~~~~

Use pdb or ipdb:

.. code-block:: python

   import pdb
   
   def problematic_function():
       # Set breakpoint
       pdb.set_trace()
       
       # Code to debug
       result = complex_calculation()
       return result

VS Code debugging configuration:

.. code-block:: json

   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: Current File",
         "type": "python",
         "request": "launch",
         "program": "${file}",
         "console": "integratedTerminal"
       },
       {
         "name": "Python: Pytest",
         "type": "python",
         "request": "launch",
         "module": "pytest",
         "args": ["-v"]
       }
     ]
   }

Makefile Commands
-----------------

Common commands:

.. code-block:: makefile

   # Install dependencies
   make install
   
   # Run linting
   make lint
   
   # Run tests
   make test
   make test-cov
   
   # Build documentation
   make docs
   
   # Clean build artifacts
   make clean
   
   # Build package
   make build
   
   # All checks
   make check

See Also
--------

* :doc:`contributing` - Contribution guidelines
* :doc:`testing` - Testing guide
* :doc:`../api/core` - Core API for extending models
