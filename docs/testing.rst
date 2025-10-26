Testing Guide
=============

Comprehensive guide to testing KRL Model Zoo.

Testing Philosophy
------------------

Testing Principles
~~~~~~~~~~~~~~~~~~

1. **Test behavior, not implementation**: Focus on what code does, not how
2. **Test edge cases**: Include boundary conditions and error cases
3. **Keep tests independent**: Each test should run in isolation
4. **Make tests readable**: Tests are documentation
5. **Fast feedback**: Keep tests fast for rapid iteration

Test Pyramid
~~~~~~~~~~~~

Our testing strategy follows the test pyramid:

* **70% Unit Tests**: Fast, isolated, test individual components
* **20% Integration Tests**: Test component interactions
* **10% End-to-End Tests**: Test complete workflows

Running Tests
-------------

Basic Commands
~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with verbose output
   pytest -v
   
   # Run specific test file
   pytest tests/test_arima.py
   
   # Run specific test
   pytest tests/test_arima.py::TestARIMAModel::test_fit
   
   # Run tests matching pattern
   pytest -k "arima"
   
   # Run with coverage
   pytest --cov=krl_models --cov=krl_core
   
   # Generate HTML coverage report
   pytest --cov=krl_models --cov=krl_core --cov-report=html
   
   # Stop at first failure
   pytest -x
   
   # Run last failed tests
   pytest --lf

Using Makefile
~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   make test
   
   # Run with coverage
   make test-cov
   
   # Run integration tests only
   make test-integration
   
   # Run unit tests only
   make test-unit

Writing Unit Tests
------------------

Basic Test Structure
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   import pandas as pd
   import numpy as np
   from krl_models.econometric import ARIMAModel
   from krl_core import ModelResults
   
   
   class TestARIMAModel:
       """Test suite for ARIMA model."""
       
       def test_initialization(self):
           """Test model can be initialized with valid parameters."""
           model = ARIMAModel(
               time_col='date',
               target_col='value',
               order=(1, 1, 1)
           )
           
           assert model.order == (1, 1, 1)
           assert model.time_col == 'date'
           assert model.target_col == 'value'
       
       def test_initialization_invalid_order(self):
           """Test model raises error with invalid order."""
           with pytest.raises(ValueError, match="Order must be non-negative"):
               ARIMAModel(
                   time_col='date',
                   target_col='value',
                   order=(-1, 1, 1)
               )

Using Fixtures
~~~~~~~~~~~~~~

Fixtures provide reusable test data:

.. code-block:: python

   import pytest
   import pandas as pd
   import numpy as np
   
   
   @pytest.fixture
   def simple_time_series():
       """Generate simple time series for testing."""
       np.random.seed(42)
       return pd.DataFrame({
           'date': pd.date_range('2020-01-01', periods=100, freq='M'),
           'value': np.cumsum(np.random.normal(0, 1, 100)) + 100
       })
   
   
   @pytest.fixture
   def seasonal_time_series():
       """Generate seasonal time series."""
       np.random.seed(42)
       t = np.arange(120)
       trend = t * 0.5
       seasonal = 10 * np.sin(2 * np.pi * t / 12)
       noise = np.random.normal(0, 2, 120)
       
       return pd.DataFrame({
           'date': pd.date_range('2010-01-01', periods=120, freq='M'),
           'value': trend + seasonal + noise + 100
       })
   
   
   class TestARIMAModel:
       """Test ARIMA model."""
       
       def test_fit_simple_series(self, simple_time_series):
           """Test fitting on simple time series."""
           model = ARIMAModel(
               time_col='date',
               target_col='value',
               order=(1, 1, 1)
           )
           
           results = model.fit(simple_time_series)
           
           assert isinstance(results, ModelResults)
           assert len(results.fitted_values) == len(simple_time_series)
       
       def test_fit_seasonal_series(self, seasonal_time_series):
           """Test fitting on seasonal time series."""
           model = ARIMAModel(
               time_col='date',
               target_col='value',
               order=(1, 1, 1)
           )
           
           results = model.fit(seasonal_time_series)
           assert results is not None

Parametrized Tests
~~~~~~~~~~~~~~~~~~

Test multiple scenarios with one test:

.. code-block:: python

   import pytest
   
   
   @pytest.mark.parametrize("order,expected_params", [
       ((1, 0, 0), 1),  # AR(1) has 1 parameter
       ((2, 0, 0), 2),  # AR(2) has 2 parameters
       ((0, 0, 1), 1),  # MA(1) has 1 parameter
       ((1, 0, 1), 2),  # ARMA(1,1) has 2 parameters
   ])
   def test_parameter_count(order, expected_params, simple_time_series):
       """Test correct number of parameters for different orders."""
       model = ARIMAModel(
           time_col='date',
           target_col='value',
           order=order
       )
       
       results = model.fit(simple_time_series)
       assert len(results.params) == expected_params
   
   
   @pytest.mark.parametrize("invalid_order", [
       (-1, 1, 1),
       (1, -1, 1),
       (1, 1, -1),
       (None, 1, 1),
   ])
   def test_invalid_orders(invalid_order):
       """Test various invalid order specifications."""
       with pytest.raises((ValueError, TypeError)):
           ARIMAModel(
               time_col='date',
               target_col='value',
               order=invalid_order
           )

Testing Exceptions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   
   
   def test_fit_with_missing_data(simple_time_series):
       """Test that model raises error with missing data."""
       # Add missing values
       data_with_na = simple_time_series.copy()
       data_with_na.loc[10:15, 'value'] = np.nan
       
       model = ARIMAModel(
           time_col='date',
           target_col='value',
           order=(1, 1, 1)
       )
       
       with pytest.raises(ValueError, match="Missing values detected"):
           model.fit(data_with_na)
   
   
   def test_predict_before_fit():
       """Test that predict raises error if called before fit."""
       model = ARIMAModel(
           time_col='date',
           target_col='value',
           order=(1, 1, 1)
       )
       
       with pytest.raises(RuntimeError, match="Model must be fitted"):
           model.predict(steps=10)

Integration Tests
-----------------

Testing Component Interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from krl_models.econometric import SARIMAModel
   from krl_core import ModelRegistry
   
   
   class TestModelIntegration:
       """Test integration between models and core components."""
       
       def test_model_registry_integration(self, simple_time_series):
           """Test model registration and retrieval."""
           registry = ModelRegistry()
           
           # Register model
           registry.register('sarima', SARIMAModel, {
               'description': 'Seasonal ARIMA',
               'category': 'econometric'
           })
           
           # Retrieve and use
           ModelClass = registry.get_model('sarima')
           model = ModelClass(
               time_col='date',
               target_col='value',
               order=(1, 1, 1),
               seasonal_order=(1, 1, 1, 12)
           )
           
           results = model.fit(simple_time_series)
           assert results is not None
       
       def test_results_export_integration(self, simple_time_series, tmp_path):
           """Test model results export functionality."""
           model = SARIMAModel(
               time_col='date',
               target_col='value',
               order=(1, 1, 1),
               seasonal_order=(1, 1, 1, 12)
           )
           
           results = model.fit(simple_time_series)
           
           # Export to CSV
           csv_path = tmp_path / "results.csv"
           results.export_to_csv(csv_path)
           assert csv_path.exists()
           
           # Export to JSON
           json_path = tmp_path / "results.json"
           results.export_to_json(json_path)
           assert json_path.exists()

Testing Workflows
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_complete_forecasting_workflow(simple_time_series):
       """Test complete workflow from data to forecast."""
       # Split data
       train = simple_time_series[:-12]
       test = simple_time_series[-12:]
       
       # Fit model
       model = SARIMAModel(
           time_col='date',
           target_col='value',
           order=(1, 1, 1),
           seasonal_order=(1, 1, 1, 12)
       )
       
       results = model.fit(train)
       
       # Generate forecast
       forecast = model.predict(steps=12)
       
       # Validate forecast
       assert len(forecast) == 12
       assert 'forecast' in forecast.columns
       
       # Calculate accuracy
       from sklearn.metrics import mean_squared_error
       mse = mean_squared_error(test['value'], forecast['forecast'])
       assert mse > 0  # Basic sanity check

End-to-End Tests
----------------

Complete Application Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from krl_models.econometric import ARIMAModel, SARIMAModel, ProphetModel
   from sklearn.metrics import mean_absolute_error
   
   
   class TestEndToEnd:
       """End-to-end tests for complete use cases."""
       
       def test_model_comparison_workflow(self, seasonal_time_series):
           """Test complete model comparison workflow."""
           # Split data
           train = seasonal_time_series[:-24]
           test = seasonal_time_series[-24:]
           
           # Define models
           models = {
               'ARIMA': ARIMAModel(
                   time_col='date',
                   target_col='value',
                   order=(1, 1, 1)
               ),
               'SARIMA': SARIMAModel(
                   time_col='date',
                   target_col='value',
                   order=(1, 1, 1),
                   seasonal_order=(1, 1, 1, 12)
               ),
           }
           
           # Fit and evaluate
           results = {}
           for name, model in models.items():
               model.fit(train)
               forecast = model.predict(steps=24)
               mae = mean_absolute_error(test['value'], forecast['forecast'])
               results[name] = mae
           
           # Validate results
           assert all(mae > 0 for mae in results.values())
           assert 'ARIMA' in results
           assert 'SARIMA' in results

Mocking and Patching
--------------------

Mocking External Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from unittest.mock import Mock, patch, MagicMock
   
   
   def test_with_mocked_data_source():
       """Test model with mocked external data source."""
       # Mock data loader
       mock_loader = Mock()
       mock_loader.load_data.return_value = pd.DataFrame({
           'date': pd.date_range('2020-01-01', periods=100, freq='M'),
           'value': np.random.normal(100, 10, 100)
       })
       
       # Use mocked data
       data = mock_loader.load_data()
       model = ARIMAModel(time_col='date', target_col='value', order=(1,1,1))
       results = model.fit(data)
       
       assert results is not None
       mock_loader.load_data.assert_called_once()

Patching Functions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @patch('krl_models.econometric.arima.some_expensive_function')
   def test_with_patched_function(mock_func, simple_time_series):
       """Test with patched expensive function."""
       mock_func.return_value = "mocked_result"
       
       model = ARIMAModel(time_col='date', target_col='value', order=(1,1,1))
       results = model.fit(simple_time_series)
       
       mock_func.assert_called()

Performance Testing
-------------------

Benchmarking
~~~~~~~~~~~~

.. code-block:: python

   import pytest
   
   
   @pytest.mark.benchmark
   def test_arima_fit_performance(benchmark, simple_time_series):
       """Benchmark ARIMA model fitting."""
       model = ARIMAModel(time_col='date', target_col='value', order=(1,1,1))
       
       result = benchmark(model.fit, simple_time_series)
       
       assert result is not None
   
   
   @pytest.mark.benchmark
   def test_prediction_performance(benchmark, fitted_arima_model):
       """Benchmark prediction generation."""
       result = benchmark(fitted_arima_model.predict, steps=12)
       
       assert len(result) == 12

Run benchmarks:

.. code-block:: bash

   pytest tests/benchmarks/ --benchmark-only
   
   # Compare with baseline
   pytest tests/benchmarks/ --benchmark-compare

Memory Profiling
~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from memory_profiler import profile
   
   
   @profile
   def test_memory_usage(large_time_series):
       """Profile memory usage during fitting."""
       model = ARIMAModel(time_col='date', target_col='value', order=(1,1,1))
       results = model.fit(large_time_series)
       return results

Test Coverage
-------------

Measuring Coverage
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate coverage report
   pytest --cov=krl_models --cov=krl_core --cov-report=term-missing
   
   # HTML report
   pytest --cov=krl_models --cov=krl_core --cov-report=html
   open htmlcov/index.html
   
   # XML report (for CI)
   pytest --cov=krl_models --cov=krl_core --cov-report=xml

Coverage Goals
~~~~~~~~~~~~~~

* **Overall**: 90%+ coverage
* **Core modules**: 95%+ coverage
* **Models**: 90%+ coverage
* **Utilities**: 85%+ coverage

Improving Coverage
~~~~~~~~~~~~~~~~~~

Identify uncovered lines:

.. code-block:: bash

   pytest --cov=krl_models --cov-report=term-missing

Add tests for uncovered code:

.. code-block:: python

   def test_uncovered_edge_case():
       """Test previously uncovered edge case."""
       # Test specific condition that wasn't covered
       model = ARIMAModel(time_col='date', target_col='value', order=(1,1,1))
       
       # Test edge case
       with pytest.raises(ValueError):
           model.some_uncovered_method(invalid_input)

Testing Best Practices
-----------------------

Test Organization
~~~~~~~~~~~~~~~~~

1. **Group related tests**: Use test classes
2. **Descriptive names**: test_what_when_expected
3. **One assertion per test**: Keep tests focused
4. **Arrange-Act-Assert**: Clear test structure
5. **DRY with fixtures**: Reuse test data

.. code-block:: python

   class TestARIMAModel:
       """Tests for ARIMA model."""
       
       def test_fit_succeeds_with_valid_data(self, simple_time_series):
           """Test that fit succeeds with valid input data."""
           # Arrange
           model = ARIMAModel(
               time_col='date',
               target_col='value',
               order=(1, 1, 1)
           )
           
           # Act
           results = model.fit(simple_time_series)
           
           # Assert
           assert isinstance(results, ModelResults)

Test Data Management
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # conftest.py - shared fixtures
   import pytest
   import pandas as pd
   import numpy as np
   
   
   @pytest.fixture(scope="session")
   def sample_data_dir(tmp_path_factory):
       """Create temporary directory for test data."""
       return tmp_path_factory.mktemp("data")
   
   
   @pytest.fixture(scope="module")
   def large_time_series():
       """Generate large time series for performance tests."""
       np.random.seed(42)
       return pd.DataFrame({
           'date': pd.date_range('2000-01-01', periods=5000, freq='D'),
           'value': np.cumsum(np.random.normal(0, 1, 5000)) + 1000
       })

Continuous Testing
------------------

Pre-commit Testing
~~~~~~~~~~~~~~~~~~

Run fast tests before commit:

.. code-block:: bash

   # Add to .pre-commit-config.yaml
   - repo: local
     hooks:
       - id: pytest-quick
         name: Run quick tests
         entry: pytest tests/unit -x --tb=short
         language: system
         pass_filenames: false

CI/CD Integration
~~~~~~~~~~~~~~~~~

GitHub Actions workflow:

.. code-block:: yaml

   name: Tests
   
   on: [push, pull_request]
   
   jobs:
     test:
       runs-on: ${{ matrix.os }}
       strategy:
         matrix:
           os: [ubuntu-latest, macos-latest, windows-latest]
           python-version: ["3.9", "3.10", "3.11"]
       
       steps:
         - uses: actions/checkout@v3
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: ${{ matrix.python-version }}
         - name: Install dependencies
           run: |
             pip install -e ".[test]"
         - name: Run tests
           run: |
             pytest --cov=krl_models --cov=krl_core --cov-report=xml
         - name: Upload coverage
           uses: codecov/codecov-action@v3
           with:
             file: ./coverage.xml

Troubleshooting Tests
---------------------

Common Issues
~~~~~~~~~~~~~

**Tests pass locally but fail in CI:**

* Check Python version differences
* Verify all dependencies installed
* Check for timezone issues
* Look for filesystem path issues

**Flaky tests:**

* Add explicit waits for async operations
* Set random seeds
* Avoid time-dependent assertions
* Isolate external dependencies

**Slow tests:**

* Use pytest-xdist for parallel execution
* Mock expensive operations
* Reduce test data size
* Profile to find bottlenecks

Debug Failed Tests
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run with verbose output
   pytest -vv
   
   # Show local variables on failure
   pytest -l
   
   # Drop into debugger on failure
   pytest --pdb
   
   # Run only failed tests
   pytest --lf
   
   # Stop at first failure
   pytest -x

See Also
--------

* :doc:`development` - Development setup
* :doc:`contributing` - Contribution guidelines
* `pytest documentation <https://docs.pytest.org/>`_
