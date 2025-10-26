Contributing to KRL Model Zoo
===============================

Thank you for your interest in contributing to KRL Model Zoo! This guide will help you get started.

Overview
--------

KRL Model Zoo welcomes contributions in the form of:

* Bug reports and fixes
* New model implementations
* Documentation improvements
* Performance optimizations
* Test coverage improvements
* Example notebooks and tutorials

Getting Started
---------------

1. Fork the Repository
~~~~~~~~~~~~~~~~~~~~~~~

Fork the repository on GitHub and clone your fork:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/krl-model-zoo.git
   cd krl-model-zoo

2. Set Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a virtual environment and install development dependencies:

.. code-block:: bash

   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install in editable mode with dev dependencies
   pip install -e ".[dev,test,docs]"
   
   # Install pre-commit hooks
   pre-commit install

3. Create a Branch
~~~~~~~~~~~~~~~~~~

Create a feature branch for your changes:

.. code-block:: bash

   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix

Development Guidelines
----------------------

Code Style
~~~~~~~~~~

We follow PEP 8 style guidelines with these tools:

* **Black**: Code formatting (line length: 100)
* **isort**: Import sorting
* **flake8**: Linting
* **mypy**: Type checking

Format your code before committing:

.. code-block:: bash

   # Format code
   black krl_models/ krl_core/ tests/
   
   # Sort imports
   isort krl_models/ krl_core/ tests/
   
   # Check linting
   flake8 krl_models/ krl_core/ tests/
   
   # Type check
   mypy krl_models/ krl_core/

Or run all checks at once:

.. code-block:: bash

   make lint

Type Hints
~~~~~~~~~~

All code must include type hints:

.. code-block:: python

   from typing import Optional, List, Dict, Any
   import pandas as pd
   from krl_core import ModelResults
   
   def fit(
       self,
       data: pd.DataFrame,
       exog: Optional[pd.DataFrame] = None
   ) -> ModelResults:
       """Fit model with type hints."""
       ...

Docstrings
~~~~~~~~~~

Use Google-style docstrings:

.. code-block:: python

   def predict(self, steps: int = 1, **kwargs) -> pd.DataFrame:
       """Generate multi-step forecasts.
       
       Args:
           steps: Number of periods to forecast.
           **kwargs: Additional keyword arguments.
       
       Returns:
           DataFrame with forecast values and prediction intervals.
       
       Raises:
           ValueError: If steps is not positive.
       
       Example:
           >>> model = ARIMAModel(time_col='date', target_col='value', order=(1,1,1))
           >>> results = model.fit(data)
           >>> forecast = model.predict(steps=12)
       """
       ...

Testing
-------

Writing Tests
~~~~~~~~~~~~~

All new code must include tests. We use pytest:

.. code-block:: python

   import pytest
   import pandas as pd
   import numpy as np
   from krl_models.econometric import ARIMAModel
   
   
   class TestARIMAModel:
       """Test suite for ARIMA model."""
       
       @pytest.fixture
       def sample_data(self):
           """Create sample time series data."""
           return pd.DataFrame({
               'date': pd.date_range('2020-01-01', periods=100, freq='M'),
               'value': np.cumsum(np.random.normal(0, 1, 100)) + 100
           })
       
       def test_initialization(self):
           """Test model initialization."""
           model = ARIMAModel(
               time_col='date',
               target_col='value',
               order=(1, 1, 1)
           )
           assert model.order == (1, 1, 1)
       
       def test_fit(self, sample_data):
           """Test model fitting."""
           model = ARIMAModel(
               time_col='date',
               target_col='value',
               order=(1, 1, 1)
           )
           results = model.fit(sample_data)
           
           assert results is not None
           assert len(results.fitted_values) == len(sample_data)
       
       def test_predict(self, sample_data):
           """Test prediction."""
           model = ARIMAModel(
               time_col='date',
               target_col='value',
               order=(1, 1, 1)
           )
           results = model.fit(sample_data)
           forecast = model.predict(steps=12)
           
           assert len(forecast) == 12
           assert 'forecast' in forecast.columns

Running Tests
~~~~~~~~~~~~~

Run tests with pytest:

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run specific test file
   pytest tests/test_arima.py
   
   # Run with coverage
   pytest --cov=krl_models --cov=krl_core --cov-report=html
   
   # Run specific test
   pytest tests/test_arima.py::TestARIMAModel::test_fit

Or use make:

.. code-block:: bash

   make test
   make test-cov

Test Coverage
~~~~~~~~~~~~~

We aim for 90%+ test coverage. Check coverage:

.. code-block:: bash

   pytest --cov=krl_models --cov=krl_core --cov-report=term-missing

View detailed HTML report:

.. code-block:: bash

   pytest --cov=krl_models --cov=krl_core --cov-report=html
   open htmlcov/index.html

Adding New Models
-----------------

Model Implementation
~~~~~~~~~~~~~~~~~~~~

New models must inherit from BaseModel:

.. code-block:: python

   from krl_core import BaseModel, ModelResults
   from typing import Optional
   import pandas as pd
   
   
   class MyNewModel(BaseModel):
       """Brief description of the model.
       
       Longer description with mathematical background,
       use cases, and references.
       
       Args:
           time_col: Name of datetime column.
           target_col: Name of target variable column.
           custom_param: Description of custom parameter.
       
       Example:
           >>> model = MyNewModel(time_col='date', target_col='value', custom_param=1.0)
           >>> results = model.fit(data)
           >>> forecast = model.predict(steps=12)
       """
       
       def __init__(
           self,
           time_col: str,
           target_col: str,
           custom_param: float = 1.0,
           **kwargs
       ):
           super().__init__(
               time_col=time_col,
               target_col=target_col,
               model_type='my_new_model',
               **kwargs
           )
           self.custom_param = custom_param
           self._validate_params()
       
       def _validate_params(self) -> None:
           """Validate model parameters."""
           if self.custom_param <= 0:
               raise ValueError("custom_param must be positive")
       
       def fit(self, data: pd.DataFrame) -> ModelResults:
           """Fit the model.
           
           Args:
               data: Time series data.
           
           Returns:
               Model results object.
           """
           # Validate input
           self.validate_input(data)
           
           # Your implementation here
           fitted_values = self._fit_implementation(data)
           residuals = data[self.target_col] - fitted_values
           
           # Return results
           return ModelResults(
               model=self,
               fitted_values=fitted_values,
               residuals=residuals,
               metadata=self.get_metadata()
           )
       
       def predict(
           self,
           steps: int = 1,
           **kwargs
       ) -> pd.DataFrame:
           """Generate predictions.
           
           Args:
               steps: Number of periods to forecast.
               **kwargs: Additional arguments.
           
           Returns:
               DataFrame with forecasts.
           """
           # Your implementation here
           predictions = self._predict_implementation(steps)
           
           return pd.DataFrame({
               'forecast': predictions
           })

Required Components
~~~~~~~~~~~~~~~~~~~

Every new model needs:

1. **Model class** in appropriate module (krl_models/)
2. **Unit tests** in tests/
3. **Documentation** in docs/api/ and docs/user_guide/
4. **Example script** in examples/
5. **Entry in model registry**

Example Script
~~~~~~~~~~~~~~

Create an example in examples/:

.. code-block:: python

   """
   Example: My New Model
   =====================
   
   Demonstrates usage of MyNewModel for forecasting.
   """
   
   import pandas as pd
   import numpy as np
   from krl_models.my_module import MyNewModel
   
   
   def main():
       # Generate sample data
       data = pd.DataFrame({
           'date': pd.date_range('2020-01-01', periods=100, freq='M'),
           'value': np.cumsum(np.random.normal(0, 1, 100)) + 100
       })
       
       # Initialize and fit model
       model = MyNewModel(
           time_col='date',
           target_col='value',
           custom_param=1.5
       )
       
       results = model.fit(data)
       
       # Generate forecast
       forecast = model.predict(steps=12)
       
       # Visualize
       results.plot()
       
       print("Forecast:")
       print(forecast)
   
   
   if __name__ == '__main__':
       main()

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Build docs locally:

.. code-block:: bash

   cd docs
   make html
   open _build/html/index.html

Or use make from root:

.. code-block:: bash

   make docs

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

Documentation uses reStructuredText (RST) format:

* Add API docs to docs/api/
* Add user guide to docs/user_guide/
* Add examples to examples/
* Update CHANGELOG.md

API Documentation
~~~~~~~~~~~~~~~~~

API docs are auto-generated from docstrings:

.. code-block:: rst

   My New Model
   ------------
   
   .. autoclass:: krl_models.my_module.MyNewModel
      :members:
      :undoc-members:
      :show-inheritance:

Pull Request Process
---------------------

Before Submitting
~~~~~~~~~~~~~~~~~

1. Ensure all tests pass
2. Update documentation
3. Add entry to CHANGELOG.md
4. Run linting and formatting
5. Update examples if needed

Submitting PR
~~~~~~~~~~~~~

1. Push your branch to your fork
2. Create pull request on GitHub
3. Fill out PR template
4. Wait for CI checks to pass
5. Respond to review comments

PR Template:

.. code-block:: text

   ## Description
   Brief description of changes.
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] All tests pass
   - [ ] Added new tests
   - [ ] Updated documentation
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Type hints included
   - [ ] Docstrings complete
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated

Code Review
~~~~~~~~~~~

All PRs require:

* Passing CI checks
* Code review approval
* No merge conflicts
* Updated documentation
* Adequate test coverage

Release Process
---------------

Versioning
~~~~~~~~~~

We follow Semantic Versioning (SemVer):

* MAJOR: Incompatible API changes
* MINOR: New features, backward compatible
* PATCH: Bug fixes, backward compatible

Example: 1.2.3 (MAJOR.MINOR.PATCH)

Creating Release
~~~~~~~~~~~~~~~~

1. Update version in pyproject.toml
2. Update CHANGELOG.md
3. Create git tag
4. Push to GitHub
5. GitHub Actions builds and publishes to PyPI

.. code-block:: bash

   # Update version and changelog
   vim pyproject.toml
   vim CHANGELOG.md
   
   # Commit changes
   git add pyproject.toml CHANGELOG.md
   git commit -m "Bump version to 1.2.0"
   
   # Create tag
   git tag -a v1.2.0 -m "Release version 1.2.0"
   
   # Push
   git push origin main --tags

Community
---------

Communication Channels
~~~~~~~~~~~~~~~~~~~~~~

* GitHub Issues: Bug reports and feature requests
* GitHub Discussions: Questions and discussions
* Pull Requests: Code contributions

Code of Conduct
~~~~~~~~~~~~~~~

We follow the Contributor Covenant Code of Conduct. Be respectful and inclusive in all interactions.

Getting Help
~~~~~~~~~~~~

* Check documentation
* Search existing issues
* Ask in GitHub Discussions
* Tag issues appropriately

Recognition
-----------

Contributors are recognized in:

* CONTRIBUTORS.md file
* Release notes
* GitHub contributors page

Thank you for contributing to KRL Model Zoo!

See Also
--------

* :doc:`development` - Development setup and workflows
* :doc:`testing` - Testing guidelines
* :doc:`../api/core` - Core utilities for extending models
