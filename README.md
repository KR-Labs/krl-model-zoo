---
© 2025 KR-Labs. All rights reserved.  
KR-Labs™ is a trademark of Quipu Research Labs, LLC, a subsidiary of Sudiata Giddasira, Inc.

SPDX-License-Identifier: Apache-2.0
---

# KRL Model Zoo

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Tests](https://github.com/KR-Labs/krl-model-zoo/workflows/tests/badge.svg)](https://github.com/KR-Labs/krl-model-zoo/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.krlabs.dev/model-zoo)

**Production-ready models for causal inference, forecasting, and policy evaluation.**

Part of the [KRL Analytics Suite](https://krlabs.dev) - an open-source platform for economic analysis, causal inference, and policy evaluation.

## Overview

KRL Model Zoo provides battle-tested implementations of econometric and machine learning models for:

- **Causal Inference** - Difference-in-Differences (DiD), Synthetic Control, RDD
- **Forecasting** - ARIMA, VAR, SARIMA, Prophet
- **Policy Evaluation** - Treatment effects, propensity score matching
- **Time Series Analysis** - Seasonality, trend decomposition, structural breaks

### Key Features

✨ **Unified API** - Consistent interface across all models  
⚡ **Production-Ready** - Comprehensive validation and error handling  
🔒 **Type-Safe** - Full type hints and pydantic validation  
📊 **Rich Output** - Automatic plots, tables, and diagnostics  
🚀 **Performant** - Optimized implementations with caching  
🧪 **Well-Tested** - >90% test coverage  
📚 **Well-Documented** - Extensive examples and tutorials  

## Installation

```bash
# Basic installation
pip install krl-model-zoo

# With all optional dependencies
pip install krl-model-zoo[all]

# Development installation
pip install krl-model-zoo[dev]
```

## Quick Start

### Difference-in-Differences

```python
from krl_models import DifferenceInDifferences
import pandas as pd

# Load data
data = pd.read_csv("policy_data.csv")

# Initialize model
did = DifferenceInDifferences(
    data=data,
    outcome="employment_rate",
    treatment="treated",
    time="year",
    unit="state",
    treatment_period=2020
)

# Fit model
results = did.fit()

# Get treatment effect
print(f"Average Treatment Effect: {results.ate:.3f}")
print(f"Standard Error: {results.se:.3f}")
print(f"95% CI: [{results.ci_lower:.3f}, {results.ci_upper:.3f}]")

# Parallel trends test
did.plot_parallel_trends()
```

### ARIMA Forecasting

```python
from krl_models import ARIMAModel
import pandas as pd

# Load time series data
data = pd.read_csv("economic_series.csv", index_col="date", parse_dates=True)

# Initialize model
arima = ARIMAModel(
    data=data["gdp"],
    order=(1, 1, 1),  # (p, d, q)
    seasonal_order=(1, 1, 1, 4)  # (P, D, Q, s) for quarterly data
)

# Fit model
arima.fit()

# Forecast 8 quarters ahead
forecast = arima.forecast(steps=8)

# Plot results
arima.plot_forecast(forecast)
```

## Available Models

### Causal Inference

#### Difference-in-Differences (DiD)
- Standard DiD with parallel trends assumption
- Staggered adoption support
- Robust standard errors
- Parallel trends testing

#### Synthetic Control
- Donor pool selection
- Pre-treatment fit diagnostics
- Placebo tests
- In-space placebo tests

#### Regression Discontinuity Design (RDD)
- Sharp and fuzzy RDD
- Bandwidth selection (IK, CCT)
- Polynomial order selection
- Robustness checks

### Forecasting Models

#### ARIMA / SARIMA
- Auto ARIMA (automatic order selection)
- Seasonal decomposition
- Model diagnostics (ACF, PACF, residuals)
- Out-of-sample validation

#### VAR (Vector Autoregression)
- Multi-variate time series
- Granger causality tests
- Impulse response functions
- Forecast error variance decomposition

### Policy Evaluation

#### Propensity Score Matching
- Multiple matching algorithms (nearest neighbor, kernel, radius)
- Balance diagnostics
- Common support checking
- Sensitivity analysis

## Architecture

### BaseModel

All models inherit from `BaseModel`, which provides:

- **Data Validation** - Pydantic-based schema validation
- **Caching** - Fitted model caching
- **Logging** - Structured logging of fit/predict operations
- **Serialization** - Save/load models to disk
- **Diagnostics** - Automatic diagnostic plots and tests

```python
from abc import ABC, abstractmethod
from krl_core import get_logger, FileCache

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, data, **kwargs):
        self.logger = get_logger(self.__class__.__name__)
        self.cache = FileCache(namespace="models")
        self.data = data
        self._fitted = False
    
    @abstractmethod
    def fit(self):
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, **kwargs):
        """Make predictions."""
        pass
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific model tests
pytest tests/unit/test_did_model.py -v

# Run integration tests
pytest tests/integration/ -v
```

## Development

```bash
# Clone repository
git clone https://github.com/KR-Labs/krl-model-zoo.git
cd krl-model-zoo

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev,test]"

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the **Apache License 2.0** - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: https://docs.krlabs.dev/model-zoo
- **Issue Tracker**: https://github.com/KR-Labs/krl-model-zoo/issues
- **Discussions**: https://github.com/KR-Labs/krl-model-zoo/discussions

## Related Projects

- **[krl-open-core](https://github.com/KR-Labs/krl-open-core)** - Core utilities
- **[krl-data-connectors](https://github.com/KR-Labs/krl-data-connectors)** - Data access layer
- **[krl-dashboard](https://github.com/KR-Labs/krl-dashboard)** - Interactive visualization

## Citation

```bibtex
@software{krl_model_zoo,
  title = {KRL Model Zoo: Production-Ready Econometric Models},
  author = {KR-Labs Foundation},
  year = {2025},
  url = {https://github.com/KR-Labs/krl-model-zoo}
}
```

---

**Built with ❤️ by [KR-Labs Foundation](https://krlabs.dev)**
