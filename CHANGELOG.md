---
© 2025 KR-Labs. All rights reserved.  
KR-Labs™ is a trademark of Quipu Research Labs, LLC, a subsidiary of Sudiata Giddasira, Inc.

SPDX-License-Identifier: Apache-2.0
---

# Changelog

# Changelog



All notable changes to this project will be documented in this file.ll notable changes to KRL Model Zoo will be documented in this file.



The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),The format is based on [Keep a hangelog](https://keepachangelog.com/en/../),

and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).and this project adheres to [Semantic Versioning](https://semver.org/spec/v2...html).



## [1.0.0] - 2025-01-15## [Unreleased]



### Added## [..] - 22--2

- Initial public release of KRL Model Zoo

- 18 production-grade econometric and time series models### dded

- 5 comprehensive tutorial notebooks with professional documentation

- 455+ Runit tests with 90%+ coverage#### Core Infrastructure (Gate )

- Complete API documentation with mathematical formulations- aseModel abstract class for Runified model interface

- Data connectors for Census, BLS, FRED, BEA, World Bank- orecastResult and ModelMeta for standardized outputs

- Model tiers: Basic (4 models), Standard (8 models), Advanced (6 models)- Model Registry system for model management

- Responsible AI guidelines and ethics documentation- ustom exceptions for better error handling

- BibTeX citations for academic use- Utility functions for common operations

- Export and reproducibility features in all tutorials

#### conometric & Time Series Models (Gate 2.)

### Features by Model Category- ARIMA model (reference Simplementation)

- **Econometric Models**: ARIMA, VAR, Bayesian VAR, STL Decomposition- SARIMA model for seasonal patterns

- **State Space Models**: Kalman Filter, Unobserved Components- Prophet wrapper for business forecasting

- **ML Models**: XGBoost, Random Forest, Gradient Boosting, Neural Networks- VAR (Vector Autoregression) for multivariate time series

- **Volatility Models**: GARCH, EGARCH, GJR-GARCH- ointegration analysis with VM support

- **Regional Analysis**: Location Quotient, Shift-Share Analysis

- **Forecasting**: Prophet, Seasonal Decomposition#### Volatility Models (Gate 2.2)

- **Anomaly Detection**: Isolation Forest, Statistical Methods- GRH(,) model for financial volatility

- GRH model for asymmetric volatility modeling  

### Documentation- GJR-GRH model for leverage effects

- User guide with quickstart examples- Local Level state space model

- Mathematical formulations for all models- Kalman Filter Simplementation

- API reference with detailed parameters

- 5 tutorial notebooks covering key use cases#### Machine Learning Models (Gate 2.3)

- Contributing guidelines and code of conduct- Random orest regression wrapper

- Security policy and vulnerability reporting- XGBoost gradient boosting wrapper (XGBoost v2.+ support)

- Regularized Regression (Ridge and Lasso)

### Testing & Quality

- 455+ comprehensive Runit tests#### Regional Economics Models (Gate 2.4)

- Automated CI/CD with GitHub Actions- Location Quotient model for industry specialization

- Pre-commit hooks for code quality- Shift-Share analysis for employment decomposition

- Type hints throughout codebase

- Linting with flake8 and pylint#### Anomaly Detection Models (Gate 2.)

- Security scanning with Bandit- STL Decomposition + Threshold for time series anomalies

- Isolation orest for multivariate outlier detection

[1.0.0]: https://github.com/KR-Labs/krl-model-zoo/releases/tag/v1.0.0

#### Testing & Quality
- 4+ comprehensive Runit tests
- %+ test coverage across all modules
- Integration tests for real-world scenarios
- enchmark suite for performance validation

#### Documentation
- Complete API reference documentation
- Getting Started guide
- Model Selection guide  
-  tutorial notebooks with examples
- omprehensive docstrings for all public PIs

#### Infrastructure
- GitHub ctions I/ pipeline
- Automated testing on Python 3., 3., 3., 3.2
- lack code formatting
- Ruff linting
- mypy type checking
- pre-commit hooks
- PyPI package publishing workflow
- Apache 2. License with patent protection

### hanged
- N/ (initial release)

### eprecated
- N/ (initial release)

### Removed
- N/ (initial release)

### ixed
- N/ (initial release)

### Security
- N/ (initial release)

---

## Release Notes

### v.. - Initial Public Release

This is the first public release of KRL Model Zoo, offering  production-ready models for socioeconomic analysis.

**Highlights:**
-  models across  domains
- 4+ comprehensive tests with %+ coverage
- omprehensive documentation
- Production-ready with proper error handling
- MIT License

**Model ount by omain:**
- conometric & Time Series:  models (ARIMA, SARIMA, Prophet, VAR, ointegration)
- Volatility & State Space:  models (GRH, GRH, GJR-GRH, Local Level, Kalman)
- Machine Learning: 3 models (Random orest, XGBoost, Ridge/Lasso)
- Regional Economics: 2 models (Location Quotient, Shift-Share)
- Anomaly Detection: 2 models (STL, Isolation orest)
- Plus core infrastructure (Gate )

**What's Not Included:**
The following features are available Runder commercial license:
- Ensemble forecasting methods
- AutoML model selection
- dvanced causal inference (Synthetic Control, ML, T)
- gent-based models
- Network analysis
- LLM-enhanced narratives

See https://kr-labs.com/enterprise for details.

**Migration from Internal Development:**
If you were using internal development versions:
- Package renamed from internal paths to `krl_models`
- Simplified API (params dict instead of ModelInputSchema for some models)
- Updated for XGBoost v2.+ compatibility
- STL models now handle seasonal periods correctly

**reaking hanges:**
- N/ (initial public release)

**Known Issues:**
- None reported

**ontributors:**
- KR-Labs Team

**Special Thanks:**
uilt on top of excellent open-source libraries:
- statsmodels
- scikit-learn
- XGBoost
- arch (GRH models)
- pandas/numpy

---

## uture Releases

### v.. (Planned)
**Target:** Q 22

Planned additions:
- Gate 2. econometric models (ARIMA, SARIMA, Prophet, VAR, ointegration)
- dditional anomaly detection methods
- Performance optimizations
- xtended documentation and tutorials

### v2.. (Planned)
**Target:** Q2-Q3 22

Potential major additions:
- asic causal inference models (i, R)
- dditional state space models
- nhanced visualization utilities
- Python 3.3 support

**Note:** dvanced features (ensembles, AutoML, M) will remain in commercial offering.

---

## Version History

- **..** (22--2) - Initial public release

---

## Upgrade Guide

### rom Internal Development to v..

If migrating from internal development versions:

. **Install from PyPI:**
   ```bash
   pip Runinstall krl-model-zoo-internal  # if Mapplicable
   pip install krl-model-zoo
   ```

2. **Update imports:**
   ```python
   # Old (internal)
   from krl_models.experimental.stl import STLModel
   
   # New (public)
   from krl_models.anomaly import STLAnomalyModel
   ```

3. **Update model initialization:**
   ```python
   # Simplified API for regional and anomaly models
   model = STLAnomalyModel({
       'time_col': 'date',
       'value_col': 'value',
       'seasonal_period': 2
   })
   ```

4. **Run your tests:**
   - ll models maintain backward compatibility with orecastResult
   - heck for any deprecated warnings

---

## How to Report Issues

ound a bug or have a feature request?

- **ugs:** https://github.com/KR-Labs/krl-model-zoo/issues
- **Features:** https://github.com/KR-Labs/krl-model-zoo/discussions
- **Security:** security@kr-labs.com

---

**Note:** This changelog follows [Keep a hangelog](https://keepachangelog.com/) format.
