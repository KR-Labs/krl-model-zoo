# KRAnalytics

**Open-Source Socioeconomic Data Science Framework**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Package](https://img.shields.io/badge/Package-v1.0.0-blue)](#installation)

---

## Overview

KRAnalytics is an open-source analytics framework for socioeconomic data science, designed to make advanced analytics accessible to researchers, policymakers, and data scientists.

### What Can You Do With KRAnalytics?

- ** Analyze Socioeconomic Data** - Income inequality, employment trends, housing markets, health outcomes, and more
- ** Build Predictive Models** - Use state-of-the-art machine learning for forecasting and pattern detection
- ** Create Interactive Visualizations** - Generate publication-quality charts with our ML-driven visualization engine
- ** Manage API Keys Securely** - Built-in secure credential management for government data APIs
- ** Ensure Reproducibility** - Complete provenance tracking and execution logging

### Key Capabilities

- **Multi-Domain Analytics** - Ready-to-use workflows for income, employment, education, health, housing, and inequality analysis
- **Advanced ML Algorithms** - XGBoost, Random Forest, Neural Networks, Time Series Forecasting, Clustering, and more
- **Real API Integration** - Direct connections to Census, BLS, FRED, BEA, and other government data sources
- **Interactive Dashboards** - Plotly-powered visualizations with automatic chart selection
- **Academic Standards** - Proper citations, reproducibility packages, and provenance tracking

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/KR-Labs/KRAnalytics.git
cd KRAnalytics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
make install-dev
# Or: pip install -e ".[dev,ml,viz]"

# Verify installation
python -c "from kranalytics import __version__; print(f' KRAnalytics v{__version__} installed')"
```

### Your First Analysis

```python
from kranalytics.utils.plotly_visualization_engine import PlotlyVisualizationEngine
from kranalytics.utils.api_key_manager import load_api_key
import pandas as pd

# Load API key securely (no hardcoding!)
census_key = load_api_key('CENSUS_API_KEY')

# Your analysis code here...
# (See examples/ folder for complete workflows)

# Create visualizations automatically
viz_engine = PlotlyVisualizationEngine()
charts = viz_engine.generate_tier_visualizations(
    data=your_dataframe,
    tier_type="tier_2",
    analysis_focus="income_analysis"
)

# Display charts
for chart in charts:
    chart.show()
```

---

## Features

###  Secure API Management

Never hardcode credentials again. KRAnalytics provides secure API key management:

```python
from kranalytics.utils.api_key_manager import load_api_key

# Loads from environment variables or secure config file
api_key = load_api_key('YOUR_API_NAME')
```

###  ML-Driven Visualizations

Automatic chart generation based on your data characteristics:

```python
from kranalytics.utils.plotly_visualization_engine import PlotlyVisualizationEngine

viz_engine = PlotlyVisualizationEngine()
charts = viz_engine.generate_tier_visualizations(
    data=df,
    tier_type="tier_1",  # descriptive, predictive, clustering, etc.
    analysis_focus="employment",
    domain="Labor Markets"
)
```

###  Advanced Machine Learning

Pre-configured pipelines for common socioeconomic analyses:

- **Predictive Modeling** - Income prediction, employment forecasting, health outcome estimation
- **Time Series Analysis** - ARIMA, Prophet, exponential smoothing for trend forecasting
- **Clustering Analysis** - k-means, DBSCAN, hierarchical clustering for pattern discovery
- **Causal Inference** - Difference-in-differences, propensity score matching for policy evaluation
- **Ensemble Methods** - XGBoost, Random Forest, Gradient Boosting for robust predictions

###  Example Notebooks

Learn by example with our tutorial notebooks:

```
notebooks/
 examples/           # Getting started tutorials
    01_basic_income_analysis.ipynb
    02_employment_forecasting.ipynb
    03_housing_market_clustering.ipynb
    04_health_disparity_analysis.ipynb
 templates/          # Customizable templates
     predictive_model_template.ipynb
     time_series_template.ipynb
     clustering_template.ipynb
```

---

## Supported Data Sources

### Government APIs (Free Registration Required)

- ** U.S. Census Bureau** - Demographics, income, poverty, housing
- ** Bureau of Labor Statistics** - Employment, wages, unemployment
- ** Federal Reserve (FRED)** - Macroeconomic indicators, interest rates
- ** Bureau of Economic Analysis** - GDP, trade, national accounts

### Optional Enhanced Data Sources

- ** National Center for Education Statistics** - Education outcomes
- ** FBI Crime Data API** - Crime statistics
- ** NOAA Climate API** - Weather and climate data
- ** HUD Fair Market Rent** - Housing affordability data

**Getting API Keys:**
- [Census API Key](https://api.census.gov/data/key_signup.html) (Free)
- [BLS API Key](https://www.bls.gov/developers/home.htm) (Free)
- [FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html) (Free)
- [BEA API Key](https://apps.bea.gov/api/signup/) (Free)

---

## Documentation

###  Quick References

- **[Quick Start Guide](./docs/quick-references/QUICK_START_GUIDE.md)** - Get up and running in 5 minutes
- **[API Setup Guide](./docs/api-documentation/)** - Configure your data sources
- **[Visualization Guide](./docs/quick-references/ANALYTICS_ENGINE_QUICK_GUIDE.md)** - Create publication-quality charts

###  Architecture

- **[Platform Overview](./docs/architecture/PLATFORM_OVERVIEW.md)** - System architecture and design
- **[Package Structure](./docs/)** - Understanding the codebase organization

###  Tutorials

- **[Example Notebooks](./notebooks/examples/)** - Learn by doing
- **[Templates](./notebooks/templates/)** - Customize for your analysis

---

## Use Cases

###  Academic Research

- Analyze socioeconomic disparities across regions
- Study employment trends and labor market dynamics
- Investigate health outcome determinants
- Examine housing affordability and displacement patterns

###  Policy Analysis

- Evaluate policy impacts with causal inference methods
- Forecast economic indicators for planning
- Identify underserved communities for resource allocation
- Monitor inequality trends over time

###  Data Science Education

- Learn ML techniques with real-world socioeconomic data
- Practice API integration and data pipelines
- Develop interactive visualization skills
- Build end-to-end analytics projects

###  Nonprofit & Community Organizations

- Demonstrate community needs with data
- Support grant applications with evidence
- Track program outcomes over time
- Advocate for policy changes with analysis

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./docs/CONTRIBUTING.md) for guidelines on:

- Setting up your development environment
- Code quality standards
- Testing requirements
- Pull request process
- Community standards

---

## Development Tools

### Makefile Commands

```bash
make install         # Install package
make install-dev     # Install with development dependencies
make test            # Run test suite
make format          # Auto-format code (black, isort)
make lint            # Run linting (flake8, mypy)
make clean           # Clean build artifacts
```

### Pre-commit Hooks

We use automated code quality checks:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Project Structure

```
KRAnalytics/
 src/kranalytics/              # Main package
    utils/                    # Utility modules
       plotly_visualization_engine.py
       api_key_manager.py
       kaggle_dataset_manager.py
    ...
 notebooks/
    examples/                 # Tutorial notebooks
    templates/                # Customizable templates
 tests/                        # Test suite
 docs/                         # Documentation
 pyproject.toml                # Package configuration
 Makefile                      # Development automation
 README.md                     # This file
```

---

## Requirements

- **Python:** 3.9 or higher
- **Dependencies:** See `pyproject.toml` for complete list
- **Key Packages:**
  - pandas, numpy - Data manipulation
  - scikit-learn - Machine learning
  - xgboost, lightgbm - Gradient boosting
  - plotly - Interactive visualizations
  - statsmodels - Statistical models
  - requests - API integration

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## Acknowledgments

### Data Sources

- **U.S. Census Bureau** - Demographic and economic data
- **Bureau of Labor Statistics** - Employment statistics
- **Federal Reserve Economic Data** - Macroeconomic indicators
- **Bureau of Economic Analysis** - National accounts

### Open Source Libraries

This project builds on excellent open-source tools:
- **Scikit-learn, XGBoost, LightGBM** - Machine learning frameworks
- **Plotly** - Interactive visualization library
- **Pandas, NumPy** - Data manipulation tools
- **Statsmodels** - Statistical modeling
- **Requests** - HTTP library for API integration

---

## Citation

If you use KRAnalytics in your research, please cite:

```bibtex
@software{kranalytics2025,
  author = {Deloatch, Brandon},
  title = {KRAnalytics: Open-Source Socioeconomic Data Science Framework},
  year = {2025},
  publisher = {KR-Labs},
  url = {https://github.com/KR-Labs/KRAnalytics}
}
```

---

## Support & Community

- ** Documentation:** [docs/](./docs/)
- ** Report Issues:** [GitHub Issues](https://github.com/KR-Labs/KRAnalytics/issues)
- ** Discussions:** [GitHub Discussions](https://github.com/KR-Labs/KRAnalytics/discussions)
- ** Email:** info@krlabs.dev

---

## Roadmap

### Current Features (v1.0.0)
-  Core analytics framework
-  API integration modules
-  Visualization engine
-  Example notebooks
-  Comprehensive documentation

### Planned Features
-  Additional example notebooks for more domains
-  Interactive web dashboard
-  Plugin system for custom data sources
-  Cloud deployment templates
-  Automated report generation

See [docs/roadmaps/](./docs/roadmaps/) for detailed plans.

---

## Related Projects

- **[Quipu Analytics Suite](https://github.com/QuipuAnalytics)** - Enterprise analytics platform
- **[Analytics Model Matrix](./docs/)** - Comprehensive analytics framework documentation

---

**Made with  by KR-Labs**

**Version:** 1.0.0 | **Last Updated:** October 2025 | **Status:** Production Ready
