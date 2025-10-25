# Contributing to KRAnalytics

Thank you for your interest in contributing to KRAnalytics! This document provides guidelines for contributing to our open-source socioeconomic data science framework.

---

##  Project Vision

KRAnalytics aims to make advanced socioeconomic data analysis accessible to researchers, policymakers, and data scientists through:

- **Open Access** - Free, open-source tools for public good
- **Academic Rigor** - Sound methodologies with proper documentation
- **Code Quality** - Professional standards and comprehensive testing
- **Community Driven** - Collaborative development and peer review

---

##  Getting Started

### Before You Contribute

1. **Read the Documentation** - Familiarize yourself with [README.md](./README.md)
2. **Check Existing Issues** - See if someone is already working on your idea
3. **Discuss Major Changes** - Open an issue to discuss significant contributions before starting work
4. **Fork the Repository** - Create your own fork to work on

### Development Environment Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/KRAnalytics.git
cd KRAnalytics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
make install-dev

# Install pre-commit hooks
pre-commit install

# Verify setup
make test
```

---

##  Contribution Types

We welcome various types of contributions:

###  Bug Reports

Found a bug? Help us fix it:

1. **Check Existing Issues** - Make sure it hasn't been reported
2. **Create Detailed Report** - Include:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version)
   - Error messages/stack traces
3. **Use Bug Report Template** - Follow `.github/ISSUE_TEMPLATE/bug_report.md`

###  Feature Requests

Have an idea for improvement?

1. **Check Roadmap** - See if it's already planned
2. **Open Feature Request** - Describe:
   - Problem you're solving
   - Proposed solution
   - Use cases
   - Alternatives considered
3. **Discuss First** - For major features, discuss before implementing

###  Documentation

Help improve our docs:

- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation
- Create how-to guides
- Translate documentation

###  Example Notebooks

Contribute tutorial notebooks:

- Demonstrate specific analysis techniques
- Show integration with new data sources
- Provide domain-specific examples
- Include clear explanations and comments

###  Code Contributions

Improve the codebase:

- Fix bugs
- Implement new features
- Optimize performance
- Improve test coverage
- Enhance error handling

---

##  Development Guidelines

### Code Quality Standards

We maintain high code quality standards:

```bash
# Format code (runs automatically with pre-commit)
make format

# Run linting
make lint

# Run tests
make test

# Run tests with coverage
make test-cov
```

#### Style Guidelines

- **PEP 8** - Follow Python style guide
- **100 character line length** - For readability
- **Type hints** - Use where appropriate
- **Docstrings** - Document all public functions/classes
- **Comments** - Explain complex logic

#### Example Code Style

```python
from typing import Dict, List, Optional
import pandas as pd


def analyze_income_distribution(
    data: pd.DataFrame,
    income_column: str = "income",
    group_by: Optional[str] = None
) -> Dict[str, float]:
    """
    Analyze income distribution with summary statistics.

    Args:
        data: DataFrame containing income data
        income_column: Name of the income column
        group_by: Optional column to group analysis by

    Returns:
        Dictionary with summary statistics (mean, median, std, etc.)

    Example:
        >>> df = pd.DataFrame({"income": [30000, 45000, 60000]})
        >>> analyze_income_distribution(df)
        {'mean': 45000.0, 'median': 45000.0, 'std': 15000.0}
    """
    # Implementation here
    pass
```

### Testing Requirements

All code contributions must include tests:

```python
# tests/test_your_module.py
import pytest
from kranalytics.your_module import your_function


def test_your_function_basic():
    """Test basic functionality."""
    result = your_function(test_input)
    assert result == expected_output


def test_your_function_edge_cases():
    """Test edge cases."""
    with pytest.raises(ValueError):
        your_function(invalid_input)
```

Run tests before submitting:

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_your_module.py -v

# Run with coverage
make test-cov
```

### Documentation Standards

Document all public APIs:

```python
def load_census_data(
    api_key: str,
    year: int,
    geography: str = "state"
) -> pd.DataFrame:
    """
    Load demographic data from Census API.

    This function retrieves American Community Survey (ACS) data
    from the U.S. Census Bureau API for specified geography.

    Args:
        api_key: Census API key (get from api.census.gov)
        year: Year of data (2010-2023)
        geography: Geographic level ('state', 'county', 'tract')

    Returns:
        DataFrame with Census data indexed by geography

    Raises:
        ValueError: If year is out of range
        requests.HTTPError: If API request fails

    Example:
        >>> api_key = load_api_key('CENSUS_API_KEY')
        >>> df = load_census_data(api_key, 2020, 'state')
        >>> df.head()

    Note:
        Requires valid Census API key. Register at:
        https://api.census.gov/data/key_signup.html
    """
    pass
```

---

##  Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
# or
git checkout -b docs/documentation-update
```

### 2. Make Your Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed
- Follow code style guidelines
- Commit frequently with clear messages

### 3. Commit Standards

Use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <description>

# Examples:
git commit -m "feat(api): add BEA API integration"
git commit -m "fix(viz): resolve chart rendering issue"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(utils): add tests for api_key_manager"
git commit -m "refactor(models): optimize prediction pipeline"
```

**Commit Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only
- `test` - Adding or updating tests
- `refactor` - Code refactoring
- `perf` - Performance improvement
- `chore` - Maintenance tasks

### 4. Push and Create PR

```bash
# Push your branch
git push origin feature/your-feature-name

# Create pull request on GitHub
# Fill out the PR template completely
```

### 5. PR Checklist

Before submitting, ensure:

- [ ] All tests pass (`make test`)
- [ ] Code follows style guidelines (`make lint`)
- [ ] Code is formatted (`make format`)
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] PR description is complete
- [ ] No sensitive data or credentials included
- [ ] Related issues are referenced

### 6. Code Review

- Be responsive to feedback
- Make requested changes promptly
- Discuss disagreements constructively
- Keep commits focused and atomic

---

##  Security Guidelines

### Never Commit Sensitive Data

-  API keys, passwords, tokens
-  Personal or confidential data
-  Private configuration files
-  Credentials of any kind

### Use Environment Variables

```python
#  Good - Load from environment
from kranalytics.utils.api_key_manager import load_api_key
api_key = load_api_key('CENSUS_API_KEY')

#  Bad - Hardcoded credential
api_key = "abc123def456"  # NEVER DO THIS
```

### Reporting Security Issues

**DO NOT** open public issues for security vulnerabilities.

Instead, email: security@krlabs.dev

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

We'll respond within 48 hours.

---

##  Example Notebook Guidelines

When contributing example notebooks:

### Required Metadata

```python
"""
Example: Income Inequality Analysis

Author: Your Name
Date: YYYY-MM-DD
Description: Demonstrates Gini coefficient calculation and visualization
Data Source: U.S. Census Bureau ACS
License: MIT

This notebook shows how to:
1. Load income data from Census API
2. Calculate inequality metrics
3. Create visualizations
4. Interpret results
"""
```

### Structure

1. **Introduction** - Clear explanation of the analysis
2. **Setup** - Import libraries, load credentials
3. **Data Loading** - Fetch data from APIs or files
4. **Analysis** - Step-by-step methodology
5. **Visualization** - Publication-quality charts
6. **Interpretation** - Explain results and implications
7. **Extensions** - Suggest further analysis

### Best Practices

- Use descriptive variable names
- Add comments explaining complex logic
- Include markdown cells with context
- Show sample outputs
- Handle errors gracefully
- Document data sources
- Use reproducible random seeds

---

##  Testing Guidelines

### Test Coverage

Aim for high test coverage:

```bash
# Check coverage
make test-cov

# Opens HTML report in browser
```

### Test Types

1. **Unit Tests** - Test individual functions
2. **Integration Tests** - Test component interactions
3. **API Tests** - Test external API integrations (with mocking)
4. **Notebook Tests** - Test example notebooks execute

### Writing Good Tests

```python
import pytest
import pandas as pd
from kranalytics.utils import calculate_gini


class TestGiniCoefficient:
    """Tests for Gini coefficient calculation."""

    def test_perfect_equality(self):
        """Test Gini = 0 for perfect equality."""
        data = pd.Series([100, 100, 100, 100])
        result = calculate_gini(data)
        assert result == pytest.approx(0.0)

    def test_perfect_inequality(self):
        """Test Gini approaches 1 for perfect inequality."""
        data = pd.Series([0, 0, 0, 100])
        result = calculate_gini(data)
        assert result > 0.9

    def test_negative_values_raise_error(self):
        """Test that negative values raise ValueError."""
        data = pd.Series([100, -50, 200])
        with pytest.raises(ValueError, match="negative"):
            calculate_gini(data)
```

---

##  Academic Standards

### Citations

Cite sources for:
- Algorithms and methodologies
- Data sources
- Previous research
- External tools/libraries

```python
"""
Implements the Gini coefficient for income inequality measurement.

References:
    Gini, C. (1921). "Measurement of Inequality of Incomes."
    The Economic Journal, 31(121), 124-126.

    U.S. Census Bureau (2023). "American Community Survey."
    https://www.census.gov/programs-surveys/acs
"""
```

### Reproducibility

Ensure analyses are reproducible:

- Use fixed random seeds
- Document package versions
- Provide sample data or data access instructions
- Include environment specifications

---

##  Community Guidelines

### Be Respectful

- Treat all contributors with respect
- Welcome newcomers warmly
- Provide constructive feedback
- Assume good intentions
- Be patient and helpful

### Collaborate Openly

- Share ideas and knowledge
- Help others learn and grow
- Give credit where due
- Celebrate contributions

### Stay On Topic

- Keep discussions relevant
- Use appropriate channels (issues, PRs, discussions)
- Search before posting duplicates

---

##  License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).

---

##  Recognition

Contributors are recognized:

- Listed in [CONTRIBUTORS.md](./docs/CONTRIBUTORS.md)
- Credited in release notes
- Mentioned in relevant documentation

Significant contributions may lead to:
- Co-authorship opportunities
- Speaking invitations
- Collaboration on research

---

##  Getting Help

### Questions?

- **General Questions:** [GitHub Discussions](https://github.com/KR-Labs/KRAnalytics/discussions)
- **Bug Reports:** [GitHub Issues](https://github.com/KR-Labs/KRAnalytics/issues)
- **Security Issues:** security@krlabs.dev
- **Other Inquiries:** info@krlabs.dev

### Resources

- **Documentation:** [docs/](./docs/)
- **Examples:** [notebooks/examples/](./notebooks/examples/)
- **API Reference:** [docs/api/](./docs/api/)

---

##  Roadmap

See our [development roadmap](./docs/roadmaps/) for planned features and priorities.

Want to work on something? Check issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - We need community help
- `documentation` - Improve our docs
- `enhancement` - New features

---

**Thank you for contributing to KRAnalytics and helping make socioeconomic data analysis more accessible!** 
