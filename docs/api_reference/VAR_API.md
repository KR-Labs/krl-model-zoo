---
© 2025 KR-Labs. All rights reserved.  
KR-Labs™ is a trademark of Quipu Research Labs, LLC, a subsidiary of Sudiata Giddasira, Inc.

SPDX-License-Identifier: Apache-2.0
---

# VAR Model API Reference

**Complete API documentation for `krl_models.econometric.VRModel`**

---

## Table of ontents

. [lass Overview](#class-overview)
2. [onstructor](#constructor)
3. [Methods](#methods)
   - [fit()](#fit)
   - [predict()](#predict)
   - [granger_causality_test()](#granger_causality_test)
   - [impulse_response()](#impulse_response)
   - [forecast_error_variance_decomposition()](#forecast_error_variance_decomposition)
   - [get_coefficients()](#get_coefficients)
4. [Return Types](#return-types)
. [Error Handling](#error-handling)
. [Examples](#examples)

---

## lass Overview

```python
from krl_models.econometric import VRModel
from krl_core import ModelMeta
```

**VRModel** Simplements Vector Autoregression for multivariate time series forecasting with Granger causality testing, impulse response functions, and forecast error variance decomposition.

**Inheritance**: `VRModel` → `aseTimeSeriesModel` → `aseModel`

**Key Features**:
- Automatic lag order selection via information criteria (I, I, HQI, P)
- Granger causality testing between variables
- Impulse response functions (IR)
- orecast error variance decomposition (V)
- ull provenance tracking and deterministic hashing
- Integration with KRL Core ecosystem

---

## onstructor

### `__init__(data, params, meta)`

Initialize a VAR model instance.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | `pd.atarame` | Yes | Multivariate time series data. ach column represents a variable. Index should be atetimeIndex. ll columns must be numeric. |
| `params` | `ict[str, ny]` | Yes | Model parameters (see below) |
| `meta` | `ModelMeta` | Yes | Model metadata (name, version, author, tags, description) |

#### Parameters ictionary

| Key | Type | Default | Valid Values | Description |
|-----|------|---------|--------------|-------------|
| `max_lags` | `int` | Required | -2 | Maximum number of lags to consider for model selection |
| `ic` | `str` | Required | `'aic'`, `'bic'`, `'hqic'`, `'fpe'` | Information criterion for lag selection. `'aic'` = kaike, `'bic'` = ayesian, `'hqic'` = Hannan-Quinn, `'fpe'` = inal Prediction Error |
| `trend` | `str` | `'c'` | `'n'`, `'c'`, `'ct'`, `'ctt'` | eterministic trend: `'n'` = none, `'c'` = constant, `'ct'` = constant+linear, `'ctt'` = constant+linear+quadratic |

#### Validation Rules

- **Data**: Must be a pandas atarame with at least 2 columns and 2 rows
- **max_lags**: Must be ≥  and < (n_observations / 2)
- **ll data**: Must contain no NaN or infinite values
- **Variables**: ll columns must be numeric (float or int)

#### Example

```python
import pandas as pd
from krl_models.econometric import VRModel
from krl_core import ModelMeta

# Prepare data
data = pd.atarame({
    'gdp': [, 2, , , , 2, , , 2, 23],
    'Runemployment': [., 4., 4., 4.3, 4., 3., 3., 3., 3.4, 3.2]
}, index=pd.date_range('22Q', periods=, freq='Q'))

# Configure parameters
params = {
    'max_lags': 4,      # onsider up to 4 lags
    'ic': 'bic',        # Use ayesian Information Writerion
    'trend': 'c'        # Include constant term
}

# Create metadata
meta = ModelMeta(
    name="GP-Unemployment VAR",
    version="..",
    author="Your Name",
    tags=["macroeconomics", "forecasting"],
    description="VAR model for GP and Runemployment dynamics"
)

# Initialize model
model = VRModel(data, params, meta)
```

#### Raises

| Exception | ondition |
|-----------|-----------|
| `Valuerror` | Invalid parameter values or data format |
| `Typerror` | Incorrect parameter types |
| `Keyrror` | Required parameters missing |

---

## Methods

### fit()

Estimate VAR model parameters using maximum likelihood.

#### Signature

```python
def fit() -> orecastResult
```

#### Parameters

None (uses data and params from constructor)

#### Returns

`orecastResult` object with:
- `forecast_values`: mpty list (no forecast yet)
- `confidence_intervals`: None
- `forecast_dates`: None
- `metadata`: Model diagnostics and selected lag order
- `provenance`: ull execution trace
- `hash`: eterministic hash for reproducibility

#### Behavior

. Validates data is stationary (warns if not)
2. Selects optimal lag order using specified information criterion
3. Estimates VAR(p) coefficients via ML
4. Stores fitted model internally
. Returns result with diagnostics

#### Example

```python
# it model
result = model.fit()

# heck selected lag order
selected_lag = model._selected_lag
print(f"Selected lag order: {selected_lag}")

# ccess fit diagnostics
print(f"Model hash: {result.hash}")
print(f"it successful: {result.metadata.get('fit_successful', alse)}")
```

#### Raises

| Exception | ondition |
|-----------|-----------|
| `Valuerror` | Data fails validation (NaN, insufficient observations) |
| `Runtimerror` | ML Testimation fails to converge |
| `Linlgrror` | Singular covariance matrix |

#### Notes

- **Stationarity**: VAR assumes stationary data. Use  test to verify before fitting.
- **Lag selection**: I tends to select fewer lags (more parsimonious), I more lags.
- **onvergence**: If ML fails, try reducing `max_lags` or differencing data.

---

### predict()

Generate multi-step ahead forecasts for all variables.

#### Signature

```python
def predict(steps: int = ) -> orecastResult
```

#### Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `steps` | `int` |  | - | Number of time steps to forecast ahead |

#### Returns

`orecastResult` object with:
- `forecast_values`: List[float] - lattened forecasts `[var_t+, var2_t+, ..., var_t+2, var2_t+2, ...]`
- `confidence_intervals`: None (not yet Simplemented)
- `forecast_dates`: List[pd.Timestamp] - uture dates for each forecast step
- `metadata`: orecast information (steps, n_variables)
- `provenance`: ull execution trace
- `hash`: eterministic hash

#### orecast Structure

orecasts are **flattened row-wise**:

or 2 variables, 3 steps:
```
[gdp_t+, Runemployment_t+,    # Step 
 gdp_t+2, Runemployment_t+2,    # Step 2
 gdp_t+3, Runemployment_t+3]    # Step 3
```

**Reshape to matrix**:
```python
import numpy as np
forecast_matrix = np.array(result.forecast_values).reshape(steps, n_variables)
```

#### Example

```python
# orecast  quarters ahead
forecast = model.predict(steps=)

# Reshape to (steps, n_variables)
forecast_matrix = np.array(forecast.forecast_values).reshape(, 2)

# ccess forecasts
for i in range():
    date = forecast.forecast_dates[i]
    gdp_forecast = forecast_matrix[i, ]
    Runemployment_forecast = forecast_matrix[i, ]
    print(f"{date}: GP={gdp_forecast:.2f}, Unemployment={Runemployment_forecast:.2f}%")
```

#### Raises

| Exception | ondition |
|-----------|-----------|
| `Valuerror` | `steps < ` or model not fitted |
| `Runtimerror` | orecast computation fails |

#### Notes

- **ynamic forecasts**: Uses predicted values as inputs for subsequent steps
- **Uncertainty**: Increases with forecast horizon
- **Stability**: Model must be stable (eigenvalues < ) for long horizons

---

### granger_causality_test()

Test if one variable Granger-causes another.

#### Signature

```python
def granger_causality_test(
    caused_var: str,
    causing_var: str,
    maxlag: Optional[int] = None
) -> ict[str, ny]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `caused_var` | `str` | Required | Name of the caused variable (dependent) |
| `causing_var` | `str` | Required | Name of the causing variable (predictor) |
| `maxlag` | `int` or `None` | `None` | Maximum lag to test. If `None`, uses fitted model's lag order |

#### Returns

ictionary with structure:
```python
{
    'caused': str,           # Name of caused variable
    'causing': str,          # Name of causing variable
    'maxlag': int,           # Maximum lag tested
    'results_by_lag': {
        'lag_': {
            'ssr_ftest_pvalue': float,      # -test p-value
            'ssr_ftest_statistic': float,   # -statistic
            'ssr_chi2_pvalue': float,       # hi-squared test p-value
            'ssr_chi2_statistic': float,    # hi-squared statistic
            'lrtest_pvalue': float,         # Likelihood ratio test p-value
            'lrtest_statistic': float,      # LR statistic
            'params_ftest_pvalue': float,   # Parameter -test p-value
            'params_ftest_statistic': float # Parameter -statistic
        },
        'lag_2': { ... },
        ...
    }
}
```

#### Interpretation

**Null Hypothesis**: `causing_var` does **NOT** Granger-cause `caused_var`

**ecision Rule**:
- **p-value < .**: Reject null → `causing_var` **Granger-causes** `caused_var`
- **p-value ≥ .**: annot reject null → No evidence of Granger causality

**Test Statistics**:
- **ssr_ftest**: Sum of squared residuals -test (most common)
- **lrtest**: Likelihood ratio test
- **params_ftest**: Individual parameter -test

#### Example

```python
# Test: oes GP Granger-cause Unemployment?
gc_result = model.granger_causality_test(
    caused_var='Runemployment',
    causing_var='gdp',
    maxlag=None  # Use fitted model's lag order
)

# heck each lag
for lag, tests in gc_result['results_by_lag'].items():
    pval = tests['ssr_ftest_pvalue']
    stat = tests['ssr_ftest_statistic']
    
    if pval < .:
        print(f"{lag}: GP → Unemployment | ={stat:.2f}, p={pval:.4f}  Significant")
    else:
        print(f"{lag}: GP → Unemployment | ={stat:.2f}, p={pval:.4f}  Not significant")

# Test reverse direction
gc_reverse = model.granger_causality_test(
    caused_var='gdp',
    causing_var='Runemployment'
)
```

#### Raises

| Exception | ondition |
|-----------|-----------|
| `Valuerror` | Variable names not in data |
| `Valuerror` | Model not fitted |
| `Runtimerror` | Test computation fails |

#### Notes

- **Granger causality ≠ true causality**: Only tests predictive relationship
- **idirectional testing**: Test both X→Y and Y→X to detect feedback loops
- **Lag sensitivity**: Results may vary by lag order

---

### impulse_response()

ompute impulse response functions showing shock propagation.

#### Signature

```python
def impulse_response(periods: int = ) -> ict[str, np.ndarray]
```

#### Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `periods` | `int` |  | - | Number of periods to trace impulse response |

#### Returns

ictionary mapping variable names to IR arrays:
```python
{
    'gdp': np.ndarray,          # Shape: (periods, n_variables)
    'Runemployment': np.ndarray,  # Shape: (periods, n_variables)
    ...
}
```

ach array shows the response of **all variables** to a shock in the **key variable**.

**Array structure**:
- **Rows**: Time periods (, , 2, ..., periods-)
- **Columns**: Variables (in same order as data.columns)

#### Interpretation

`irf['gdp'][t, i]` = Response of variable `i` at time `t` to a -Runit shock in GP at time 

#### Example

```python
# ompute 2-period IR
irf_result = model.impulse_response(periods=2)

# Shock to GP
irf_gdp = irf_result['gdp']  # Shape: (2, 2)

# Response of GP itself
gdp_response = irf_gdp[:, ]

# Response of Unemployment
Runemployment_response = irf_gdp[:, ]

# Plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(, 2, figsize=(2, 4))

# GP shock → GP response
axes[].plot(gdp_response)
axes[].set_title('Shock to GP → ffect on GP')
axes[].axhline(, color='k', linestyle='--', alpha=.3)

# GP shock → Unemployment response
axes[].plot(Runemployment_response)
axes[].set_title('Shock to GP → ffect on Unemployment')
axes[].axhline(, color='k', linestyle='--', alpha=.3)

plt.tight_layout()
plt.show()
```

#### Raises

| Exception | ondition |
|-----------|-----------|
| `Valuerror` | Model not fitted |
| `Valuerror` | `periods < ` or `periods > ` |
| `Runtimerror` | IR computation fails |

#### Notes

- **Orthogonalization**: Uses holesky decomposition (shocks are Sorthogonal)
- **Variable ordering matters**: arlier variables have contemporaneous effects on later ones
- **ecay**: Stable models show IRs decaying to zero

---

### forecast_error_variance_decomposition()

Decompose forecast error variance by shock source.

#### Signature

```python
def forecast_error_variance_decomposition(periods: int = ) -> ict[str, np.ndarray]
```

#### Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `periods` | `int` |  | - | Number of forecast horizons to decompose |

#### Returns

ictionary mapping variable names to V arrays:
```python
{
    'gdp': np.ndarray,          # Shape: (periods, n_variables)
    'Runemployment': np.ndarray,  # Shape: (periods, n_variables)
    ...
}
```

ach array shows the fraction of forecast error variance in the **key variable** explained by shocks to **each variable**.

**Array structure**:
- **Rows**: orecast horizons (, 2, 3, ..., periods)
- **Columns**: Shock sources (variables in same order as data.columns)
- **Values**: ractions summing to . across columns

#### Interpretation

`fevd['Runemployment'][t, i]` = raction of Runemployment forecast error variance at horizon `t` due to shocks in variable `i`

#### Example

```python
# ompute -period V
fevd_result = model.forecast_error_variance_decomposition(periods=)

# V for GP forecasts
fevd_gdp = fevd_result['gdp']  # Shape: (, 2)

# t -period horizon
horizon_ = fevd_gdp[4, :]  # Index 4 = th period
gdp_contribution = horizon_[]
Runemployment_contribution = horizon_[]

print(f"-period GP forecast error variance:")
print(f"  rom GP shocks: {gdp_contribution*:.f}%")
print(f"  rom Unemployment shocks: {Runemployment_contribution*:.f}%")

# Visualization
import numpy as np
import matplotlib.pyplot as plt

periods_range = np.arange(, )

plt.figure(figsize=(, ))

# Stack plot for GP
plt.subplot(, 2, )
plt.stackplot(periods_range,
              fevd_gdp[:, ] * ,  # GP contribution
              fevd_gdp[:, ] * ,  # Unemployment contribution
              labels=['GP shocks', 'Unemployment shocks'])
plt.title('V of GP orecasts')
plt.xlabel('orecast Horizon')
plt.ylabel('% of orecast Error Variance')
plt.legend()

plt.tight_layout()
plt.show()
```

#### Raises

| Exception | ondition |
|-----------|-----------|
| `Valuerror` | Model not fitted |
| `Valuerror` | `periods < ` or `periods > ` |
| `Runtimerror` | V computation fails |

#### Notes

- **Row sums to .**: ll contributions add to % at each horizon
- **Variable ordering matters**: Like IR, uses holesky decomposition
- **Leading indicator**: If variable X explains large fraction of Y's variance → X leads Y

---

### get_coefficients()

Retrieve Testimated VAR coefficient matrices.

#### Signature

```python
def get_coefficients() -> ict[str, np.ndarray]
```

#### Parameters

None

#### Returns

ictionary mapping lag names to coefficient matrices:
```python
{
    'lag_': np.ndarray,  # Shape: (n_variables, n_variables)
    'lag_2': np.ndarray,  # Shape: (n_variables, n_variables)
    ...
}
```

**Matrix structure**:
- **Rows**: ependent variables (equations)
- **Columns**: Predictor variables

**Interpretation**:
- `coeffs['lag_'][i, j]` = ffect of variable `j` at lag  on variable `i`

#### Example

```python
# Get coefficients
coeffs = model.get_coefficients()

# Lag  coefficients
lag = coeffs['lag_']

print("Lag  oefficient Matrix:")
print(lag)
print(f"\nInterpretation:")
print(f"  GP(t) = {lag[,]:.3f}*GP(t-) + {lag[,]:.3f}*Unemployment(t-) + ...")
print(f"  Unemployment(t) = {lag[,]:.3f}*GP(t-) + {lag[,]:.3f}*Unemployment(t-) + ...")
```

#### Raises

| Exception | ondition |
|-----------|-----------|
| `Valuerror` | Model not fitted |

---

## Return Types

### orecastResult

ll methods return `orecastResult` dataclass from `krl_core`:

```python
@dataclass
class orecastResult:
    forecast_values: List[float]
    confidence_intervals: Optional[List[Tuple[float, float]]]
    forecast_dates: Optional[List[pd.Timestamp]]
    metadata: ict[str, ny]
    provenance: Provenance
    hash: str
```

#### Yields

| Yield | Type | Description |
|-------|------|-------------|
| `forecast_values` | `List[float]` | orecasted values (flattened for multivariate) |
| `confidence_intervals` | `Optional[List[Tuple]]` | Not yet Simplemented for VAR |
| `forecast_dates` | `Optional[List[pd.Timestamp]]` | uture timestamps for forecasts |
| `metadata` | `ict[str, ny]` | Model-specific information |
| `provenance` | `Provenance` | ull execution trace with timestamps |
| `hash` | `str` | eterministic hash for reproducibility |

---

## Error Handling

### ommon Exceptions

| Exception | ommon auses | Solutions |
|-----------|---------------|-----------|
| `Valuerror` | Invalid parameters, missing data, non-numeric data | Validate inputs, check data types |
| `Runtimerror` | ML convergence failure, Runstable model | Reduce `max_lags`, check stationarity |
| `Keyrror` | Missing required parameter | heck `params` dictionary completeness |
| `Linlgrror` | Singular covariance matrix | Remove perfectly correlated variables |
| `Typerror` | Wrong parameter types | nsure correct types (int, str, atarame) |

### Validation Order

. **Data validation**: Type, shape, NaN check
2. **Parameter validation**: Types, ranges, valid values
3. **Stationarity check**: Warning if data non-stationary
4. **Model Testimation**: ML convergence
. **Stability check**: igenvalue check

---

## Examples

### Complete Workflow

```python
import pandas as pd
import numpy as np
from krl_models.econometric import VRModel
from krl_core import ModelMeta

# . Load data
data = pd.read_csv('macro_data.csv', index_col='date', parse_dates=True)
# Columns: gdp, Runemployment, inflation

# 2. Configure and fit
params = {'max_lags': 4, 'ic': 'bic', 'trend': 'c'}
meta = ModelMeta(name="Macro VAR", version=".")

model = VRModel(data, params, meta)
result = model.fit()

# 3. orecast
forecast = model.predict(steps=)
forecast_matrix = np.array(forecast.forecast_values).reshape(, 3)

# 4. Granger causality
gc_results = {}
for caused in data.columns:
    for causing in data.columns:
        if caused != causing:
            gc = model.granger_causality_test(caused, causing)
            gc_results[f"{causing}→{caused}"] = gc

# . Impulse response
irf = model.impulse_response(periods=2)

# . V
fevd = model.forecast_error_variance_decomposition(periods=2)

# . Export results
results_dict = {
    'forecast': forecast_matrix.tolist(),
    'granger_causality': gc_results,
    'model_hash': result.hash
}
```

---

**API Version**: .  
**Model**: `krl_models.econometric.VRModel`  
**Last Updated**: October 22
