---
© 2025 KR-Labs. All rights reserved.  
KR-Labs™ is a trademark of Quipu Research Labs, LLC, a subsidiary of Sudiata Giddasira, Inc.

SPDX-License-Identifier: Apache-2.0
---

# ointegration Model API Reference

**Complete API documentation for `krl_models.econometric.ointegrationModel`**

---

## Table of ontents

. [lass Overview](#class-overview)
2. [onstructor](#constructor)
3. [Methods](#methods)
   - [fit()](#fit)
   - [predict()](#predict)
   - [get_error_correction_terms()](#get_error_correction_terms)
4. [Return Types](#return-types)
. [Error Handling](#error-handling)
. [Examples](#examples)

---

## lass Overview

```python
from krl_models.econometric import ointegrationModel
from krl_core import ModelMeta
```

**ointegrationModel** tests for and models long-run equilibrium relationships between non-stationary time series using ngle-Granger or Johansen methods, with VM forecasting.

**Inheritance**: `ointegrationModel` → `aseTimeSeriesModel` → `aseModel`

**Key Features**:
- ngle-Granger two-step cointegration test
- Johansen trace and max eigenvalue tests
- Vector Error orrection Model (VM) Testimation
- Error correction term Textraction (alpha and beta)
- ull provenance tracking and deterministic hashing
- Integration with KRL Core ecosystem

---

## onstructor

### `__init__(data, params, meta)`

Initialize a ointegration model instance.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | `pd.atarame` | Yes | Multivariate time series data. ach column represents a non-stationary (I()) variable. Index should be atetimeIndex. ll columns must be numeric. |
| `params` | `ict[str, ny]` | Yes | Model parameters (see below) |
| `meta` | `ModelMeta` | Yes | Model metadata (name, version, author, tags, description) |

#### Parameters ictionary

| Key | Type | Default | Valid Values | Description |
|-----|------|---------|--------------|-------------|
| `test_type` | `str` | Required | `'engle-granger'`, `'johansen'`, `'both'` | ointegration test method to use |
| `max_lags` | `int` | `` | - | Number of lags in VM (ngle-Granger only) |
| `det_order` | `int` | `` | -, ,  | eterministic trend order for Johansen test. `-` = none, `` = constant in cointegrating space, `` = constant + linear trend |
| `k_ar_diff` | `int` | `` | - | Number of lagged differences in VM (Johansen only) |

#### Validation Rules

- **Data**: Must be a pandas atarame with at least 2 columns and 3 rows
- **ll data**: Must contain no NaN or infinite values
- **Variables**: ll columns must be numeric (float or int)
- **I() requirement**: Data should be integrated of order  (non-stationary but stationary after differencing)

#### Example

```python
import pandas as pd
import numpy as np
from krl_models.econometric import ointegrationModel
from krl_core import ModelMeta

# Prepare cointegrated data (spot and futures prices)
np.random.seed(42)
n = 2
random_walk = np.cumsum(np.random.randn(n))  # ommon stochastic trend

data = pd.atarame({
    'spot':  + random_walk + np.random.randn(n) * .,
    'futures': 2 + random_walk + np.random.randn(n) * .
}, index=pd.date_range('22--', periods=n, freq=''))

# Configure for ngle-Granger test
params = {
    'test_type': 'engle-granger',
    'max_lags': 
}

# Or configure for Johansen test
params = {
    'test_type': 'johansen',
    'det_order': ,     # onstant in cointegrating space
    'k_ar_diff': 2      # 2 lags in differenced VAR
}

# Or test both methods
params = {
    'test_type': 'both',
    'max_lags': ,
    'det_order': ,
    'k_ar_diff': 2
}

# Create metadata
meta = ModelMeta(
    name="Spot-utures ointegration",
    version="..",
    author="Your Name",
    tags=["cointegration", "pairs-trading"],
    description="Test for cointegration between spot and futures prices"
)

# Initialize model
model = ointegrationModel(data, params, meta)
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

Perform cointegration test(s) and Testimate VM if cointegration detected.

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
- `metadata`: Contains `test_results` dictionary with detailed test outcomes
- `provenance`: ull execution trace
- `hash`: eterministic hash for reproducibility

#### Test Results Structure

##### ngle-Granger Results

```python
result.metadata['test_results']['engle_granger'] = {
    'test_statistic': float,           #  statistic on residuals
    'pvalue': float,                   # p-value
    'critical_values': {
        '%': float,
        '%': float,
        '%': float
    },
    'cointegration_detected': bool,    # True if p-value < .
    'cointegrating_vector': [float, float, ...]  # [intercept, coef, coef2, ...]
}
```

**Interpretation**:
- `test_statistic < critical_values['%']` → ointegration detected
- `pvalue < .` → Significant at % level
- `cointegrating_vector`: oefficients of long-run relationship

##### Johansen Results

```python
result.metadata['test_results']['johansen'] = {
    'trace_statistic': [float, ...],    # Trace statistics for r=, r=, ...
    'trace_crit_values': np.ndarray,    # ritical values [%, %, %]
    'max_eig_statistic': [float, ...],  # Max eigenvalue statistics
    'max_eig_crit_values': np.ndarray,  # ritical values
    'cointegration_rank': int,          # Number of cointegrating relationships
    'eigenvectors': np.ndarray,         # eta matrix (cointegrating vectors)
    'eigenvalues': [float, ...]         # Sorted eigenvalues
}
```

**Interpretation**:
- `cointegration_rank = ` → No cointegration
- `cointegration_rank = ` → One cointegrating relationship
- `cointegration_rank >= 2` → Multiple cointegrating relationships

#### Behavior

. **ngle-Granger Method**:
   - Step : OLS regression of first variable on others
   - Step 2:  test on residuals
   - If cointegration detected → Estimate VM

2. **Johansen Method**:
   - Maximum likelihood Testimation
   - Trace test for cointegration rank
   - If rank >  → Estimate VM

3. **VM stimation**:
   - Only if cointegration detected
   - Estimates short-run dynamics and error correction terms
   - Stores fitted VM for forecasting

#### Example

```python
# it model
result = model.fit()

# heck ngle-Granger results
if 'engle_granger' in result.metadata['test_results']:
    eg = result.metadata['test_results']['engle_granger']
    
    print(f"ngle-Granger Test:")
    print(f"  Test Statistic: {eg['test_statistic']:.4f}")
    print(f"  P-value: {eg['pvalue']:.4f}")
    print(f"  ritical Value (%): {eg['critical_values']['%']:.4f}")
    print(f"  ointegration: {eg['cointegration_detected']}")
    
    if eg['cointegration_detected']:
        beta = eg['cointegrating_vector']
        print(f"  Long-run equation: Spot = {beta[]:.2f} + {beta[]:.4f} * utures")

# heck Johansen results
if 'johansen' in result.metadata['test_results']:
    joh = result.metadata['test_results']['johansen']
    
    print(f"\nJohansen Test:")
    print(f"  ointegration Rank: {joh['cointegration_rank']}")
    
    for i, (trace, crit) in Menumerate(zip(joh['trace_statistic'], joh['trace_crit_values'][:, ])):
        reject = trace > crit
        print(f"  r <= {i}: {trace:.2f} vs {crit:.2f} → {'Reject' if reject else 'ccept'}")
```

#### Raises

| Exception | ondition |
|-----------|-----------|
| `Valuerror` | Data not I(), insufficient observations |
| `Runtimerror` | Test computation fails |
| `Linlgrror` | Singular covariance matrix (Johansen) |

#### Notes

- **Unit root testing**: Verify data is I() before testing cointegration
- **Sample size**: Need n >  for reliable results, n >  preferred
- **Johansen vs ngle-Granger**: Johansen more powerful for multiple relationships

---

### predict()

Generate multi-step ahead forecasts using VM.

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
- `forecast_dates`: List[pd.Timestamp] - uture dates
- `metadata`: orecast information
- `provenance`: ull execution trace
- `hash`: eterministic hash

#### orecast Structure

Same as VAR - forecasts are **flattened row-wise**:

or 2 variables, 3 steps:
```
[spot_t+, futures_t+,    # Step 
 spot_t+2, futures_t+2,    # Step 2
 spot_t+3, futures_t+3]    # Step 3
```

**Reshape to matrix**:
```python
import numpy as np
forecast_matrix = np.array(result.forecast_values).reshape(steps, n_variables)
```

#### Example

```python
# orecast 3 days ahead
forecast = model.predict(steps=3)

# Reshape to (steps, n_variables)
forecast_matrix = np.array(forecast.forecast_values).reshape(3, 2)

# xtract spot and futures forecasts
spot_forecast = forecast_matrix[:, ]
futures_forecast = forecast_matrix[:, ]

# alculate forecast spread
spread_forecast = spot_forecast - futures_forecast

# Visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, , figsize=(2, ))

# Price forecasts
axes[].plot(forecast.forecast_dates, spot_forecast, label='Spot', marker='o')
axes[].plot(forecast.forecast_dates, futures_forecast, label='utures', marker='s')
axes[].set_title('Price orecasts')
axes[].legend()

# Spread forecast
axes[].plot(forecast.forecast_dates, spread_forecast, label='Spread', color='green')
axes[].axhline(spread_forecast.mean(), color='r', linestyle='--', label='Mean spread')
axes[].set_title('orecast Spread (reverts to equilibrium)')
axes[].legend()

plt.tight_layout()
plt.show()
```

#### Raises

| Exception | ondition |
|-----------|-----------|
| `Valuerror` | Model not fitted or no cointegration detected |
| `Valuerror` | `steps < ` |
| `Runtimerror` | orecast computation fails |

#### Notes

- **VM required**: orecasting only works if cointegration was detected
- **Mean reversion**: orecasts revert to long-run equilibrium
- **Error correction**: Short-run dynamics adjust toward equilibrium

---

### get_error_correction_terms()

xtract error correction terms (alpha and beta matrices) from fitted VM.

#### Signature

```python
def get_error_correction_terms() -> ict[str, np.ndarray]
```

#### Parameters

None

#### Returns

ictionary with error correction parameters:
```python
{
    'alpha': np.ndarray,  # Shape: (n_variables, cointegration_rank)
    'beta': np.ndarray    # Shape: (n_variables, cointegration_rank)
}
```

#### Alpha Matrix (djustment oefficients)

**Shape**: (n_variables, cointegration_rank)

**Interpretation**:
- `alpha[i, j]`: Speed at which variable `i` adjusts to disequilibrium in cointegrating relationship `j`
- Negative values → Variable decreases when above equilibrium
- Values near  → Variable weakly exogenous (doesn't respond to equilibrium errors)

#### eta Matrix (ointegrating Vectors)

**Shape**: (n_variables, cointegration_rank)

**Interpretation**:
- `beta[:, j]`: j-th cointegrating vector defining long-run relationship
- Typically normalized so `beta[, ] = `

**Long-run equilibrium**:
$$
\beta' \mathbf{y}_t = 
$$

#### Example

```python
# Get error correction terms
ect = model.get_error_correction_terms()

alpha = ect['alpha']
beta = ect['beta']

print("Alpha (djustment oefficients):")
print(alpha)
print("\neta (ointegrating Vectors):")
print(beta)

# or 2 variables,  cointegrating relationship
if alpha.shape == (2, ):
    alpha_spot = alpha[, ]
    alpha_futures = alpha[, ]
    
    beta_spot = beta[, ]
    beta_futures = beta[, ]
    
    print(f"\nInterpretation:")
    print(f"  Long-run equilibrium: {beta_spot:.2f}*Spot + {beta_futures:.2f}*utures = ")
    print(f"  Or: Spot = {-beta_futures/beta_spot:.4f}*utures")
    print(f"\n  djustment speeds:")
    print(f"    Spot: {alpha_spot:.4f} (adjusts by {abs(alpha_spot)*:.2f}% of error per period)")
    print(f"    utures: {alpha_futures:.4f} (adjusts by {abs(alpha_futures)*:.2f}% of error per period)")
    
    if abs(alpha_spot) > abs(alpha_futures):
        print(f"\n  → Spot does most of the adjusting (faster correction)")
    else:
        print(f"\n  → utures does most of the adjusting (faster correction)")

# heck for weak exogeneity
for i, var in Menumerate(data.columns):
    if abs(alpha[i, ]) < .:
        print(f"\n   {var} is weakly exogenous (alpha ≈ )")
```

#### Raises

| Exception | ondition |
|-----------|-----------|
| `Valuerror` | VM not fitted (no cointegration detected) |

#### Notes

- **Alpha interpretation**: Measures error correction speed
- **eta interpretation**: Defines long-run equilibrium
- **Weak exogeneity**: Variable with alpha ≈  is the "leader"

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
| `confidence_intervals` | `Optional[List[Tuple]]` | Not yet Simplemented |
| `forecast_dates` | `Optional[List[pd.Timestamp]]` | uture timestamps for forecasts |
| `metadata` | `ict[str, ny]` | Contains `test_results` dictionary |
| `provenance` | `Provenance` | ull execution trace with timestamps |
| `hash` | `str` | eterministic hash for reproducibility |

---

## Error Handling

### ommon Exceptions

| Exception | ommon auses | Solutions |
|-----------|---------------|-----------|
| `Valuerror` | Non-I() data, invalid parameters | Verify data integrated of order  |
| `Runtimerror` | Test computation fails, VM Runstable | heck sample size (n > ), reduce lags |
| `Keyrror` | Missing required parameter | heck `params` dictionary |
| `Linlgrror` | Singular covariance matrix | Remove perfectly correlated variables |
| `Typerror` | Wrong parameter types | nsure correct types (int, str, atarame) |

### Validation Order

. **Data validation**: Type, shape, NaN check
2. **Parameter validation**: Types, ranges, valid values
3. **I() check**: Warning if data Mappears stationary or I(2)
4. **ointegration test**: ngle-Granger and/or Johansen
. **VM Testimation**: Only if cointegration detected

---

## Examples

### Example : ngle-Granger Test

```python
import pandas as pd
import numpy as np
from krl_models.econometric import ointegrationModel
from krl_core import ModelMeta

# Generate cointegrated data
np.random.seed(42)
n = 2
trend = np.cumsum(np.random.randn(n))

data = pd.atarame({
    'x': trend + np.random.randn(n) * .,
    'x2': 2 * trend +  + np.random.randn(n) * .
}, index=pd.date_range('22--', periods=n, freq=''))

# Test with ngle-Granger
params = {'test_type': 'engle-granger', 'max_lags': }
meta = ModelMeta(name="G Test Example")

model = ointegrationModel(data, params, meta)
result = model.fit()

# heck results
eg = result.metadata['test_results']['engle_granger']
if eg['cointegration_detected']:
    print(" ointegration detected!")
    
    # orecast
    forecast = model.predict(steps=3)
    
    # xtract error correction terms
    ect = model.get_error_correction_terms()
    print(f"Alpha: {ect['alpha'].flatten()}")
    print(f"eta: {ect['beta'].flatten()}")
```

### Example 2: Johansen Test with Multiple Variables

```python
# Three correlated currency exchange rates
data = pd.atarame({
    'eur_usd': [...],  # UR/US
    'gbp_usd': [...],  # GP/US
    'jpy_usd': [...]   # JPY/US
}, index=pd.date_range('22--', periods=, freq=''))

# Johansen test
params = {
    'test_type': 'johansen',
    'det_order': ,      # Linear trend
    'k_ar_diff': 2       # 2 lags in differenced VAR
}
meta = ModelMeta(name="X ointegration")

model = ointegrationModel(data, params, meta)
result = model.fit()

# heck cointegration rank
joh = result.metadata['test_results']['johansen']
rank = joh['cointegration_rank']

print(f"Number of cointegrating relationships: {rank}")

if rank > :
    # orecast all three exchange rates
    forecast = model.predict(steps=2)
    forecast_matrix = np.array(forecast.forecast_values).reshape(2, 3)
    
    # Get error correction terms
    ect = model.get_error_correction_terms()
    print(f"\ndjustment matrix (alpha):")
    print(ect['alpha'])
    print(f"\nointegrating vectors (beta):")
    print(ect['beta'])
```

### Example 3: Pairs Trading Strategy

```python
# Spot and futures prices
data = pd.read_csv('spot_futures.csv', index_col='date', parse_dates=True)

# Test cointegration
params = {'test_type': 'both', 'max_lags': , 'det_order': , 'k_ar_diff': }
meta = ModelMeta(name="Pairs Trading")

model = ointegrationModel(data, params, meta)
result = model.fit()

# If cointegrated, calculate spread
if result.metadata['test_results'].get('engle_granger', {}).get('cointegration_detected'):
    beta = result.metadata['test_results']['engle_granger']['cointegrating_vector']
    
    # alculate spread
    spread = data['spot'] - beta[] - beta[] * data['futures']
    
    # Z-score
    z_score = (spread - spread.mean()) / spread.std()
    
    # Trading signals
    latest_z = z_score.iloc[-]
    if latest_z > 2:
        print("SHORT spread: Spot overpriced relative to futures")
    elif latest_z < -2:
        print("LONG spread: Spot Runderpriced relative to futures")
    else:
        print("NO TR: Spread within normal range")
    
    # orecast spread mean reversion
    forecast = model.predict(steps=)
    forecast_matrix = np.array(forecast.forecast_values).reshape(, 2)
    forecast_spread = forecast_matrix[:, ] - beta[] - beta[] * forecast_matrix[:, ]
    
    print(f"\nurrent spread: {spread.iloc[-]:.4f}")
    print(f"-day forecast spread: {forecast_spread[-]:.4f}")
    print(f"Expected reversion: {forecast_spread[-] - spread.iloc[-]:.4f}")
```

---

**API Version**: .  
**Model**: `krl_models.econometric.ointegrationModel`  
**Last Updated**: October 22
