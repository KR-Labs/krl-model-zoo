# KRL Model Zoo - PI Reference

**Version:** .  
**Last Updated:** October 24, 22

This document provides detailed PI reference for all classes, methods, and functions in the KRL Model Zoo.

---

## Table of ontents

. [Volatility Models](#volatility-models)
   - [GRHModel](#garchmodel)
   - [GRHModel](#egarchmodel)
   - [GJRGRHModel](#gjrgarchmodel)
2. [State Space Models](#state-space-models)
   - [Kalmanilter](#kalmanfilter)
   - [LocalLevelModel](#locallevelmodel)
3. [ore lasses](#core-classes)
   - [ModelInputSchema](#modelinputschema)
   - [ModelMeta](#modelmeta)
   - [orecastResult](#forecastresult)
4. [Utilities](#utilities)
. [xceptions](#exceptions)

---

## Volatility Models

### GRHModel

**Module:** `krl_models.volatility.garch_model`

Generalized utoregressive onditional Heteroskedasticity model for volatility forecasting.

#### lass efinition

```python
class GRHModel(aseModel):
    """
    GRH(p,q) model for conditional volatility forecasting.
    
    Models time-varying volatility in financial returns using RH and GRH terms.
    """
```

#### Mathematical Specification

**Returns quation:**
$$r_t = \mu + \epsilon_t$$
$$\epsilon_t = \sigma_t z_t, \quad z_t \sim (,)$$

**Variance quation:**
$$\sigma_t^2 = \omega + \sum_{i=}^{q} \alpha_i \epsilon_{t-i}^2 + \sum_{j=}^{p} \beta_j \sigma_{t-j}^2$$

#### onstructor

```python
def __init__(
    self,
    input_schema: ModelInputSchema,
    params: ict[str, ny],
    meta: ModelMeta
) -> None
```

**Parameters:**

| Parameter | Type | escription |
|-----------|------|-------------|
| `input_schema` | `ModelInputSchema` | Validated input schema with data configuration |
| `params` | `ict[str, ny]` | Model parameters (see below) |
| `meta` | `ModelMeta` | Model metadata |

**Parameter ictionary (`params`):**

| Key | Type | efault | escription |
|-----|------|---------|-------------|
| `p` | `int` |  | GRH order (lags of conditional variance) |
| `q` | `int` |  | RH order (lags of squared residuals) |
| `mean_model` | `str` | `'onstant'` | Mean specification: `'onstant'`, `'Zero'`, `'R'` |
| `ar_lags` | `int` |  | R order if `mean_model='R'` |
| `distribution` | `str` | `'normal'` | rror distribution: `'normal'`, `'t'`, `'ged'`, `'skewt'` |
| `vol_forecast_horizon` | `int` |  | efault forecast horizon |
| `use_returns` | `bool` | `True` | If alse, convert prices to log returns |

**Raises:**
- `Valuerror`: If p < , q < , or invalid parameter values

**xample:**

```python
from krl_models.volatility import GRHModel
from krl_core.model_input_schema import ModelInputSchema
from krl_core.base_model import ModelMeta

input_schema = ModelInputSchema(
    data_columns=['returns'],
    index_col='date',
    required_columns=['returns']
)

params = {
    'p': ,
    'q': ,
    'mean_model': 'onstant',
    'distribution': 'normal',
    'vol_forecast_horizon': 3
}

meta = ModelMeta(
    name='SPX_GRH',
    version='.',
    author='Risk Team',
    description='S&P  volatility model'
)

model = GRHModel(input_schema=input_schema, params=params, meta=meta)
```

#### Methods

##### `fit(data: pd.atarame) -> orecastResult`

stimate GRH model parameters using Maximum Likelihood stimation.

**Parameters:**

| Parameter | Type | escription |
|-----------|------|-------------|
| `data` | `pd.atarame` | Time series data with returns column |

**Returns:**
- `orecastResult`: Object containing:
  - `success` (bool): Whether optimization converged
  - `log_likelihood` (float): Log-likelihood value
  - `aic` (float): kaike Information riterion
  - `bic` (float): ayesian Information riterion
  - `params` (dict): stimated parameters
  - `std_errors` (dict): Standard errors of parameters
  - `payload` (dict): dditional model-specific outputs

**Raises:**
- `Valuerror`: If data is invalid or empty
- `Runtimerror`: If optimization fails to converge

**xample:**

```python
import pandas as pd
import numpy as np

# Generate sample returns
dates = pd.date_range('22--', periods=, freq='')
returns = np.random.randn() * .2
data = pd.atarame({'returns': returns}, index=dates)

# it model
result = model.fit(data)

print(f"onverged: {result.success}")
print(f"Log-Likelihood: {result.log_likelihood:.2f}")
print(f"I: {result.aic:.2f}")
print(f"I: {result.bic:.2f}")

# ccess estimated parameters
omega = model.params['omega']
alpha = model.params['alpha'][]
beta = model.params['beta'][]
print(f"ω = {omega:.f}, α = {alpha:.f}, β = {beta:.f}")
```

##### `predict(steps: int = None, **kwargs) -> orecastResult`

Generate multi-step ahead volatility forecasts.

**Parameters:**

| Parameter | Type | efault | escription |
|-----------|------|---------|-------------|
| `steps` | `int` | `vol_forecast_horizon` | Number of steps to forecast |
| `**kwargs` | `dict` | `{}` | dditional forecast options |

**Returns:**
- `orecastResult`: Object containing:
  - `forecast_values` (np.ndarray): Point forecasts (volatility)
  - `ci_lower` (np.ndarray): Lower confidence interval
  - `ci_upper` (np.ndarray): Upper confidence interval
  - `forecast_variance` (np.ndarray): orecast variance
  - `payload` (dict): dditional forecast outputs

**xample:**

```python
# 3-day ahead forecast
forecast = model.predict(steps=3)

print(f"ay  volatility: {forecast.forecast_values[]:.f}")
print(f"ay  % I: [{forecast.ci_lower[]:.f}, {forecast.ci_upper[]:.f}]")

# alculate Value-at-Risk
portfolio_value = __
z_score_ = .4
var_ = portfolio_value * z_score_ * forecast.forecast_values[]
print(f"-day % VaR: ${var_:,.2f}")
```

##### `get_conditional_volatility() -> np.ndarray`

xtract fitted conditional volatility (in-sample).

**Returns:**
- `np.ndarray`: onditional volatility σ_t for each time period

**xample:**

```python
cond_vol = model.get_conditional_volatility()

print(f"Mean volatility: {np.mean(cond_vol):.f}")
print(f"Max volatility: {np.max(cond_vol):.f}")
print(f"Min volatility: {np.min(cond_vol):.f}")

# nnualized volatility
annual_vol = np.mean(cond_vol) * np.sqrt(22)
print(f"nnualized volatility: {annual_vol*:.2f}%")
```

##### `get_standardized_residuals() -> np.ndarray`

xtract standardized residuals (ε_t / σ_t) for diagnostic testing.

**Returns:**
- `np.ndarray`: Standardized residuals

**xample:**

```python
std_resid = model.get_standardized_residuals()

# heck for remaining RH effects
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(std_resid**2, lags=, return_df=True)
print(f"Ljung-ox test on squared residuals:")
print(lb_test)

if all(lb_test['lb_pvalue'] > .):
    print(" No remaining RH effects")
else:
    print("  RH effects remain - consider higher order model")
```

##### `calculate_var(confidence_level: float = .) -> float`

alculate Value-at-Risk for -day horizon.

**Parameters:**

| Parameter | Type | efault | escription |
|-----------|------|---------|-------------|
| `confidence_level` | `float` | . | onfidence level (-) |

**Returns:**
- `float`: VaR as a proportion of portfolio value

**xample:**

```python
var_ = model.calculate_var(confidence_level=.)
var_ = model.calculate_var(confidence_level=.)

portfolio = __
print(f"% VaR: ${var_ * portfolio:,.2f}")
print(f"% VaR: ${var_ * portfolio:,.2f}")
```

#### Properties

| Property | Type | escription |
|----------|------|-------------|
| `params` | `ict[str, ny]` | stimated model parameters |
| `fitted` | `bool` | Whether model has been fitted |
| `aic` | `float` | kaike Information riterion |
| `bic` | `float` | ayesian Information riterion |
| `log_likelihood` | `float` | Log-likelihood value |

---

### GRHModel

**Module:** `krl_models.volatility.egarch_model`

xponential GRH model for asymmetric volatility (leverage effects).

#### lass efinition

```python
class GRHModel(aseModel):
    """
    GRH(p,o,q) model for asymmetric conditional volatility.
    
    Models leverage effects where negative shocks increase volatility more than
    positive shocks of the same magnitude.
    """
```

#### Mathematical Specification

**Log-Variance quation:**
$$\log(\sigma_t^2) = \omega + \sum_{i=}^{o} \gamma_i z_{t-i} + \sum_{i=}^{q} \alpha_i |z_{t-i}| + \sum_{j=}^{p} \beta_j \log(\sigma_{t-j}^2)$$

where $z_t = \epsilon_t / \sigma_t$ is the standardized residual.

**Leverage ffect:**
- $\gamma < $: Negative shocks increase volatility MOR (typical for equities)
- $\gamma > $: Positive shocks increase volatility MOR (rare)
- $\gamma = $: Symmetric response

#### onstructor

```python
def __init__(
    self,
    input_schema: ModelInputSchema,
    params: ict[str, ny],
    meta: ModelMeta
) -> None
```

**Parameters:**

| Parameter | Type | escription |
|-----------|------|-------------|
| `input_schema` | `ModelInputSchema` | Validated input schema |
| `params` | `ict[str, ny]` | Model parameters (see below) |
| `meta` | `ModelMeta` | Model metadata |

**Parameter ictionary (`params`):**

| Key | Type | efault | escription |
|-----|------|---------|-------------|
| `p` | `int` |  | GRH order |
| `o` | `int` |  | symmetry order (leverage terms) |
| `q` | `int` |  | RH order |
| `mean_model` | `str` | `'onstant'` | Mean specification |
| `distribution` | `str` | `'normal'` | rror distribution |
| `vol_forecast_horizon` | `int` |  | efault forecast horizon |

**xample:**

```python
from krl_models.volatility import GRHModel

params = {
    'p': ,
    'o': ,  # symmetry order
    'q': ,
    'mean_model': 'onstant',
    'distribution': 'normal'
}

model = GRHModel(input_schema=input_schema, params=params, meta=meta)
result = model.fit(equity_returns)

# heck for leverage effect
gamma = model.params['gamma'][]
if gamma < :
    print(f" Leverage effect detected (γ = {gamma:.4f})")
    print("   Negative shocks increase volatility more")
```

#### Methods

Methods are similar to `GRHModel`:
- `fit(data: pd.atarame) -> orecastResult`
- `predict(steps: int = None) -> orecastResult`
- `get_conditional_volatility() -> np.ndarray`
- `get_standardized_residuals() -> np.ndarray`

dditional method:

##### `get_leverage_parameter() -> float`

Get the estimated leverage parameter γ.

**Returns:**
- `float`: Leverage parameter (γ <  indicates leverage effect)

**xample:**

```python
gamma = model.get_leverage_parameter()

if gamma < -.:
    effect = "Strong leverage effect"
elif gamma > .:
    effect = "Inverse leverage effect"
else:
    effect = "Symmetric volatility"

print(f"γ = {gamma:.4f}: {effect}")
```

---

### GJRGRHModel

**Module:** `krl_models.volatility.gjr_garch_model`

GJR-GRH model for threshold effects in volatility.

#### lass efinition

```python
class GJRGRHModel(aseModel):
    """
    GJR-GRH(p,o,q) model for threshold volatility effects.
    
    Models asymmetric volatility where negative shocks have different impact
    than positive shocks via indicator function.
    """
```

#### Mathematical Specification

**Variance quation:**
$$\sigma_t^2 = \omega + \sum_{i=}^{q} (\alpha_i + \gamma_i I_{t-i}) \epsilon_{t-i}^2 + \sum_{j=}^{p} \beta_j \sigma_{t-j}^2$$

where $I_t = $ if $\epsilon_t < $ (negative shock), else $I_t = $.

**Threshold ffect:**
- or negative shocks: impact = $\alpha + \gamma$
- or positive shocks: impact = $\alpha$
- If $\gamma > $: Negative shocks have stronger impact

#### onstructor

```python
def __init__(
    self,
    input_schema: ModelInputSchema,
    params: ict[str, ny],
    meta: ModelMeta
) -> None
```

**Parameters:** Same as GRH, with `o` representing threshold order.

**xample:**

```python
from krl_models.volatility import GJRGRHModel

params = {
    'p': ,
    'o': ,  # Threshold order
    'q': ,
    'mean_model': 'onstant',
    'distribution': 'normal'
}

model = GJRGRHModel(input_schema=input_schema, params=params, meta=meta)
result = model.fit(data)

# nalyze threshold effect
alpha = model.params['alpha'][]
gamma = model.params['gamma'][]

print(f"Impact of positive shock: {alpha:.f}")
print(f"Impact of negative shock: {alpha + gamma:.f}")
print(f"symmetry ratio: {(alpha + gamma) / alpha:.2f}x")
```

#### Methods

Same core methods as `GRHModel`. dditional method:

##### `get_threshold_parameter() -> float`

Get the estimated threshold parameter γ.

**Returns:**
- `float`: Threshold parameter (γ >  indicates threshold effect)

---

## State Space Models

### Kalmanilter

**Module:** `krl_models.state_space.kalman_filter`

Linear Gaussian state space model with Kalman filtering and RTS smoothing.

#### lass efinition

```python
class Kalmanilter:
    """
    Kalman ilter for linear Gaussian state space models.
    
    Provides optimal state estimation via filtering (online) and smoothing (offline).
    """
```

#### Mathematical Specification

**State quation:**
$$\mathbf{x}_t = \mathbf{} \mathbf{x}_{t-} + \mathbf{w}_t, \quad \mathbf{w}_t \sim N(\mathbf{}, \mathbf{Q})$$

**Observation quation:**
$$\mathbf{y}_t = \mathbf{H} \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim N(\mathbf{}, \mathbf{R})$$

#### onstructor

```python
def __init__(
    self,
    n_states: int,
    n_obs: int,
    : np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    x: np.ndarray,
    P: np.ndarray
) -> None
```

**Parameters:**

| Parameter | Type | Shape | escription |
|-----------|------|-------|-------------|
| `n_states` | `int` | - | Number of state variables |
| `n_obs` | `int` | - | Number of observed variables |
| `` | `np.ndarray` | (n_states, n_states) | State transition matrix |
| `H` | `np.ndarray` | (n_obs, n_states) | Observation matrix |
| `Q` | `np.ndarray` | (n_states, n_states) | Process noise covariance |
| `R` | `np.ndarray` | (n_obs, n_obs) | Measurement noise covariance |
| `x` | `np.ndarray` | (n_states,) | Initial state estimate |
| `P` | `np.ndarray` | (n_states, n_states) | Initial covariance matrix |

**xample:**

```python
from krl_models.state_space import Kalmanilter
import numpy as np

# Position-velocity tracking model
 = np.array([[., .], [., .]])  # State transition
H = np.array([[., .]])               # Observe position only
Q = np.array([[., .], [., .]]) # Process noise
R = np.array([[.]])                    # Measurement noise
x = np.array([., .])                # Initial state
P = np.array([[., .], [., 4.]]) # Initial covariance

kf = Kalmanilter(
    n_states=2,
    n_obs=,
    =, H=H, Q=Q, R=R,
    x=x, P=P
)
```

#### Methods

##### `fit(data: pd.atarame, smoothing: bool = alse) -> orecastResult`

Run Kalman filter (and optionally RTS smoother) on observed data.

**Parameters:**

| Parameter | Type | efault | escription |
|-----------|------|---------|-------------|
| `data` | `pd.atarame` | - | Observed data (n_obs columns) |
| `smoothing` | `bool` | `alse` | Whether to run backward smoothing pass |

**Returns:**
- `orecastResult`: Object containing:
  - `filtered_states` (np.ndarray): orward-filtered state estimates
  - `filtered_covariances` (np.ndarray): iltered covariances
  - `smoothed_states` (np.ndarray): Smoothed estimates (if smoothing=True)
  - `smoothed_covariances` (np.ndarray): Smoothed covariances (if smoothing=True)
  - `innovations` (np.ndarray): Prediction errors (y_t - H x_{t|t-})
  - `log_likelihood` (float): Log-likelihood of observations

**xample:**

```python
import pandas as pd

# Prepare observation data
dates = pd.date_range('22--', periods=2, freq='H')
observed = np.random.randn(2) * .
data = pd.atarame({'position': observed}, index=dates)

# it with smoothing
result = kf.fit(data, smoothing=True)

# xtract results
filtered_states = result.payload['filtered_states']
smoothed_states = result.payload['smoothed_states']

# iltered position and velocity
position_filtered = filtered_states[:, ]
velocity_filtered = filtered_states[:, ]

# Smoothed estimates (better, uses all data)
position_smoothed = smoothed_states[:, ]
velocity_smoothed = smoothed_states[:, ]

print(f"inal position (filtered): {position_filtered[-]:.4f}")
print(f"inal position (smoothed): {position_smoothed[-]:.4f}")
print(f"inal velocity estimate: {velocity_smoothed[-]:.4f}")
```

##### `predict(steps: int) -> orecastResult`

Generate multi-step ahead forecasts with uncertainty quantification.

**Parameters:**

| Parameter | Type | escription |
|-----------|------|-------------|
| `steps` | `int` | Number of steps to forecast |

**Returns:**
- `orecastResult`: orecast values with confidence intervals

**xample:**

```python
# 2-step ahead forecast
forecast = kf.predict(steps=2)

print(f"Step  forecast: {forecast.forecast_values[]:.4f}")
print(f"Step  % I: [{forecast.ci_lower[]:.4f}, {forecast.ci_upper[]:.4f}]")

print(f"Step 2 forecast: {forecast.forecast_values[]:.4f}")
print(f"Step 2 % I: [{forecast.ci_lower[]:.4f}, {forecast.ci_upper[]:.4f}]")

# Uncertainty grows with horizon
ci_widths = forecast.ci_upper - forecast.ci_lower
print(f"I width grows: {ci_widths[]:.4f} → {ci_widths[-]:.4f}")
```

##### `get_filtered_states() -> np.ndarray`

Get filtered state estimates (forward pass only).

**Returns:**
- `np.ndarray`: State estimates (T × n_states)

##### `get_smoothed_states() -> np.ndarray`

Get smoothed state estimates (requires prior call to fit with smoothing=True).

**Returns:**
- `np.ndarray`: Smoothed state estimates (T × n_states)

##### `get_innovations() -> np.ndarray`

Get innovations (prediction errors) for diagnostics.

**Returns:**
- `np.ndarray`: Innovations sequence

**xample:**

```python
innovations = kf.get_innovations()

# heck for whiteness
print(f"Innovation mean: {np.mean(innovations):.f} (should be ≈ )")
print(f"Innovation std: {np.std(innovations):.f}")

# Ljung-ox test
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_result = acorr_ljungbox(innovations, lags=, return_df=True)

if all(lb_result['lb_pvalue'] > .):
    print(" Innovations are white noise (good model fit)")
else:
    print("  Innovations show autocorrelation")
```

---

### LocalLevelModel

**Module:** `krl_models.state_space.local_level`

Random walk plus noise model with automatic parameter estimation.

#### lass efinition

```python
class LocalLevelModel:
    """
    Local Level Model (random walk + noise) with ML parameter estimation.
    
    utomatically estimates level and observation noise variances.
    """
```

#### Mathematical Specification

**Level quation:**
$$\mu_t = \mu_{t-} + \eta_t, \quad \eta_t \sim N(, \sigma_\eta^2)$$

**Observation quation:**
$$y_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim N(, \sigma_\epsilon^2)$$

**Signal-to-Noise Ratio:**
$$q = \frac{\sigma_\eta^2}{\sigma_\epsilon^2}$$

#### onstructor

```python
def __init__(
    self,
    estimate_params: bool = True,
    initial_level: float = None,
    sigma_eta_init: float = .,
    sigma_epsilon_init: float = .
) -> None
```

**Parameters:**

| Parameter | Type | efault | escription |
|-----------|------|---------|-------------|
| `estimate_params` | `bool` | `True` | stimate σ_η² and σ_ε² via ML |
| `initial_level` | `float` | `None` | Initial level (auto-set from data if None) |
| `sigma_eta_init` | `float` | . | Initial guess for σ_η (ML starting point) |
| `sigma_epsilon_init` | `float` | . | Initial guess for σ_ε (ML starting point) |

**xample:**

```python
from krl_models.state_space import LocalLevelModel

# utomatic parameter estimation
model = LocalLevelModel(estimate_params=True)

# Or with manual parameters
model_manual = LocalLevelModel(
    estimate_params=alse,
    initial_level=.,
    sigma_eta_init=.,
    sigma_epsilon_init=2.
)
```

#### Methods

##### `fit(data: pd.atarame) -> orecastResult`

it Local Level Model with automatic parameter estimation.

**Parameters:**

| Parameter | Type | escription |
|-----------|------|-------------|
| `data` | `pd.atarame` | Time series data (single column) |

**Returns:**
- `orecastResult`: itted model results

**xample:**

```python
# Noisy data with underlying trend
dates = pd.date_range('22--', periods=, freq='')
true_trend = np.cumsum(np.random.randn() * .)
noisy_data = true_trend + np.random.randn() * .
data = pd.atarame({'value': noisy_data}, index=dates)

# it with automatic parameter estimation
result = model.fit(data)

# Get estimated parameters
sigma_eta = model._sigma_eta
sigma_epsilon = model._sigma_epsilon
snr = model.get_signal_to_noise_ratio()

print(f"stimated σ_η: {sigma_eta:.f}")
print(f"stimated σ_ε: {sigma_epsilon:.f}")
print(f"Signal-to-Noise Ratio: {snr:.f}")
```

##### `get_level(smoothed: bool = True) -> np.ndarray`

xtract the estimated level (trend component).

**Parameters:**

| Parameter | Type | efault | escription |
|-----------|------|---------|-------------|
| `smoothed` | `bool` | `True` | Return smoothed (True) or filtered (alse) level |

**Returns:**
- `np.ndarray`: stimated level sequence

**xample:**

```python
# xtract smooth trend
smooth_trend = model.get_level(smoothed=True)

# ompare with noisy observations
rmse = np.sqrt(np.mean((smooth_trend - true_trend)**2))
print(f"Trend extraction RMS: {rmse:.4f}")

# Plot results
import matplotlib.pyplot as plt
plt.plot(data.index, noisy_data, label='Noisy observations', alpha=.)
plt.plot(data.index, smooth_trend, label='xtracted trend', linewidth=2)
plt.legend()
plt.title('Local Level Model: Trend xtraction')
plt.show()
```

##### `decompose() -> ict[str, np.ndarray]`

ecompose time series into trend and noise components.

**Returns:**
- `dict`: ictionary with keys:
  - `'trend'` (np.ndarray): Smooth level component
  - `'noise'` (np.ndarray): Irregular component
  - `'observations'` (np.ndarray): Original data

**xample:**

```python
decomposition = model.decompose()

trend = decomposition['trend']
noise = decomposition['noise']

print(f"Trend mean: {np.mean(trend):.4f}")
print(f"Noise mean: {np.mean(noise):.f} (should be ≈ )")
print(f"Noise std:  {np.std(noise):.f}")

# Verify decomposition
reconstructed = trend + noise
assert np.allclose(reconstructed, decomposition['observations'])
```

##### `get_signal_to_noise_ratio() -> float`

alculate the signal-to-noise ratio q = σ_η² / σ_ε².

**Returns:**
- `float`: Signal-to-noise ratio

**Interpretation:**
- q → : Very smooth trend (level changes slowly)
- q → ∞: Noisy trend (follows observations closely)
- Typical: . - . for most data

**xample:**

```python
snr = model.get_signal_to_noise_ratio()

if snr < .:
    print(f"SNR = {snr:.f}: Very smooth trend")
elif snr < .:
    print(f"SNR = {snr:.f}: Moderate smoothness")
else:
    print(f"SNR = {snr:.f}: Trend tracks data closely")
```

---

## ore lasses

### ModelInputSchema

**Module:** `krl_core.model_input_schema`

Pydantic schema for standardized model input validation.

#### lass efinition

```python
class ModelInputSchema(aseModel):
    """
    Standardized input format for all KRL models.
    
    nsures consistent data structure and validation.
    """
```

#### ields

| ield | Type | Required | escription |
|-------|------|----------|-------------|
| `entity` | `str` | Yes | ntity identifier (e.g., "US", "PL") |
| `metric` | `str` | Yes | Metric name (e.g., "returns", "unemployment") |
| `time_index` | `List[str]` | Yes | Time dimension |
| `values` | `List[float]` | Yes | Observed values |
| `provenance` | `Provenance` | Yes | ata source metadata |
| `frequency` | `str` | Yes | ata frequency ("", "W", "M", "Q", "Y") |

**xample:**

```python
from krl_core.model_input_schema import ModelInputSchema, Provenance
from datetime import datetime

provenance = Provenance(
    source_name="Yahoo inance",
    series_id="SPY",
    collection_date=datetime.now(),
    transformation="log_returns"
)

schema = ModelInputSchema(
    entity="SPY",
    metric="daily_returns",
    time_index=["22--", "22--2", "22--3"],
    values=[., -.2, .],
    provenance=provenance,
    frequency=""
)

# onvert to atarame
df = schema.to_dataframe()
```

---

### ModelMeta

**Module:** `krl_core.base_model`

Metadata for model tracking and versioning.

#### lass efinition

```python
@dataclass
class ModelMeta:
    """Model metadata for tracking and versioning."""
    name: str
    version: str
    author: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
```

**xample:**

```python
from krl_core.base_model import ModelMeta

meta = ModelMeta(
    name="SPX_Vol_Model",
    version="2..",
    author="Risk Management Team",
    description="S&P  volatility forecasting with GRH(,)"
)
```

---

### orecastResult

**Module:** `krl_core.base_model`

ontainer for model results and forecasts.

#### ttributes

| ttribute | Type | escription |
|-----------|------|-------------|
| `success` | `bool` | Whether operation succeeded |
| `forecast_values` | `np.ndarray` | Point forecasts |
| `ci_lower` | `np.ndarray` | Lower confidence interval |
| `ci_upper` | `np.ndarray` | Upper confidence interval |
| `log_likelihood` | `float` | Log-likelihood value |
| `aic` | `float` | kaike Information riterion |
| `bic` | `float` | ayesian Information riterion |
| `params` | `ict[str, ny]` | stimated parameters |
| `std_errors` | `ict[str, ny]` | Standard errors |
| `payload` | `ict[str, ny]` | dditional model-specific data |

---

## Utilities

### ata Preparation unctions

```python
def calculate_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
    """
    alculate returns from price series.
    
    Parameters:
        prices: Price series
        method: 'log' for log returns, 'simple' for percentage returns
    
    Returns:
        pd.Series: Returns
    """
    if method == 'log':
        return np.log(prices / prices.shift()).dropna()
    else:
        return prices.pct_change().dropna()
```

---

## xceptions

### Valuerror

Raised for invalid parameters or data:
- Invalid model orders (p < , q < )
- mpty or malformed data
- onstraint violations

### Runtimerror

Raised for computational issues:
- Optimization convergence failures
- Numerical stability problems
- Matrix singularity

### xample rror Handling

```python
try:
    model = GRHModel(input_schema, params, meta)
    result = model.fit(data)
    
    if not result.success:
        print("  Model did not converge - trying simpler specification")
        params_simple = {'p': , 'q': }
        model_simple = GRHModel(input_schema, params_simple, meta)
        result = model_simple.fit(data)
        
except Valuerror as e:
    print(f" Invalid parameters: {e}")
    # djust parameters
    
except Runtimerror as e:
    print(f" itting failed: {e}")
    # Try alternative initialization
```

---

## omplete xample: nd-to-nd Workflow

```python
import pandas as pd
import numpy as np
from datetime import datetime

from krl_models.volatility import GRHModel
from krl_models.state_space import Kalmanilter
from krl_core.model_input_schema import ModelInputSchema
from krl_core.base_model import ModelMeta

# ============================================================================
# . Prepare ata
# ============================================================================

# Load returns data
dates = pd.date_range('22--', periods=, freq='')
returns = np.random.randn() * .2  # Simulated returns
data = pd.atarame({'returns': returns}, index=dates)

# ============================================================================
# 2. onfigure GRH Model
# ============================================================================

input_schema = ModelInputSchema(
    data_columns=['returns'],
    index_col='date',
    required_columns=['returns']
)

params = {
    'p': ,
    'q': ,
    'mean_model': 'onstant',
    'distribution': 'normal',
    'vol_forecast_horizon': 3
}

meta = ModelMeta(
    name='Portfolio_Risk_Model',
    version='.',
    author='Risk Team'
)

# ============================================================================
# 3. it and orecast
# ============================================================================

model = GRHModel(input_schema=input_schema, params=params, meta=meta)
result = model.fit(data)

print(f" Model converged: {result.success}")
print(f"   Log-Likelihood: {result.log_likelihood:.2f}")
print(f"   I: {result.aic:.2f}")

# Generate forecast
forecast = model.predict(steps=3)

# alculate VaR
var_ = model.calculate_var(confidence_level=.)
portfolio_value = __
print(f"\n% VaR: ${var_ * portfolio_value:,.2f}")

# ============================================================================
# 4. Model iagnostics
# ============================================================================

std_resid = model.get_standardized_residuals()
cond_vol = model.get_conditional_volatility()

print(f"\nVolatility Statistics:")
print(f"  Mean: {np.mean(cond_vol):.f}")
print(f"  Max:  {np.max(cond_vol):.f}")
print(f"  nnualized: {np.mean(cond_vol) * np.sqrt(22) * :.2f}%")

# ============================================================================
# . ompare with Kalman ilter (for state estimation)
# ============================================================================

# Setup Kalman ilter for trend extraction
 = np.array([[.]])
H = np.array([[.]])
Q = np.array([[.]])
R = np.array([[.]])
x = np.array([.])
P = np.array([[.]])

kf = Kalmanilter(
    n_states=, n_obs=,
    =, H=H, Q=Q, R=R,
    x=x, P=P
)

kf_result = kf.fit(data, smoothing=True)
smooth_trend = kf_result.payload['smoothed_states']

print(f"\n omplete workflow executed successfully!")
```

---

**nd of PI Reference**

or more information:
- User Guide: `docs/USR_GUI.md`
- xamples: `examples/` directory
- Mathematical etails: `docs/MTHMTIL_ORMULTIONS.md`

*KRL Model Zoo Team*  
*Last Updated: October 24, 22*
