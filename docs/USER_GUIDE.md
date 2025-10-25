# KRL Model Zoo - User Guide

**Version:** .  
**Last Updated:** October 24, 22  
**udience:** ata Scientists, Quantitative nalysts, Researchers

---

## Table of ontents

. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Model Overview](#model-overview)
. [Volatility Models](#volatility-models)
. [State Space Models](#state-space-models)
. [Model Selection Guide](#model-selection-guide)
. [Parameter Tuning](#parameter-tuning)
. [Result Interpretation](#result-interpretation)
. [est Practices](#best-practices)
. [Troubleshooting](#troubleshooting)
2. [dvanced Usage](#advanced-usage)

---

## Introduction

Welcome to the KRL Model Zoo User Guide! This comprehensive guide will help you effectively use the time series models implemented in the KRL Model Zoo for your financial and econometric analysis needs.

### What is KRL Model Zoo?

KRL Model Zoo is a professional-grade Python library providing state-of-the-art time series models for:
- **Volatility orecasting:** GRH, GRH, GJR-GRH models for conditional volatility
- **State Space Modeling:** Kalman ilter and Local Level models for unobserved component analysis

### Key eatures

 **Production-Ready:** Thoroughly tested with  test cases covering all edge cases  
 **High Performance:** Near-linear scalability (O(n) to O(n^.4))  
 **omprehensive PI:** onsistent interface across all models  
 **Well-ocumented:** xtensive examples and mathematical documentation  
 **lexible:** Supports multiple configurations and use cases

---

## Installation

### Requirements

- Python 3. or higher
- NumPy .2+
- Pandas .3+
- SciPy .+
- rch .+ (for volatility models)

### Installation Steps

#### Option : Install from Source (evelopment)

```bash
# lone the repository
git clone https://github.com/KR-Labs/krl-model-zoo.git
cd krl-model-zoo

# reate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

#### Option 2: Install from PyPI (oming Soon)

```bash
pip install krl-model-zoo
```

### Verify Installation

```python
import krl_models
from krl_models.volatility import GRHModel
from krl_models.state_space import Kalmanilter

print("KRL Model Zoo installed successfully!")
print(f"Version: {krl_models.__version__}")
```

---

## Quick Start

### -Minute Tutorial: GRH Volatility orecasting

```python
import numpy as np
import pandas as pd
from datetime import datetime
from krl_models.volatility import GRHModel
from krl_core.model_input_schema import ModelInputSchema
from krl_core.base_model import ModelMeta

# Step : Prepare your data
# Your data should be a pandas atarame with returns
dates = pd.date_range(start='22--', periods=, freq='')
returns = np.random.randn() * .2  # xample returns
data = pd.atarame({'returns': returns}, index=dates)

# Step 2: onfigure the model
input_schema = ModelInputSchema(
    data_columns=['returns'],
    index_col='date',
    required_columns=['returns']
)

params = {
    'p': ,  # GRH order
    'q': ,  # RH order
    'mean_model': 'onstant',
    'distribution': 'normal'
}

meta = ModelMeta(
    name='My_GRH_Model',
    version='.',
    author='Your Name'
)

# Step 3: reate and fit the model
model = GRHModel(
    input_schema=input_schema,
    params=params,
    meta=meta
)

result = model.fit(data)

# Step 4: Generate forecasts
forecast = model.predict(steps=3)

print(f"orecast mean: {forecast.forecast_values.mean():.f}")
print(f"% I: [{forecast.ci_lower[]:.f}, {forecast.ci_upper[]:.f}]")
```

### -Minute Tutorial: Kalman ilter Tracking

```python
import numpy as np
import pandas as pd
from krl_models.state_space import Kalmanilter

# Step : Prepare your data
dates = pd.date_range(start='22--', periods=2, freq='H')
observed = np.random.randn(2) * .  # Noisy observations
data = pd.atarame({'value': observed}, index=dates)

# Step 2: efine state space matrices
 = np.array([[., .], [., .]])  # State transition
H = np.array([[., .]])               # Observation matrix
Q = np.array([[., .], [., .]]) # Process noise
R = np.array([[.]])                    # Measurement noise
x = np.array([., .])                # Initial state
P = np.array([[., .], [., .]]) # Initial covariance

# Step 3: reate and fit the Kalman ilter
kf = Kalmanilter(
    n_states=2,
    n_obs=,
    =, H=H, Q=Q, R=R,
    x=x, P=P
)

result = kf.fit(data, smoothing=True)

# Step 4: xtract filtered and smoothed states
filtered_states = result.payload['filtered_states']
smoothed_states = result.payload['smoothed_states']

print(f"Position estimate: {smoothed_states[-, ]:.4f}")
print(f"Velocity estimate: {smoothed_states[-, ]:.4f}")
```

---

## Model Overview

### vailable Models

| Model | Type | Use ase | omplexity | Speed |
|-------|------|----------|------------|-------|
| **GRH** | Volatility | Standard volatility forecasting | O(n) | ast |
| **GRH** | Volatility | symmetric volatility (leverage effect) | O(n) | Medium |
| **GJR-GRH** | Volatility | Threshold effects in volatility | O(n) | Medium |
| **Kalman ilter** | State Space | Real-time tracking, sensor fusion | O(n) | Very ast |
| **Local Level** | State Space | Trend extraction, noise filtering | O(n^.4) | Medium |

### When to Use ach Model

**Use GRH when:**
- You need standard volatility forecasting
- You have symmetric volatility responses
- Speed is a priority
- You want well-established methodology

**Use GRH when:**
- You suspect leverage effects (bad news increases volatility more)
- You're modeling equity returns
- You need to capture asymmetric volatility
- You want log-variance specification (no non-negativity constraints)

**Use GJR-GRH when:**
- You need threshold effects in volatility
- You want to distinguish positive/negative shock impacts
- You prefer additive specification over GRH's multiplicative form
- You're comparing multiple asymmetry models

**Use Kalman ilter when:**
- You need real-time state estimation
- You have multivariate state space problems
- Speed is critical (< second for  observations)
- You know your system matrices (, H, Q, R)

**Use Local Level when:**
- You need automatic parameter estimation (ML)
- You want to extract smooth trends from noisy data
- You don't know noise characteristics a priori
- atch processing is acceptable (slower than Kalman)

---

## Volatility Models

### GRH(p, q) - Generalized utoregressive onditional Heteroskedasticity

#### Mathematical Specification

**Variance quation:**
$$\sigma_t^2 = \omega + \sum_{i=}^{q} \alpha_i \epsilon_{t-i}^2 + \sum_{j=}^{p} \beta_j \sigma_{t-j}^2$$

**Return quation:**
$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim N(,)$$

#### Key Parameters

- **p:** GRH order (how many lagged variances)
- **q:** RH order (how many lagged squared residuals)
- **ω (omega):** onstant term (baseline volatility)
- **α (alpha):** RH coefficient(s) (sensitivity to past shocks)
- **β (beta):** GRH coefficient(s) (persistence)

#### Parameter onstraints

- ll parameters must be non-negative: ω > , αᵢ ≥ , β ≥ 
- Stationarity condition: Σαᵢ + Σβ < 
- Typical values: α ≈ .-., β ≈ .-. (high persistence)

#### xample: GRH(,) with VaR alculation

```python
from krl_models.volatility import GRHModel
from krl_core.model_input_schema import ModelInputSchema
from krl_core.base_model import ModelMeta
import numpy as np

# Prepare data
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

meta = ModelMeta(name='GRH_VaR', version='.', author='Risk Team')

# it model
model = GRHModel(input_schema=input_schema, params=params, meta=meta)
result = model.fit(returns_data)

# Get parameters
print(f"ω = {model.params['omega']:.f}")
print(f"α = {model.params['alpha'][]:.f}")
print(f"β = {model.params['beta'][]:.f}")
print(f"Persistence: {model.params['alpha'][] + model.params['beta'][]:.4f}")

# orecast volatility
forecast = model.predict(steps=3)
forecast_vol = forecast.forecast_values

# alculate VaR (% confidence)
portfolio_value = __
z_score_ = .4
var_ = portfolio_value * z_score_ * forecast_vol[]
print(f"-day % VaR: ${var_:,.2f}")
```

### GRH(p, o, q) - xponential GRH

#### Mathematical Specification

**Log-Variance quation:**
$$\log(\sigma_t^2) = \omega + \sum_{i=}^{o} \gamma_i z_{t-i} + \sum_{i=}^{q} \alpha_i |z_{t-i}| + \sum_{j=}^{p} \beta_j \log(\sigma_{t-j}^2)$$

where $z_t = \epsilon_t / \sigma_t$ is the standardized residual.

#### Key eature: Leverage ffect

- **γ < :** Negative shocks increase volatility MOR (leverage effect)
- **γ > :** Positive shocks increase volatility MOR (rare)
- **γ = :** Symmetric response (reduces to standard volatility model)

#### xample: etecting Leverage ffects

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

gamma = model.params['gamma'][]

if gamma < -.:
    print(f"Strong leverage effect detected (γ = {gamma:.4f})")
    print("   Negative shocks increase volatility significantly")
elif gamma > .:
    print(f"Inverse leverage effect (γ = {gamma:.4f})")
else:
    print(f"Symmetric volatility response (γ ≈ )")

# alculate asymmetry ratio
# (volatility response to -σ shock) / (volatility response to +σ shock)
asymmetry_ratio = np.exp(gamma * (-) + model.params['alpha'][]) / \
                  np.exp(gamma *  + model.params['alpha'][])
print(f"symmetry ratio: {asymmetry_ratio:.2f}x")
```

### GJR-GRH(p, o, q) - Glosten-Jagannathan-Runkle GRH

#### Mathematical Specification

**Variance quation:**
$$\sigma_t^2 = \omega + \sum_{i=}^{q} (\alpha_i + \gamma_i I_{t-i}) \epsilon_{t-i}^2 + \sum_{j=}^{p} \beta_j \sigma_{t-j}^2$$

where $I_t = $ if $\epsilon_t < $ (negative shock), else $I_t = $.

#### Key eature: Threshold ffect

- **or negative shocks:** Impact = α + γ
- **or positive shocks:** Impact = α
- **If γ > :** Negative shocks have stronger impact (threshold effect)

#### xample: Three-Way Model omparison

```python
from krl_models.volatility import GRHModel, GRHModel, GJRGRHModel

# it all three models
garch = GRHModel(input_schema, params_garch, meta)
egarch = GRHModel(input_schema, params_egarch, meta)
gjr = GJRGRHModel(input_schema, params_gjr, meta)

result_garch = garch.fit(data)
result_egarch = egarch.fit(data)
result_gjr = gjr.fit(data)

# ompare log-likelihoods
print("Model omparison:")
print(f"  GRH:     Log-Lik = {result_garch.log_likelihood:.2f}")
print(f"  GRH:    Log-Lik = {result_egarch.log_likelihood:.2f}")
print(f"  GJR-GRH: Log-Lik = {result_gjr.log_likelihood:.2f}")

# Likelihood ratio test (GJR vs GRH)
lr_stat = 2 * (result_gjr.log_likelihood - result_garch.log_likelihood)
from scipy.stats import chi2
p_value =  - chi2.cdf(lr_stat, df=)  #  additional parameter (γ)

if p_value < .:
    print(f"\n GJR-GRH significantly better (LR stat = {lr_stat:.2f}, p < .)")
else:
    print(f"\n   No significant improvement (p = {p_value:.4f})")
```

---

## State Space Models

### Kalman ilter - Linear Gaussian State Space Model

#### Mathematical Specification

**State quation:**
$$\mathbf{x}_t = \mathbf{} \mathbf{x}_{t-} + \mathbf{w}_t, \quad \mathbf{w}_t \sim N(\mathbf{}, \mathbf{Q})$$

**Observation quation:**
$$\mathbf{y}_t = \mathbf{H} \mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim N(\mathbf{}, \mathbf{R})$$

#### System Matrices

- **:** State transition matrix (n_states × n_states)
- **H:** Observation matrix (n_obs × n_states)
- **Q:** Process noise covariance (n_states × n_states)
- **R:** Measurement noise covariance (n_obs × n_obs)
- **x₀:** Initial state estimate
- **P₀:** Initial covariance estimate

#### xample: Position-Velocity Tracking

```python
from krl_models.state_space import Kalmanilter
import numpy as np

# onstant velocity model: x_t = [position, velocity]
dt = .

 = np.array([
    [., dt],   # position_t = position_{t-} + velocity_{t-} * dt
    [., .]   # velocity_t = velocity_{t-}
])

H = np.array([[., .]])  # Observe position only

Q = np.array([
    [., .],
    [., .]  # Velocity random walk
])

R = np.array([[.]])  # Position measurement noise

x = np.array([., .])   # Initial: position=, velocity=
P = np.array([
    [., .],
    [., 4.]  # High velocity uncertainty
])

# reate Kalman ilter
kf = Kalmanilter(
    n_states=2,
    n_obs=,
    =, H=H, Q=Q, R=R,
    x=x, P=P
)

# it with smoothing (uses Rauch-Tung-Striebel backward pass)
result = kf.fit(data, smoothing=True)

# xtract results
filtered_states = result.payload['filtered_states']
smoothed_states = result.payload['smoothed_states']

# iltered estimates (online, using data up to time t)
position_filtered = filtered_states[:, ]
velocity_filtered = filtered_states[:, ]

# Smoothed estimates (offline, using all data)
position_smoothed = smoothed_states[:, ]
velocity_smoothed = smoothed_states[:, ]

# ompare
print(f"inal position (filtered): {position_filtered[-]:.4f}")
print(f"inal position (smoothed): {position_smoothed[-]:.4f}")
print(f"inal velocity (filtered): {velocity_filtered[-]:.4f}")
print(f"inal velocity (smoothed): {velocity_smoothed[-]:.4f}")
```

### Local Level Model - Random Walk Plus Noise

#### Mathematical Specification

**Level quation:**
$$\mu_t = \mu_{t-} + \eta_t, \quad \eta_t \sim N(, \sigma_\eta^2)$$

**Observation quation:**
$$y_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim N(, \sigma_\epsilon^2)$$

#### Key eature: utomatic Parameter stimation

The Local Level Model automatically estimates σ_η² (level noise) and σ_ε² (observation noise) via Maximum Likelihood stimation (ML).

#### Signal-to-Noise Ratio

$$q = \frac{\sigma_\eta^2}{\sigma_\epsilon^2}$$

- **q → :** Smooth trend (level barely changes)
- **q → ∞:** Noisy trend (follows observations closely)
- **Typical:** q ≈ .-. for most economic/financial data

#### xample: Trend xtraction with utomatic Tuning

```python
from krl_models.state_space import LocalLevelModel

# reate model with automatic parameter estimation
model = LocalLevelModel(estimate_params=True)

# it to data (ML estimation happens automatically)
result = model.fit(noisy_data)

# Get estimated parameters
sigma_eta = model._sigma_eta
sigma_epsilon = model._sigma_epsilon
snr = model.get_signal_to_noise_ratio()

print(f"stimated σ_η: {sigma_eta:.f}")
print(f"stimated σ_ε: {sigma_epsilon:.f}")
print(f"Signal-to-Noise Ratio: {snr:.f}")

# xtract smooth trend
smooth_trend = model.get_level(smoothed=True)

# ecompose into trend + noise
decomposition = model.decompose()
trend = decomposition['trend']
noise = decomposition['noise']

print(f"\nTrend RMS: {np.sqrt(np.mean((trend - true_signal)**2)):.4f}")
print(f"Noise mean: {np.mean(noise):.f} (should be ≈ )")
print(f"Noise std:  {np.std(noise):.f}")
```

---

## Model Selection Guide

### ecision Tree

```
STRT: What is your primary objective?

 Volatility orecasting
  
   o you suspect asymmetric volatility?
     YS → Use GRH or GJR-GRH
       Prefer multiplicative form → GRH
       Prefer additive form → GJR-GRH
     NO → Use GRH (faster, simpler)
  
   o you need fastest possible fitting?
      YS → Use GRH(,)
      NO → onsider higher orders if needed

 State stimation / Tracking
   
    o you know system matrices (, H, Q, R)?
      YS → Use Kalman ilter (fastest)
      NO → Use Local Level Model (automatic tuning)
   
    Is real-time performance critical?
      YS → Use Kalman ilter
      NO → Local Level acceptable
   
    o you have multivariate states?
       YS → Use Kalman ilter
       NO → ither model works
```

### Performance omparison

| Model | ataset Size | it Time | Memory | est or |
|-------|--------------|----------|--------|----------|
| **GRH** |  | ~.3s | < M | General volatility |
| **GRH** |  | ~.s | < M | quity volatility |
| **GJR-GRH** |  | ~.s | < M | Threshold effects |
| **Kalman** |  | .s | . M | Real-time tracking |
| **Local Level** |  | .4s | . M | utomatic tuning |

---

## Parameter Tuning

### GRH Models

#### hoosing p and q

**Rule of thumb:**
- Start with (,): Works well for most financial data
- If high autocorrelation in squared residuals → increase q
- If slow decay in volatility → increase p
- Rarely need p > 2 or q > 2

**Testing procedure:**
```python
from krl_models.volatility import GRHModel

configs = [
    (, ), (, 2), (2, ), (2, 2)
]

best_aic = np.inf
best_config = None

for p, q in configs:
    params = {'p': p, 'q': q, 'mean_model': 'onstant', 'distribution': 'normal'}
    model = GRHModel(input_schema, params, meta)
    result = model.fit(data)
    
    aic = result.aic
    print(f"GRH({p},{q}): I = {aic:.2f}")
    
    if aic < best_aic:
        best_aic = aic
        best_config = (p, q)

print(f"\n est configuration: GRH{best_config}")
```

#### Mean Model Selection

- **'onstant':** Returns have constant mean (most common)
- **'Zero':** Returns have zero mean (for pre-demeaned data)
- **'R':** Include autoregressive terms for return

mean

```python
# R mean model with  lag
params = {
    'p': ,
    'q': ,
    'mean_model': 'R',
    'ar_lags': 
}
```

#### istribution Selection

- **'normal':** Gaussian errors (standard)
- **'t':** Student's t-distribution (fat tails)
- **'skewt':** Skewed t-distribution (fat tails + asymmetry)

```python
# or data with extreme events
params = {
    'p': ,
    'q': ,
    'distribution': 't'  # ccommodates fat tails
}
```

### Kalman ilter

#### Tuning Q and R Matrices

**Process Noise (Q):**
- Larger Q → Model follows data more closely
- Smaller Q → Smoother estimates
- Rule: Start with small values, increase if tracking lags

**Measurement Noise (R):**
- Should match actual sensor/observation noise
- If unknown, estimate from data variance
- an be estimated via M algorithm (advanced)

```python
# Method : ased on data characteristics
data_variance = data['value'].var()
Q_scale = . * data_variance  # % of data variance
R_estimate = . * data_variance  # % of data variance

Q = np.array([[Q_scale]])
R = np.array([[R_estimate]])

# Method 2: Grid search over different Q/R ratios
q_r_ratios = [., ., ., .]
best_ll = -np.inf

for ratio in q_r_ratios:
    Q_test = np.array([[ratio]])
    R_test = np.array([[.]])
    
    kf = Kalmanilter(n_states=, n_obs=, =, H=H, Q=Q_test, R=R_test, x=x, P=P)
    result = kf.fit(data)
    
    if result.log_likelihood > best_ll:
        best_ll = result.log_likelihood
        best_ratio = ratio
        
print(f"est Q/R ratio: {best_ratio}")
```

### Local Level Model

The Local Level Model automatically estimates parameters via ML, but you can control the optimization:

```python
model = LocalLevelModel(
    estimate_params=True,
    initial_level=None,  # uto-initialize from data
    sigma_eta_init=.,  # Initial guess for ML
    sigma_epsilon_init=.
)

# dvanced: Provide bounds for optimization
model._q_bounds = (e-, .)  # ounds on q = σ_η² / σ_ε²
```

---

## Result Interpretation

### GRH Model Results

```python
result = model.fit(data)

# Parameter estimates
print("stimated Parameters:")
print(f"  ω: {model.params['omega']:.f}")
print(f"  α: {model.params['alpha']}")
print(f"  β: {model.params['beta']}")

# Model diagnostics
print("\nModel iagnostics:")
print(f"  Log-Likelihood: {result.log_likelihood:.2f}")
print(f"  I: {result.aic:.2f}")
print(f"  I: {result.bic:.2f}")

# Interpretation
persistence = sum(model.params['alpha']) + sum(model.params['beta'])
print(f"\nPersistence: {persistence:.4f}")
if persistence > .:
    print(" Very high persistence - volatility shocks decay slowly")
elif persistence > .:
    print(" High persistence - typical for financial data")
else:
    print(" Moderate persistence - volatility mean-reverts faster")

# Unconditional volatility
uncond_var = model.params['omega'] / ( - persistence)
uncond_vol = np.sqrt(uncond_var)
print(f"\nUnconditional volatility: {uncond_vol:.4f} ({uncond_vol*22**.*:.2f}% annualized)")
```

### orecast Interpretation

```python
forecast = model.predict(steps=3)

# Point forecasts
print("3-day Volatility orecast:")
print(f"  ay :  {forecast.forecast_values[]:.f}")
print(f"  ay : {forecast.forecast_values[]:.f}")
print(f"  ay 3: {forecast.forecast_values[2]:.f}")

# onfidence intervals
print(f"\nay  % I: [{forecast.ci_lower[]:.f}, {forecast.ci_upper[]:.f}]")
print(f"I Width: {forecast.ci_upper[] - forecast.ci_lower[]:.f}")

# heck for convergence
long_run_vol = np.sqrt(uncond_var)
final_forecast = forecast.forecast_values[-]
convergence = abs(final_forecast - long_run_vol) / long_run_vol

print(f"\nonvergence to long-run mean:")
print(f"  inal forecast: {final_forecast:.f}")
print(f"  Long-run mean:  {long_run_vol:.f}")
print(f"  ifference:     {convergence*:.2f}%")
```

### Kalman ilter Results

```python
result = kf.fit(data, smoothing=True)

# iltered vs Smoothed comparison
filtered_states = result.payload['filtered_states']
smoothed_states = result.payload['smoothed_states']

rmse_filtered = np.sqrt(np.mean((filtered_states[:, ] - true_states)**2))
rmse_smoothed = np.sqrt(np.mean((smoothed_states[:, ] - true_states)**2))

print("State stimation Quality:")
print(f"  iltered RMS: {rmse_filtered:.4f}")
print(f"  Smoothed RMS: {rmse_smoothed:.4f}")
print(f"  Improvement:   {( - rmse_smoothed/rmse_filtered)*:.f}%")

# Innovation diagnostics
innovations = result.payload['innovations']
print(f"\nInnovation Statistics:")
print(f"  Mean: {np.mean(innovations):.f} (should be ≈ )")
print(f"  Std:  {np.std(innovations):.f}")

# Test for whiteness (Ljung-ox test)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_result = acorr_ljungbox(innovations, lags=, return_df=True)
if all(lb_result['lb_pvalue'] > .):
    print(" Innovations are white noise (good fit)")
else:
    print(" Innovations show autocorrelation (model may be misspecified)")
```

---

## est Practices

### ata Preparation

#### . Returns alculation

```python
# or volatility models, use log returns
prices = data['close']
returns = np.log(prices / prices.shift()).dropna()

# Or percentage returns
returns_pct = prices.pct_change().dropna()

# Important: Remove outliers carefully
from scipy.stats import zscore
z_scores = np.abs(zscore(returns))
returns_clean = returns[z_scores < ]  # Remove extreme outliers
```

#### 2. heck for Stationarity

```python
from statsmodels.tsc.stattools import adfuller

result = adfuller(returns)
print(f" Statistic: {result[]:.4f}")
print(f"p-value: {result[]:.4f}")

if result[] < .:
    print("Series is stationary")
else:
    print("Series is non-stationary - consider differencing")
```

#### 3. Handle Missing ata

```python
# orward fill (use previous value)
data_filled = data.fillna(method='ffill')

# Interpolation (for state space models)
data_interpolated = data.interpolate(method='linear')

# Kalman ilter can handle missing observations natively
# Just pass data with NaN values - they'll be skipped
```

### Model Validation

#### . Out-of-Sample Testing

```python
# Split data
train_size = int(len(data) * .)
train_data = data[:train_size]
test_data = data[train_size:]

# it on training data
model.fit(train_data)

# Rolling forecast on test data
predictions = []
for i in range(len(test_data)):
    forecast = model.predict(steps=)
    predictions.append(forecast.forecast_values[])
    
    # Refit with new observation (rolling window)
    current_data = data[:train_size + i + ]
    model.fit(current_data)

# alculate forecast errors
forecast_errors = test_data['value'].values - np.array(predictions)
mse = np.mean(forecast_errors**2)
mae = np.mean(np.abs(forecast_errors))

print(f"Out-of-sample MS: {mse:.4f}")
print(f"Out-of-sample M: {mae:.4f}")
```

#### 2. Residual iagnostics

```python
# xtract standardized residuals
residuals = model.get_standardized_residuals()

# Test for autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=2)

# Test for RH effects (should be none if GRH fit is good)
lb_test_squared = acorr_ljungbox(residuals**2, lags=2)

# Normality test
from scipy.stats import jarque_bera
jb_stat, jb_pvalue = jarque_bera(residuals)

print(f"Jarque-era test: stat={jb_stat:.2f}, p={jb_pvalue:.4f}")
if jb_pvalue < .:
    print(" Residuals not normal - consider t-distribution")
```

### Production eployment

#### . Model Persistence

```python
import pickle

# Save fitted model
with open('garch_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('garch_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Use loaded model
forecast = loaded_model.predict(steps=)
```

#### 2. rror Handling

```python
try:
    result = model.fit(data)
    
    if not result.success:
        print("Model did not converge")
        # allback strategy
        model_simple = GRHModel(input_schema, {'p': , 'q': }, meta)
        result = model_simple.fit(data)
        
except Valuerror as e:
    print(f"ata validation error: {e}")
    # Handle invalid data
    
except Runtimerror as e:
    print(f"itting error: {e}")
    # Handle convergence failures
```

#### 3. Performance Monitoring

```python
import time

# Measure fitting time
start = time.time()
result = model.fit(data)
fit_time = time.time() - start

print(f"itting took {fit_time:.2f} seconds")

if fit_time > :
    print("   itting is slow - consider:")
    print("   • Reducing data size")
    print("   • Using simpler model")
    print("   • hecking for data issues")
```

---

## Troubleshooting

### ommon Issues and Solutions

#### Issue: "Model failed to converge"

**auses:**
- Poor initial guesses
- ata issues (non-stationarity, outliers)
- Model too complex for data

**Solutions:**
```python
# . Try simpler model
params_simple = {'p': , 'q': , 'mean_model': 'Zero'}

# 2. Increase optimization iterations
params['max_iter'] = 

# 3. lean data
data_clean = data[np.abs(zscore(data)) < 4]

# 4. Try different starting values
model.fit(data, starting_values=custom_starts)
```

#### Issue: "Non-positive definite covariance matrix"

**ause:** Process/measurement noise matrices are ill-conditioned

**Solution:**
```python
# dd small regularization
Q_reg = Q + np.eye(n_states) * e-
R_reg = R + np.eye(n_obs) * e-

kf = Kalmanilter(n_states, n_obs, , H, Q_reg, R_reg, x, P)
```

#### Issue: "Predictions are too smooth/noisy"

**or Kalman ilter:**
```python
# Too smooth → Increase Q (more process noise)
Q = Q * 

# Too noisy → ecrease Q (less process noise)
Q = Q / 
```

**or GRH:**
```python
# heck persistence
persistence = alpha + beta
if persistence > .:
    print("Very high persistence - forecasts converge slowly")
    # This is normal for financial data
```

#### Issue: "Local Level Model is very slow"

**Solutions:**
```python
# . Reduce optimization iterations
model._max_iter =   # efault is higher

# 2. Provide better initial guesses
model._sigma_eta_init = np.std(data) * .
model._sigma_epsilon_init = np.std(data)

# 3. or very large datasets, use Kalman ilter with fixed parameters
sigma_eta = .
sigma_epsilon = .
Q = np.array([[sigma_eta**2]])
R = np.array([[sigma_epsilon**2]])
# ... create Kalman ilter with these fixed values
```

---

## dvanced Usage

### ustom State Space Models

You can create custom state space models by extending the Kalman ilter:

```python
from krl_models.state_space import Kalmanilter
import numpy as np

class SeasonalModel(Kalmanilter):
    """ustom seasonal state space model"""
    
    def __init__(self, period=2):
        self.period = period
        n_states = period
        n_obs = 
        
        # State transition: rotate seasonal components
         = np.zeros((period, period))
        [, :] = -
        [:, :-] = np.eye(period-)
        
        # Observe first seasonal component
        H = np.array([[] + []*(period-)])
        
        # Noise matrices
        Q = np.eye(period) * .
        R = np.array([[.]])
        
        # Initial conditions
        x = np.zeros(period)
        P = np.eye(period)
        
        super().__init__(n_states, n_obs, , H, Q, R, x, P)

# Use custom model
seasonal_model = SeasonalModel(period=2)
result = seasonal_model.fit(monthly_data)
```

### Parallel Model itting

or fitting multiple models in parallel:

```python
from concurrent.futures import ProcessPoolxecutor
from functools import partial

def fit_model_wrapper(params, data, input_schema, meta):
    model = GRHModel(input_schema, params, meta)
    result = model.fit(data)
    return params, result.aic

# efine parameter grid
param_grid = [
    {'p': , 'q': },
    {'p': , 'q': 2},
    {'p': 2, 'q': },
    {'p': 2, 'q': 2}
]

# it in parallel
with ProcessPoolxecutor(max_workers=4) as executor:
    fit_func = partial(fit_model_wrapper, 
                       data=data, 
                       input_schema=input_schema, 
                       meta=meta)
    results = list(executor.map(fit_func, param_grid))

# ind best model
best_params, best_aic = min(results, key=lambda x: x[])
print(f"est model: {best_params} (I = {best_aic:.2f})")
```

### ombining Multiple Models

nsemble forecasting:

```python
# it multiple models
models = {
    'GRH': GRHModel(input_schema, params_garch, meta),
    'GRH': GRHModel(input_schema, params_egarch, meta),
    'GJR': GJRGRHModel(input_schema, params_gjr, meta)
}

forecasts = {}
weights = {}

for name, model in models.items():
    result = model.fit(data)
    forecast = model.predict(steps=3)
    
    # Weight by inverse I (better models get higher weight)
    weight = np.exp(-result.aic / 2)
    
    forecasts[name] = forecast.forecast_values
    weights[name] = weight

# Normalize weights
total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

# nsemble forecast (weighted average)
ensemble_forecast = sum(weights[name] * forecasts[name] 
                       for name in models.keys())

print("nsemble Weights:")
for name, weight in weights.items():
    print(f"  {name}: {weight*:.f}%")
```

---

## Next Steps

### urther Learning

. **xamples:** heck the `examples/` directory for complete workflows
   - `example__garch_volatility_forecasting.py`
   - `example_2_egarch_leverage_analysis.py`
   - `example_3_gjr_garch_threshold_detection.py`
   - `example_4_kalman_filter_tracking.py`

2. **PI Reference:** See `docs/PI_RRN.md` for detailed class documentation

3. **Mathematical etails:** See `docs/MTHMTIL_ORMULTIONS.md` for equations and derivations

4. **Performance:** See `benchmarks/PRORMN_RPORT.md` for timing and scalability analysis

### Getting Help

- **Issues:** https://github.com/KR-Labs/krl-model-zoo/issues
- **iscussions:** https://github.com/KR-Labs/krl-model-zoo/discussions
- **mail:** support@kr-labs.ai

### ontributing

We welcome contributions! See `ONTRIUTING.md` for guidelines.

---

**Happy Modeling!**

*KRL Model Zoo Team*  
*Last Updated: October 24, 22*
