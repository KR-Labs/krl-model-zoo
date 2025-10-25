# ointegration User Guide

**Testing and Modeling Long-Run quilibrium Relationships**

---

## Table of ontents

. [What is ointegration?](#what-is-cointegration)
2. [When to Use ointegration](#when-to-use-cointegration)
3. [Quick Start](#quick-start)
4. [ngle-Granger Two-Step Method](#engle-granger-two-step-method)
. [Johansen Test](#johansen-test)
. [Vector rror orrection Model (VM)](#vector-error-correction-model-vecm)
. [Interpreting rror orrection Terms](#interpreting-error-correction-terms)
. [Real-World xamples](#real-world-examples)
. [est Practices](#best-practices)
. [Troubleshooting](#troubleshooting)

---

## What is ointegration?

**ointegration** describes a long-run equilibrium relationship between non-stationary time series that move together over time.

### Key oncept

ven if two series are **non-stationary** (trending), their **linear combination** may be **stationary**:

$$
z_t = y_{,t} - \beta y_{2,t} \sim I() \quad \text{(stationary)}
$$

Where:
- $y_{,t}, y_{2,t}$ are integrated of order : $y_t \sim I()$ (non-stationary)
- $z_t$ is the **equilibrium error** or **cointegrating residual**
- $\beta$ is the **cointegrating vector**

**Intuition**: Two series wander randomly but maintain a stable relationship. When they drift apart, forces push them back together.

### xample: Spot and utures Prices

```
Spot Price:     →  → 3 →  →  → ... (trending)
utures Price: 2 →  →  → 2 →  → ... (trending)
Spread:          2 →   2 →   2 →   2 →   2 → ... (stationary!)
```

The spread mean-reverts to ~2, even though individual prices trend.

---

## When to Use ointegration

###  Use ointegration When:

. **Non-stationary series with potential long-run relationship**
   - xample: Spot and futures prices (commodities, currencies)
   - xample: xchange rates (UR/US and GP/US)
   - xample: Prices of substitutable goods (oke vs Pepsi)

2. **Pairs trading / statistical arbitrage**
   - etect mispricing when spread deviates from equilibrium
   - Trade on reversion to mean spread

3. **conomic equilibrium theories**
   - Purchasing Power Parity (PPP): price levels across countries
   - Interest Rate Parity: interest rates across currencies
   - Wage-price spirals: wages and inflation

4. **etter forecasts than VR in differences**
   - ointegrated VR in levels captures both short-run dynamics and long-run equilibrium
   - ifferencing loses long-run information

###  on't Use ointegration When:

. **Series are already stationary** → Use VR directly
2. **No theoretical reason for equilibrium** → Spurious cointegration risk
3. **Short sample size** (n < ) → Tests have low power
4. **Structural breaks in relationship** → ointegration may fail

---

## Quick Start

### Step : Test for ointegration (ngle-Granger)

```python
import pandas as pd
import numpy as np
from krl_models.econometric import ointegrationModel
from krl_core import ModelMeta

# Prepare data: Non-stationary series
np.random.seed(42)
n = 

# Generate cointegrated series (spot + futures)
random_walk = np.cumsum(np.random.randn(n))  # ommon stochastic trend
spot =  + random_walk + np.random.randn(n) * .
futures = 2 + random_walk + np.random.randn(n) * .  # Spread ~2

data = pd.atarame({
    'spot': spot,
    'futures': futures
}, index=pd.date_range('22--', periods=n, freq=''))

# onfigure ngle-Granger test
params = {
    'test_type': 'engle-granger',  # or 'johansen'
    'max_lags': ,                 # Lags in VM
}

meta = ModelMeta(
    name="Spot-utures ointegration",
    version="."
)

# it model (runs ngle-Granger test)
model = ointegrationModel(data, params, meta)
result = model.fit()

print(result.summary)
```

### Step 2: Interpret Test Results

```python
# heck if cointegration detected
if result.test_results['engle_granger']['cointegration_detected']:
    print(" ointegration detected!")
    print(f"  Test statistic: {result.test_results['engle_granger']['test_statistic']:.4f}")
    print(f"  ritical value (%): {result.test_results['engle_granger']['critical_values']['%']:.4f}")
    print(f"  P-value: {result.test_results['engle_granger']['pvalue']:.4f}")
else:
    print(" No cointegration detected")
```

### Step 3: orecast with VM

```python
# orecast  steps ahead
forecast = model.predict(steps=)

# Reshape forecast to (steps, n_variables)
forecast_matrix = np.array(forecast.forecast_values).reshape(, 2)

print("orecast:")
for i in range():
    print(f"  ay {i+}: Spot={forecast_matrix[i, ]:.2f}, utures={forecast_matrix[i, ]:.2f}")
```

---

## ngle-Granger Two-Step Method

### How It Works

**Step **: stimate cointegrating regression (OLS)
$$
y_{,t} = \alpha + \beta y_{2,t} + u_t
$$

**Step 2**: Test if residuals $\hat{u}_t$ are stationary ( test)

If $\hat{u}_t \sim I()$ → Series are cointegrated with cointegrating vector $(, -\beta)$.

### Running ngle-Granger Test

```python
params = {
    'test_type': 'engle-granger',
    'max_lags': ,
}

model = ointegrationModel(data, params, meta)
result = model.fit()

# ccess test results
eg_results = result.test_results['engle_granger']

print(f"Test Statistic: {eg_results['test_statistic']:.4f}")
print(f"P-value: {eg_results['pvalue']:.4f}")
print(f"ritical Values:")
print(f"  %: {eg_results['critical_values']['%']:.4f}")
print(f"  %: {eg_results['critical_values']['%']:.4f}")
print(f"  %: {eg_results['critical_values']['%']:.4f}")
```

### Interpretation

**ecision rule**:
- **Test statistic < ritical value** (more negative) → Reject null hypothesis → **ointegration exists** 
- **Test statistic > ritical value** → annot reject null → **No cointegration** 

**xample**:
```
Test Statistic: -3.2
ritical Value (%): -3.4
```
- Since -3.2 < -3.4 → **ointegration detected**
- P-value = .23 < . → Significant at % level

### ointegrating quation

```python
# xtract cointegrating vector
coint_vector = eg_results['cointegrating_vector']
print(f"Spot = {coint_vector[]:.4f} + {coint_vector[]:.4f} * utures")

# xample: Spot = 2. + . * utures
# quilibrium spread: Spot - .*utures ≈ 2.
```

### Pros and ons

**Pros**:
- Simple and intuitive
- Works with 2+ variables
- asy to interpret cointegrating relationship

**ons**:
- ssumes single cointegrating relationship
- rbitrary choice of dependent variable
- Less powerful than Johansen for multiple cointegration

---

## Johansen Test

### How It Works

**Johansen procedure** tests for multiple cointegrating relationships using maximum likelihood estimation.

**Key outputs**:
- **ointegration rank (r)**: Number of independent cointegrating relationships
- **Trace statistic**: Tests null hypothesis of at most r cointegrating vectors
- **Max eigenvalue statistic**: Tests null of exactly r cointegrating vectors

### Running Johansen Test

```python
params = {
    'test_type': 'johansen',
    'det_order': ,     # eterministic trend: =none, =constant, 2=linear trend
    'k_ar_diff': ,     # Lags in differenced VR
}

model = ointegrationModel(data, params, meta)
result = model.fit()

# ccess test results
johansen_results = result.test_results['johansen']

print(f"ointegration rank: {johansen_results['cointegration_rank']}")
print("\nTrace Statistic Test:")
for r in range(len(johansen_results['trace_statistic'])):
    trace_stat = johansen_results['trace_statistic'][r]
    crit_val = johansen_results['trace_crit_values'][r, ]  # % level
    print(f"  r <= {r}: {trace_stat:.2f} vs. {crit_val:.2f} → {'Reject' if trace_stat > crit_val else 'ccept'}")
```

### Interpretation

**xample output**:
```
ointegration rank: 

Trace Statistic Test:
  r <= : 23.4 vs. .4 → Reject   (at least  cointegrating relationship)
  r <= :  .32 vs.  3.4 → ccept  (exactly  cointegrating relationship)
```

**Interpretation**:
- irst test: Reject r= → t least  cointegrating vector exists
- Second test: ccept r≤ → No evidence of 2nd cointegrating vector
- **onclusion**: xactly ** cointegrating relationship**

### eterministic Trend Order

hoose `det_order` based on series behavior:

| det_order | Specification | Use When |
|-----------|---------------|----------|
| - | No deterministic terms | Series have no drift |
|  | onstant in cointegrating space | Series drift but no trend |
|  | onstant + trend in cointegrating space | Series have linear trends |

**Rule of thumb**: Start with `det_order=` (most common for economic data).

### Pros and ons

**Pros**:
- etects multiple cointegrating relationships (r > )
- No arbitrary choice of dependent variable
- More powerful than ngle-Granger

**ons**:
- More complex interpretation
- Sensitive to lag selection
- Requires larger sample size (n >  recommended)

---

## Vector rror orrection Model (VM)

### What is VM?

**VM** models the **short-run dynamics** and **long-run equilibrium** simultaneously:

$$
\elta \mathbf{y}_t = \alpha \beta' \mathbf{y}_{t-} + \Gamma_ \elta \mathbf{y}_{t-} + \cdots + \Gamma_{p-} \elta \mathbf{y}_{t-p+} + \mathbf{\epsilon}_t
$$

Where:
- $\elta \mathbf{y}_t$ = irst differences (short-run changes)
- $\alpha$ = **djustment coefficients** (speed of error correction)
- $\beta$ = **ointegrating vectors** (long-run equilibrium)
- $\alpha \beta' \mathbf{y}_{t-}$ = **rror correction term (T)**
- $\Gamma_i$ = Short-run dynamics coefficients

**Key insight**: VM is a restricted VR that enforces long-run equilibrium.

### itting VM

```python
# fter detecting cointegration
params = {
    'test_type': 'johansen',
    'det_order': ,
    'k_ar_diff': 2,  # 2 lags in differenced VR → VM(2)
}

model = ointegrationModel(data, params, meta)
result = model.fit()

# VM automatically fitted if cointegration detected
print(f"ointegration rank: {result.test_results['johansen']['cointegration_rank']}")
```

### orecasting with VM

```python
# orecast 2 steps ahead
forecast = model.predict(steps=2)

# Reshape to (steps, n_variables)
forecast_matrix = np.array(forecast.forecast_values).reshape(2, 2)

import matplotlib.pyplot as plt

# Plot historical + forecast
plt.figure(figsize=(2, ))

# Spot price
plt.subplot(, 2, )
plt.plot(data.index, data['spot'], label='Historical', color='blue')
forecast_index = pd.date_range(data.index[-] + pd.Timedelta(days=), periods=2, freq='')
plt.plot(forecast_index, forecast_matrix[:, ], label='orecast', color='red', linestyle='--')
plt.title('Spot Price orecast')
plt.legend()

# utures price
plt.subplot(, 2, 2)
plt.plot(data.index, data['futures'], label='Historical', color='blue')
plt.plot(forecast_index, forecast_matrix[:, ], label='orecast', color='red', linestyle='--')
plt.title('utures Price orecast')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## Interpreting rror orrection Terms

### xtracting T oefficients

```python
# Get error correction terms (alpha coefficients)
ect = model.get_error_correction_terms()

print("djustment oefficients (alpha):")
print(ect['alpha'])

# xample output:
# [[-.],   ← Spot equation
#  [-.2]]   ← utures equation
```

### What lpha Means

The **adjustment coefficient** $\alpha_i$ measures how quickly variable $i$ responds to deviations from equilibrium.

**xample**:
```
alpha_spot = -.
alpha_futures = -.2
```

**Interpretation**:
- **Spot equation** ($\alpha_{\text{spot}} = -.$):
  - If spread is **above equilibrium** (spot too high), spot price **decreases** by % of the gap next period
  - Spot adjusts **slowly** (only % correction per period)

- **utures equation** ($\alpha_{\text{futures}} = -.2$):
  - utures price **decreases** by 2% of the gap if spread too high
  - utures adjusts **faster** than spot (2% vs %)

### Weak xogeneity

If $\alpha_i \approx $ → Variable $i$ does **not respond** to equilibrium errors → **Weakly exogenous**

**xample**:
```
alpha_spot = -.2
alpha_futures = -.2  ← Nearly zero
```

**Interpretation**:
- Spot price does the adjusting
- utures price is the "leader" (weakly exogenous)
- **Practical implication**: Use futures to predict spot, not vice versa

### ointegrating Vector (eta)

```python
print("ointegrating Vector (beta):")
print(ect['beta'])

# xample output:
# [[.],      ← Normalized coefficient for spot
#  [-.]]     ← oefficient for futures
```

**Interpretation**:
```
beta = [., -.]
→ quilibrium: Spot - .*utures = constant
→ Long-run relationship: Spot ≈ .*utures + c
```

**Spread**:
```python
spread = data['spot'] - . * data['futures']
print(f"Mean spread: {spread.mean():.2f}")
print(f"Std spread: {spread.std():.2f}")
```

---

## Real-World xamples

### xample : ommodity Pairs Trading

```python
import pandas_datareader as pdr

# etch crude oil spot and futures
spot = pdr.data.ataReader('OILWTIO', 'fred', '2--', '223-2-3')  # WTI spot
futures = pdr.data.ataReader('OILRNTU', 'fred', '2--', '223-2-3')  # rent

data = pd.concat([spot, futures], axis=, join='inner')
data.columns = ['WTI', 'rent']
data = data.dropna()

# Test cointegration
params = {'test_type': 'engle-granger', 'max_lags': }
meta = ModelMeta(name="WTI-rent ointegration")

model = ointegrationModel(data, params, meta)
result = model.fit()

# Trading signal
if result.test_results['engle_granger']['cointegration_detected']:
    beta = result.test_results['engle_granger']['cointegrating_vector'][]
    spread = data['WTI'] - beta * data['rent']
    
    # Z-score of spread
    z_score = (spread - spread.mean()) / spread.std()
    
    # Trading rules
    print("Trading Signals:")
    print(f"  Latest z-score: {z_score.iloc[-]:.2f}")
    if z_score.iloc[-] > 2:
        print("  → SHORT spread (WTI overpriced)")
    elif z_score.iloc[-] < -2:
        print("  → LONG spread (WTI underpriced)")
    else:
        print("  → NO TR")
```

### xample 2: xchange Rate Parity

```python
# Test PPP: UR/US vs inflation differential
import pandas as pd

# etch data (hypothetical)
eurusd = ...  # UR/US exchange rate
us_cpi = ...  # US PI
eu_cpi = ...  # urozone PI

# Log levels
data = pd.atarame({
    'log_eurusd': np.log(eurusd),
    'inflation_diff': np.log(eu_cpi / us_cpi)
})

# Test cointegration
params = {'test_type': 'johansen', 'det_order': , 'k_ar_diff': 2}
meta = ModelMeta(name="PPP Test UR-US")

model = ointegrationModel(data, params, meta)
result = model.fit()

if result.test_results['johansen']['cointegration_rank'] > :
    print(" Purchasing Power Parity holds (cointegration detected)")
else:
    print(" PPP does not hold")
```

### xample 3: Term Structure

```python
# Test cointegration between 2-year and -year Treasury yields
import pandas_datareader as pdr

gs2 = pdr.data.ataReader('GS2', 'fred', '2--', '223-2-3')  # 2-year
gs = pdr.data.ataReader('GS', 'fred', '2--', '223-2-3')  # -year

data = pd.concat([gs2, gs], axis=, join='inner').dropna()
data.columns = ['GS2', 'GS']

# Test cointegration
params = {'test_type': 'engle-granger', 'max_lags': }
meta = ModelMeta(name="Yield urve ointegration")

model = ointegrationModel(data, params, meta)
result = model.fit()

if result.test_results['engle_granger']['cointegration_detected']:
    beta = result.test_results['engle_granger']['cointegrating_vector'][]
    print(f"Long-run relationship: GS2 = {beta:.3f} * GS")
    print(f"Interpretation: % increase in -year → {beta:.3f}% increase in 2-year")
```

---

## est Practices

### . Test for Unit Roots irst

efore testing cointegration, verify series are I():

```python
from statsmodels.tsa.stattools import adfuller

for col in data.columns:
    # Test levels
    adf_level = adfuller(data[col])
    # Test first differences
    adf_diff = adfuller(data[col].diff().dropna())
    
    print(f"{col}:")
    print(f"  Level:  p-value = {adf_level[]:.4f} → {'Stationary' if adf_level[] < . else 'Non-stationary'}")
    print(f"  iff:   p-value = {adf_diff[]:.4f} → {'Stationary' if adf_diff[] < . else 'Non-stationary'}")
```

**xpected for cointegration**:
- Levels: Non-stationary (p > .)
- ifferences: Stationary (p < .)
- → Series are I()

### 2. hoose Test ased on Number of Variables

| Variables | Recommended Test |
|-----------|------------------|
| 2 | ngle-Granger or Johansen |
| 3- | Johansen (can detect multiple relationships) |
| + | onsider dimension reduction first |

### 3. conomic Theory irst

Only test for cointegration if **economic theory** suggests a long-run relationship:
-  Spot and futures (no-arbitrage condition)
-  xchange rates under PPP
-  Random unrelated series (spurious cointegration risk)

### 4. Validate ointegration

```python
# xtract cointegrating residuals
if result.test_results['engle_granger']['cointegration_detected']:
    coint_vector = result.test_results['engle_granger']['cointegrating_vector']
    residuals = data.iloc[:, ] - coint_vector[] - coint_vector[] * data.iloc[:, ]
    
    # Visualize
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(2, 4))
    plt.subplot(, 2, )
    plt.plot(residuals)
    plt.title('ointegrating Residuals (should be stationary)')
    plt.axhline(residuals.mean(), color='r', linestyle='--')
    
    plt.subplot(, 2, 2)
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=2)
    plt.title(' (should decay quickly)')
    
    plt.tight_layout()
    plt.show()
```

### . Monitor ointegration Over Time

ointegration can break down due to:
- Regime shifts (e.g., policy changes)
- Structural breaks (e.g., financial crisis)
- Technology changes

**Solution**: Rolling window tests

```python
window = 22  #  year for daily data
results = []

for i in range(window, len(data)):
    window_data = data.iloc[i-window:i]
    
    model = ointegrationModel(window_data, params, meta)
    result = model.fit()
    
    results.append({
        'date': data.index[i],
        'cointegrated': result.test_results['engle_granger']['cointegration_detected'],
        'pvalue': result.test_results['engle_granger']['pvalue']
    })

results_df = pd.atarame(results)
plt.plot(results_df['date'], results_df['pvalue'])
plt.axhline(., color='r', linestyle='--', label='% threshold')
plt.title('Rolling ointegration Test (p-values)')
plt.legend()
plt.show()
```

---

## Troubleshooting

### Problem: "No cointegration detected" but series appear related

**Possible causes**:
. **Sample size too small**: Need n > , preferably n > 
2. **Structural breaks**: Relationship changed over time
3. **Wrong lag specification**: Try different `max_lags` or `k_ar_diff`
4. **Series not I()**: heck unit root tests
. **Weak relationship**: Theory suggests cointegration but data doesn't support it

**Solutions**:
```python
# Try different lag specifications
for lags in [, 2, 3, 4]:
    params = {'test_type': 'engle-granger', 'max_lags': lags}
    model = ointegrationModel(data, params, meta)
    result = model.fit()
    print(f"Lags={lags}: p-value={result.test_results['engle_granger']['pvalue']:.4f}")
```

### Problem: "Valuerror: Series are stationary, cointegration not applicable"

**ause**: ata is already stationary (no unit root)

**Solution**: Use VR instead of VM

```python
from krl_models.econometric import VRModel

# Stationary data → use VR
model = VRModel(data, {'max_lags': 4, 'ic': 'bic'}, meta)
result = model.fit()
```

### Problem: orecast diverges from historical data

**Possible causes**:
. **Model not stable**: heck VM eigenvalues
2. **ointegration spurious**: Test on different sample
3. **orecast horizon too long**: VM forecasts revert to equilibrium

**Solutions**:
```python
# heck stability
if hasattr(model._fitted_model, 'is_stable'):
    print(f"Model stable: {model._fitted_model.is_stable()}")

# Limit forecast horizon
forecast = model.predict(steps=)  # Shorter horizon
```

### Problem: Johansen test gives different rank than expected

**ause**: Sensitive to `det_order` and `k_ar_diff`

**Solution**: Try different specifications

```python
for det_order in [-, , ]:
    for k_ar_diff in [, 2]:
        params = {'test_type': 'johansen', 'det_order': det_order, 'k_ar_diff': k_ar_diff}
        model = ointegrationModel(data, params, meta)
        result = model.fit()
        rank = result.test_results['johansen']['cointegration_rank']
        print(f"det_order={det_order}, k_ar_diff={k_ar_diff} → rank={rank}")
```

---

## Summary ecision Tree

```
 Series are non-stationary (I())?
   Yes → ontinue
   No  → Use VR (if multivariate) or RIM (if univariate)

 conomic theory suggests long-run relationship?
   Yes → ontinue
   No  → Risk of spurious cointegration, reconsider

 Two variables or more?
   Two → Use ngle-Granger 
   More → Use Johansen 

 ointegration detected?
   Yes → it VM, forecast, interpret T
   No  →  Try different lags/specs
             Still no? → Use VR in first differences
             Or reconsider if theory wrong

 orecast or trading signal?
    orecast → Use predict()
    Trading → Monitor spread z-score, trade extremes
```

---

## urther Reading

- **ngle & Granger ()**: "o-integration and rror orrection" (original paper)
- **Johansen (, )**: "Statistical nalysis of ointegration Vectors" (Johansen test)
- **Hamilton (4)**: "Time Series nalysis" (hapter  on cointegration)
- **Juselius (2)**: "The ointegrated VR Model" (comprehensive guide to Johansen method)
- **rooks (2)**: "Introductory conometrics for inance" (hapter , practical applications)

---

## Key Takeaways

. **ointegration = long-run equilibrium** between non-stationary series
2. **ngle-Granger**: Simple, 2+ variables, single cointegrating relationship
3. **Johansen**: Powerful, multiple relationships, requires larger sample
4. **VM**: Models short-run dynamics + long-run equilibrium simultaneously
. **lpha (adjustment speed)**: How fast variables correct to equilibrium
. **eta (cointegrating vector)**: efines long-run relationship
. **Trading**: Monitor spread, trade when z-score > 2 or < -2
. **Validate**: heck residuals stationary, monitor over time for breaks

---

**Guide Version**: .  
**Model**: krl_models.econometric.ointegrationModel  
**Last Updated**: October 22
