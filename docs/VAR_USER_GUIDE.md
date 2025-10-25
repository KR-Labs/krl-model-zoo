# VR Model User Guide

**Vector utoregression (VR) for Multivariate Time Series orecasting**

---

## Table of ontents

. [What is VR?](#what-is-var)
2. [When to Use VR](#when-to-use-var)
3. [Quick Start](#quick-start)
4. [Understanding VR omponents](#understanding-var-components)
. [Granger ausality nalysis](#granger-causality-analysis)
. [Impulse Response unctions (IR)](#impulse-response-functions-irf)
. [orecast rror Variance ecomposition (V)](#forecast-error-variance-decomposition-fevd)
. [Model iagnostics](#model-diagnostics)
. [Real-World xamples](#real-world-examples)
. [est Practices](#best-practices)
. [Troubleshooting](#troubleshooting)

---

## What is VR?

**Vector utoregression (VR)** is a statistical model for analyzing the dynamic relationships between multiple time series variables. Unlike univariate models (RIM, SRIM), VR captures:

- **ross-variable dependencies**: How changes in one variable affect others
- **eedback loops**: idirectional relationships between variables
- **ynamic systems**: volution of multiple indicators over time

### Mathematical ormulation

 VR(p) model with k variables:

$$
\mathbf{y}_t = \mathbf{c} + \mathbf{}_ \mathbf{y}_{t-} + \mathbf{}_2 \mathbf{y}_{t-2} + \cdots + \mathbf{}_p \mathbf{y}_{t-p} + \mathbf{\epsilon}_t
$$

Where:
- $\mathbf{y}_t$ = $(y_{,t}, y_{2,t}, \ldots, y_{k,t})'$ is a k× vector of variables at time t
- $\mathbf{c}$ is a k× vector of constants
- $\mathbf{}_i$ are k×k coefficient matrices for lag i
- $\mathbf{\epsilon}_t$ is a k× vector of white noise error terms

**Key Insight**: ach variable is modeled as a function of:
. Its own past values (autoregressive component)
2. Past values of **all other variables** (cross-variable effects)

---

## When to Use VR

###  Use VR When:

. **Multiple interrelated time series**
   - xample: GP, unemployment, inflation (macroeconomic system)
   - xample: Stock prices of correlated companies
   
2. **idirectional causality suspected**
   - xample: Interest rates ↔ inflation
   - xample: Supply ↔ demand

3. **orecasting with cross-variable information**
   - Using past GP to improve unemployment forecasts
   - Leveraging lead-lag relationships

4. **Understanding system dynamics**
   - Impulse response analysis (how shocks propagate)
   - Policy impact assessment

. **ll variables are stationary** (or can be made stationary)
   - ifferencing, detrending, or logging may be needed
   - ointegrated non-stationary series → use VM instead

###  on't Use VR When:

. **Variables are independent** → Use separate univariate models
2. **Non-stationary without cointegration** → Spurious regression risk
3. **Many variables (k>)** → urse of dimensionality (too many parameters)
4. **Structural breaks** → onsider regime-switching models
. **High-frequency data with microstructure noise** → Use state-space models

---

## Quick Start

### asic VR Model

```python
import pandas as pd
from krl_models.econometric import VRModel
from krl_core import ModelMeta

# Prepare data: multivariate atarame
data = pd.atarame({
    'gdp': [, 2, , , , 2, , , 2, 23],
    'unemployment': [., 4., 4., 4.3, 4., 3., 3., 3., 3.4, 3.2]
}, index=pd.date_range('22Q', periods=, freq='Q'))

# onfigure VR with automatic lag selection
params = {
    'max_lags': 4,        # Maximum lags to consider
    'ic': 'aic',          # Information criterion ('aic', 'bic', 'hqic', 'fpe')
}

meta = ModelMeta(
    name="GP-Unemployment VR",
    version=".",
    author="Your Name"
)

# it model
model = VRModel(data, params, meta)
result = model.fit()

# heck selected lag order
print(f"Selected lag order: {model._selected_lag}")

# orecast 4 quarters ahead
forecast = model.predict(steps=4)
print(forecast.forecast_values)  # lattened: [gdp_t+, unemp_t+, gdp_t+2, unemp_t+2, ...]
```

### Output Interpretation

```python
# Reshape forecast to (steps, n_variables)
import numpy as np
forecast_matrix = np.array(forecast.forecast_values).reshape(4, 2)

print("orecast:")
print(f"  Q: GP={forecast_matrix[,]:.2f}, Unemployment={forecast_matrix[,]:.2f}%")
print(f"  Q2: GP={forecast_matrix[,]:.2f}, Unemployment={forecast_matrix[,]:.2f}%")
```

---

## Understanding VR omponents

### . Lag Order Selection

**What it means**: How many past periods to include in the model

**Selection criteria**:
- **I (kaike Information riterion)**: alances fit and complexity, prefers more lags
- **I (ayesian Information riterion)**: Penalizes complexity more, prefers fewer lags
- **HQI (Hannan-Quinn)**: Middle ground between I and I
- **P (inal Prediction rror)**: ocuses on forecast accuracy

**Rule of thumb**:
- Quarterly data: Try `max_lags=4` ( year of history)
- Monthly data: Try `max_lags=2` ( year of history)
- nnual data: Try `max_lags=3` (3 years)

```python
# ompare different criteria
for ic in ['aic', 'bic', 'hqic', 'fpe']:
    params = {'max_lags': , 'ic': ic}
    model = VRModel(data, params, meta)
    model.fit()
    print(f"{ic.upper()}: Selected {model._selected_lag} lags")
```

**Interpretation**:
- If I selects fewer lags than I → More parsimonious model
- If all criteria agree → Strong evidence for that lag order
- If criteria disagree → Use domain knowledge or cross-validation

### 2. oefficient Matrices

VR stores coefficient matrices $\mathbf{}_, \mathbf{}_2, \ldots, \mathbf{}_p$:

```python
coeffs = model.get_coefficients()

# xample: VR(2) with 2 variables
# coeffs['lag_'] = [[a, a2],   # y(t) depends on y(t-), y2(t-)
#                    [a2, a22]]   # y2(t) depends on y(t-), y2(t-)

print("Lag  coefficients:")
print(coeffs['lag_'])
```

**Interpretation**:
- `coeffs['lag_'][, ]`: ffect of GP(t-) on GP(t)
- `coeffs['lag_'][, ]`: ffect of Unemployment(t-) on GP(t) ← **ross-variable effect**
- `coeffs['lag_'][, ]`: ffect of GP(t-) on Unemployment(t) ← **ross-variable effect**

**xample**:
```
 = [[ ., -.],
      [-.2,  .]]
```
- Row : GP(t) = .×GP(t-) - .×Unemployment(t-) + ...
  - Interpretation: Higher past unemployment predicts **lower** future GP (Okun's Law!)
  
- Row 2: Unemployment(t) = -.2×GP(t-) + .×Unemployment(t-) + ...
  - Interpretation: Higher past GP predicts **lower** future unemployment

---

## Granger ausality nalysis

### What is Granger ausality?

**Granger causality** tests whether past values of variable X help predict variable Y, beyond what Y's own past values provide.

**Important**: Granger causality ≠ true causality! It's a **predictive relationship** test.

### Running the Test

```python
# Test: oes GP Granger-cause Unemployment?
gc_result = model.granger_causality_test(
    caused_var='unemployment',
    causing_var='gdp',
    maxlag=None  # Uses fitted model's lag order
)

print(gc_result)
```

### Interpreting Results

```python
{
    'caused': 'unemployment',
    'causing': 'gdp',
    'maxlag': 2,
    'results_by_lag': {
        'lag_': {
            'ssr_ftest_pvalue': .4,    # p-value for -test
            'lrtest_pvalue': .42,       # p-value for likelihood ratio test
            ...
        },
        'lag_2': {
            'ssr_ftest_pvalue': .2,
            'lrtest_pvalue': .,
            ...
        }
    }
}
```

**ecision rule**:
- **p-value < .**: Reject null hypothesis → **GP Granger-causes Unemployment**
- **p-value > .**: annot reject null → No evidence of Granger causality

**xample interpretation**:
```python
for lag, tests in gc_result['results_by_lag'].items():
    pval = tests['ssr_ftest_pvalue']
    if pval < .:
        print(f"{lag}: GP → Unemployment (p={pval:.4f})  Significant")
    else:
        print(f"{lag}: GP → Unemployment (p={pval:.4f})  Not significant")
```

### idirectional Testing

Test both directions to detect feedback loops:

```python
# GP → Unemployment
gc = model.granger_causality_test('unemployment', 'gdp')

# Unemployment → GP
gc2 = model.granger_causality_test('gdp', 'unemployment')

# Interpret
if gc['results_by_lag']['lag_']['ssr_ftest_pvalue'] < .:
    print("GP → Unemployment: Significant")
    
if gc2['results_by_lag']['lag_']['ssr_ftest_pvalue'] < .:
    print("Unemployment → GP: Significant")
```

**Possible outcomes**:
. **Unidirectional**: Only X → Y (or only Y → X)
2. **idirectional**: oth X → Y and Y → X (feedback loop)
3. **No causality**: Neither direction significant
4. **Instantaneous**: orrelation, but no predictive relationship

---

## Impulse Response unctions (IR)

### What are IRs?

**Impulse Response unctions** show how a one-time shock to one variable affects all variables over time.

**Use cases**:
- Policy analysis: ffect of interest rate increase on GP and inflation
- Shock propagation: How oil price spike affects economy
- ynamic multipliers: umulative impact over time

### omputing IRs

```python
irf_result = model.impulse_response(periods=)

print(irf_result.keys())  # ['gdp', 'unemployment']

# IR for shock to GP
irf_gdp = irf_result['gdp']
print(irf_gdp.shape)  # (, 2) →  periods, 2 variables
```

### Interpreting IRs

```python
import matplotlib.pyplot as plt

# Shock to GP
irf_gdp_shock = irf_result['gdp']

fig, axes = plt.subplots(, 2, figsize=(2, 4))

# ffect on GP itself
axes[].plot(irf_gdp_shock[:, ])
axes[].set_title('Shock to GP → ffect on GP')
axes[].set_xlabel('Periods head')
axes[].set_ylabel('Response')
axes[].axhline(, color='k', linestyle='--', alpha=.3)

# ffect on Unemployment
axes[].plot(irf_gdp_shock[:, ])
axes[].set_title('Shock to GP → ffect on Unemployment')
axes[].set_xlabel('Periods head')
axes[].axhline(, color='k', linestyle='--', alpha=.3)

plt.tight_layout()
plt.show()
```

### xample Interpretation

```
Shock to GP:
  Period : GP +., Unemployment -.
  Period 2: GP +., Unemployment -.2
  Period 3: GP +., Unemployment -.
  Period 4: GP +.4, Unemployment -.4
  ...
```

**Reading**:
- Initial -unit shock to GP
- Unemployment decreases by . in first period
- Maximum unemployment response at period 3 (-.)
- ffects decay over time (system returns to equilibrium)

---

## orecast rror Variance ecomposition (V)

### What is V?

**V** quantifies the percentage of forecast error variance in each variable explained by shocks to each variable.

**nswers**: "What fraction of uncertainty in GP forecasts comes from GP shocks vs. Unemployment shocks?"

### omputing V

```python
fevd_result = model.forecast_error_variance_decomposition(periods=)

# xample: V for GP
fevd_gdp = fevd_result['gdp']
print(fevd_gdp)
```

### Interpreting V

```python
{
    'gdp': array([
        [., .],  # Period : % GP shock, % Unemployment shock
        [., .],  # Period 2: % GP, % Unemployment
        [., .2],  # Period 3: % GP, 2% Unemployment
        [.2, .],  # Period 4: 2% GP, % Unemployment
        ...
    ]),
    'unemployment': array([
        [.3, .],  # Period : 3% GP shock, % Unemployment shock
        [.4, .],  # Period 2: 4% GP, % Unemployment
        [.2, .4],  # Period 3: 2% GP, 4% Unemployment
        ...
    ])
}
```

**Interpretation**:

**or GP**:
- Short-term (Period ): GP forecast errors almost entirely due to GP shocks (%)
- Long-term (Period ): Unemployment shocks explain % of GP forecast uncertainty

**or Unemployment**:
- Short-term: Mix of shocks (3% GP, % Unemployment)
- Long-term: GP shocks become more important (2%), suggesting **GP is a leading indicator**

### Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

periods = np.arange(, )
fevd_gdp = fevd_result['gdp']

plt.figure(figsize=(, ))

plt.subplot(, 2, )
plt.stackplot(periods, 
              fevd_gdp[:, ] * ,  # GP contribution
              fevd_gdp[:, ] * ,  # Unemployment contribution
              labels=['GP shocks', 'Unemployment shocks'])
plt.title('V of GP')
plt.xlabel('Periods head')
plt.ylabel('% of orecast rror Variance')
plt.legend(loc='upper right')

# Repeat for Unemployment...
plt.tight_layout()
plt.show()
```

---

## Model iagnostics

### . heck Residuals

```python
# Get fitted values and residuals
fitted_values = model._fitted_model.fittedvalues
residuals = model._fitted_model.resid

# heck for autocorrelation
import statsmodels.stats.diagnostic as sm_diag

for i, var in enumerate(['gdp', 'unemployment']):
    lb_stat, lb_pval = sm_diag.acorr_ljungbox(residuals.iloc[:, i], lags=, return_df=alse)
    print(f"{var} residuals: Ljung-ox p-value = {lb_pval[]:.4f}")
    # p-value > . → No autocorrelation (good!)
```

### 2. Stability heck

VR is stable if eigenvalues of companion matrix are inside unit circle:

```python
# heck stability
eigenvalues = model._fitted_model.is_stable(verbose=True)
# Should print: True (all eigenvalues <  in modulus)
```

### 3. Residual Normality

```python
from scipy.stats import jarque_bera

for i, var in enumerate(['gdp', 'unemployment']):
    jb_stat, jb_pval = jarque_bera(residuals.iloc[:, i])
    print(f"{var}: Jarque-era p-value = {jb_pval:.4f}")
    # p-value > . → Normality assumption holds
```

---

## Real-World xamples

### xample : Macroeconomic orecasting

```python
import pandas_datareader as pdr

# etch quarterly GP and unemployment
gdp = pdr.data.ataReader('GP', 'fred', '2--', '223-2-3')
unemp = pdr.data.ataReader('UNRT', 'fred', '2--', '223-2-3')

# Resample unemployment to quarterly
unemp_q = unemp.resample('QS').mean()

# Merge
data = pd.concat([gdp, unemp_q], axis=, join='inner')
data.columns = ['GP', 'Unemployment']

# it VR
params = {'max_lags': 4, 'ic': 'bic'}
meta = ModelMeta(name="US Macro VR")

model = VRModel(data, params, meta)
result = model.fit()

# Granger causality: oes GP predict Unemployment?
gc = model.granger_causality_test('Unemployment', 'GP')
print(f"GP → Unemployment: p-value = {gc['results_by_lag']['lag_']['ssr_ftest_pvalue']:.4f}")

# orecast next 4 quarters
forecast = model.predict(steps=4)
```

### xample 2: inancial Markets

```python
# Stock prices of related companies
import yfinance as yf

# ownload data
aapl = yf.Ticker("PL").history(start="22--", end="223-2-3")['lose']
msft = yf.Ticker("MST").history(start="22--", end="223-2-3")['lose']

data = pd.atarame({'PL': aapl, 'MST': msft}).dropna()

# Log returns (stationary)
returns = np.log(data / data.shift()).dropna()

# it VR on returns
params = {'max_lags': , 'ic': 'aic'}
meta = ModelMeta(name="Tech Stock VR")

model = VRModel(returns, params, meta)
result = model.fit()

# Impulse response: shock to PL affects MST?
irf = model.impulse_response(periods=)
print("PL shock effect on MST:")
print(irf['PL'][:, ])  # olumn  is MST
```

---

## est Practices

### . Stationarity is ritical

 **on't**: it VR on non-stationary data (levels with trends)
 **o**: ifference, detrend, or log-transform first

```python
# heck stationarity with  test
from statsmodels.tsa.stattools import adfuller

for col in data.columns:
    result = adfuller(data[col])
    print(f"{col}:  p-value = {result[]:.4f}")
    # p-value < . → Stationary

# If non-stationary, difference
data_diff = data.diff().dropna()
```

### 2. Start Simple

- egin with 2-3 variables
- Use I for lag selection (prefers parsimony)
- Validate with diagnostics before adding complexity

### 3. Interpret with aution

- Granger causality ≠ true causality
- IRs show average effects (nonlinear relationships not captured)
- V sensitive to variable ordering (use holesky decomposition awareness)

### 4. ross-Validate

```python
# Rolling forecast validation
from sklearn.metrics import mean_absolute_percentage_error

train_size = len(data) - 2
predictions = []
actuals = []

for i in range(train_size, len(data) - 4):
    train = data.iloc[:i]
    test = data.iloc[i:i+4]
    
    model = VRModel(train, params, meta)
    model.fit()
    forecast = model.predict(steps=4)
    
    predictions.append(forecast.forecast_values)
    actuals.append(test.values.flatten())

# alculate MP
mape = mean_absolute_percentage_error(
    np.concatenate(actuals),
    np.concatenate(predictions)
)
print(f"Rolling MP: {mape:.2%}")
```

---

## Troubleshooting

### Problem: "Valuerror: ata is not stationary"

**Solution**: ifference or detrend data

```python
# irst difference
data_diff = data.diff().dropna()

# Or log difference (returns)
data_logreturns = np.log(data / data.shift()).dropna()
```

### Problem: "Too many parameters to estimate"

**ause**: Too many variables or too many lags

**Solution**:
. Reduce `max_lags`
2. Remove less important variables
3. Use I instead of I (prefers fewer lags)

### Problem: "Model is not stable"

**ause**: xplosive dynamics (eigenvalues outside unit circle)

**Solution**:
. heck for non-stationarity → difference data
2. Reduce lag order
3. dd constraints or use ayesian VR (future)

### Problem: "Granger causality test results are unclear"

**Interpretation**:
- heck multiple lags: significance at one lag may be spurious
- ompare p-values across lags for consistency
- onsider economic theory: does the result make sense?

---

## Summary ecision Tree

```
 Multiple time series?
   Yes → ontinue
   No  → Use SRIM or Prophet

 Variables interrelated?
   Yes → ontinue
   No  → Use separate models

 ll stationary?
   Yes → Use VR 
   No  →  ointegrated? → Use VM
             Not cointegrated? → ifference then use VR

 orecast needed?
    Yes → Use predict()
    No  → Use for Granger causality / IR analysis
```

---

## urther Reading

- **Lütkepohl (2)**: "New Introduction to Multiple Time Series nalysis" (comprehensive textbook)
- **Hamilton (4)**: "Time Series nalysis" (hapter  on VR)
- **Sims ()**: "Macroeconomics and Reality" (original VR paper)
- **Granger ()**: "Investigating ausal Relations" (Granger causality)

---

**Guide Version**: .  
**Model**: krl_models.econometric.VRModel  
**Last Updated**: October 22
