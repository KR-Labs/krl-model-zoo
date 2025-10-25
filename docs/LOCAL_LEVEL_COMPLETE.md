# Task 4 omplete: Local Level Model 

**ate:** October 24, 224  
**Status:** OMPLT  
**Phase:** 2.2 - KRL Model Zoo oundation (Task 4 of 2)

---

## Summary

Successfully implemented a Local Level Model (also known as "random walk plus noise"), the simplest structural time series model. The implementation provides automatic parameter estimation via Maximum Likelihood, trend extraction, signal-noise decomposition, and forecasting capabilities.

**eliverable:** `krl_models/state_space/local_level.py` (44 lines)

---

## Mathematical oundation

### Model Specification

The Local Level Model decomposes a time series into two components:

**Level quation (Hidden State):**
```
μ_t = μ_{t-} + η_t,  where η_t ~ N(, σ²_η)
```

**Observation quation (Measurement):**
```
y_t = μ_t + ε_t,  where ε_t ~ N(, σ²_ε)
```

**Parameters:**
- `μ_t`: Unobserved level (stochastic trend) at time t
- `y_t`: Observed value at time t
- `σ²_η`: Level noise variance (controls trend smoothness)
- `σ²_ε`: Observation noise variance (measurement error)
- `η_t`, `ε_t`: Independent Gaussian white noise

### Signal-to-Noise Ratio

The signal-to-noise ratio (SNR) determines the smoothness of the extracted trend:

```
q = σ²_η / σ²_ε
```

**Interpretation:**
- **q → :** Smooth trend (level changes slowly, observations are noisy)
- **q → ∞:** Noisy trend (level changes rapidly, follows observations closely)
- **q = :** qual level and observation variability

### Special ases

. **σ²_η = :** onstant level plus noise (no trend variation)
2. **σ²_ε = :** Pure random walk (observations = level exactly)
3. **oth small:** Smooth, predictable series
4. **oth large:** High overall variability

---

## Implementation eatures

### ore apabilities

. **utomatic Parameter stimation (ML)**
   - stimates σ²_η and σ²_ε via Maximum Likelihood
   - Uses numerical optimization (L-GS-)
   - Kalman ilter computes log-likelihood efficiently

2. **ixed Parameter Mode**
   - ccepts pre-specified variance parameters
   - Useful when parameters are known a priori
   - aster (skips optimization)

3. **Level xtraction**
   - iltered estimates: Using data up to time t
   - Smoothed estimates: Using all available data (optimal)
   - Smoothing typically improves accuracy by 2-3%

4. **ecomposition**
   - Separates observations into level + noise
   - xtracts stochastic trend
   - Identifies observation error component

. **orecasting**
   - Multi-step ahead predictions
   - % confidence intervals
   - or local level: forecasts are constant (last level)
   - Uncertainty grows with horizon

. **iagnostics**
   - Innovation statistics (forecast errors)
   - Normality tests (Jarque-era)
   - Model fit criteria (I, I)
   - Skewness and kurtosis of residuals

### State Space Representation

The Local Level Model can be written in standard state space form (used internally by Kalman ilter):

```
State transition:  x_t =  * x_{t-} + w_t,  where  = []
Observation:       y_t = H * x_t + v_t,      where H = []
Process noise:     Q = [σ²_η]
Observation noise: R = [σ²_ε]
```

This allows leveraging the Kalman ilter for optimal state estimation.

---

## Testing ramework

**ile:** `tests/state_space/test_local_level_smoke.py` (4 lines)

### Test overage ( comprehensive tests):

#### . **ML Parameter stimation Test**
- **Synthetic ata:** Known σ_η = ., σ_ε = .
- **Validates:** Parameter recovery via Maximum Likelihood
- **Results:**
  - stimated σ_η: .44 (true: .) → % accuracy
  - stimated σ_ε: .2 (true: .) → % accuracy
  - Smoothed RMS: .43 (2.% improvement over filtered)
- **Outcome:**  ML successfully recovers parameters

#### 2. **ixed Parameters Test**
- **Validates:** Model with pre-specified variances
- **hecks:** ecomposition (observations = level + noise)
- **Results:**
  - Noise mean: -. (near zero )
  - ecomposition successful
  - 2-step forecast: constant (as expected for local level)
- **Outcome:**  ixed parameter mode works correctly

#### 3. **Signal-to-Noise Ratio omparison**
- **ase :** Smooth trend (q = .)
  - Very low level variability
  - Level changes slowly
  - Level RMS: .23
- **ase 2:** Noisy trend (q = .)
  - qual level and observation noise
  - Level changes rapidly
  - Level RMS: .444
- **Insight:** x difference in SNR ratio produces dramatically different trends
- **Outcome:**  SNR correctly controls trend smoothness

#### 4. **iagnostics Test**
- **Validates:** iagnostic statistics computation
- **hecks:**
  - Innovation mean near zero
  - Parameter estimates
  - Signal-to-noise ratio
  - Noise component extraction
- **Results:**
  - Noise mean: .2 (effectively zero )
  - ll diagnostics computed correctly
- **Outcome:**  iagnostics working properly

#### . **dge ases Test**
- **ase :** Nearly constant level (σ_η → )
  - stimated σ_η: . (correctly identifies no trend variation)
  - SNR: . (smooth as expected)
- **ase 2:** Short time series (T = 2)
  - Model fits successfully
  - -step forecast works
  - Handles limited data gracefully
- **Outcome:**  dge cases handled robustly

### Test Results Summary

```
Testing Local Level Model: ML Parameter stimation
 stimated σ_η: .44 (true: .) - % accuracy
 stimated σ_ε: .2 (true: .) - % accuracy
 Smoothed RMS improvement: 2.%
 -step forecast successful

Testing Local Level Model: ixed Parameters
 ecomposition: noise mean = -. (near zero)
 ixed parameter mode works
 2-step forecast successful

Testing Local Level Model: Signal-to-Noise omparison
 Smooth trend (q=.): RMS = .23
 Noisy trend (q=.): RMS = .444
 SNR controls smoothness correctly

Testing Local Level Model: iagnostics
 Innovation mean: .2 (near zero)
 ll diagnostics computed
 Noise extraction working

Testing Local Level Model: dge ases
 Nearly constant level detected (σ_η = )
 Short time series (T=2) handled
 dge cases robust

 ll  Local Level Model tests passed!
```

---

## Use ases

### . **Trend xtraction**
- xtract smooth trend from noisy data
- conomic indicators (GP, inflation smoothing)
- Remove measurement noise from sensor data

### 2. **Short-Term orecasting**
- orecasts are constant (last level estimate)
- Useful when recent level is best predictor
- Growing uncertainty over time

### 3. **Signal Processing**
- Separate signal (level) from noise
- Quality control (process monitoring)
- nomaly detection (deviations from level)

### 4. **Missing ata Imputation**
- Smoothed level can fill gaps
- Provides optimal estimates given available data
- Uncertainty quantification via covariances

### . **Model uilding lock**
- oundation for more complex structural models
- Local Linear Trend (adds slope component)
- Seasonal models (adds seasonal component)

---

## Key Implementation etails

### Parameter stimation (ML)

The model uses numerical optimization to maximize the log-likelihood:

```python
def neg_log_likelihood(sigma_eta, sigma_epsilon):
    # Set up Kalman ilter with these parameters
    kf = Kalmanilter(Q=sigma_eta^2, R=sigma_epsilon^2, ...)
    
    # it and compute log-likelihood
    result = kf.fit(data)
    log_lik = result.payload['log_likelihood']
    
    return -log_lik  # Minimize negative
```

**Optimization Method:** L-GS- with bounds (σ > )

**Initial Guess:** qual split of variance
```
σ_η_init = sqrt(Var(y) / 2)
σ_ε_init = sqrt(Var(y) / 2)
```

**onvergence:** Typically -3 iterations

### Level stimation ccuracy

rom test results:

| Metric | iltered | Smoothed | Improvement |
|--------|----------|----------|-------------|
| **RMS** | .2 | .43 | 2.% |
| **Uses ata** | Up to t | ll data | ackward pass |
| **Speed** | ast | Slower | 2x compute |

**Recommendation:** Use smoothed estimates for analysis, filtered for real-time applications.

### orecasting Properties

or the Local Level Model:

. **Point orecast:** onstant at last level estimate
   ```
   ŷ_{T+h|T} = μ̂_T  for all h > 
   ```

2. **orecast Variance:** Grows linearly with horizon
   ```
   Var(y_{T+h|T}) = h * σ²_η + σ²_ε
   ```

3. **onfidence Intervals:** Widen over time
   ```
   I_ = ŷ_{T+h} ± . * sqrt(Var(y_{T+h|T}))
   ```

### Signal-to-Noise Ratio Impact

Test results demonstrate SNR effects:

| SNR (q) | σ_η | σ_ε | Level RMS | Interpretation |
|---------|-----|-----|------------|----------------|
| **.** | . | . | .23 | Very smooth, slow changes |
| **.2** | . | . | .43 | Moderate smoothing |
| **.** | . | . | .444 | Noisy, tracks data closely |

**Rule of Thumb:**
- q < .: Smooth trend extraction
- . < q < : alanced trend-noise
- q > : Trend follows data closely

---

## ode Statistics

| Metric | Value |
|--------|-------|
| **Lines of ode** | 44 |
| **lasses** |  (LocalLevelModel) |
| **Methods** |  |
| **Test Lines** | 4 |
| **Test unctions** |  |
| **Test Success Rate** | % (/) |

**Target:** 2-3 lines  
**elivered:** 44 lines (2% of target upper bound)  
**Reason:** dded ML estimation, comprehensive diagnostics, decomposition capabilities

---

## PI Overview

### Initialization

```python
# utomatic parameter estimation
model = LocalLevelModel(estimate_params=True)

# ixed parameters
model = LocalLevelModel(
    sigma_eta=.,
    sigma_epsilon=.,
    estimate_params=alse
)
```

### itting

```python
result = model.fit(data)  # Returns orecastResult

# ccess results
log_lik = result.payload['log_likelihood']
aic = result.metadata['aic']
snr = result.payload['signal_to_noise_ratio']
```

### Level xtraction

```python
# Smoothed level (optimal, uses all data)
level = model.get_level(smoothed=True)

# iltered level (real-time, uses data up to t)
level_filtered = model.get_level(smoothed=alse)
```

### ecomposition

```python
decomp = model.decompose()
# Returns dict with:
#   'observations': original data
#   'level': smoothed trend
#   'noise': observation error
#   'level_filtered': filtered trend
```

### orecasting

```python
forecast = model.predict(steps=)
# Returns orecastResult with:
#   forecast_values: point forecasts
#   ci_lower: lower confidence bounds
#   ci_upper: upper confidence bounds
```

### iagnostics

```python
sigma_eta, sigma_epsilon = model.get_variances()
snr = model.get_signal_to_noise_ratio()
diagnostics = model.get_diagnostics()
noise = model.get_noise()
```

---

## iles reated/Modified

### reated:
. **`krl_models/state_space/local_level.py`** (44 lines)
   - omplete Local Level Model implementation
   - ML parameter estimation
   - Kalman ilter integration
   - ecomposition and diagnostics

2. **`tests/state_space/test_local_level_smoke.py`** (4 lines)
   -  comprehensive smoke tests
   - ML validation, fixed parameters, SNR comparison
   - iagnostics and edge cases
   - Synthetic data generation

3. **`docs/LOL_LVL_OMPLT.md`** (This document)
   - omplete implementation summary

### Modified:
. **`krl_models/state_space/__init__.py`**
   - dded LocalLevelModel export

---

## omparison with Related Models

| Model | Trend | Slope | Seasonal | omplexity |
|-------|-------|-------|----------|------------|
| **Local Level** |  |  |  | Low |
| Local Linear Trend |  |  |  | Medium |
| asic Structural |  |  |  | High |
| RIM(,,) |  |  |  | Low |

**Local Level vs RIM(,,):**
- Local Level: explicit state space form, interpretable components
- RIM(,,): equivalent in some cases, different parameterization
- Local Level: easier to extend (add slope, seasonality)

---

## Mathematical Guarantees

### Optimality

. **Kalman ilter:** Provides LU (est Linear Unbiased stimator) for level
2. **Smoothed stimates:** Minimum variance given all observations
3. **ML:** symptotically efficient parameter estimates
4. **orecasts:** Optimal given model assumptions

### ssumptions

- **Linearity:** Level evolves linearly (random walk)
- **Gaussianity:** Noise terms are Gaussian
- **Independence:** η_t and ε_t are independent
- **Stationarity of innovations:** σ²_η and σ²_ε constant over time

### When ssumptions ail

- **Non-Gaussian errors:** ML still consistent but not efficient
- **Structural breaks:** onsider regime-switching models
- **Time-varying variance:** Use GRH-type extensions
- **Nonlinearity:** xtended Kalman ilter or particle filter

---

## Next Steps

### Immediate (Task ):
- omprehensive unit tests (3+ tests)
- over all models: GRH, GRH, GJR-GRH, Kalman ilter, Local Level
- dge cases, parameter validation, error handling

### Short-term (Tasks -):
- Integration tests (+ tests)
- rror handling tests (+ tests)
- nd-to-end workflows

### Medium-term (Tasks -2):
- 4 detailed examples
- Performance benchmarking
- omplete documentation suite

---

## References

**ooks:**
- Harvey, . . (). *orecasting, Structural Time Series Models and the Kalman ilter*. ambridge University Press.
- urbin, J., & Koopman, S. J. (22). *Time Series nalysis by State Space Methods*. Oxford University Press.

**Papers:**
- Harvey, . ., & Todd, P. H. J. (3). "orecasting conomic Time Series with Structural and ox-Jenkins Models." *Journal of usiness & conomic Statistics*, (4), 2-3.
- Snyder, R. . (). "Recursive stimation of ynamic Linear Models." *Journal of the Royal Statistical Society *, 4(2), 22-2.

**pplications:**
- Trend extraction in economic time series
- Structural time series modeling
- Signal processing and filtering

---

## Validation hecklist

- [x] Local Level Model implemented with complete functionality
- [x] ML parameter estimation (L-GS- optimization)
- [x] ixed parameter mode (user-specified variances)
- [x] Level extraction (filtered and smoothed)
- [x] ecomposition (observations = level + noise)
- [x] Multi-step forecasting with confidence intervals
- [x] Signal-to-noise ratio computation
- [x] iagnostic statistics (innovations, normality tests)
- [x]  comprehensive smoke tests created
- [x] ll tests passing (% success rate)
- [x] ML parameter recovery validated (-% accuracy)
- [x] ixed parameter mode tested
- [x] SNR impact demonstrated (smooth vs noisy trends)
- [x] iagnostics working correctly
- [x] dge cases handled robustly
- [x] Kalman ilter integration working
- [x] Package initialization updated
- [x] ompletion summary documented

---

**Phase 2.2 Progress:** 4/2 tasks complete (33.3%)  
**Task 4 Status:**  OMPLT  
**Total Implementation:**  lines (44 model + 4 tests)  
**Test Success Rate:** % (/ tests passing)  
**Parameter Recovery:** -% accuracy (ML estimation)  
**Smoothing Improvement:** 2.% RMS reduction
