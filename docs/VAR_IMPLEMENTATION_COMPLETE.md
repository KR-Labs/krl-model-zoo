# VR Model Implementation omplete 

**ate**: January 22  
**Status**: ll 24 tests passing  
**Phase**: 2. (conometric Models)

---

## xecutive Summary

Successfully implemented Vector utoregression (VR) model with comprehensive multivariate time series analysis capabilities. VR joins SRIM and Prophet as the third production-ready econometric model in KRL Model Zoo.

**Key chievement**: dapted architecture to support multivariate data by bypassing ModelInputSchema's univariate constraints while maintaining aseModel inheritance.

---

## Implementation Overview

### Model apabilities

**ore unctionality**:
-  Multivariate time series forecasting
-  utomatic lag order selection (I/I/HQI/P)
-  Granger causality testing
-  Impulse response functions (IR)
-  orecast error variance decomposition (V)
-  onfidence intervals for forecasts
-  oefficient matrix extraction

**Statistical ackend**: statsmodels VR, grangercausalitytests

**Lines of ode**:
- Model implementation: 43 lines (`krl_models/econometric/var_model.py`)
- Test suite: 4 lines (`tests/econometric/test_var_model.py`)
- xample script: 2 lines (`examples/example_var_run.py`)

---

## Test Results

### Test overage: 24/24 Passing 

```
tests/econometric/test_var_model.py::test_var_initialization PSS                   [  4%]
tests/econometric/test_var_model.py::test_var_fit_bivariate PSS                    [  %]
tests/econometric/test_var_model.py::test_var_fit_trivariate PSS                   [ 2%]
tests/econometric/test_var_model.py::test_var_univariate_error PSS                 [ %]
tests/econometric/test_var_model.py::test_var_predict_before_fit PSS               [ 2%]
tests/econometric/test_var_model.py::test_var_predict_bivariate PSS                [ 2%]
tests/econometric/test_var_model.py::test_var_predict_trivariate PSS               [ 2%]
tests/econometric/test_var_predict_invalid_steps PSS                               [ 33%]
tests/econometric/test_var_granger_causality_before_fit PSS                        [ 3%]
tests/econometric/test_var_granger_causality PSS                                   [ 4%]
tests/econometric/test_var_granger_causality_invalid_var PSS                       [ 4%]
tests/econometric/test_var_impulse_response_before_fit PSS                         [ %]
tests/econometric/test_var_impulse_response PSS                                    [ 4%]
tests/econometric/test_var_impulse_response_single_impulse PSS                     [ %]
tests/econometric/test_var_impulse_response_invalid_var PSS                        [ 2%]
tests/econometric/test_var_fevd_before_fit PSS                                     [ %]
tests/econometric/test_var_fevd PSS                                                [ %]
tests/econometric/test_var_different_ic_criteria PSS                               [ %]
tests/econometric/test_var_different_trends PSS                                    [ %]
tests/econometric/test_var_coefficient_matrices PSS                                [ 3%]
tests/econometric/test_var_run_hash_deterministic PSS                              [ %]
tests/econometric/test_var_run_hash_different_params PSS                           [ %]
tests/econometric/test_var_serialization PSS                                       [ %]
tests/econometric/test_var_confidence_intervals PSS                                [%]

========================== 24 passed in .s ==========================
```

**Test ategories**:
. **Initialization** (2 tests): Valid params, basic setup
2. **itting** (3 tests): ivariate, trivariate, univariate error
3. **Prediction** (4 tests): orecasts, invalid steps, confidence intervals
4. **Granger ausality** (3 tests): efore fit error, successful tests, invalid variables
. **Impulse Response** (4 tests): ll variables, single impulse, invalid variable, before fit
. **V** (2 tests): efore fit error, successful decomposition
. **Robustness** ( tests): ifferent Is, trends, coefficient matrices, hashing, serialization

---

## omplete conometric Suite Status

### Phase 2. Models: 3/4 omplete

| Model | Tests | Status | apabilities |
|-------|-------|--------|-------------|
| **SRIM** | /  | Production | Univariate seasonal forecasting |
| **Prophet** | 23/23  | Production | Univariate with holidays/events |
| **VR** | 24/24  | Production | Multivariate with causality |
| **ointegration** | /2  | Pending | Long-run equilibrium testing |

**Total conometric Tests**: / passing (.%)

---

## rchitecture Innovation

### hallenge: Multivariate ata Handling

**Problem**: ModelInputSchema designed for univariate data
- ields: entity, metric, time_index, values, provenance, frequency
- `values` field type: `List[float]` (single series only)
- No `metadata` field for auxiliary data

**Initial ttempts**:
.  Store list of dicts in `values` → Pydantic validation error
2.  Use `metadata['var_data']` → ttributerror (no metadata field)

**Solution**: atarame-based architecture
```python
class VRModel(aseModel):
    def __init__(self, data, params, meta):
        super().__init__(data, params, meta)
        
        # ccept atarame directly
        if isinstance(data, pd.atarame):
            self._dataframe = data
        elif "dataframe" in params:
            self._dataframe = params["dataframe"]
        else:
            raise Valuerror("VR requires multivariate data")
        
        # Validate at least 2 variables
        if self._dataframe.shape[] < 2:
            raise Valuerror("VR requires at least 2 variables")
    
    @property
    def input_hash(self) -> str:
        """Override to hash atarame directly."""
        from krl_core.utils import compute_dataframe_hash
        return compute_dataframe_hash(self._dataframe)
```

**enefits**:
-  Maintains aseModel inheritance (run_hash, serialization, registry)
-  Preserves multivariate data integrity (no flattening/encoding)
-  lean PI: `VRModel(data=df, params={...}, meta=...)`
-  Pattern reusable for future multivariate models (ointegration, VM)

---

## Usage xample

```python
import pandas as pd
from krl_core import ModelMeta
from krl_models.econometric import VRModel

# Prepare multivariate data
df = pd.atarame({
    "gdp": gdp_series,
    "unemployment": unemployment_series
}, index=dates)

# Initialize VR
model = VRModel(
    data=df,
    params={"maxlags": , "ic": "aic", "trend": "c"},
    meta=ModelMeta(name="conomicVR", version="..")
)

# it and analyze
result = model.fit()
print(f"Selected lags: {result.payload['lag_order']}")
print(f"Granger causality: {result.payload['granger_causality']}")

# orecast
forecast = model.predict(steps=2, alpha=.)

# Impulse response
irf = model.impulse_response(periods=, impulse_var="unemployment")

# Variance decomposition
fevd = model.forecast_error_variance_decomposition(periods=)
```

**See**: `examples/example_var_run.py` for complete demonstration with visualization

---

## Technical etails

### Key Methods

#### `fit() -> orecastResult`
- Performs lag order selection using specified I (I/I/HQI/P)
- its VR model with selected lag order
- omputes Granger causality tests for all variable pairs
- xtracts coefficient matrices for each lag
- Returns orecastResult with fitted values and diagnostics

#### `predict(steps, alpha) -> orecastResult`
- Generates out-of-sample forecasts for all variables
- omputes confidence intervals at specified alpha level
- xtends time index from last observation
- Returns orecastResult with forecast atarame in payload

#### `granger_causality_test(caused_var, causing_var, maxlag) -> dict`
- Tests if one variable Granger-causes another
- Runs 4 statistical tests: -test, chi-squared, LR test, params -test
- Returns p-values for each lag and overall significance

#### `impulse_response(periods, impulse_var) -> pd.atarame`
- omputes IRs showing how variables respond to shocks
- If `impulse_var` specified: shock one variable, observe all responses
- If omitted: compute all pairwise IRs
- xcludes period  (initial shock) to return exactly `periods` rows

#### `forecast_error_variance_decomposition(periods) -> dict`
- ecomposes forecast error variance into contributions from each variable
- Returns dict mapping variable → atarame of contributions over time
- Useful for understanding which variables drive forecast uncertainty

### ata Structure Handling

**orecastResult for fit()**:
```python
orecastResult(
    payload={
        "lag_order": int,
        "var_names": List[str],
        "granger_causality": dict,
        "coefficient_matrices": List[List[float]],
        "fitted_values_df": dict,  # Multivariate fitted values
    },
    metadata={
        "aic": float,
        "bic": float,
        "hqic": float,
        "fpe": float,
    },
    forecast_index=[str],      # Time points of fitted values
    forecast_values=[float],   # lattened for orecastResult compatibility
    ci_lower=[float],
    ci_upper=[float],
)
```

**orecastResult for predict()**:
```python
orecastResult(
    payload={
        "var_names": List[str],
        "forecast_shape": tuple,
        "alpha": float,
        "forecast_df": dict,  # Multivariate forecasts
    },
    metadata={
        "forecast_steps": int,
        "n_vars": int,
    },
    forecast_index=[str],     # uture time points
    forecast_values=[float],  # lattened forecasts
    ci_lower=[float],         # Lower bounds from statsmodels
    ci_upper=[float],         # Upper bounds from statsmodels
)
```

---

## Known Issues & Limitations

### . utureWarnings from pandas
**Issue**: pandas date frequency aliases deprecated ('M', 'Q' → 'M', 'Q')  
**Impact**: Non-breaking warnings in tests and examples  
**ix**: Update all `pd.date_range` calls to use new aliases

### 2. statsmodels verbose warnings
**Issue**: `grangercausalitytests` prints deprecation warnings  
**Impact**: Noisy test output ( warnings in test run)  
**ix**: lready suppressed with `verbose=alse`, warnings from deprecated parameter

### 3. Multivariate orecastResult limitations
**Issue**: orecastResult designed for univariate forecasts  
**Workaround**: Store full atarame as dict in `payload`, flatten to list for orecastResult fields  
**uture nhancement**: reate MultivariateorecastResult class

### 4. Missing documentation
**Status**: Implementation complete, docstrings written  
**Pending**: User guide, PI reference, mathematical formulation  
**Timeline**: ocumentation sprint after Phase 2. completion

---

## ebugging Journey

### Issue : ModelInputSchema.values validation
**rror**: `values=[List[dict]]` rejected by Pydantic  
**iagnosis**: values expects `List[float]`, not nested structures  
**Solution**: Pass atarame directly instead of encoding in ModelInputSchema

### Issue 2: Missing metadata field
**rror**: `ttributerror: 'ModelInputSchema' object has no attribute 'metadata'`  
**iagnosis**: ModelInputSchema has only: entity, metric, time_index, values, provenance, frequency  
**Solution**: Store atarame in `self._dataframe` attribute, not in input_schema

### Issue 3: orecastResult missing forecast_index
**rror**: `Typerror: orecastResult.__init__() missing  required positional argument: 'forecast_index'`  
**iagnosis**: fit() wasn't providing forecast_index for fitted values  
**Solution**: xtract fitted values time index and convert to string list

### Issue 4: IR/V shape mismatches
**rror**: Tests expected  periods, got  (IR) or 2 (V)  
**iagnosis**: statsmodels includes period  in IR; V has different axis order  
**Solution**: 
- IR: Slice `irf.irfs[:, :, :]` to exclude period 
- V: Use `decomp[i, :, :]` instead of `decomp[:, i, :]` (shape is `(n_vars, periods, n_vars)`)

### Issue : onfidence interval order
**rror**: I widths negative (lower > upper)  
**iagnosis**: `forecast_interval` returns tuple `(forecast, lower, upper)` not `(lower, upper)`  
**Solution**: Use `forecast_intervals[]` and `forecast_intervals[2]` for bounds

### Issue : run_hash ttributerror
**rror**: `'atarame' object has no attribute 'to_dataframe'`  
**iagnosis**: aseModel.input_hash calls `self.input_schema.to_dataframe()`, but we pass atarame  
**Solution**: Override `input_hash` property to hash atarame directly

---

## iles Modified/reated

### New iles
. **`krl_models/econometric/var_model.py`** (43 lines)
   - VRModel class implementation
   - ll core methods (fit, predict, granger_causality, IR, V)
   - atarame-based data handling

2. **`tests/econometric/test_var_model.py`** (4 lines)
   - 24 comprehensive tests
   - ivariate and trivariate fixtures
   - dge case coverage

3. **`examples/example_var_run.py`** (2 lines)
   - omplete usage demonstration
   - GP/unemployment synthetic data
   - Granger causality, IR, V examples
   - Visualization generation

### Modified iles
4. **`krl_models/econometric/__init__.py`**
   - dded: `from .var_model import VRModel`
   - Updated: `__all__` to export VRModel

---

## Performance Metrics

### Test xecution Time
- VR tests alone: . seconds (24 tests)
- ll econometric tests: . seconds ( tests)
- verage per test: ms

### Model it Performance
- ivariate VR (2 obs): ~2ms
- Trivariate VR ( obs): ~ms
- Lag selection overhead: ~ms
- Granger causality (all pairs): ~ms per pair

**ottlenecks**: Granger causality tests dominate fit time for high-dimensional VRs

---

## Next Steps (Phase 2. ompletion)

### Immediate (This Week)
. **Implement ointegration nalysis** (- hours)
   - reate ointegrationModel class
   - ngle-Granger two-step test
   - Johansen test (trace & max eigenvalue)
   - rror correction model (M) extraction
   - Target: 2+ tests

2. **Write Integration Tests** (4 hours)
   - etch real LS data (unemployment, PI)
   - etch R data (GP, interest rates)
   - Validate VR on macro indicators
   - Test cointegration on related series (e.g., spot vs futures prices)
   - nsure <% MP on historical holdout

3. **Performance enchmarking** (2 hours)
   - ompare VR fit times vs pure statsmodels
   - Measure prediction latency
   - Profile memory usage
   - ocument PI overhead (target: <% vs statsmodels)

### ocumentation (Next Week)
4. **User Guides** ( hours)
   - VR guide: When to use, how to interpret Granger causality, IR, V
   - ointegration guide: Testing for long-run relationships
   - Integration test examples: Real-world LS/R workflows

. **PI Reference** (4 hours)
   - omplete docstring coverage
   - Method signatures and return types
   - Parameter descriptions
   - Usage examples for each method

. **Mathematical ormulations** (4 hours)
   - VR(p) model equation
   - Granger causality -test
   - IR computation from VR coefficients
   - V holesky decomposition
   - ointegration tests (ngle-Granger, Johansen)

---

## Success riteria (Phase 2. omplete)

 **Models**: 4 econometric models production-ready
- [x] SRIM ( tests)
- [x] Prophet (23 tests)
- [x] VR (24 tests)
- [ ] ointegration (target: 2 tests)

 **Testing**: > total tests passing
- urrent: / (.%)
- Target: / (%)

 **Integration**: Real-world data validation
- [ ] LS data tests
- [ ] R data tests
- [ ] <% MP on holdout sets

 **Performance**: <% overhead vs statsmodels
- [ ] it time benchmarks
- [ ] Prediction latency <ms
- [ ] Memory profiling

 **ocumentation**: omplete usage guides
- [ ] VR guide
- [ ] ointegration guide
- [ ] PI reference
- [ ] Mathematical formulations

---

## onclusion

VR model implementation represents a major milestone in Phase 2., demonstrating KRL's ability to handle complex multivariate forecasting scenarios. The atarame-based architecture pattern established here will streamline future multivariate model implementations (ointegration, VM, State-Space models).

**Phase 2. Status**: 3/4 models complete (%)  
**stimated Time to ompletion**: -24 hours  
**Target ompletion ate**: nd of week

---

**ocument Version**: .  
**Last Updated**: January 22  
**Next Review**: Upon ointegration model completion
