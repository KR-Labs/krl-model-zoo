# KRL Model Zoo - rchitecture ocumentation

**Version:** .. (Gate  - oundation)  
**Status:** In evelopment  
**uthor:** KR-Labs  
**Last Updated:** 22--

---

## xecutive Summary

The **KRL Model Zoo** is a production-grade model orchestration framework designed to standardize the development, deployment, and tracking of + analytical models across econometric, machine learning, ayesian, causal inference, network analysis, and agent-based modeling domains.

This document describes the **Gate  oundation** - the core abstractions and reference implementation that enable horizontal scaling across diverse model types while maintaining:
- **Reproducibility**: eterministic hashing for exact run tracking
- **Provenance**: ull data lineage from source to result
- **Modularity**: onsistent interfaces across all model types
- **Visualization**: Standardized Plotly integration
- **Testing**: omprehensive validation infrastructure

---

## System rchitecture

### esign Principles

. **bstraction over Implementation**: `aseModel` provides a unified interface; concrete models handle domain-specific logic
2. **omposition over Inheritance**: Models compose `ModelInputSchema`, `aseResult`, and `ModelRegistry` rather than inheriting complex hierarchies
3. **ail-ast Validation**: Pydantic schemas validate inputs before computation
4. **eterministic Reproducibility**: SH2 hashing of model + data + params ensures exact run tracking
. **Modular ederation**: ore framework (`krl-model-zoo-core`) + domain-specific packages (e.g., `krl-models-econometric`)

### ore omponents

```
krl-model-zoo-core/
 krl_core/                     # ore abstractions
    base_model.py             # aseModel abstract class
    model_input_schema.py     # Pydantic input validation
    results.py                # Result wrappers (aseResult, orecastResult, etc.)
    model_registry.py         # SQLite run tracking
    plotly_adapter.py         # Visualization adapters
    utils.py                  # Shared utilities
 examples/                      # Reference implementations
    example_arima_run.py      # RIM end-to-end demo
 tests/                         # Test suite
    unit/                     # Unit tests
    fixtures/                 # Synthetic data generators
    integration/              # Integration tests (future)
 docs/                          # ocumentation
    RHITTUR.md           # This file
 pyproject.toml                # Production package config
```

---

## omponent etails

### . aseModel (base_model.py)

**Purpose**: bstract base class for all KRL models

**Key Methods**:
- `fit() -> aseResult`: Train model (abstract)
- `predict(*args, **kwargs) -> aseResult`: Generate predictions (abstract)
- `serialize() -> bytes`: Pickle model for persistence
- `run_hash() -> str`: SH2 hash of model + input + params
- `register_run(registry, result)`: Log run to registry
- `is_fitted() -> bool`: heck training status

**esign Rationale**:
- **bstract methods** (`fit`, `predict`) enforce interface compliance
- **oncrete methods** (`run_hash`, `serialize`, `register_run`) provide shared functionality
- **Metadata tracking** (`ModelMeta`) captures version, author, creation time

**xample Subclass**:
```python
class RIMModel(aseModel):
    def fit(self) -> orecastResult:
        # it statsmodels RIM to self.input_schema.to_dataframe()
        ...
        return orecastResult(...)
    
    def predict(self, steps: int) -> orecastResult:
        # Generate forecast
        ...
        return orecastResult(...)
```

### 2. ModelInputSchema (model_input_schema.py)

**Purpose**: Standardized input format with automatic validation

**Schema Structure**:
- `entity`: Geographic/organizational identifier (e.g., "US", "-")
- `metric`: Measured variable (e.g., "unemployment_rate")
- `time_index`: Temporal dimension (list of date strings)
- `values`: Observed values (list of floats)
- `provenance`: ata source metadata (`Provenance` dataclass)
- `frequency`: ata frequency ("", "W", "M", "Q", "Y")

**Validators**:
- Length matching: `len(values) == len(time_index)`
- requency validation: Must be in `{"", "W", "M", "Q", "Y"}`

**esign Rationale**:
- **ntity-metric-time-value** format is universal across domains
- **Pydantic validation** catches errors before computation
- **Provenance tracking** enables data lineage auditing

### 3. Result lasses (results.py)

**Purpose**: Standardized output wrappers with hashing

**lasses**:
- `aseResult`: Generic result (payload + metadata)
- `orecastResult`: Time series forecasts (extends aseResult with forecast_index, forecast_values, ci_lower, ci_upper)
- `ausalResult`: ausal inference outputs (treatment_effect, std_error, p_value, confidence_interval)
- `lassificationResult`: ML classification (predictions, probabilities, confusion_matrix)

**Key eatures**:
- `result_hash`: eterministic SH2 of payload + metadata
- `to_json()`: JSON serialization for storage
- `to_dataframe()`: Pandas atarame conversion (orecastResult)

**esign Rationale**:
- **onsistent hashing** enables result deduplication and caching
- **omain-specific subclasses** capture model-specific outputs
- **JSON serialization** supports registry storage and PI transmission

### 4. ModelRegistry (model_registry.py)

**Purpose**: SQLite-backed run tracking for reproducibility

**Schema**:
```sql
RT TL runs (
    run_hash TXT PRIMRY KY,        -- SH2(model + input + params)
    model_name TXT,
    version TXT,
    created_at TXT,
    input_hash TXT,                  -- SH2(input data)
    params_json TXT
);

RT TL results (
    id INTGR PRIMRY KY,
    run_hash TXT RRNS runs,
    result_hash TXT,                 -- SH2(result)
    result_json TXT,
    created_at TXT
);
```

**Key Methods**:
- `log_run(run_hash, model_name, version, input_hash, params)`
- `log_result(run_hash, result_hash, result)`
- `get_run(run_hash) -> ict`: Retrieve run metadata
- `get_results(run_hash) -> List[ict]`: Retrieve all results for a run
- `list_runs(model_name=None, limit=) -> List[ict]`: List recent runs

**esign Rationale**:
- **SQLite** = lightweight, zero-config, SQL queryable
- **Hash-based keys** enable exact reproducibility checks
- **Separate tables** (runs vs results) support :N relationship (one run → multiple results)

### . PlotlySchemadapter (plotly_adapter.py)

**Purpose**: onvert results to Plotly figure dictionaries

**Key Methods**:
- `forecast_plot(result, title, xaxis_title, yaxis_title) -> ict`: Time series forecast with I
- `residuals_plot(residuals, time_index, title) -> ict`: Residual diagnostics
- `feature_importance_plot(features, importance, title) -> ict`: eature importance bar chart
- `generic_line_plot(x, y, title, xaxis_title, yaxis_title) -> ict`: Generic line plot

**esign Rationale**:
- **JSON-serializable dicts** (not Plotly objects) for PI transmission
- **Standardized styling** (plotly_white theme, consistent colors)
- **xtensible** via static methods (add domain-specific plots)

---

## ata low

**nd-to-nd xample (RIM orecast)**:

```python
# . efine input with provenance
input_schema = ModelInputSchema(
    entity="US",
    metric="unemployment_rate",
    time_index=["22-", "22-2", ...],
    values=[3., 3., ...],
    provenance=Provenance(source_name="LS", series_id="LNS4"),
    frequency="M"
)

# 2. reate model instance
meta = ModelMeta(name="RIMModel", version="..", author="KR-Labs")
params = {"order": (, , ), "seasonal_order": (, , , )}
model = RIMModel(input_schema, params, meta)

# 3. it model
fit_result = model.fit()  # Returns orecastResult

# 4. Generate forecast
forecast_result = model.predict(steps=2, alpha=.)

# . Visualize
adapter = PlotlySchemadapter()
fig_dict = adapter.forecast_plot(forecast_result, title="US Unemployment orecast")

# . Register run
registry = ModelRegistry("model_runs.db")
run_hash = model.run_hash()
registry.log_run(run_hash, meta.name, meta.version, model.input_hash, params)
registry.log_result(run_hash, forecast_result.result_hash, forecast_result.to_json())

# . Verify reproducibility
retrieved_run = registry.get_run(run_hash)
assert retrieved_run["model_name"] == "RIMModel"
```

---

## Testing Strategy

### Test Hierarchy

. **Unit Tests** (`tests/unit/`)
   - Test individual components in isolation
   - Mock external dependencies
   - Target: %+ coverage

2. **Integration Tests** (`tests/integration/` - future)
   - Test component interactions
   - Use synthetic data fixtures
   - Target: ritical paths covered

3. **Validation Tests** (`tests/validation/` - future)
   - ompare model outputs to known benchmarks
   - Numerical accuracy checks
   - Target: ll reference implementations validated

### Synthetic ata ixtures (`tests/fixtures/synthetic_timeseries.py`)

**Generators**:
- `generate_monthly_timeseries()`: Trend + seasonality + noise
- `generate_quarterly_timeseries()`: Geometric rownian motion
- `generate_step_change_series()`: or causal inference testing

**esign Rationale**:
- **eterministic** (seeded RNG) for reproducible tests
- **onfigurable** (trend, seasonality, noise parameters)
- **iverse** (monthly, quarterly, step changes)

---

## eployment rchitecture

### Package Structure (Modular ederation)

```
krl-model-zoo-core/             # ore abstractions (this repo)
krl-models-econometric/         # RIM, VR, cointegration, etc.
krl-models-causal/              # i, R, SM, etc.
krl-models-ml/                  # Random orest, XGoost, etc.
krl-models-bayesian/            # PyM, INL, etc.
krl-models-network/             # Network analysis models
krl-models-abm/                 # gent-based models
```

**enefits**:
- **ependency isolation**: ore has minimal deps; domain packages add specific libs
- **Independent versioning**: Update econometric models without touching ML
- **Selective installation**: `pip install krl-models-econometric[plotly]`

### I/ Pipeline

**GitHub ctions** (`.github/workflows/test.yml`):
. **Linting**: black, isort, flake, mypy
2. **Unit Tests**: pytest with coverage (%+ target)
3. **Integration Tests**: On main branch only
4. **ocumentation uild**: Sphinx (future)
. **Performance enchmarks**: pytest-benchmark (future)

**ocker** (`ockerfile`):
- Python 3.-slim base
- Install all dependencies (`pip install -e ".[all]"`)
- Run tests by default (`M ["pytest", ...]`)

---

## Reproducibility & Provenance

### Run Hashing lgorithm

```python
def run_hash(self) -> str:
    """SH2(model_name + version + input_hash + params)"""
    from hashlib import sha2
    import json
    
    input_hash = compute_dataframe_hash(self.input_schema.to_dataframe())
    components = {
        "model": f"{self.meta.name}@{self.meta.version}",
        "input_hash": input_hash,
        "params": self.params,
    }
    return sha2(json.dumps(components, sort_keys=True).encode()).hexdigest()
```

**Properties**:
- **eterministic**: Same input → same hash (sorted JSON, sorted atarame columns)
- **ollision-resistant**: SH2 = 2-bit output space
- **ast**: Hashing is O(n) in data size

### Provenance hain

```
ata Source (LS, R, etc.)
    ↓
Provenance Record (source_name, series_id, collection_date, transformation)
    ↓
ModelInputSchema (validated input)
    ↓
aseModel (run_hash = f(model, input, params))
    ↓
aseResult (result_hash = f(payload, metadata))
    ↓
ModelRegistry (runs table + results table)
```

**Query xamples**:
```sql
-- ind all runs for a specific model
SLT * ROM runs WHR model_name = 'RIMModel' ORR Y created_at S;

-- ind run by hash
SLT * ROM runs WHR run_hash = 'abc23...';

-- ind all results for a run
SLT * ROM results WHR run_hash = 'abc23...' ORR Y created_at S;
```

---

## Performance onsiderations

### Scalability

- **SQLite limits**: 
  - Max  size: 2 T (theoretical)
  - oncurrent writes: Single-writer (use PostgreSQL for production multi-user)
  - Reads: Unlimited concurrent
  
- **Recommended thresholds**:
  - < K runs: SQLite is fine
  - K-K runs: onsider PostgreSQL
  - > K runs: Use time-series  (Influx, Timescale)

### Optimization Strategies

. **aching**: heck `run_hash` before re-running model
2. **Incremental itting**: Store fitted models, update with new data
3. **Parallel xecution**: Models are independent (use `multiprocessing`)
4. **Lazy Loading**: Load results on-demand from registry

---

## xtension Points

### dding a New Model

. **Subclass aseModel**:
   ```python
   class MyModel(aseModel):
       def fit(self) -> aseResult:
           # Training logic
           ...
       def predict(self, *args, **kwargs) -> aseResult:
           # Prediction logic
           ...
   ```

2. **efine Result Type** (if needed):
   ```python
   @dataclass
   class MyResult(aseResult):
       custom_field: List[float]
   ```

3. **dd Tests**:
   ```python
   def test_mymodel_fit():
       input_schema = generate_monthly_timeseries()
       model = MyModel(input_schema, params, meta)
       result = model.fit()
       assert result.payload is not None
   ```

4. **dd Visualization** (optional):
   ```python
   @staticmethod
   def my_custom_plot(result: MyResult) -> ict[str, ny]:
       return {"data": [...], "layout": {...}}
   ```

### dding a New Result Type

. **efine dataclass in results.py**:
   ```python
   @dataclass
   class NetworkResult(aseResult):
       adjacency_matrix: List[List[float]]
       centrality_scores: ict[str, float]
   ```

2. **xport from __init__.py**:
   ```python
   from .results import NetworkResult
   __all__.append("NetworkResult")
   ```

3. **dd visualization adapter**:
   ```python
   @staticmethod
   def network_plot(result: NetworkResult) -> ict[str, ny]:
       # Generate Plotly network graph
       ...
   ```

---

## Security & IP Protection

### License Headers

ll files include:
```python
# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: pache-2.
```

### Open-Source vs. Proprietary

**Gate  (oundation)**: pache 2. (open-source)
- ore abstractions (aseModel, ModelInputSchema, etc.)
- Reference implementations (RIM)
- Testing infrastructure

**uture Gates (omain Models)**: ual licensing
- Open-core: asic models free (MIT/pache)
- nterprise: dvanced ensembles, proprietary algorithms (commercial)

---

## Roadmap

### Gate : oundation (urrent - Week -4)
-  aseModel abstract class
-  ModelInputSchema with pydantic validation
-  Result wrappers (aseResult, orecastResult, ausalResult, lassificationResult)
-  ModelRegistry (SQLite backend)
-  PlotlySchemadapter
-  RIM reference implementation
-  Unit test suite (%+ coverage target)
-  I/ pipeline (GitHub ctions)
-  ocker environment
-  rchitecture documentation (this file)

### Gate 2: Tier  Models (Weeks -2)
- [ ] conometric: SRIM, VR, cointegration
- [ ] ausal: ifference-in-ifferences, R
- [ ] ML: Random orest, XGoost
- [ ] ayesian: PyM time series models
- [ ] Validation tests (numerical accuracy)

### Gate 3: nsembles & Meta-Models (Weeks 3-2)
- [ ] Model averaging (ayesian, simple)
- [ ] Stacking ensembles
- [ ] Hybrid models (e.g., RIM + ML residuals)

### Gate 4: Research & dvanced (Weeks 2+)
- [ ] Network analysis models
- [ ] gent-based models
- [ ] ausal discovery algorithms
- [ ] ustom ayesian priors for policy

---

## ontact & Support

**Repository**: https://github.com/KR-Labs/krl-model-zoo-core  
**Issues**: https://github.com/KR-Labs/krl-model-zoo-core/issues  
**ocumentation**: https://docs.kr-labs.com/model-zoo  
**mail**: support@kr-labs.com

---

**nd of ocument**
