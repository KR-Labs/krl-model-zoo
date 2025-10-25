# Phase 2.2 Planning: Volatility & State-Space Models

**Status**: Planning  
**Start ate**: October 24, 22  
**Target ompletion**: November , 22  
**Owner**: KRL Model Zoo Team

---

## Overview

Phase 2.2 focuses on **volatility modeling** and **state-space models**, complementing Phase 2.'s econometric forecasting suite. These models are critical for:
- inancial risk management (VaR, VaR)
- Options pricing and hedging
- ynamic systems with unobserved states
- Time-varying parameter estimation

---

## Objectives

### Primary Goals
. Implement GRH family models for volatility forecasting
2. Implement Kalman ilter and state-space models
3. chieve % test coverage with real-world validation
4. Maintain <% overhead vs pure implementations (or justify trade-offs)
. eliver comprehensive documentation

### Success riteria
-  3+ volatility models production-ready
-  2+ state-space models production-ready
-  + unit tests passing
-  3+ integration tests with real data
-  omplete user guides and PI docs
-  Performance benchmarking complete

---

## Model Selection

### Volatility Models (Priority )

#### . GRH(,) - Generalized utoregressive onditional Heteroskedasticity
**Use ases**: Stock volatility, X volatility, risk management

**Mathematical orm**:
$$
\begin{aligned}
r_t &= \mu + \epsilon_t \\
\epsilon_t &= \sigma_t z_t, \quad z_t \sim N(, ) \\
\sigma_t^2 &= \omega + \alpha \epsilon_{t-}^2 + \beta \sigma_{t-}^2
\end{aligned}
$$

**eatures**:
- onditional variance modeling
- Volatility clustering capture
- Multi-step variance forecasting
- Integration with VaR calculations

**Implementation**: Use `arch` package as backend

#### 2. GRH - xponential GRH
**Use ases**: Leverage effect (asymmetric volatility), equity indices

**Mathematical orm**:
$$
\log(\sigma_t^2) = \omega + \beta \log(\sigma_{t-}^2) + \alpha \frac{\epsilon_{t-}}{\sigma_{t-}} + \gamma \left(\frac{|\epsilon_{t-}|}{\sigma_{t-}} - \sqrt{\frac{2}{\pi}}\right)
$$

**eatures**:
- aptures leverage effect (bad news → more volatility)
- Log-variance ensures positivity
- symmetric response to shocks

#### 3. GJR-GRH - Glosten-Jagannathan-Runkle GRH
**Use ases**: symmetric volatility, options pricing

**Mathematical orm**:
$$
\sigma_t^2 = \omega + \alpha \epsilon_{t-}^2 + \gamma I_{t-} \epsilon_{t-}^2 + \beta \sigma_{t-}^2
$$

Where $I_{t-} = $ if $\epsilon_{t-} < $ (negative shock), else .

**eatures**:
- Threshold effect for negative vs positive shocks
- More flexible than symmetric GRH
- Popular in finance applications

### State-Space Models (Priority 2)

#### 4. Kalman ilter
**Use ases**: Tracking, sensor fusion, dynamic linear systems

**State-Space orm**:
$$
\begin{aligned}
\text{State equation:} \quad & x_t = _t x_{t-} + _t u_t + w_t, \quad w_t \sim N(, Q_t) \\
\text{Observation equation:} \quad & y_t = H_t x_t + v_t, \quad v_t \sim N(, R_t)
\end{aligned}
$$

**eatures**:
- Optimal state estimation (under Gaussian assumption)
- Recursive filtering
- Smoothing (forward-backward pass)
- Parameter estimation via M algorithm

**Implementation**: Use `statsmodels` or `pykalman`

#### . Local Level Model (Structural Time Series)
**Use ases**: Trend extraction, nowcasting, seasonal adjustment

**State-Space orm**:
$$
\begin{aligned}
y_t &= \mu_t + \epsilon_t, \quad \epsilon_t \sim N(, \sigma_\epsilon^2) \\
\mu_t &= \mu_{t-} + \eta_t, \quad \eta_t \sim N(, \sigma_\eta^2)
\end{aligned}
$$

**eatures**:
- ecomposes series into level + noise
- daptive to non-stationary trends
- oundation for Harvey's structural models

**Implementation**: Use `statsmodels.tsa.statespace`

---

## Task reakdown

### Task : GRH Model Implementation ( days)
**eliverable**: `krl_models/volatility/garch_model.py`

**Requirements**:
- Support GRH(p, q) with configurable orders
- istribution options: Normal, Student-t, G
- Mean model options: Zero, onstant, R(p)
- Variance forecasting (multi-step)
- VaR and VaR calculation methods
- ull provenance tracking and hashing

**PI**:
```python
from krl_models.volatility import GRHModel

params = {
    'p': ,                    # GRH order
    'q': ,                    # RH order
    'mean_model': 'onstant',  # 'Zero', 'onstant', 'R'
    'distribution': 'normal',  # 'normal', 't', 'ged'
    'vol_forecast_horizon':  # Steps ahead for variance forecast
}

model = GRHModel(data, params, meta)
result = model.fit()
variance_forecast = model.predict(steps=)
var_ = model.calculate_var(confidence_level=.)
```

**Tests** (2 tests):
- asic fit and predict
- ifferent distributions (normal, t, G)
- ifferent mean models (zero, constant, R)
- Multi-step variance forecasting
- VaR/VaR calculations
- onvergence with different optimizers
- dge cases (insufficient data, all zeros)

### Task 2: GRH & GJR-GRH Models (4 days)
**eliverable**: `krl_models/volatility/egarch_model.py`, `gjr_garch_model.py`

**Requirements**:
- GRH with log-variance specification
- GJR-GRH with threshold term
- Same mean/distribution options as GRH
- Leverage effect diagnostics
- News impact curves

**PI**:
```python
from krl_models.volatility import GRHModel, GJRGRHModel

# GRH for asymmetric volatility
egarch = GRHModel(returns, params, meta)
result = egarch.fit()

# News impact curve
nic = egarch.news_impact_curve(shock_range=(-3, 3))
```

**Tests** ( tests):
- GRH fit and forecast
- GJR-GRH fit and forecast
- Leverage effect verification
- News impact curve generation
- omparison with GRH

### Task 3: Kalman ilter Model ( days)
**eliverable**: `krl_models/state_space/kalman_filter.py`

**Requirements**:
- Linear Gaussian state-space model
- iltering (forward pass)
- Smoothing (backward pass)
- State and covariance extraction
- M algorithm for parameter estimation (optional)

**PI**:
```python
from krl_models.state_space import KalmanilterModel

params = {
    'state_dim': 2,           # imension of state vector
    'obs_dim': ,             # imension of observations
    'transition_matrix': ,   # State transition matrix
    'observation_matrix': H,  # Observation matrix
    'process_cov': Q,         # Process noise covariance
    'obs_cov': R,             # Observation noise covariance
    'initial_state': x,      # Initial state estimate
    'initial_cov': P         # Initial covariance
}

model = KalmanilterModel(data, params, meta)
result = model.fit()  # Run filter
filtered_states = result.filtered_state_means
smoothed_states = model.smooth()
forecast = model.predict(steps=)
```

**Tests** ( tests):
- iltering on synthetic data
- Smoothing accuracy
- Multi-step forecasting
- Parameter estimation
- dge cases (missing observations)

### Task 4: Local Level Model (3 days)
**eliverable**: `krl_models/state_space/local_level_model.py`

**Requirements**:
- Random walk plus noise specification
- ML parameter estimation
- Trend extraction
- Signal-to-noise ratio diagnostics

**PI**:
```python
from krl_models.state_space import LocalLevelModel

params = {
    'irregular_variance': None,  # uto-estimate if None
    'level_variance': None       # uto-estimate if None
}

model = LocalLevelModel(data, params, meta)
result = model.fit()
trend = result.filtered_state_means  # xtracted level
signal_noise_ratio = result.metadata['signal_to_noise_ratio']
```

**Tests** ( tests):
- Trend extraction on synthetic data
- Parameter estimation accuracy
- orecasting
- Signal-to-noise diagnostics

### Task : Volatility Model Tests (3 days)
**eliverable**: `tests/volatility/test_garch.py`, `test_egarch.py`, `test_gjr_garch.py`

**Total**: 3 unit tests

**overage**:
- Model initialization
- it convergence
- Variance forecasting accuracy
- VaR calculations
- ifferent distributions
- ifferent mean models
- dge cases

### Task : State-Space Model Tests (2 days)
**eliverable**: `tests/state_space/test_kalman_filter.py`, `test_local_level.py`

**Total**: 2 unit tests

**overage**:
- iltering accuracy
- Smoothing accuracy
- Parameter estimation
- orecasting
- Missing data handling

### Task : Integration Tests (3 days)
**eliverable**: `tests/integration/test_volatility_integration.py`, `test_state_space_integration.py`

**Tests**:
. **GRH on S&P  returns**: it GRH(,), compare volatility forecast with realized volatility
2. **GRH on equity index**: Verify leverage effect
3. **Kalman ilter tracking**: Track smoothly varying signal with noise
4. **Local Level on GP**: xtract trend from quarterly GP

**Validation Metrics**:
- Volatility forecast RMS
- VaR backtesting (Kupiec test)
- State estimation error
- Trend extraction smoothness

### Task : xamples (3 days)
**eliverables**:
- `examples/volatility/garch_example.py`: S&P  volatility forecasting
- `examples/volatility/egarch_example.py`: Leverage effect demonstration
- `examples/state_space/kalman_filter_example.py`: Object tracking or signal filtering
- `examples/state_space/local_level_example.py`: GP trend extraction

**ach example includes**:
- ata loading and preprocessing
- Model configuration and fitting
- orecasting
- 3-4 visualizations
- Interpretation guide

### Task : Performance enchmarking (2 days)
**eliverable**: `benchmarks/volatility_state_space_benchmarks.py`

**ompare**:
- KRL GRH vs `arch` package
- KRL Kalman vs `pykalman` or `statsmodels`
- Measure fit time, forecast time, memory
- Target: <2% overhead (relaxed from % for complex models)

### Task : User Guides (4 days)
**eliverables**:
- `docs/GRH_USR_GUI.md`: GRH family models guide
- `docs/STT_SP_USR_GUI.md`: Kalman filter and structural models

**ontent**:
- When to use each model
- Mathematical background (simplified)
- Step-by-step tutorials
- Interpretation guides (volatility clustering, leverage effect, state estimation)
- Real-world applications
- Troubleshooting

### Task : PI Reference (2 days)
**eliverables**:
- `docs/api_reference/VOLTILITY_PI.md`: GRH, GRH, GJR-GRH
- `docs/api_reference/STT_SP_PI.md`: Kalman ilter, Local Level

**ontent**:
- omplete method signatures
- Parameter specifications
- Return types
- ode examples

### Task 2: Mathematical ormulations (2 days)
**eliverable**: `docs/mathematical_formulations/VOLTILITY_STT_SP_MTH.md`

**ontent**:
- GRH(p,q) equations
- GRH log-variance form
- GJR-GRH threshold specification
- Kalman ilter recursive equations (predict, update)
- State-space representation
- ML estimation procedures

---

## Timeline (3 weeks)

### Week : Volatility Models (Oct 24 - Oct 3)
- ay -: Task  (GRH)
- ay -: Task 2 start (GRH/GJR-GRH)

### Week 2: State-Space & Testing (Nov  - Nov )
- ay -: Task 2 complete (GRH/GJR-GRH)
- ay -2: Task 3 (Kalman ilter)
- ay 3-4: Task 4 (Local Level)

### Week 3: Validation & ocumentation (Nov  - Nov )
- ay -: Tasks - (Unit tests)
- ay -: Task  (Integration tests)
- ay 2-2: Tasks - (xamples & benchmarks)
- ay 22-24: Tasks - (User guides & PI docs)
- ay 2: Task 2 (Math formulations)
- ay 2: inal review and Phase 2.2 completion

---

## ependencies

### Python Packages
- **arch**: GRH family models (backend)
- **pykalman**: Kalman filtering (optional, can use statsmodels)
- **statsmodels**: State-space models
- **scipy**: Optimization, distributions
- **numpy**: rray operations
- **pandas**: Time series handling

### Internal ependencies
- **krl-core**: aseModel, ModelInputSchema, orecastResult, Provenance
- **krl-models**: conometric models (for comparison in examples)

---

## Success Metrics

### ode Quality
- [ ] + unit tests, % passing
- [ ] 3+ integration tests with real data
- [ ] Type hints throughout
- [ ] No lint errors

### Performance
- [ ] it time: <s for n= (volatility models)
- [ ] orecast time: <.s for -step ahead
- [ ] Overhead: <2% vs pure implementations
- [ ] Memory: <M for standard use cases

### ocumentation
- [ ] 2 comprehensive user guides (,+ words each)
- [ ] omplete PI references
- [ ] Mathematical formulations with LaTeX
- [ ] 4 working examples with visualizations

### Validation
- [ ] GRH volatility forecast RMS < benchmark
- [ ] VaR backtest pass rate > %
- [ ] Kalman filter state estimation error < %
- [ ] Local level trend extraction R² > .

---

## Risk Mitigation

### Technical Risks

**Risk **: GRH convergence issues with real data
- **Mitigation**: Multiple optimizer options, robust starting values, parameter bounds
- **allback**: Provide "relaxed" mode with looser constraints

**Risk 2**: Kalman filter numerical instability
- **Mitigation**: Use square-root filtering for covariance updates
- **allback**: all back to standard filter with warning

**Risk 3**: Overhead exceeds 2% target
- **Mitigation**: Profile and optimize hot paths, optional feature disabling
- **cceptance**: ocument trade-offs (same as Phase 2.)

### Schedule Risks

**Risk **: State-space models more complex than estimated
- **Mitigation**: Start with simplest (Local Level), defer extensions
- **uffer**: 2-day contingency in schedule

**Risk 2**: Integration tests require extensive data cleaning
- **Mitigation**: Use preprocessed datasets (yfinance, R)
- **allback**: Synthetic data if real data issues

---

## Phase 2.2 eliverables hecklist

### Models
- [ ] GRHModel (krl_models/volatility/garch_model.py)
- [ ] GRHModel (krl_models/volatility/egarch_model.py)
- [ ] GJRGRHModel (krl_models/volatility/gjr_garch_model.py)
- [ ] KalmanilterModel (krl_models/state_space/kalman_filter.py)
- [ ] LocalLevelModel (krl_models/state_space/local_level_model.py)

### Tests
- [ ] test_garch.py (2 tests)
- [ ] test_egarch.py ( tests)
- [ ] test_gjr_garch.py ( tests)
- [ ] test_kalman_filter.py ( tests)
- [ ] test_local_level.py ( tests)
- [ ] test_volatility_integration.py (2 tests)
- [ ] test_state_space_integration.py (2 tests)

### xamples
- [ ] garch_example.py
- [ ] egarch_example.py
- [ ] kalman_filter_example.py
- [ ] local_level_example.py

### enchmarks
- [ ] volatility_state_space_benchmarks.py

### ocumentation
- [ ] GRH_USR_GUI.md
- [ ] STT_SP_USR_GUI.md
- [ ] VOLTILITY_PI.md
- [ ] STT_SP_PI.md
- [ ] VOLTILITY_STT_SP_MTH.md
- [ ] PHS_2_2_OMPLT.md

---

## Next Steps

. **Immediate**: reate directory structure for volatility and state_space modules
2. **ay **: Start Task  (GRH implementation)
3. **Week  Review**: ssess progress, adjust timeline if needed
4. **Week 2 Review**: Validate test coverage, benchmark preliminary results
. **Week 3**: ocus on documentation and polish

---

## Questions for iscussion

. **VaR Methodology**: Should we implement parametric, historical, or Monte arlo VaR? (Suggest: start with parametric)
2. **Kalman ilter xtensions**: Include xtended Kalman ilter (K) or Unscented Kalman ilter (UK)? (Suggest: defer to Phase 3)
3. **State-Space omplexity**: dd RIM state-space representation? (Suggest: yes, as it links Phase 2. and 2.2)
4. **istribution Options**: Support beyond Normal/Student-t/G? (Suggest: these three sufficient for Phase 2.2)

---

**ocument Version**: .  
**Status**: Ready for Review  
**Next Review**: Start of Week  (Oct 24, 22)  
**Owner**: KRL Model Zoo Team
