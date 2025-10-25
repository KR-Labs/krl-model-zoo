# Mathematical ormulations

**KRL Model Zoo - Rigorous Mathematical oundations**

**Version:** .  
**Last Updated:** October 24, 22

This document provides comprehensive mathematical formulations, derivations, and theoretical foundations for all models in the KRL Model Zoo.

---

## Table of ontents

. [Volatility Models](#volatility-models)
   - [GRH Model](#garch-model)
   - [GRH Model](#egarch-model)
   - [GJR-GRH Model](#gjr-garch-model)
2. [State Space Models](#state-space-models)
   - [Linear Gaussian State Space ramework](#linear-gaussian-state-space-framework)
   - [Kalman ilter](#kalman-filter)
   - [Local Level Model](#local-level-model)
3. [stimation Methods](#estimation-methods)
   - [Maximum Likelihood stimation](#maximum-likelihood-estimation)
   - [Quasi-Maximum Likelihood](#quasi-maximum-likelihood)
4. [orecasting Theory](#forecasting-theory)
. [Model Selection riteria](#model-selection-criteria)
. [References](#references)

---

## Volatility Models

### GRH Model

#### . Model Specification

The **Generalized utoregressive onditional Heteroskedasticity (GRH)** model was introduced by ollerslev () as an extension of ngle's (2) RH model.

**Mean quation:**

$$r_t = \mu_t + \epsilon_t$$

where:
- $r_t$ is the return at time $t$
- $\mu_t$ is the conditional mean
- $\epsilon_t$ is the error term

**rror Specification:**

$$\epsilon_t = \sigma_t z_t$$

where:
- $\sigma_t$ is the conditional standard deviation
- $z_t \sim \text{i.i.d.} \; (,)$ with distribution $$

**Variance quation (GRH(p,q)):**

$$\sigma_t^2 = \omega + \sum_{i=}^{q} \alpha_i \epsilon_{t-i}^2 + \sum_{j=}^{p} \beta_j \sigma_{t-j}^2$$

**Parameters:**
- $\omega > $ (intercept/baseline volatility)
- $\alpha_i \geq $ for $i = , \ldots, q$ (RH coefficients)
- $\beta_j \geq $ for $j = , \ldots, p$ (GRH coefficients)

**Stationarity ondition:**

or covariance stationarity, we require:

$$\sum_{i=}^{q} \alpha_i + \sum_{j=}^{p} \beta_j < $$

**Unconditional Variance:**

When the stationarity condition holds:

$$\mathbb{}[\epsilon_t^2] = \frac{\omega}{ - \sum_{i=}^{q} \alpha_i - \sum_{j=}^{p} \beta_j}$$

#### .2 GRH(,) Special ase

The most commonly used specification:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-}^2 + \beta \sigma_{t-}^2$$

**Persistence:**

The persistence parameter measures volatility memory:

$$\text{Persistence} = \alpha + \beta$$

Typical values: $. < \alpha + \beta < .$ for financial returns.

**Recursive Substitution:**

$$\sigma_t^2 = \frac{\omega}{-\beta} + \alpha \sum_{j=}^{\infty} \beta^j \epsilon_{t--j}^2$$

This shows GRH(,) as an RH(∞) process with exponentially declining weights.

#### .3 Mean Specifications

**onstant Mean:**

$$r_t = \mu + \epsilon_t$$

**Zero Mean:**

$$r_t = \epsilon_t$$

**R(k) Mean:**

$$r_t = \mu + \sum_{i=}^{k} \phi_i r_{t-i} + \epsilon_t$$

#### .4 istribution Specifications

**Normal istribution:**

$$z_t \sim N(, )$$

$$f(z_t) = \frac{}{\sqrt{2\pi}} \exp\left(-\frac{z_t^2}{2}\right)$$

**Student's t-istribution:**

$$z_t \sim t_\nu$$

$$f(z_t; \nu) = \frac{\Gamma((\nu+)/2)}{\Gamma(\nu/2)\sqrt{\pi(\nu-2)}} \left( + \frac{z_t^2}{\nu-2}\right)^{-(\nu+)/2}$$

where $\nu > 2$ is the degrees of freedom parameter.

**Generalized rror istribution (G):**

$$f(z_t; \lambda) = \frac{\lambda \exp(-.|z_t/\eta|^\lambda)}{2^{+/\lambda} \eta \Gamma(/\lambda)}$$

where:
- $\lambda > $ is the shape parameter
- $\eta = \sqrt{2^{-2/\lambda} \Gamma(/\lambda) / \Gamma(3/\lambda)}$
- $\lambda = 2$ gives normal, $\lambda < 2$ gives heavy tails

#### . Volatility orecasting

**One-Step head:**

$$\mathbb{}_t[\sigma_{t+}^2] = \omega + \alpha \epsilon_t^2 + \beta \sigma_t^2$$

**Multi-Step head (GRH(,)):**

$$\mathbb{}_t[\sigma_{t+h}^2] = \bar{\sigma}^2 + (\alpha + \beta)^{h-} (\sigma_{t+}^2 - \bar{\sigma}^2)$$

where $\bar{\sigma}^2 = \omega/(-\alpha-\beta)$ is the unconditional variance.

**orecast onvergence:**

s $h \to \infty$:

$$\mathbb{}_t[\sigma_{t+h}^2] \to \bar{\sigma}^2$$

orecasts converge to unconditional variance at rate $(\alpha + \beta)^h$.

#### . Value-at-Risk (VaR)

**VaR efinition:**

$$\text{VaR}_\alpha(R_{t+} | \mathcal{}_t) = -\mu_{t+} + \sigma_{t+} q_\alpha$$

where:
- $\alpha$ is the confidence level (e.g., ., .)
- $q_\alpha$ is the $\alpha$-quantile of the standardized distribution
- $\mathcal{}_t$ is the information set at time $t$

**or Normal istribution:**

$$q_{.} = .4, \quad q_{.} = 2.32$$

**or Student's t:**

$$q_\alpha = t^{-}_\nu(\alpha) \sqrt{\frac{\nu-2}{\nu}}$$

---

### GRH Model

#### 2. Model Specification

The **xponential GRH** model by Nelson () captures asymmetric volatility effects.

**Log-Variance quation:**

$$\log(\sigma_t^2) = \omega + \sum_{i=}^{o} \gamma_i z_{t-i} + \sum_{i=}^{q} \alpha_i \left(|z_{t-i}| - \mathbb{}[|z_{t-i}|]\right) + \sum_{j=}^{p} \beta_j \log(\sigma_{t-j}^2)$$

where $z_t = \epsilon_t / \sigma_t$ is the standardized residual.

#### 2.2 GRH(,) Specification

$$\log(\sigma_t^2) = \omega + \gamma z_{t-} + \alpha (|z_{t-}| - \sqrt{2/\pi}) + \beta \log(\sigma_{t-}^2)$$

**Key eatures:**

. **No Parameter Restrictions:** $\gamma, \alpha, \beta$ can be any real numbers
2. **Positivity Guaranteed:** $\sigma_t^2 > $ always (due to log transformation)
3. **Leverage ffect:** $\gamma < $ captures asymmetry

#### 2.3 symmetric Response

**or positive shocks ($z_{t-} > $):**

$$\log(\sigma_t^2) - \omega - \beta \log(\sigma_{t-}^2) = (\gamma + \alpha) z_{t-} - \alpha\sqrt{2/\pi}$$

**or negative shocks ($z_{t-} < $):**

$$\log(\sigma_t^2) - \omega - \beta \log(\sigma_{t-}^2) = (\gamma - \alpha) |z_{t-}| - \alpha\sqrt{2/\pi}$$

**Leverage ffect:**

If $\gamma < $:
- Negative shocks: impact = $|\gamma - \alpha|$
- Positive shocks: impact = $|\gamma + \alpha|$
- Typically $|\gamma - \alpha| > |\gamma + \alpha|$

#### 2.4 Persistence

**Persistence Parameter:**

$$\text{Persistence} = \beta$$

Unlike GRH, persistence is directly given by $\beta$.

**Stationarity:**

GRH is stationary if $|\beta| < $.

#### 2. News Impact urve

The **news impact curve** shows how volatility responds to shocks:

$$\text{NI}(z) = \exp\left(\omega + \gamma z + \alpha(|z| - \sqrt{2/\pi})\right)$$

**Properties:**
- symmetric around zero if $\gamma \neq $
- Minimum at $z = $ if $\alpha > , \gamma = $
- Typical: steeper for $z < $ (negative shocks)

---

### GJR-GRH Model

#### 3. Model Specification

The **GJR-GRH** model by Glosten, Jagannathan, and Runkle (3) uses indicator functions for asymmetry.

**Variance quation:**

$$\sigma_t^2 = \omega + \sum_{i=}^{q} \alpha_i \epsilon_{t-i}^2 + \sum_{i=}^{o} \gamma_i I_{t-i} \epsilon_{t-i}^2 + \sum_{j=}^{p} \beta_j \sigma_{t-j}^2$$

where:

$$I_t = \begin{cases}
 & \text{if } \epsilon_t <  \text{ (negative shock)} \\
 & \text{if } \epsilon_t \geq  \text{ (positive shock)}
\end{cases}$$

#### 3.2 GJR-GRH(,,) Specification

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-}^2 + \gamma I_{t-} \epsilon_{t-}^2 + \beta \sigma_{t-}^2$$

**onditional Response:**

**or positive shocks ($\epsilon_{t-} \geq $):**

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-}^2 + \beta \sigma_{t-}^2$$

**or negative shocks ($\epsilon_{t-} < $):**

$$\sigma_t^2 = \omega + (\alpha + \gamma) \epsilon_{t-}^2 + \beta \sigma_{t-}^2$$

**Leverage ffect:**

If $\gamma > $: negative shocks have impact $\alpha + \gamma > \alpha$.

#### 3.3 Parameter onstraints

**Non-negativity:**
- $\omega > $
- $\alpha \geq $
- $\gamma \geq $
- $\beta \geq $

**Stationarity:**

$$\alpha + \frac{\gamma}{2} + \beta < $$

(assuming $\mathbb{P}(\epsilon_t < ) = .$)

#### 3.4 Unconditional Variance

$$\mathbb{}[\epsilon_t^2] = \frac{\omega}{ - \alpha - \gamma/2 - \beta}$$

#### 3. News Impact urve

$$\text{NI}(\epsilon) = \begin{cases}
\omega + (\alpha + \gamma) \epsilon^2 & \text{if } \epsilon <  \\
\omega + \alpha \epsilon^2 & \text{if } \epsilon \geq 
\end{cases}$$

**Properties:**
- Piecewise quadratic
- Kink at $\epsilon = $
- Steeper for $\epsilon < $ if $\gamma > $

---

## State Space Models

### Linear Gaussian State Space ramework

#### 4. General ormulation

**State quation:**

$$\mathbf{x}_t = \mathbf{}_t \mathbf{x}_{t-} + \mathbf{}_t \mathbf{u}_t + \mathbf{w}_t$$

$$\mathbf{w}_t \sim N(\mathbf{}, \mathbf{Q}_t)$$

**Observation quation:**

$$\mathbf{y}_t = \mathbf{H}_t \mathbf{x}_t + \mathbf{}_t \mathbf{u}_t + \mathbf{v}_t$$

$$\mathbf{v}_t \sim N(\mathbf{}, \mathbf{R}_t)$$

**Notation:**
- $\mathbf{x}_t \in \mathbb{R}^n$: unobserved state vector
- $\mathbf{y}_t \in \mathbb{R}^m$: observed data vector
- $\mathbf{u}_t \in \mathbb{R}^k$: known inputs (optional)
- $\mathbf{}_t \in \mathbb{R}^{n \times n}$: state transition matrix
- $\mathbf{H}_t \in \mathbb{R}^{m \times n}$: observation matrix
- $\mathbf{Q}_t \in \mathbb{R}^{n \times n}$: process noise covariance (positive semi-definite)
- $\mathbf{R}_t \in \mathbb{R}^{m \times m}$: measurement noise covariance (positive definite)
- $\mathbf{}_t, \mathbf{}_t$: input matrices

**ssumptions:**
. $\mathbf{w}_t$ and $\mathbf{v}_s$ are independent for all $t, s$
2. $\mathbf{w}_t$ and $\mathbf{x}_$ are independent
3. $\mathbf{v}_t$ and $\mathbf{x}_$ are independent

---

### Kalman ilter

#### . Recursive lgorithm

The Kalman filter provides optimal state estimates under the linear Gaussian assumptions.

**Notation:**
- $\mathbf{x}_{t|t-}$: predicted state (a priori)
- $\mathbf{x}_{t|t}$: filtered state (a posteriori)
- $\mathbf{P}_{t|t-}$: predicted covariance
- $\mathbf{P}_{t|t}$: filtered covariance

#### .2 Prediction Step

**State Prediction:**

$$\mathbf{x}_{t|t-} = \mathbf{}_t \mathbf{x}_{t-|t-} + \mathbf{}_t \mathbf{u}_t$$

**ovariance Prediction:**

$$\mathbf{P}_{t|t-} = \mathbf{}_t \mathbf{P}_{t-|t-} \mathbf{}_t^T + \mathbf{Q}_t$$

**Innovation (Prediction rror):**

$$\mathbf{\nu}_t = \mathbf{y}_t - \mathbf{H}_t \mathbf{x}_{t|t-} - \mathbf{}_t \mathbf{u}_t$$

**Innovation ovariance:**

$$\mathbf{S}_t = \mathbf{H}_t \mathbf{P}_{t|t-} \mathbf{H}_t^T + \mathbf{R}_t$$

#### .3 Update Step

**Kalman Gain:**

$$\mathbf{K}_t = \mathbf{P}_{t|t-} \mathbf{H}_t^T \mathbf{S}_t^{-}$$

**State Update:**

$$\mathbf{x}_{t|t} = \mathbf{x}_{t|t-} + \mathbf{K}_t \mathbf{\nu}_t$$

**ovariance Update:**

$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t|t-}$$

**lternative (Joseph) orm:**

$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t|t-} (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t)^T + \mathbf{K}_t \mathbf{R}_t \mathbf{K}_t^T$$

(numerically stable)

#### .4 Initialization

**Initial State:**

$$\mathbf{x}_{|} = \mathbb{}[\mathbf{x}_]$$

**Initial ovariance:**

$$\mathbf{P}_{|} = \mathbb{}[(\mathbf{x}_ - \mathbf{x}_{|})(\mathbf{x}_ - \mathbf{x}_{|})^T]$$

**iffuse Initialization:**

or non-stationary states:

$$\mathbf{P}_{|} = \kappa \mathbf{I}, \quad \kappa \to \infty$$

#### . RTS Smoother

The **Rauch-Tung-Striebel (RTS)** smoother provides optimal backward smoothing.

**Smoothing Gain:**

$$\mathbf{J}_{t-} = \mathbf{P}_{t-|t-} \mathbf{}_t^T \mathbf{P}_{t|t-}^{-}$$

**State Smoothing:**

$$\mathbf{x}_{t-|T} = \mathbf{x}_{t-|t-} + \mathbf{J}_{t-} (\mathbf{x}_{t|T} - \mathbf{x}_{t|t-})$$

**ovariance Smoothing:**

$$\mathbf{P}_{t-|T} = \mathbf{P}_{t-|t-} + \mathbf{J}_{t-} (\mathbf{P}_{t|T} - \mathbf{P}_{t|t-}) \mathbf{J}_{t-}^T$$

**ackward Recursion:**

Run for $t = T, T-, \ldots, $ after forward filtering pass.

#### . Log-Likelihood

**Innovation-ased Likelihood:**

$$\log L(\theta) = -\frac{mT}{2} \log(2\pi) - \frac{}{2} \sum_{t=}^{T} \left(\log|\mathbf{S}_t| + \mathbf{\nu}_t^T \mathbf{S}_t^{-} \mathbf{\nu}_t\right)$$

where $\theta$ represents all model parameters.

**Prediction rror ecomposition:**

The log-likelihood naturally decomposes into contributions from each time period.

#### . orecasting

**h-Step head orecast:**

$$\mathbf{x}_{T+h|T} = \mathbf{}^h \mathbf{x}_{T|T}$$

**orecast ovariance:**

$$\mathbf{P}_{T+h|T} = \mathbf{}^h \mathbf{P}_{T|T} (\mathbf{}^h)^T + \sum_{i=}^{h-} \mathbf{}^i \mathbf{Q} (\mathbf{}^i)^T$$

**Observation orecast:**

$$\mathbf{y}_{T+h|T} = \mathbf{H} \mathbf{x}_{T+h|T}$$

**orecast Variance:**

$$\text{Var}[\mathbf{y}_{T+h|T}] = \mathbf{H} \mathbf{P}_{T+h|T} \mathbf{H}^T + \mathbf{R}$$

---

### Local Level Model

#### . Model Specification

The **Local Level Model** (or **Random Walk Plus Noise**) is the simplest structural time series model.

**Level quation:**

$$\mu_t = \mu_{t-} + \eta_t, \quad \eta_t \sim N(, \sigma_\eta^2)$$

**Observation quation:**

$$y_t = \mu_t + \epsilon_t, \quad \epsilon_t \sim N(, \sigma_\epsilon^2)$$

**State Space orm:**

$$\mu_t =  \cdot \mu_{t-} + \eta_t$$

$$y_t =  \cdot \mu_t + \epsilon_t$$

**Parameters:**
- $\mu_t$: unobserved level (state)
- $\sigma_\eta^2$: level variance (process noise)
- $\sigma_\epsilon^2$: observation variance (measurement noise)

#### .2 Signal-to-Noise Ratio

**efinition:**

$$q = \frac{\sigma_\eta^2}{\sigma_\epsilon^2}$$

**Interpretation:**
- $q \to $: smooth trend (level changes slowly)
- $q \to \infty$: noisy trend (follows observations closely)
- $q = $: equal signal and noise variances

**Typical Values:**

or most time series: $. < q < .$

#### .3 Kalman ilter pplication

**Matrices:**

$$ = , \quad H = , \quad Q = \sigma_\eta^2, \quad R = \sigma_\epsilon^2$$

**Prediction:**

$$\mu_{t|t-} = \mu_{t-|t-}$$

$$P_{t|t-} = P_{t-|t-} + \sigma_\eta^2$$

**Update:**

$$K_t = \frac{P_{t|t-}}{P_{t|t-} + \sigma_\epsilon^2}$$

$$\mu_{t|t} = \mu_{t|t-} + K_t (y_t - \mu_{t|t-})$$

$$P_{t|t} = ( - K_t) P_{t|t-}$$

#### .4 Steady-State Kalman ilter

s $t \to \infty$, $P_{t|t}$ converges to steady state $P_\infty$:

$$P_\infty = \frac{-\sigma_\epsilon^2 + \sqrt{\sigma_\epsilon^4 + 4\sigma_\eta^2\sigma_\epsilon^2}}{2}$$

**Steady-State Gain:**

$$K_\infty = \frac{P_\infty}{P_\infty + \sigma_\epsilon^2}$$

#### . Maximum Likelihood stimation

**Log-Likelihood:**

$$\log L(\sigma_\eta^2, \sigma_\epsilon^2) = -\frac{T}{2}\log(2\pi) - \frac{}{2}\sum_{t=}^{T}\left(\log S_t + \frac{\nu_t^2}{S_t}\right)$$

where:
- $\nu_t = y_t - \mu_{t|t-}$ (innovation)
- $S_t = P_{t|t-} + \sigma_\epsilon^2$ (innovation variance)

**oncentrated Likelihood:**

Often maximize with respect to $q = \sigma_\eta^2 / \sigma_\epsilon^2$ and $\sigma_\epsilon^2$.

**Optimization:**

Use numerical optimization (e.g., L-GS-) with constraints:
- $\sigma_\eta^2 > $
- $\sigma_\epsilon^2 > $

#### . Smoothed stimates

**Level ecomposition:**

$$y_t = \mu_{t|T} + \epsilon_{t|T}$$

where:
- $\mu_{t|T}$: smoothed level (trend)
- $\epsilon_{t|T} = y_t - \mu_{t|T}$: irregular component

**Properties:**
- $\mathbb{}[\epsilon_{t|T}] = $
- $\text{Var}[\epsilon_{t|T}] \approx \sigma_\epsilon^2 ( - K_\infty)$

#### . orecasting

**h-Step head orecast:**

$$\mathbb{}[y_{T+h} | y_{:T}] = \mu_{T|T}$$

**orecast Mean Squared rror:**

$$\text{MS}[y_{T+h}] = P_{T|T} + h\sigma_\eta^2 + \sigma_\epsilon^2$$

**orecast Intervals (%):**

$$\mu_{T|T} \pm . \sqrt{P_{T|T} + h\sigma_\eta^2 + \sigma_\epsilon^2}$$

**Uncertainty Growth:**

Uncertainty grows linearly with horizon: $\text{MS} \propto h$.

---

## stimation Methods

### Maximum Likelihood stimation

#### . General ramework

**Likelihood unction:**

$$L(\theta; y_{:T}) = \prod_{t=}^{T} f(y_t | y_{:t-}, \theta)$$

**Log-Likelihood:**

$$\ell(\theta) = \sum_{t=}^{T} \log f(y_t | y_{:t-}, \theta)$$

#### .2 GRH ML

**onditional ensity:**

$$f(\epsilon_t | \mathcal{}_{t-}, \theta) = \frac{}{\sigma_t} f\left(\frac{\epsilon_t}{\sigma_t}\right)$$

**Normal istribution:**

$$\ell(\theta) = -\frac{T}{2}\log(2\pi) - \frac{}{2}\sum_{t=}^{T}\left(\log\sigma_t^2 + \frac{\epsilon_t^2}{\sigma_t^2}\right)$$

**Student's t istribution:**

$$\ell(\theta, \nu) = T\left[\log\Gamma\left(\frac{\nu+}{2}\right) - \log\Gamma\left(\frac{\nu}{2}\right) - \frac{}{2}\log(\pi(\nu-2))\right]$$

$$- \frac{}{2}\sum_{t=}^{T}\left[\log\sigma_t^2 + (\nu+)\log\left( + \frac{\epsilon_t^2}{\sigma_t^2(\nu-2)}\right)\right]$$

#### .3 Optimization

**irst-Order onditions:**

$$\frac{\partial \ell(\theta)}{\partial \theta} = $$

**Numerical Methods:**
- GS (royden-letcher-Goldfarb-Shanno)
- L-GS- (with box constraints)
- Nelder-Mead (derivative-free)

**Standard rrors:**

rom inverse Hessian (observed information matrix):

$$\text{Var}[\hat{\theta}] \approx \left[-\frac{\partial^2 \ell(\hat{\theta})}{\partial \theta \partial \theta^T}\right]^{-}$$

---

### Quasi-Maximum Likelihood

#### . Robustness

**QML stimation:**

Maximize Gaussian log-likelihood even if true distribution is non-Gaussian.

**onsistency:**

Under regularity conditions, QML is consistent for variance parameters even with misspecified density.

**symptotic Normality:**

$$\sqrt{T}(\hat{\theta}_{QML} - \theta_) \xrightarrow{d} N(, ^{-}^{-})$$

where:
- $ = -\mathbb{}\left[\frac{\partial^2 \ell}{\partial\theta\partial\theta^T}\right]$ (Hessian)
- $ = \mathbb{}\left[\frac{\partial\ell}{\partial\theta}\frac{\partial\ell}{\partial\theta^T}\right]$ (outer product of gradients)

**Robust Standard rrors:**

Use "sandwich" covariance matrix $^{-}^{-}$.

---

## orecasting Theory

### . Optimal orecasts

**Mean Squared rror riterion:**

The optimal h-step ahead forecast minimizes MS:

$$\hat{y}_{T+h|T} = \mathbb{}[y_{T+h} | \mathcal{}_T]$$

**orecast rror:**

$$e_{T+h} = y_{T+h} - \hat{y}_{T+h|T}$$

**Mean Squared rror:**

$$\text{MS}(h) = \mathbb{}[e_{T+h}^2]$$

### .2 GRH Volatility orecasts

**One-Step head:**

$$\hat{\sigma}_{T+}^2 = \omega + \alpha \epsilon_T^2 + \beta \sigma_T^2$$

**Multi-Step (GRH(,)):**

$$\hat{\sigma}_{T+h}^2 = \bar{\sigma}^2 + (\alpha+\beta)^{h-}(\hat{\sigma}_{T+}^2 - \bar{\sigma}^2)$$

**Long-Horizon Limit:**

$$\lim_{h\to\infty} \hat{\sigma}_{T+h}^2 = \bar{\sigma}^2$$

### .3 State Space orecasts

**Point orecast:**

$$\hat{\mathbf{y}}_{T+h|T} = \mathbf{H} \mathbf{}^h \mathbf{x}_{T|T}$$

**orecast Uncertainty:**

$$\text{Var}[\mathbf{y}_{T+h} | \mathcal{}_T] = \mathbf{H}\mathbf{P}_{T+h|T}\mathbf{H}^T + \mathbf{R}$$

### .4 orecast valuation

**Mean bsolute rror:**

$$\text{M} = \frac{}{H}\sum_{h=}^{H}|y_{T+h} - \hat{y}_{T+h|T}|$$

**Root Mean Squared rror:**

$$\text{RMS} = \sqrt{\frac{}{H}\sum_{h=}^{H}(y_{T+h} - \hat{y}_{T+h|T})^2}$$

**Mean bsolute Percentage rror:**

$$\text{MP} = \frac{}{H}\sum_{h=}^{H}\left|\frac{y_{T+h} - \hat{y}_{T+h|T}}{y_{T+h}}\right|$$

---

## Model Selection riteria

### . Information riteria

**kaike Information riterion (I):**

$$\text{I} = -2\ell(\hat{\theta}) + 2k$$

where $k$ is the number of parameters.

**ayesian Information riterion (I):**

$$\text{I} = -2\ell(\hat{\theta}) + k\log(T)$$

**Interpretation:**

Lower values indicate better fit penalized for complexity.

**Model Selection:**

hoose model with minimum I or I.

**I vs I:**
- I penalizes complexity more heavily (for $T > $)
- I better for prediction
- I better for identifying true model (asymptotically)

### .2 Likelihood Ratio Tests

or nested models $M_ \subset M_$:

**Test Statistic:**

$$LR = 2(\ell_ - \ell_)$$

**symptotic istribution:**

$$LR \xrightarrow{d} \chi^2_{k_ - k_}$$

where $k_ - k_$ is difference in number of parameters.

**ecision Rule:**

Reject $M_$ if $LR > \chi^2_{k_-k_, \alpha}$ at significance level $\alpha$.

### .3 Residual iagnostics

**Ljung-ox Test:**

Tests for autocorrelation in standardized residuals:

$$Q(m) = T(T+2)\sum_{k=}^{m}\frac{\hat{\rho}_k^2}{T-k}$$

$$Q(m) \sim \chi^2_{m-p}$$

**Null Hypothesis:** No autocorrelation up to lag $m$.

**RH-LM Test:**

Tests for remaining RH effects:

$$LM = TR^2 \sim \chi^2_q$$

from auxiliary regression of $\hat{z}_t^2$ on lags.

---

## References

### Volatility Models

. **ngle, R. . (2).** "utoregressive onditional Heteroscedasticity with stimates of the Variance of United Kingdom Inflation." *conometrica*, (4), -.

2. **ollerslev, T. ().** "Generalized utoregressive onditional Heteroskedasticity." *Journal of conometrics*, 3(3), 3-32.

3. **Nelson, . . ().** "onditional Heteroskedasticity in sset Returns:  New pproach." *conometrica*, (2), 34-3.

4. **Glosten, L. R., Jagannathan, R., & Runkle, . . (3).** "On the Relation between the xpected Value and the Volatility of the Nominal xcess Return on Stocks." *Journal of inance*, 4(), -.

. **Hansen, P. R., & Lunde, . (2).** " orecast omparison of Volatility Models: oes nything eat a GRH(,)?" *Journal of pplied conometrics*, 2(), 3-.

### State Space Models

. **Kalman, R. . ().** " New pproach to Linear iltering and Prediction Problems." *Journal of asic ngineering*, 2(), 3-4.

. **Rauch, H. ., Tung, ., & Striebel, . T. ().** "Maximum Likelihood stimates of Linear ynamic Systems." *I Journal*, 3(), 44-4.

. **Harvey, . . ().** *orecasting, Structural Time Series Models and the Kalman ilter*. ambridge University Press.

. **urbin, J., & Koopman, S. J. (22).** *Time Series nalysis by State Space Methods* (2nd ed.). Oxford University Press.

### stimation Theory

. **ollerslev, T., & Wooldridge, J. M. (2).** "Quasi-Maximum Likelihood stimation and Inference in ynamic Models with Time-Varying ovariances." *conometric Reviews*, (2), 43-2.

. **White, H. (2).** "Maximum Likelihood stimation of Misspecified Models." *conometrica*, (), -2.

### General conometrics

2. **Hamilton, J. . (4).** *Time Series nalysis*. Princeton University Press.

3. **Tsay, R. S. (2).** *nalysis of inancial Time Series* (3rd ed.). Wiley.

4. **rancq, ., & Zakoïan, J.-M. (2).** *GRH Models: Structure, Statistical Inference and inancial pplications* (2nd ed.). Wiley.

---

## Notation Summary

### General Notation

| Symbol | escription |
|--------|-------------|
| $t$ | Time index |
| $T$ | Sample size |
| $h$ | orecast horizon |
| $\theta$ | Parameter vector |
| $\mathcal{}_t$ | Information set at time $t$ |
| $\mathbb{}[\cdot]$ | xpectation operator |
| $\text{Var}[\cdot]$ | Variance operator |
| $\sim$ | istributed as |
| $\xrightarrow{d}$ | onverges in distribution |

### GRH Notation

| Symbol | escription |
|--------|-------------|
| $r_t$ | Return at time $t$ |
| $\epsilon_t$ | rror term |
| $\sigma_t^2$ | onditional variance |
| $z_t$ | Standardized residual |
| $\omega$ | Intercept parameter |
| $\alpha_i$ | RH coefficient |
| $\beta_j$ | GRH coefficient |
| $\gamma$ | symmetry/leverage parameter |

### State Space Notation

| Symbol | escription |
|--------|-------------|
| $\mathbf{x}_t$ | State vector |
| $\mathbf{y}_t$ | Observation vector |
| $\mathbf{}$ | State transition matrix |
| $\mathbf{H}$ | Observation matrix |
| $\mathbf{Q}$ | Process noise covariance |
| $\mathbf{R}$ | Measurement noise covariance |
| $\mathbf{K}_t$ | Kalman gain |
| $\mathbf{P}_t$ | State covariance |
| $\mu_t$ | Level (Local Level Model) |

---

**nd of Mathematical ormulations**

or practical implementation details, see:
- PI Reference: `docs/PI_RRN.md`
- User Guide: `docs/USR_GUI.md`
- xamples: `examples/` directory

*KRL Model Zoo - Rigorous Mathematical oundations*  
*October 24, 22*
