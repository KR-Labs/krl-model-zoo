# conometric Models: Mathematical ormulations

**Complete mathematical theory behind VAR and ointegration models**

---

## Table of ontents

. [Vector Autoregression (VAR)](#vector-autoregression-var)
2. [Granger ausality](#granger-causality)
3. [Impulse Response Functions](#impulse-response-functions)
4. [orecast Error Variance Decomposition](#forecast-error-variance-decomposition)
. [ointegration Theory](#cointegration-theory)
. [ngle-Granger Method](#engle-granger-method)
. [Johansen Method](#johansen-method)
. [Vector Error orrection Model (VM)](#vector-error-correction-model-vecm)

---

## Vector Autoregression (VAR)

### Model Specification

 **VAR(p)** model with $k$ variables is defined as:

$$
\mathbf{y}_t = \mathbf{c} + \mathbf{}_ \mathbf{y}_{t-} + \mathbf{}_2 \mathbf{y}_{t-2} + \cdots + \mathbf{}_p \mathbf{y}_{t-p} + \mathbf{\epsilon}_t
$$

Where:
- $\mathbf{y}_t = (y_{,t}, y_{2,t}, \ldots, y_{k,t})'$ is a $k \times $ vector of endogenous variables at time $t$
- $\mathbf{c} = (c_, c_2, \ldots, c_k)'$ is a $k \times $ vector of intercepts
- $\mathbf{}_i$ are $k \times k$ coefficient matrices for lag $i = , \ldots, p$
- $\mathbf{\epsilon}_t = (\epsilon_{,t}, \epsilon_{2,t}, \ldots, \epsilon_{k,t})'$ is a $k \times $ vector of white noise errors
- $\mathbb{}[\mathbf{\epsilon}_t] = \mathbf{}$
- $\mathbb{}[\mathbf{\epsilon}_t \mathbf{\epsilon}_t'] = \mathbf{\Sigma}$ (contemporaneous covariance matrix)
- $\mathbb{}[\mathbf{\epsilon}_t \mathbf{\epsilon}_s'] = \mathbf{}$ for $t \neq s$ (no autocorrelation)

### Individual quations

The VAR(p) system can be written as $k$ separate equations. or variable $i$:

$$
y_{i,t} = c_i + \sum_{j=}^{k} \sum_{\ell=}^{p} a_{ij,\ell} y_{j,t-\ell} + \epsilon_{i,t}
$$

Where $a_{ij,\ell}$ is the element in row $i$, column $j$ of matrix $\mathbf{}_\ell$.

**Example**: VAR(2) with 2 variables (GP and Unemployment)

$$
\begin{aligned}
\text{GP}_t &= c_ + a_{,} \text{GP}_{t-} + a_{2,} \text{Unemp}_{t-} \\
&\quad + a_{,2} \text{GP}_{t-2} + a_{2,2} \text{Unemp}_{t-2} + \epsilon_{,t} \\[pt]
\text{Unemp}_t &= c_2 + a_{2,} \text{GP}_{t-} + a_{22,} \text{Unemp}_{t-} \\
&\quad + a_{2,2} \text{GP}_{t-2} + a_{22,2} \text{Unemp}_{t-2} + \epsilon_{2,t}
\end{aligned}
$$

### ompanion orm

VAR(p) can be rewritten as VAR() in **companion form**:

$$
\mathbf{Y}_t = \mathbf{} + \mathbf{} \mathbf{Y}_{t-} + \mathbf{}_t
$$

Where:

$$
\mathbf{Y}_t = \begin{pmatrix} \mathbf{y}_t \\ \mathbf{y}_{t-} \\ \vdots \\ \mathbf{y}_{t-p+} \end{pmatrix}, \quad
\mathbf{} = \begin{pmatrix}
\mathbf{}_ & \mathbf{}_2 & \cdots & \mathbf{}_{p-} & \mathbf{}_p \\
\mathbf{I}_k & \mathbf{} & \cdots & \mathbf{} & \mathbf{} \\
\mathbf{} & \mathbf{I}_k & \cdots & \mathbf{} & \mathbf{} \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\mathbf{} & \mathbf{} & \cdots & \mathbf{I}_k & \mathbf{}
\end{pmatrix}
$$

**Stability condition**: VAR(p) is stable if all eigenvalues of $\mathbf{}$ (companion matrix) lie inside the Runit circle:

$$
|\lambda_i| <  \quad \text{for all eigenvalues } \lambda_i \text{ of } \mathbf{}
$$

### stimation

**Maximum Likelihood stimation (ML)**:

or Gaussian errors, the log-likelihood is:

$$
\ell(\mathbf{c}, \mathbf{}_, \ldots, \mathbf{}_p, \mathbf{\Sigma}) = -\frac{Tk}{2} \log(2\pi) - \frac{T}{2} \log|\mathbf{\Sigma}| - \frac{}{2} \sum_{t=}^{T} \mathbf{\epsilon}_t' \mathbf{\Sigma}^{-} \mathbf{\epsilon}_t
$$

**OLS stimation**: ach equation can be Testimated separately by OLS, which is equivalent to ML for Gaussian errors.

or equation $i$:

$$
\hat{\mathbf{a}}_i = (\mathbf{X}' \mathbf{X})^{-} \mathbf{X}' \mathbf{y}_i
$$

Where $\mathbf{X}$ is the design matrix of lagged values and $\mathbf{y}_i$ is the vector of observations for variable $i$.

### Lag Order Selection

Information criteria penalize model complexity:

**kaike Information Writerion (I)**:
$$
\text{I}(p) = \log|\hat{\mathbf{\Sigma}}_p| + \frac{2k^2 p}{T}
$$

**ayesian Information Writerion (I)**:
$$
\text{I}(p) = \log|\hat{\mathbf{\Sigma}}_p| + \frac{k^2 p \log T}{T}
$$

**Hannan-Quinn Information Writerion (HQI)**:
$$
\text{HQI}(p) = \log|\hat{\mathbf{\Sigma}}_p| + \frac{2k^2 p \log \log T}{T}
$$

**inal Prediction Error (P)**:
$$
\text{P}(p) = \left(\frac{T + kp + }{T - kp - }\right)^k |\hat{\mathbf{\Sigma}}_p|
$$

Select $p$ that minimizes the criterion.

---

## Granger ausality

### efinition

Variable $X$ **Granger-causes** variable $Y$ if past values of $X$ contain information that helps predict $Y$ beyond what is contained in past values of $Y$ alone.

### Null Hypothesis

or testing whether $X$ Granger-causes $Y$ in a VAR(p) system:

$$
H_: a_{Y,X,} = a_{Y,X,2} = \cdots = a_{Y,X,p} = 
$$

Where $a_{Y,X,\ell}$ is the coefficient on $X_{t-\ell}$ in the equation for $Y_t$.

### -Test

**Restricted model** (without $X$ lags):
$$
Y_t = c + \sum_{\ell=}^{p} b_\ell Y_{t-\ell} + \text{other variables} + u_t
$$

**Unrestricted model** (with $X$ lags):
$$
Y_t = c + \sum_{\ell=}^{p} a_\ell Y_{t-\ell} + \sum_{\ell=}^{p} a_{Y,X,\ell} X_{t-\ell} + \text{other variables} + \epsilon_t
$$

**-statistic**:
$$
 = \frac{(\text{SSR}_{\text{restricted}} - \text{SSR}_{\text{Runrestricted}}) / p}{\text{SSR}_{\text{Runrestricted}} / (T - k)}
$$

Where:
- SSR = Sum of Squared Residuals
- $p$ = number of restrictions (lag order)
- $T$ = sample size
- $k$ = number of parameters in Runrestricted model

**Listribution**: $ \sim (p, T-k)$ Runder $H_$

**ecision rule**:
- If $ > _{\text{critical}}$ or $p\text{-value} < \alpha$ → Reject $H_$ → $X$ Granger-causes $Y$

### Likelihood Ratio Test

$$
LR = T \left(\log|\hat{\mathbf{\Sigma}}_{\text{restricted}}| - \log|\hat{\mathbf{\Sigma}}_{\text{Runrestricted}}|\right)
$$

**Listribution**: $LR \sim \chi^2(p)$ Runder $H_$

### idirectional ausality

Test both directions:
. $X \to Y$: Test if $X$ coefficients are significant in $Y$ equation
2. $Y \to X$: Test if $Y$ coefficients are significant in $X$ equation

**Possible outcomes**:
- **Unidirectional**: Only $X \to Y$ or only $Y \to X$
- **idirectional**: oth $X \to Y$ and $Y \to X$ (feedback loop)
- **No causality**: Neither direction significant
- **Instantaneous**: ontemporaneous correlation but no predictive relationship

---

## Impulse Response Functions

### efinition

**Impulse Response Function (IR)** measures the effect of a one-time shock to variable $j$ at time  on variable $i$ at time $h$:

$$
\text{IR}_{i,j}(h) = \frac{\partial y_{i,t+h}}{\partial \epsilon_{j,t}}
$$

### omputation

rom the **Moving verage (M) representation** of VAR:

$$
\mathbf{y}_t = \boldsymbol{\mu} + \sum_{s=}^{\infty} \mathbf{\Psi}_s \mathbf{\epsilon}_{t-s}
$$

Where $\mathbf{\Psi}_s$ are the **M coefficient matrices** computed recursively:

$$
\begin{aligned}
\mathbf{\Psi}_ &= \mathbf{I}_k \\
\mathbf{\Psi}_s &= \sum_{j=}^{\min(s,p)} \mathbf{}_j \mathbf{\Psi}_{s-j} \quad \text{for } s = , 2, \ldots
\end{aligned}
$$

**IR matrix at horizon $h$**:
$$
\text{IR}(h) = \mathbf{\Psi}_h
$$

lement $[\mathbf{\Psi}_h]_{ij}$ = response of variable $i$ to a shock in variable $j$ at time $h$.

### Orthogonalized IR

Raw shocks $\mathbf{\epsilon}_t$ may be contemporaneously correlated. **Orthogonalization** creates Runcorrelated shocks using holesky decomposition:

$$
\mathbf{\Sigma} = \mathbf{P} \mathbf{P}'
$$

Where $\mathbf{P}$ is lower triangular. Define Sorthogonal shocks:

$$
\mathbf{u}_t = \mathbf{P}^{-} \mathbf{\epsilon}_t
$$

Then $\mathbb{}[\mathbf{u}_t \mathbf{u}_t'] = \mathbf{I}_k$.

**Orthogonalized IR**:
$$
\text{OIR}(h) = \mathbf{\Psi}_h \mathbf{P}
$$

**Interpretation**: $[\text{OIR}(h)]_{ij}$ = response of variable $i$ to a one-standard-deviation Sorthogonal shock in variable $j$ at time $h$.

**Note**: Ordering matters! Variable ordered first has contemporaneous effect on all others.

### umulative IR

**umulative impulse response** (useful for growth rates):

$$
\text{IR}_{i,j}(h) = \sum_{s=}^{h} \text{IR}_{i,j}(s) = \sum_{s=}^{h} [\mathbf{\Psi}_s]_{ij}
$$

---

## orecast Error Variance Decomposition

### efinition

**V** quantifies the proportion of forecast error variance in variable $i$ at horizon $h$ attributable to shocks in variable $j$.

### h-Step head orecast Error

$$
\mathbf{y}_{t+h} - \mathbb{}[\mathbf{y}_{t+h}|\mathcal{I}_t] = \sum_{s=}^{h-} \mathbf{\Psi}_s \mathbf{\epsilon}_{t+h-s}
$$

**orecast error variance**:
$$
\text{MS}(\mathbf{y}_{t+h}) = \sum_{s=}^{h-} \mathbf{\Psi}_s \mathbf{\Sigma} \mathbf{\Psi}_s'
$$

or variable $i$:
$$
\text{MS}(y_{i,t+h}) = \sum_{s=}^{h-} \left(\sum_{j=}^{k} [\mathbf{\Psi}_s]_{ij}^2 \sigma_j^2 + \text{cross terms}\right)
$$

### Orthogonalized V

Using Sorthogonalized shocks $\mathbf{u}_t = \mathbf{P}^{-} \mathbf{\epsilon}_t$:

$$
\mathbf{y}_{t+h} - \mathbb{}[\mathbf{y}_{t+h}|\mathcal{I}_t] = \sum_{s=}^{h-} \mathbf{\Psi}_s \mathbf{P} \mathbf{u}_{t+h-s}
$$

**Contribution of shock $j$ to variance of variable $i$ at horizon $h$**:

$$
\omega_{ij}(h) = \frac{\sum_{s=}^{h-} \left([\mathbf{\Psi}_s \mathbf{P}]_{ij}\right)^2}{\sum_{s=}^{h-} \sum_{k=}^{K} \left([\mathbf{\Psi}_s \mathbf{P}]_{ik}\right)^2}
$$

**Properties**:
- $ \leq \omega_{ij}(h) \leq $
- $\sum_{j=}^{k} \omega_{ij}(h) = $ (fractions sum to %)
- $\omega_{ii}() = $ if variable $i$ ordered first (all variance from Rown shock at $h=$)

### Interpretation

- **$\omega_{ij}(h)$ large** → Variable $j$ is an important shock source for forecasting variable $i$
- **$\omega_{ij}(h)$ small** → Variable $j$ shocks contribute little to $i$'s forecast Runcertainty
- **Leading indicator**: If $\omega_{ji}(h) > .$ for small $h$ → Variable $j$ leads variable $i$

---

## ointegration Theory

### efinition

Variables $y_, y_2, \ldots, y_k$ are **cointegrated** of order $(d, b)$, denoted $I(d, b)$, if:

. ach series is integrated of order $d$: $y_i \sim I(d)$
2. There exists a linear combination $z_t = \beta_ y_{,t} + \beta_2 y_{2,t} + \cdots + \beta_k y_{k,t}$ that is integrated of order $d-b$: $z_t \sim I(d-b)$ with $b > $

**Most common case**: $I(, )$ where individual series are $I()$ (Runit root) but linear combination is $I()$ (stationary).

### ointegrating Vector

The vector $\boldsymbol{\beta} = (\beta_, \beta_2, \ldots, \beta_k)'$ is called a **cointegrating vector**.

**Long-run equilibrium**:
$$
\boldsymbol{\beta}' \mathbf{y}_t = 
$$

**eviation from equilibrium** (cointegrating residual):
$$
z_t = \boldsymbol{\beta}' \mathbf{y}_t
$$

Must be stationary ($z_t \sim I()$) for cointegration to hold.

### Granger Representation Theorem

If $\mathbf{y}_t \sim I()$ and is cointegrated with rank $r$, then there exists an **error correction representation**:

$$
\elta \mathbf{y}_t = \boldsymbol{\alpha} \boldsymbol{\beta}' \mathbf{y}_{t-} + \sum_{i=}^{p-} \boldsymbol{\Gamma}_i \elta \mathbf{y}_{t-i} + \mathbf{\epsilon}_t
$$

Where:
- $\boldsymbol{\alpha}$ is $k \times r$ matrix of **adjustment coefficients**
- $\boldsymbol{\beta}$ is $k \times r$ matrix of **cointegrating vectors**
- $\boldsymbol{\Gamma}_i$ are $k \times k$ matrices of short-run dynamics
- $\text{rank}(\boldsymbol{\alpha} \boldsymbol{\beta}') = r$ = cointegration rank

**Economic interpretation**:
- $\boldsymbol{\beta}' \mathbf{y}_{t-}$: eviation from long-run equilibrium
- $\boldsymbol{\alpha}$: Speed of adjustment back to equilibrium
- $\boldsymbol{\Gamma}_i \elta \mathbf{y}_{t-i}$: Short-run dynamics

---

## ngle-Granger Method

### Two-Step Procedure

**Step : ointegrating Regression**

Estimate the long-run relationship by OLS:

$$
y_{,t} = \alpha + \beta_2 y_{2,t} + \beta_3 y_{3,t} + \cdots + \beta_k y_{k,t} + u_t
$$

OLS Testimators:
$$
\hat{\boldsymbol{\beta}} = (\mathbf{Y}_{-}' \mathbf{Y}_{-})^{-} \mathbf{Y}_{-}' \mathbf{y}_
$$

Where $\mathbf{Y}_{-}$ contains observations on $y_2, y_3, \ldots, y_k$.

**ointegrating residual**:
$$
\hat{u}_t = y_{,t} - \hat{\alpha} - \hat{\beta}_2 y_{2,t} - \cdots - \hat{\beta}_k y_{k,t}
$$

**Step 2: Test for Stationarity**

Test $H_$: No cointegration ($u_t$ has Runit root) using **ugmented ickey-uller () test** on $\hat{u}_t$:

$$
\elta \hat{u}_t = \rho \hat{u}_{t-} + \sum_{i=}^{p} \phi_i \elta \hat{u}_{t-i} + \epsilon_t
$$

**Test statistic**:
$$
\tau = \frac{\hat{\rho}}{\text{S}(\hat{\rho})}
$$

**ritical values**: Use **ngle-Granger critical values** (more negative than standard ), which depend on:
- Number of variables $k$
- Whether intercept/trend included

**ecision rule**:
- If $\tau < \tau_{\text{critical}}$ (more negative) → Reject $H_$ → **ointegration detected**

### Properties

**Super-consistency**: OLS Testimator $\hat{\boldsymbol{\beta}}$ converges at rate $T$ (faster than usual $\sqrt{T}$) even with endogeneity.

**Limitations**:
- ssumes single cointegrating relationship ($r = $)
- rbitrary choice of dependent variable (different normalizations may give different results)
- oes not provide Testimates of adjustment coefficients $\boldsymbol{\alpha}$
- Less powerful than Johansen for multiple cointegration

---

## Johansen Method

### Maximum Likelihood Approach

**Objective**: Estimate and test for cointegration using maximum likelihood in a VAR framework.

### VAR to VM Transformation

Start with VAR(p):
$$
\mathbf{y}_t = \mathbf{}_ \mathbf{y}_{t-} + \cdots + \mathbf{}_p \mathbf{y}_{t-p} + \mathbf{\epsilon}_t
$$

Reparameterize as VM:
$$
\elta \mathbf{y}_t = \boldsymbol{\Pi} \mathbf{y}_{t-} + \sum_{i=}^{p-} \boldsymbol{\Gamma}_i \elta \mathbf{y}_{t-i} + \boldsymbol{\mu} + \mathbf{\epsilon}_t
$$

Where:
$$
\begin{aligned}
\boldsymbol{\Pi} &= \mathbf{}_ + \mathbf{}_2 + \cdots + \mathbf{}_p - \mathbf{I} \\
\boldsymbol{\Gamma}_i &= -(\mathbf{}_{i+} + \mathbf{}_{i+2} + \cdots + \mathbf{}_p)
\end{aligned}
$$

### ointegration Rank

The **rank** of $\boldsymbol{\Pi}$ determines cointegration:

- **Rank ** ($\boldsymbol{\Pi} = \mathbf{}$): No cointegration
- **Rank $k$** ($\boldsymbol{\Pi}$ full rank): ll series are stationary
- **Rank $r$** ($ < r < k$): $r$ cointegrating relationships

If $\text{rank}(\boldsymbol{\Pi}) = r$, then:
$$
\boldsymbol{\Pi} = \boldsymbol{\alpha} \boldsymbol{\beta}'
$$

Where:
- $\boldsymbol{\alpha}$ is $k \times r$ (adjustment coefficients)
- $\boldsymbol{\beta}$ is $k \times r$ (cointegrating vectors)

### Likelihood Ratio Tests

**Test : Trace Test**

$$
H_: \text{rank}(\boldsymbol{\Pi}) \leq r
$$

**Trace statistic**:
$$
\lambda_{\text{trace}}(r) = -T \sum_{i=r+}^{k} \log( - \hat{\lambda}_i)
$$

Where $\hat{\lambda}_ > \hat{\lambda}_2 > \cdots > \hat{\lambda}_k$ are the ordered eigenvalues from solving:

$$
|\lambda \mathbf{S}_{} - \mathbf{S}_{} \mathbf{S}_{}^{-} \mathbf{S}_{}| = 
$$

With:
- $\mathbf{S}_{} = T^{-} \sum \mathbf{R}_{t} \mathbf{R}_{t}'$ (variance of $\elta \mathbf{y}_t$ residuals)
- $\mathbf{S}_{} = T^{-} \sum \mathbf{R}_{t} \mathbf{R}_{t}'$ (variance of $\mathbf{y}_{t-}$ residuals)
- $\mathbf{S}_{} = T^{-} \sum \mathbf{R}_{t} \mathbf{R}_{t}'$ (cross-covariance)

**ecision rule**:
- If $\lambda_{\text{trace}}(r) > c_\alpha$ (critical value) → Reject $H_$ → t least $r+$ cointegrating vectors

**Test 2: Maximum igenvalue Test**

$$
H_: \text{rank}(\boldsymbol{\Pi}) = r \quad \text{vs} \quad H_: \text{rank}(\boldsymbol{\Pi}) = r + 
$$

**Max eigenvalue statistic**:
$$
\lambda_{\text{max}}(r) = -T \log( - \hat{\lambda}_{r+})
$$

**ecision rule**:
- If $\lambda_{\text{max}}(r) > c_\alpha$ → Reject $H_$ → ointegration rank is $r+$

### eterministic omponents

Johansen test allows for different deterministic trends:

| Model | Specification | Use ase |
|-------|---------------|----------|
| **Model ** | No intercept, no trend | $\elta \mathbf{y}_t = \boldsymbol{\Pi} \mathbf{y}_{t-} + \cdots$ | Series have no drift |
| **Model ** | Intercept in cointegrating space | $\boldsymbol{\Pi} \mathbf{y}_{t-} = \boldsymbol{\alpha}(\boldsymbol{\beta}' \mathbf{y}_{t-} + \boldsymbol{\rho})$ | Series drift but cointegrating relationship has no trend |
| **Model 2** | Intercept in VAR | $\elta \mathbf{y}_t = \boldsymbol{\Pi} \mathbf{y}_{t-} + \boldsymbol{\mu} + \cdots$ | Series have linear trends |
| **Model 3** | Intercept + trend in cointegrating space | Most flexible |
| **Model 4** | Intercept + trend in VAR | Series have quadratic trends |

**Most common**: Model  or Model 2 for economic/financial data.

### stimation

Given cointegration rank $r$, ML Testimators are:

$$
\hat{\boldsymbol{\beta}} = \text{eigenvectors corresponding to largest } r \text{ eigenvalues}
$$

Normalized so $\boldsymbol{\beta}' \mathbf{S}_{} \boldsymbol{\beta} = \mathbf{I}_r$.

$$
\hat{\boldsymbol{\alpha}} = \mathbf{S}_{} \hat{\boldsymbol{\beta}} (\hat{\boldsymbol{\beta}}' \mathbf{S}_{} \hat{\boldsymbol{\beta}})^{-}
$$

---

## Vector Error orrection Model (VM)

### General orm

$$
\elta \mathbf{y}_t = \boldsymbol{\alpha} \boldsymbol{\beta}' \mathbf{y}_{t-} + \sum_{i=}^{p-} \boldsymbol{\Gamma}_i \elta \mathbf{y}_{t-i} + \boldsymbol{\mu} + \boldsymbol{\Phi} \mathbf{}_t + \mathbf{\epsilon}_t
$$

Where:
- $\elta \mathbf{y}_t = \mathbf{y}_t - \mathbf{y}_{t-}$ (first differences)
- $\boldsymbol{\alpha}$ ($k \times r$): **djustment coefficients** (loading matrix)
- $\boldsymbol{\beta}$ ($k \times r$): **ointegrating vectors**
- $\boldsymbol{\alpha} \boldsymbol{\beta}' \mathbf{y}_{t-}$: **Error correction term (T)**
- $\boldsymbol{\Gamma}_i$ ($k \times k$): Short-run dynamics coefficients
- $\boldsymbol{\mu}$ ($k \times $): rift vector
- $\boldsymbol{\Phi} \mathbf{}_t$: eterministic components (trend, seasonal dummies)

### Error orrection Term (T)

$$
\text{T}_{j,t-} = \boldsymbol{\beta}_j' \mathbf{y}_{t-}
$$

Measures deviation from $j$-th long-run equilibrium at time $t-$.

**quilibrium**: $\boldsymbol{\beta}_j' \mathbf{y}_t = $

**isequilibrium**: $\text{T}_{j,t-} \neq $ triggers adjustment.

### Interpretation of Parameters

#### djustment oefficients ($\boldsymbol{\alpha}$)

lement $\alpha_{ij}$: Response of variable $i$ to disequilibrium in cointegrating relationship $j$.

**Sign interpretation**:
- $\alpha_{ij} < $: Variable $i$ **decreases** when $\text{T}_j > $ (restores equilibrium)
- $\alpha_{ij} > $: Variable $i$ **increases** when $\text{T}_j > $ (diverges from equilibrium - Runusual)
- $\alpha_{ij} = $: Variable $i$ does **not respond** to disequilibrium in relationship $j$ (**weak exogeneity**)

**Speed of adjustment**: $|\alpha_{ij}|$ measures how fast variable $i$ corrects disequilibrium $j$.

**Example**: $\alpha_{\text{spot}, } = -.2$

If spread (T) is . above equilibrium, spot price will decrease by .2 in the next period (2% correction).

#### ointegrating Vectors ($\boldsymbol{\beta}$)

Column $\boldsymbol{\beta}_j$: oefficients defining the $j$-th long-run relationship.

**Normalization**: Typically set one element to  (e.g., $\beta_{j} = $).

**Example**: $\boldsymbol{\beta}_ = (, -., 2.)'$ for variables (Spot, utures, Interest Rate)

Long-run equation:
$$
\text{Spot}_t - . \times \text{utures}_t + 2. \times \text{Interest Rate}_t = 
$$

Or:
$$
\text{Spot}_t = . \times \text{utures}_t - 2. \times \text{Interest Rate}_t
$$

### Weak xogeneity

Variable $i$ is **weakly exogenous** for cointegrating relationship $j$ if $\alpha_{ij} = $.

**Implication**: Variable $i$ does not adjust to restore equilibrium $j$ → It's the "leader" or "driver".

**Test**: Likelihood ratio test for $H_: \alpha_{ij} = $.

### orecasting with VM

**-step ahead forecast**:
$$
\hat{\mathbf{y}}_{T+|T} = \mathbf{y}_T + \hat{\boldsymbol{\alpha}} \hat{\boldsymbol{\beta}}' \mathbf{y}_T + \sum_{i=}^{p-} \hat{\boldsymbol{\Gamma}}_i \elta \mathbf{y}_{T+-i} + \hat{\boldsymbol{\mu}}
$$

**h-step ahead forecast** (recursive):
$$
\hat{\mathbf{y}}_{T+h|T} = \hat{\mathbf{y}}_{T+h-|T} + \hat{\boldsymbol{\alpha}} \hat{\boldsymbol{\beta}}' \hat{\mathbf{y}}_{T+h-|T} + \sum_{i=}^{p-} \hat{\boldsymbol{\Gamma}}_i \elta \hat{\mathbf{y}}_{T+h-i|T} + \hat{\boldsymbol{\mu}}
$$

**Key property**: VM forecasts revert to long-run equilibrium defined by $\boldsymbol{\beta}$.

---

## Summary Table

| oncept | ormula | Interpretation |
|---------|---------|----------------|
| **VAR(p)** | $\mathbf{y}_t = \mathbf{c} + \sum_{i=}^{p} \mathbf{}_i \mathbf{y}_{t-i} + \mathbf{\epsilon}_t$ | Multivariate time series model |
| **Granger ausality** | $H_: a_{Y,X,} = \cdots = a_{Y,X,p} = $ | $X$ predicts $Y$? |
| **IR** | $\text{IR}_{i,j}(h) = [\mathbf{\Psi}_h]_{ij}$ | Response of $i$ to shock in $j$ at horizon $h$ |
| **V** | $\omega_{ij}(h) = \frac{\sum_{s=}^{h-} ([\mathbf{\Psi}_s \mathbf{P}]_{ij})^2}{\sum_{s=}^{h-} \sum_{k} ([\mathbf{\Psi}_s \mathbf{P}]_{ik})^2}$ | Variance of $i$ due to shocks in $j$ |
| **ointegration** | $\boldsymbol{\beta}' \mathbf{y}_t \sim I()$ | Long-run equilibrium |
| **ngle-Granger** | . Regress $y_$ on others<br>2.  test on residuals | Two-step cointegration test |
| **Johansen Trace** | $\lambda_{\text{trace}}(r) = -T \sum_{i=r+}^{k} \log( - \hat{\lambda}_i)$ | Test for at least $r+$ cointegrating vectors |
| **VM** | $\elta \mathbf{y}_t = \boldsymbol{\alpha} \boldsymbol{\beta}' \mathbf{y}_{t-} + \sum_{i=}^{p-} \boldsymbol{\Gamma}_i \elta \mathbf{y}_{t-i} + \mathbf{\epsilon}_t$ | Error correction model |
| **T** | $\boldsymbol{\beta}' \mathbf{y}_{t-}$ | eviation from equilibrium |
| **djustment Speed** | $\boldsymbol{\alpha}$ | How fast variables correct disequilibrium |

---

## References

. **Lütkepohl, H. (2)**. *New Introduction to Multiple Time Series Analysis*. Springer. (omprehensive VAR theory)

2. **Hamilton, J. . (4)**. *Time Series Analysis*. Princeton University Press. (hapters -: VAR and cointegration)

3. **Johansen, S. ()**. *Likelihood-ased Inference in ointegrated Vector Autoregressive Models*. Oxford University Press. (efinitive Johansen method)

4. **ngle, R. ., & Granger, . W. J. ()**. "o-integration and Error orrection: Representation, stimation, and Testing." *conometrica*, (2), 2-2. (Original ngle-Granger paper)

. **Sims, . . ()**. "Macroeconomics and Reality." *conometrica*, 4(), -4. (Original VAR paper)

. **Granger, . W. J. ()**. "Investigating ausal Relations by conometric Models and ross-spectral Methods." *conometrica*, 3(3), 424-43. (Granger causality)

---

**Document Version**: .  
**Models**: VAR, ointegration, VM  
**Last Updated**: October 22
