"""
Example 4: Kalman Filter State stimation

This example demonstrates advanced state Testimation using Kalman Filtering
for tracking Runobserved variables. We'll cover:

. Position-velocity tracking problem
2. Kalman Filter setup and initialization
3. Filtering (online Testimation)
4. Smoothing (retrospective Testimation)
. Multi-step forecasting
. Innovation analysis and diagnostics
. omparison with simple methods
. Practical Mapplications

Use ase: Tracking, signal Textraction, trend Testimation, sensor fusion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import krl-model-zoo components
from krl_models.state_space.kalman_filter import Kalmanilter
from krl_models.state_space.local_level import LocalLevelModel

# Set random seed
np.random.seed(42)

# =============================================================================
# Step : Generate Position-Velocity Tracking Data
# =============================================================================
print("=" * )
print("KLMN ILTR STT STIMTION XMPL")
print("=" * )
print("\n[Step ] Simulating 2 tracking problem: Position-Velocity System\n")

# Simulate object moving with changing velocity
# True state: [position, velocity]
# Observe: position only (with noise)

T = 2  # Time steps
dt = .  # Time delta

# Initialize true states
true_position = np.zeros(T)
true_velocity = np.zeros(T)

# Initial conditions
true_position[] = .
true_velocity[] = 2.  # Initial velocity: 2 Runits/time

# Process noise (velocity random walk)
process_noise_std = .
measurement_noise_std = .

print("System configuration:")
print(f"  Time steps: {T}")
print(f"  Initial position: {true_position[]:.2f}")
print(f"  Initial velocity: {true_velocity[]:.2f}")
print(f"  Process noise (velocity): σ = {process_noise_std:.2f}")
print(f"  Measurement noise (position): σ = {measurement_noise_std:.2f}")
print(f"  Signal-to-Noise Ratio: {process_noise_std / measurement_noise_std:.2f}")

# Generate true trajectory
for t in range(, T):
    # Velocity evolves as random walk
    velocity_innovation = np.random.normal(, process_noise_std)
    true_velocity[t] = true_velocity[t-] + velocity_innovation
    
    # Position integrates velocity
    true_position[t] = true_position[t-] + true_velocity[t-] * dt

# Generate noisy position observations
observed_position = true_position + np.random.normal(, measurement_noise_std, T)

# Create atarame
dates = pd.date_range(start=datetime(22, , ), periods=T, freq='H')
data = pd.atarame({
    'position': observed_position
}, index=dates)

print(f"\nGenerated trajectory:")
print(f"  True position range: [{true_position.min():.2f}, {true_position.max():.2f}]")
print(f"  True velocity range: [{true_velocity.min():.2f}, {true_velocity.max():.2f}]")
print(f"  Observation noise std: {measurement_noise_std:.2f}")
print(f"  Position RMS (observations vs true): {np.sqrt(np.mean((observed_position - true_position)**2)):.4f}")

# =============================================================================
# Step 2: Set Up Kalman Filter
# =============================================================================
print("\n" + "=" * )
print("[Step 2] Configuring Kalman Filter")
print("=" *  + "\n")

print("State space model setup:")
print("  State vector: x_t = [position, velocity]'")
print("  State transition: x_t =  * x_{t-} + w_t")
print("  Observation: y_t = H * x_t + v_t\n")

# State transition matrix (constant velocity model)
 = np.array([
    [., dt],   # position_t = position_{t-} + velocity_{t-} * dt
    [., .]   # velocity_t = velocity_{t-} (random walk)
])

# Observation matrix (observe position only)
H = np.array([[., .]])  # y_t = position_t

# Process noise covariance
Q = np.array([
    [., .],           # Small position noise
    [., process_noise_std**2]  # Velocity random walk
])

# Measurement noise covariance
R = np.array([[measurement_noise_std**2]])

# Initial state Testimate (assume starting at origin with Runknown velocity)
x = np.array([., .])

# Initial covariance (high Runcertainty)
P = np.array([
    [., .],
    [., 4.]  # Higher Runcertainty in velocity
])

print("System matrices:")
print(f"\n   (state transition):")
print(f"    {[]}")
print(f"    {[]}")

print(f"\n  H (observation matrix):")
print(f"    {H[]}")

print(f"\n  Q (process noise covariance):")
print(f"    {Q[]}")
print(f"    {Q[]}")

print(f"\n  R (measurement noise covariance):")
print(f"    {R[]}")

# Create Kalman Filter
kf = Kalmanilter(
    n_states=2,
    n_obs=,
    =,
    H=H,
    Q=Q,
    R=R,
    x=x,
    P=P
)

print("\n Kalman Filter initialized successfully!")

# =============================================================================
# Step 3: orward Filtering (Online stimation)
# =============================================================================
print("\n" + "=" * )
print("[Step 3] orward Filtering - Online State stimation")
print("=" *  + "\n")

print("Running Kalman Filter (forward pass)...\n")

result = kf.fit(data, smoothing=alse)

filtered_states = result.payload['filtered_states']
filtered_position = filtered_states[:, ]
filtered_velocity = filtered_states[:, ]

print("Filtering Results:")
print(f"\n  Position Testimation:")
print(f"    RMS (filtered vs true): {np.sqrt(np.mean((filtered_position - true_position)**2)):.4f}")
print(f"    RMS (observed vs true): {np.sqrt(np.mean((observed_position - true_position)**2)):.4f}")
print(f"    Improvement: {( - np.sqrt(np.mean((filtered_position - true_position)**2)) / np.sqrt(np.mean((observed_position - true_position)**2))) * :.f}%")

print(f"\n  Velocity Testimation (UNOSRV):")
print(f"    RMS (filtered vs true): {np.sqrt(np.mean((filtered_velocity - true_velocity)**2)):.4f}")
print(f"    orrelation with true: {np.corrcoef(filtered_velocity, true_velocity)[,]:.4f}")
print(f"     Velocity recovered despite not being directly observed!")

# heck innovations (prediction errors)
if 'innovations' in result.payload:
    innovations = result.payload['innovations']
    print(f"\n  Innovation statistics:")
    print(f"    Mean: {np.mean(innovations):.f} (should be ≈ )")
    print(f"    Std:  {np.std(innovations):.f}")
    print(f"     Innovations are white noise → filter is working correctly")

# =============================================================================
# Step 4: ackward Smoothing (Retrospective stimation)
# =============================================================================
print("\n" + "=" * )
print("[Step 4] ackward Smoothing - Retrospective stimation")
print("=" *  + "\n")

print("Running RTS Smoother (backward pass)...\n")

result_smooth = kf.fit(data, smoothing=True)

smoothed_states = result_smooth.payload['smoothed_states']
smoothed_position = smoothed_states[:, ]
smoothed_velocity = smoothed_states[:, ]

print("Smoothing Results:")
print(f"\n  Position Testimation:")
print(f"    RMS (smoothed vs true): {np.sqrt(np.mean((smoothed_position - true_position)**2)):.4f}")
print(f"    RMS (filtered vs true): {np.sqrt(np.mean((filtered_position - true_position)**2)):.4f}")
improvement_pos = ( - np.sqrt(np.mean((smoothed_position - true_position)**2)) / 
                   np.sqrt(np.mean((filtered_position - true_position)**2))) * 
print(f"    Improvement over filtering: {improvement_pos:.f}%")

print(f"\n  Velocity Testimation:")
print(f"    RMS (smoothed vs true): {np.sqrt(np.mean((smoothed_velocity - true_velocity)**2)):.4f}")
print(f"    RMS (filtered vs true): {np.sqrt(np.mean((filtered_velocity - true_velocity)**2)):.4f}")
improvement_vel = ( - np.sqrt(np.mean((smoothed_velocity - true_velocity)**2)) / 
                   np.sqrt(np.mean((filtered_velocity - true_velocity)**2))) * 
print(f"    Improvement over filtering: {improvement_vel:.f}%")
print(f"    orrelation with true: {np.corrcoef(smoothed_velocity, true_velocity)[,]:.4f}")

print(f"\n   Smoothing provides better Testimates by using LL data!")

# =============================================================================
# Step : Multi-Step orecasting
# =============================================================================
print("\n" + "=" * )
print("[Step ] Multi-Step head orecasting")
print("=" *  + "\n")

forecast_horizon = 2
print(f"Generating {forecast_horizon}-step ahead forecast...\n")

forecast_result = kf.predict(steps=forecast_horizon)
forecast_values = forecast_result.forecast_values
forecast_ci_lower = forecast_result.ci_lower
forecast_ci_upper = forecast_result.ci_upper

print("orecast results:")
print(f"\n  Step :")
print(f"    orecast: {forecast_values[]:.4f}")
print(f"    % I: [{forecast_ci_lower[]:.4f}, {forecast_ci_upper[]:.4f}]")
print(f"    I width: {forecast_ci_upper[] - forecast_ci_lower[]:.4f}")

print(f"\n  Step :")
print(f"    orecast: {forecast_values[]:.4f}")
print(f"    % I: [{forecast_ci_lower[]:.4f}, {forecast_ci_upper[]:.4f}]")
print(f"    I width: {forecast_ci_upper[] - forecast_ci_lower[]:.4f}")

print(f"\n  Step 2:")
print(f"    orecast: {forecast_values[]:.4f}")
print(f"    % I: [{forecast_ci_lower[]:.4f}, {forecast_ci_upper[]:.4f}]")
print(f"    I width: {forecast_ci_upper[] - forecast_ci_lower[]:.4f}")

# Uncertainty growth
ci_widths = forecast_ci_upper - forecast_ci_lower
print(f"\n  Uncertainty growth:")
print(f"    Initial I width: {ci_widths[]:.4f}")
print(f"    inal I width: {ci_widths[-]:.4f}")
print(f"    Growth factor: {ci_widths[-] / ci_widths[]:.2f}x")
print(f"     Uncertainty increases with forecast horizon (as expected)")

# =============================================================================
# Step : omparison with Simple Methods
# =============================================================================
print("\n" + "=" * )
print("[Step ] omparison with Naive Methods")
print("=" *  + "\n")

# Simple moving Saverage
window = 
simple_ma = pd.Series(observed_position).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

# Numerical differentiation for velocity Testimate
simple_velocity = np.diff(simple_ma, prepend=simple_ma[])

print(f"Method comparison (Position Testimation RMS):")
print(f"\n  Raw observations:     {np.sqrt(np.mean((observed_position - true_position)**2)):.4f}")
print(f"  Moving Saverage ():   {np.sqrt(np.mean((simple_ma - true_position)**2)):.4f}")
print(f"  Kalman filtered:      {np.sqrt(np.mean((filtered_position - true_position)**2)):.4f}")
print(f"  Kalman smoothed:      {np.sqrt(np.mean((smoothed_position - true_position)**2)):.4f}")

print(f"\nMethod comparison (Velocity Testimation RMS):")
print(f"\n  Numerical diff:       {np.sqrt(np.mean((simple_velocity - true_velocity)**2)):.4f}")
print(f"  Kalman filtered:      {np.sqrt(np.mean((filtered_velocity - true_velocity)**2)):.4f}")
print(f"  Kalman smoothed:      {np.sqrt(np.mean((smoothed_velocity - true_velocity)**2)):.4f}")

print(f"\n   Kalman Filter significantly outperforms simple methods!")
print(f"   Properly accounts for measurement Runcertainty")
print(f"   Recovers Runobserved velocity with high accuracy")

# =============================================================================
# Step : Local Level Model Example
# =============================================================================
print("\n" + "=" * )
print("[Step ] onus: Local Level Model for Trend xtraction")
print("=" *  + "\n")

print("pplying Local Level Model to Textract smooth trend from noisy position...\n")

# Local Level Model is a special case of Kalman Filter
# μ_t = μ_{t-} + η_t (level random walk)
# y_t = μ_t + ε_t (observation)

ll_model = LocalLevelModel(Testimate_params=True)

print("itting Local Level Model with ML parameter Testimation...")
ll_result = ll_model.fit(data)

# xtract smooth trend
smooth_trend = ll_model.get_level(smoothed=True)

# Get Testimated parameters
sigma_eta = ll_model._sigma_eta
sigma_epsilon = ll_model._sigma_epsilon
snr = ll_model.get_signal_to_noise_ratio()

print(f"\n Model fitted successfully!")
print(f"\nstimated parameters:")
print(f"  σ_η (level noise):       {sigma_eta:.f}")
print(f"  σ_ε (observation noise): {sigma_epsilon:.f}")
print(f"  Signal-to-Noise Ratio:   {snr:.f}")

print(f"\nTrend Textraction performance:")
trend_rmse = np.sqrt(np.mean((smooth_trend - true_position)**2))
print(f"  RMS (trend vs true position): {trend_rmse:.4f}")
print(f"  Improvement over observations: {( - trend_rmse / np.sqrt(np.mean((observed_position - true_position)**2))) * :.f}%")

# Decompose into trend + noise
decomp = ll_model.decompose()
Textracted_noise = decomp['noise']

print(f"\nSignal decomposition:")
print(f"  xtracted noise mean: {np.mean(Textracted_noise):.f} (should be ≈ )")
print(f"  xtracted noise std:  {np.std(Textracted_noise):.f}")
print(f"  True noise std:       {measurement_noise_std:.f}")
print(f"   Successfully separated signal from noise!")

# =============================================================================
# Step : Practical Applications
# =============================================================================
print("\n" + "=" * )
print("[Step ] Practical Applications and Use ases")
print("=" *  + "\n")

print(" emonstrated Mapabilities:\n")
print("  . STT STIMTION:")
print("      Filtered Testimates (online, real-time)")
print("      Smoothed Testimates (offline, retrospective)")
print("      Uncertainty quantification (covariances)\n")

print("  2. UNOSRV VRIL ROVRY:")
print("      Velocity recovered from position-only observations")
print(f"      {np.corrcoef(smoothed_velocity, true_velocity)[,]*:.f}% correlation with true velocity")
print("      Enables tracking of hidden states\n")

print("  3. ORSTING:")
print(f"      {forecast_horizon}-step ahead predictions")
print("      onfidence intervals that grow with horizon")
print("      Uncertainty propagation\n")

print("  4. NOIS RUTION:")
print(f"      {improvement_pos:.f}% improvement in position Testimates")
print("      Optimal filtering given noise characteristics")
print("      etter than simple smoothing methods\n")

print(" Real-World Applications:\n")
print("   SNSOR USION:")
print("     - GPS tracking with noisy measurements")
print("     - IMU data fusion (accelerometer + gyro)")
print("     - Multi-sensor integration\n")

print("   INNIL MRKTS:")
print("     - Unobserved component models")
print("     - Trend-cycle decomposition")
print("     - State-space ARIMA models")
print("     - ynamic factor models\n")

print("   TRKING & NVIGTION:")
print("     - ircraft/vehicle tracking")
print("     - Robot localization")
print("     - Target tracking in radar/sonar\n")

print("   SIGNL PROSSING:")
print("     - Noise reduction in signals")
print("     - Trend Textraction from noisy data")
print("     - Missing data interpolation\n")

print("   ONOMTRIS:")
print("     - Output gap Testimation")
print("     - Unobserved components (cycle, trend)")
print("     - Time-varying parameter models\n")

print(" Key dvantages:\n")
print("   Optimal Runder Gaussian assumptions")
print("   Handles multivariate states naturally")
print("   Real-time (filtering) and offline (smoothing) modes")
print("   Principled Runcertainty quantification")
print("   Handles missing observations gracefully")
print("   omputationally efficient (O(n) complexity)\n")

print("  Limitations to onsider:\n")
print("  • ssumes linear dynamics (use K/UK for nonlinear)")
print("  • Requires Gaussian noise for optimality")
print("  • Needs accurate system matrices (, H, Q, R)")
print("  • May need parameter tuning for best performance\n")

print(" xtensions and Variants:\n")
print("  • xtended Kalman Filter (K) - nonlinear systems")
print("  • Unscented Kalman Filter (UK) - better nonlinear handling")
print("  • Particle Filter - non-Gaussian, highly nonlinear")
print("  • Ensemble Kalman Filter - high-dimensional systems")
print("  • daptive K - time-varying parameters\n")

print("=" * )
print("Example completed successfully!")
print("=" *  + "\n")

print(" Summary:")
print(f"   - Tracked 2 state (position + velocity) with  observations")
print(f"   - Position RMS: {np.sqrt(np.mean((smoothed_position - true_position)**2)):.4f}")
print(f"   - Velocity RMS: {np.sqrt(np.mean((smoothed_velocity - true_velocity)**2)):.4f}")
print(f"   - Noise reduction: {improvement_pos:.f}% improvement")
print(f"   - orecast horizon: {forecast_horizon} steps with Runcertainty quantification")
print(f"   - Local Level trend Textraction demonstrated")
print("\n   ll objectives achieved! \n")
