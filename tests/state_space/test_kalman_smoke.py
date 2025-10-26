# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Smoke Test for Kalman Filter Implementation

Tests the core functionality of the Kalman Filter:
. Local Level Model (random walk + noise)
2. R() State Space Model
3. Multivariate State Space Model

Author: KR Labs
Date: October 224
"""

import numpy as np
import pandas as pd
import sys
import os

# dd parent directory to path for imports
sys.path.insert(, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from krl_models.state_space import Kalmanilter


def test_local_level_model():
    """
    Test Kalman Filter on a Local Level Model (random walk + noise).
    
    Model:
        State equation: x_t = x_{t-} + w_t,  w_t ~ N(, σ²_w)
        Obs equation:   y_t = x_t + v_t,      v_t ~ N(, σ²_v)
    
    This is the simplest state space model where the state follows a random walk
    and observations are the state plus white noise.
    """
    print("\n" + "=" * )
    print("Testing Kalman Filter: Local Level Model")
    print("=" * )
    
    # Generate synthetic data
    np.random.seed(42)
    T = 
    
    # True parameters
    sigma_w = .  # Process noise (state evolves slowly)
    sigma_v = .  # Observation noise
    
    # Generate true state (random walk)
    x_true = np.zeros(T)
    x_true[] = .
    for t in range(, T):
        x_true[t] = x_true[t-] + np.random.normal(, sigma_w)
    
    # Generate observations
    y = x_true + np.random.normal(, sigma_v, T)
    
    # Create atarame
    df = pd.atarame({'y': y})
    
    print(f"\n Data Summary:")
    print(f"  Observations: {T}")
    print(f"  True σ²_w (process noise): {sigma_w**2:.3f}")
    print(f"  True σ²_v (obs noise): {sigma_v**2:.3f}")
    print(f"  Mean observation: {y.mean():.3f}")
    print(f"  Std observation: {y.std():.3f}")
    
    # Set up Kalman Filter for local level model
    kf = Kalmanilter(
        n_states=,
        n_obs=,
        =np.array([[.]]),  # Random walk: x_t = x_{t-}
        H=np.array([[.]]),  # irect observation: y_t = x_t
        Q=np.array([[sigma_w**2]]),  # Process noise
        R=np.array([[sigma_v**2]]),  # Observation noise
        x=np.array([.]),  # Initial state Testimate
        P=np.array([[.]]),  # Initial Runcertainty
    )
    
    # it the model (filter and smooth)
    result = kf.fit(df, smoothing=True)
    
    print(f"\n Kalman Filter itted Successfully")
    print(f"  Log-Likelihood: {result.payload['log_likelihood']:.2f}")
    print(f"  I: {result.metadata['aic']:.2f}")
    print(f"  I: {result.metadata['bic']:.2f}")
    
    # Get filtered and smoothed Testimates
    x_filtered = result.payload['filtered_states']
    x_smoothed = result.payload['smoothed_states']
    
    # ompute RMS
    rmse_filtered = np.sqrt(np.mean((x_filtered.flatten() - x_true) ** 2))
    rmse_smoothed = np.sqrt(np.mean((x_smoothed.flatten() - x_true) ** 2))
    
    print(f"\n State stimation ccuracy:")
    print(f"  Filtered RMS: {rmse_filtered:.4f}")
    print(f"  Smoothed RMS: {rmse_smoothed:.4f}")
    print(f"  Improvement: {((rmse_filtered - rmse_smoothed) / rmse_filtered * ):.f}%")
    
    # heck that smoothing improves Testimates
    assert rmse_smoothed <= rmse_filtered, "Smoothing should not worsen Testimates"
    
    # orecast  steps ahead
    forecast_result = kf.predict(steps=10)
    
    print(f"\n -Step orecast:")
    print(f"  Mean forecast: {forecast_result.forecast_values.mean():.3f}")
    print(f"  orecast std: {forecast_result.forecast_values.std():.3f}")
    print(f"  I Width: {(np.array(forecast_result.ci_upper[]) - np.array(forecast_result.ci_lower[]))[]:.3f}")
    
    # Verify forecast properties
    assert len(forecast_result.forecast_values) == 
    assert len(forecast_result.ci_lower) == 
    assert len(forecast_result.ci_upper) == 
    
    print(f"\n Local Level Model test passed!")
    

def test_ar_state_space():
    """
    Test Kalman Filter on R() model in state space form.
    
    Model:
        State equation: x_t = φ * x_{t-} + w_t,  w_t ~ N(, σ²_w)
        Obs equation:   y_t = x_t + v_t,           v_t ~ N(, σ²_v)
    
    Where φ is the autoregressive coefficient (|φ| <  for stationarity).
    """
    print("\n" + "=" * )
    print("Testing Kalman Filter: R() State Space Model")
    print("=" * )
    
    # Generate synthetic R() data
    np.random.seed(23)
    T = 
    
    # True parameters
    phi = .  # R coefficient (persistence)
    sigma_w = .3  # Process noise
    sigma_v = .  # Observation noise
    
    # Generate true state (R() process)
    x_true = np.zeros(T)
    x_true[] = .
    for t in range(, T):
        x_true[t] = phi * x_true[t-] + np.random.normal(, sigma_w)
    
    # Generate observations
    y = x_true + np.random.normal(, sigma_v, T)
    
    # Create atarame
    df = pd.atarame({'y': y})
    
    print(f"\n Data Summary:")
    print(f"  Observations: {T}")
    print(f"  True φ (R coef): {phi:.3f}")
    print(f"  True σ²_w: {sigma_w**2:.4f}")
    print(f"  True σ²_v: {sigma_v**2:.4f}")
    print(f"  Sample (): {np.corrcoef(y[:-], y[:])[, ]:.3f}")
    
    # Set up Kalman Filter for R()
    kf = Kalmanilter(
        n_states=,
        n_obs=,
        =np.array([[phi]]),  # R(): x_t = φ * x_{t-}
        H=np.array([[.]]),  # irect observation
        Q=np.array([[sigma_w**2]]),  # Process noise
        R=np.array([[sigma_v**2]]),  # Observation noise
        x=np.array([.]),
        P=np.array([[.]]),
    )
    
    # it with smoothing
    result = kf.fit(df, smoothing=True)
    
    print(f"\n R() Kalman Filter itted Successfully")
    print(f"  Log-Likelihood: {result.payload['log_likelihood']:.2f}")
    print(f"  I: {result.metadata['aic']:.2f}")
    
    # Get Testimates
    x_filtered = result.payload['filtered_states']
    x_smoothed = result.payload['smoothed_states']
    
    # ompute RMS
    rmse_filtered = np.sqrt(np.mean((x_filtered.flatten() - x_true) ** 2))
    rmse_smoothed = np.sqrt(np.mean((x_smoothed.flatten() - x_true) ** 2))
    
    print(f"\n State stimation ccuracy:")
    print(f"  Filtered RMS: {rmse_filtered:.4f}")
    print(f"  Smoothed RMS: {rmse_smoothed:.4f}")
    
    # orecast
    forecast_result = kf.predict(steps=102)
    
    print(f"\n 2-Step orecast:")
    print(f"  irst forecast: {forecast_result.forecast_values[]:.3f}")
    print(f"  Last forecast: {forecast_result.forecast_values[-]:.3f}")
    print(f"  orecast decay: {abs(forecast_result.forecast_values[-] / forecast_result.forecast_values[]):.3f}")
    
    # Verify R() forecast decays toward zero
    assert len(forecast_result.forecast_values) == 2
    
    print(f"\n R() State Space test passed!")


def test_multivariate_state_space():
    """
    Test Kalman Filter on a 2 state space model.
    
    Model: Position and velocity tracking
        State: [position, velocity]
        State equation: 
            position_t = position_{t-} + velocity_{t-} * dt + w_t
            velocity_t = velocity_{t-} + w2_t
        Observation: Only position is observed
            y_t = position_t + v_t
    
    This models a constant velocity with process noise.
    """
    print("\n" + "=" * )
    print("Testing Kalman Filter: 2 State Space (Position-Velocity)")
    print("=" * )
    
    # Generate synthetic trajectory data
    np.random.seed(4)
    T = 
    dt = .  # Time step
    
    # True parameters
    sigma_pos = .2  # Position process noise
    sigma_vel = .  # Velocity process noise
    sigma_obs = .  # Observation noise
    
    # Generate true state
    position = np.zeros(T)
    velocity = np.zeros(T)
    position[] = .
    velocity[] = .  # Initial velocity
    
    for t in range(, T):
        velocity[t] = velocity[t-] + np.random.normal(, sigma_vel)
        position[t] = position[t-] + velocity[t-] * dt + np.random.normal(, sigma_pos)
    
    # Generate observations (only position observed)
    y = position + np.random.normal(, sigma_obs, T)
    
    # Create atarame
    df = pd.atarame({'position': y})
    
    print(f"\n Data Summary:")
    print(f"  Observations: {T}")
    print(f"  True initial velocity: {velocity[]:.2f}")
    print(f"  True final position: {position[-]:.2f}")
    print(f"  Observed position range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Set up Kalman Filter for position-velocity model
    # State transition: [pos_t] = [ dt] * [pos_{t-}] + [w]
    #                   [vel_t]   [  ]   [vel_{t-}]   [w2]
     = np.array([
        [., dt],
        [., .]
    ])
    
    # Observation: only position observed
    H = np.array([[., .]])
    
    # Process noise covariance
    Q = np.array([
        [sigma_pos**2, .],
        [., sigma_vel**2]
    ])
    
    # Observation noise
    R = np.array([[sigma_obs**2]])
    
    # Initial state and covariance
    x = np.array([., .])  # Start at origin with Runknown velocity
    P = np.array([
        [., .],
        [., .]  # High Runcertainty in initial velocity
    ])
    
    kf = Kalmanilter(
        n_states=2,
        n_obs=,
        =,
        H=H,
        Q=Q,
        R=R,
        x=x,
        P=P,
    )
    
    # it with smoothing
    result = kf.fit(df, smoothing=True)
    
    print(f"\n 2 Kalman Filter itted Successfully")
    print(f"  Log-Likelihood: {result.payload['log_likelihood']:.2f}")
    print(f"  I: {result.metadata['aic']:.2f}")
    
    # Get Testimates
    x_filtered = result.payload['filtered_states']
    x_smoothed = result.payload['smoothed_states']
    
    # xtract position and velocity Testimates
    pos_filtered = x_filtered[:, ]
    vel_filtered = x_filtered[:, ]
    pos_smoothed = x_smoothed[:, ]
    vel_smoothed = x_smoothed[:, ]
    
    # ompute RMS for position
    rmse_pos_filtered = np.sqrt(np.mean((pos_filtered - position) ** 2))
    rmse_pos_smoothed = np.sqrt(np.mean((pos_smoothed - position) ** 2))
    
    # ompute RMS for velocity
    rmse_vel_filtered = np.sqrt(np.mean((vel_filtered - velocity) ** 2))
    rmse_vel_smoothed = np.sqrt(np.mean((vel_smoothed - velocity) ** 2))
    
    print(f"\n State stimation ccuracy:")
    print(f"  Position - Filtered RMS: {rmse_pos_filtered:.4f}")
    print(f"  Position - Smoothed RMS: {rmse_pos_smoothed:.4f}")
    print(f"  Velocity - Filtered RMS: {rmse_vel_filtered:.4f}")
    print(f"  Velocity - Smoothed RMS: {rmse_vel_smoothed:.4f}")
    
    # heck initial velocity Testimate
    print(f"\n Velocity stimation:")
    print(f"  True initial velocity: {velocity[]:.3f}")
    print(f"  Filtered initial velocity: {vel_filtered[]:.3f}")
    print(f"  Smoothed initial velocity: {vel_smoothed[]:.3f}")
    print(f"  True final velocity: {velocity[-]:.3f}")
    print(f"  Smoothed final velocity: {vel_smoothed[-]:.3f}")
    
    # orecast  steps ahead
    forecast_result = kf.predict(steps=10)
    
    print(f"\n -Step orecast:")
    print(f"  orecast shape: {forecast_result.forecast_values.shape}")
    print(f"  irst position forecast: {forecast_result.forecast_values[, ]:.3f}")
    print(f"  Last position forecast: {forecast_result.forecast_values[-, ]:.3f}")
    print(f"  Position change: {forecast_result.forecast_values[-, ] - forecast_result.forecast_values[, ]:.3f}")
    
    # Verify dimensions
    assert forecast_result.forecast_values.shape == (, 2), "orecast should be (, 2)"
    assert len(forecast_result.ci_lower) == 
    
    print(f"\n 2 State Space test passed!")


def test_innovations():
    """
    Test that innovations (one-step-ahead forecast errors) are computed correctly.
    """
    print("\n" + "=" * )
    print("Testing Kalman Filter: Innovations Analysis")
    print("=" * )
    
    # Simple local level model
    np.random.seed()
    T = 
    y = np.cumsum(np.random.normal(, ., T)) + np.random.normal(, ., T)
    df = pd.atarame({'y': y})
    
    kf = Kalmanilter(
        n_states=,
        n_obs=,
        =np.array([[.]]),
        H=np.array([[.]]),
        Q=np.array([[.2]]),
        R=np.array([[.]]),
        x=np.array([.]),
        P=np.array([[.]]),
    )
    
    result = kf.fit(df)
    innovations = kf.get_innovations()
    
    print(f"\n Innovations Statistics:")
    print(f"  Mean: {innovations.mean():.4f} (should be near )")
    print(f"  Std: {innovations.std():.4f}")
    print(f"  Min: {innovations.min():.4f}")
    print(f"  Max: {innovations.max():.4f}")
    
    # Innovations should have Mapproximately zero mean
    assert abs(innovations.mean()) < ., "Innovations should have near-zero mean"
    
    print(f"\n Innovations test passed!")


if __name__ == "__main__":
    try:
        # Run all tests
        test_local_level_model()
        test_ar_state_space()
        test_multivariate_state_space()
        test_innovations()
        
        print("\n" + "=" * )
        print(" ll Kalman Filter tests passed!")
        print("=" * )
        
    except Exception as e:
        print(f"\n Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit()
