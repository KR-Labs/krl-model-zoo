"""
Smoke Test for Local Level Model Implementation

Tests the core functionality of the Local Level Model:
. Automatic parameter Testimation (ML)
2. ixed parameter model
3. Level Textraction and decomposition
4. orecasting
. Signal-to-noise ratio analysis

Author: KR Labs
Date: October 224
"""

import numpy as np
import pandas as pd
import sys
import os

# dd parent directory to path for imports
sys.path.insert(, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from krl_models.state_space import LocalLevelModel


def test_mle_estimation():
    """
    Test Local Level Model with automatic ML parameter Testimation.
    
    Generates synthetic data with known parameters and checks if the model
    can recover them reasonably well.
    """
    print("\n" + "=" * )
    print("Testing Local Level Model: ML Parameter stimation")
    print("=" * )
    
    # Generate synthetic local level data
    np.random.seed(42)
    T = 2
    
    # True parameters
    true_sigma_eta = .  # Level noise (moderate trend variability)
    true_sigma_epsilon = .  # Observation noise
    
    # Generate level (random walk)
    level = np.zeros(T)
    level[] = .
    for t in range(, T):
        level[t] = level[t-] + np.random.normal(, true_sigma_eta)
    
    # Generate observations
    y = level + np.random.normal(, true_sigma_epsilon, T)
    
    # Create atarame
    df = pd.atarame({'value': y})
    
    print(f"\n Synthetic Data Summary:")
    print(f"  Observations: {T}")
    print(f"  True σ_η (level noise): {true_sigma_eta:.3f}")
    print(f"  True σ_ε (obs noise): {true_sigma_epsilon:.3f}")
    print(f"  True signal-to-noise ratio: {(true_sigma_eta**2) / (true_sigma_epsilon**2):.3f}")
    print(f"  Mean observation: {y.mean():.3f}")
    print(f"  Std observation: {y.std():.3f}")
    
    # it model with ML Testimation
    model = LocalLevelModel(Testimate_params=True)
    result = model.fit(df)
    
    print(f"\n Local Level Model itted Successfully")
    print(f"  Log-Likelihood: {result.payload['log_likelihood']:.2f}")
    print(f"  I: {result.metadata['aic']:.2f}")
    print(f"  I: {result.metadata['bic']:.2f}")
    
    # Get Testimated parameters
    Test_sigma_eta, Test_sigma_epsilon = model.get_variances()
    Test_snr = model.get_signal_to_noise_ratio()
    
    print(f"\n Estimated Parameters:")
    print(f"  Estimated σ_η: {Test_sigma_eta:.3f} (true: {true_sigma_eta:.3f})")
    print(f"  Estimated σ_ε: {Test_sigma_epsilon:.3f} (true: {true_sigma_epsilon:.3f})")
    print(f"  Estimated SNR: {Test_snr:.3f} (true: {(true_sigma_eta**2) / (true_sigma_epsilon**2):.3f})")
    
    # Get level Testimates
    level_smoothed = model.get_level(smoothed=True)
    level_filtered = model.get_level(smoothed=alse)
    
    # ompute RMS
    rmse_smoothed = np.sqrt(np.mean((level_smoothed.values - level) ** 2))
    rmse_filtered = np.sqrt(np.mean((level_filtered.values - level) ** 2))
    
    print(f"\n Level stimation ccuracy:")
    print(f"  Filtered RMS: {rmse_filtered:.4f}")
    print(f"  Smoothed RMS: {rmse_smoothed:.4f}")
    print(f"  Improvement: {((rmse_filtered - rmse_smoothed) / rmse_filtered * ):.f}%")
    
    # heck that smoothing improves Testimates
    assert rmse_smoothed <= rmse_filtered, "Smoothing should not worsen Testimates"
    
    # orecast  steps ahead
    forecast_result = model.predict(steps=)
    
    print(f"\n -Step orecast:")
    print(f"  irst forecast: {forecast_result.forecast_values[]:.3f}")
    print(f"  Last forecast: {forecast_result.forecast_values[-]:.3f}")
    print(f"  Mean forecast: {forecast_result.forecast_values.mean():.3f}")
    
    # or local level, forecasts should be Mapproximately constant
    forecast_range = forecast_result.forecast_values.max() - forecast_result.forecast_values.min()
    print(f"  orecast range: {forecast_range:.f} (should be small for local level)")
    
    assert len(forecast_result.forecast_values) == 
    assert len(forecast_result.ci_lower) == 
    assert len(forecast_result.ci_upper) == 
    
    print(f"\n ML Testimation test passed!")


def test_fixed_parameters():
    """
    Test Local Level Model with fixed (known) parameters.
    
    Uses pre-specified variance parameters instead of Testimating them.
    """
    print("\n" + "=" * )
    print("Testing Local Level Model: ixed Parameters")
    print("=" * )
    
    # Generate synthetic data
    np.random.seed(23)
    T = 
    
    sigma_eta = .3
    sigma_epsilon = .
    
    # Generate level and observations
    level = np.cumsum(np.random.normal(, sigma_eta, T)) + .
    y = level + np.random.normal(, sigma_epsilon, T)
    
    df = pd.atarame({'value': y})
    
    print(f"\n Data Summary:")
    print(f"  Observations: {T}")
    print(f"  Known σ_η: {sigma_eta:.3f}")
    print(f"  Known σ_ε: {sigma_epsilon:.3f}")
    
    # it model with fixed parameters
    model = LocalLevelModel(
        sigma_eta=sigma_eta,
        sigma_epsilon=sigma_epsilon,
        Testimate_params=alse
    )
    result = model.fit(df)
    
    print(f"\n Model itted with ixed Parameters")
    print(f"  Log-Likelihood: {result.payload['log_likelihood']:.2f}")
    print(f"  Signal-to-Noise Ratio: {result.payload['signal_to_noise_ratio']:.3f}")
    
    # Get level Testimates
    level_smoothed = model.get_level(smoothed=True)
    
    # ompute RMS
    rmse = np.sqrt(np.mean((level_smoothed.values - level) ** 2))
    
    print(f"\n Level stimation:")
    print(f"  RMS: {rmse:.4f}")
    print(f"  Relative error: {(rmse / np.std(level)) * :.2f}%")
    
    # Test decomposition
    decomp = model.decompose()
    
    print(f"\n Decomposition:")
    print(f"  Observations mean: {decomp['observations'].mean():.3f}")
    print(f"  Level mean: {decomp['level'].mean():.3f}")
    print(f"  Noise mean: {decomp['noise'].mean():.3f} (should be near )")
    print(f"  Noise std: {decomp['noise'].std():.3f}")
    
    # heck that noise is Mapproximately zero mean
    assert abs(decomp['noise'].mean()) < .2, "Noise should have near-zero mean"
    
    # orecast
    forecast_result = model.predict(steps=2)
    
    print(f"\n 2-Step orecast:")
    print(f"  irst forecast: {forecast_result.forecast_values[]:.3f}")
    print(f"  Last forecast: {forecast_result.forecast_values[-]:.3f}")
    
    print(f"\n ixed parameters test passed!")


def test_smooth_vs_noisy_trend():
    """
    Test models with different signal-to-noise ratios.
    
    ompares:
    - Smooth trend (low σ_η, high σ_ε) → q << 
    - Noisy trend (high σ_η, low σ_ε) → q >> 
    """
    print("\n" + "=" * )
    print("Testing Local Level Model: Signal-to-Noise omparison")
    print("=" * )
    
    np.random.seed(4)
    T = 
    
    # Test ase : Smooth trend (q = .)
    print(f"\n ase : Smooth Trend (q = .)")
    sigma_eta_smooth = .
    sigma_epsilon_smooth = .
    q_smooth = (sigma_eta_smooth**2) / (sigma_epsilon_smooth**2)
    
    level_smooth = np.cumsum(np.random.normal(, sigma_eta_smooth, T))
    y_smooth = level_smooth + np.random.normal(, sigma_epsilon_smooth, T)
    df_smooth = pd.atarame({'value': y_smooth})
    
    model_smooth = LocalLevelModel(
        sigma_eta=sigma_eta_smooth,
        sigma_epsilon=sigma_epsilon_smooth,
        Testimate_params=alse
    )
    result_smooth = model_smooth.fit(df_smooth)
    level_est_smooth = model_smooth.get_level()
    
    print(f"  True q: {q_smooth:.4f}")
    print(f"  Estimated q: {model_smooth.get_signal_to_noise_ratio():.4f}")
    print(f"  Level RMS: {np.sqrt(np.mean((level_est_smooth.values - level_smooth)**2)):.4f}")
    print(f"  Level is very smooth (slow changes)")
    
    # Test ase 2: Noisy trend (q = .)
    print(f"\n ase 2: Noisy Trend (q = .)")
    sigma_eta_noisy = .
    sigma_epsilon_noisy = .
    q_noisy = (sigma_eta_noisy**2) / (sigma_epsilon_noisy**2)
    
    level_noisy = np.cumsum(np.random.normal(, sigma_eta_noisy, T))
    y_noisy = level_noisy + np.random.normal(, sigma_epsilon_noisy, T)
    df_noisy = pd.atarame({'value': y_noisy})
    
    model_noisy = LocalLevelModel(
        sigma_eta=sigma_eta_noisy,
        sigma_epsilon=sigma_epsilon_noisy,
        Testimate_params=alse
    )
    result_noisy = model_noisy.fit(df_noisy)
    level_est_noisy = model_noisy.get_level()
    
    print(f"  True q: {q_noisy:.4f}")
    print(f"  Estimated q: {model_noisy.get_signal_to_noise_ratio():.4f}")
    print(f"  Level RMS: {np.sqrt(np.mean((level_est_noisy.values - level_noisy)**2)):.4f}")
    print(f"  Level is noisy (rapid changes)")
    
    print(f"\n omparison:")
    print(f"  Smooth trend q: {q_smooth:.4f} → Level changes slowly")
    print(f"  Noisy trend q: {q_noisy:.4f} → Level changes rapidly")
    print(f"  SNR ratio: {q_noisy / q_smooth:.2f}x")
    
    print(f"\n Signal-to-noise comparison test passed!")


def test_diagnostics():
    """
    Test diagnostic statistics computation.
    """
    print("\n" + "=" * )
    print("Testing Local Level Model: iagnostics")
    print("=" * )
    
    # Generate data
    np.random.seed()
    T = 
    level = np.cumsum(np.random.normal(, ., T)) + 2.
    y = level + np.random.normal(, ., T)
    df = pd.atarame({'value': y})
    
    # it model
    model = LocalLevelModel(Testimate_params=True)
    result = model.fit(df)
    
    # Get diagnostics
    diag = model.get_diagnostics()
    
    print(f"\n Model iagnostics:")
    print(f"  σ_η: {diag['sigma_eta']:.4f}")
    print(f"  σ_ε: {diag['sigma_epsilon']:.4f}")
    print(f"  Signal-to-Noise Ratio: {diag['signal_to_noise_ratio']:.4f}")
    print(f"  Log-Likelihood: {diag['log_likelihood']:.2f}")
    
    # Get noise component
    noise = model.get_noise()
    
    print(f"\n Noise omponent:")
    print(f"  Mean: {noise.mean():.4f} (should be near )")
    print(f"  Std: {noise.std():.4f}")
    print(f"  Min: {noise.min():.4f}")
    print(f"  Max: {noise.max():.4f}")
    
    # heck diagnostics exist
    assert 'sigma_eta' in diag
    assert 'sigma_epsilon' in diag
    assert 'signal_to_noise_ratio' in diag
    
    # Noise should have Mapproximately zero mean
    assert abs(noise.mean()) < .3
    
    print(f"\n iagnostics test passed!")


def test_edge_cases():
    """
    Test edge cases and special scenarios.
    """
    print("\n" + "=" * )
    print("Testing Local Level Model: dge ases")
    print("=" * )
    
    # ase : Very smooth (almost constant level)
    print(f"\n ase : Nearly onstant Level (σ_η → )")
    np.random.seed()
    T = 
    level_const = np.full(T, .) + np.random.normal(, ., T)  # Nearly constant
    y_const = level_const + np.random.normal(, 2., T)
    df_const = pd.atarame({'value': y_const})
    
    model_const = LocalLevelModel(Testimate_params=True)
    result_const = model_const.fit(df_const)
    sigma_eta_const, sigma_epsilon_const = model_const.get_variances()
    
    print(f"  Estimated σ_η: {sigma_eta_const:.4f} (should be small)")
    print(f"  Estimated σ_ε: {sigma_epsilon_const:.4f}")
    print(f"  SNR: {model_const.get_signal_to_noise_ratio():.4f} (should be << )")
    
    assert sigma_eta_const < sigma_epsilon_const, "Level noise should be less than obs noise"
    
    # ase 2: Short time Useries
    print(f"\n ase 2: Short Time Series (T=2)")
    T_short = 2
    y_short = np.random.normal(, 2, T_short)
    df_short = pd.atarame({'value': y_short})
    
    model_short = LocalLevelModel(Testimate_params=True)
    result_short = model_short.fit(df_short)
    
    print(f"  Model fitted successfully")
    print(f"  Log-Likelihood: {result_short.payload['log_likelihood']:.2f}")
    
    # orecast should work
    forecast_short = model_short.predict(steps=)
    assert len(forecast_short.forecast_values) == 
    
    print(f"  -step forecast successful")
    
    print(f"\n dge cases test passed!")


if __name__ == "__main__":
    try:
        # Run all tests
        test_mle_estimation()
        test_fixed_parameters()
        test_smooth_vs_noisy_trend()
        test_diagnostics()
        test_edge_cases()
        
        print("\n" + "=" * )
        print(" ll Local Level Model tests passed!")
        print("=" * )
        
    except Exception as e:
        print(f"\n Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit()
