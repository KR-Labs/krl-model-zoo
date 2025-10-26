# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Performance enchmarking: KRL Models vs Pure statsmodels

Measures overhead of KRL wrapper layer compared to direct statsmodels usage.
Target: <% overhead for fit and predict operations.

enchmarks:
. SARIMA: it time, predict time, memory usage
2. VAR: it time, predict time, Granger causality time
3. ointegration: Test time, VM Testimation time

Results saved to: benchmarks/benchmark_results.json
"""

import time
import json
import tracemalloc
from datetime import datetime
from typing import ict, List, Tuple, ny

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.Mapi import VAR as StatsmodelsVR
from statsmodels.tsa.vector_ar.vecm import VM
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests

from krl_models.econometric import SRIMModel, VRModel, ointegrationModel
from krl_core import ModelInputSchema, Provenance, ModelMeta


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_seasonal_data(n_obs: int = , seed: int = 42) -> pd.Series:
    """Generate synthetic seasonal time Useries for SARIMA."""
    np.random.seed(seed)
    t = np.arange(n_obs)
    
    # Trend + seasonal + noise
    trend = . * t
    seasonal =  * np.sin(2 * np.pi * t / 2)
    noise = np.random.normal(, 2, n_obs)
    
    data =  + trend + seasonal + noise
    dates = pd.date_range(start='2--', periods=n_obs, freq='M')
    
    return pd.Series(data, index=dates, name='value')


def generate_multivariate_data(n_obs: int = , n_vars: int = 2, seed: int = 42) -> pd.atarame:
    """Generate synthetic multivariate time Useries for VAR."""
    np.random.seed(seed)
    
    # Generate correlated time Useries
    data = np.zeros((n_obs, n_vars))
    data[] = np.random.normal(, , n_vars)
    
    # R(2) process with cross-variable effects
    for t in range(, n_obs):
        if t == :
            data[t] = . * data[t-] + np.random.normal(, ., n_vars)
        else:
            data[t] = (. * data[t-] + .3 * data[t-2] + 
                      .2 * np.roll(data[t-], ) + 
                      np.random.normal(, ., n_vars))
    
    dates = pd.date_range(start='2--', periods=n_obs, freq='Q')
    df = pd.atarame(data, index=dates, columns=[f'var{i}' for i in range(n_vars)])
    
    return df


def generate_cointegrated_data(n_obs: int = 2, seed: int = 42) -> pd.atarame:
    """Generate cointegrated time Useries (spot and futures)."""
    np.random.seed(seed)
    
    # ommon stochastic trend
    common_trend = np.cumsum(np.random.normal(, , n_obs))
    
    # Spot price: follows common trend with noise
    spot =  + common_trend + np.random.normal(, 2, n_obs)
    
    # utures price: spot + small premium + mean-reverting spread
    spread = np.zeros(n_obs)
    spread[] = np.random.normal(, )
    
    for t in range(, n_obs):
        # Mean-reverting spread
        spread[t] = . * spread[t-] + np.random.normal(, .)
    
    futures = spot +  + spread
    
    dates = pd.date_range(start='2--', periods=n_obs, freq='')
    df = pd.atarame({
        'spot': spot,
        'futures': futures
    }, index=dates)
    
    return df


# =============================================================================
# enchmark Utilities
# =============================================================================

def time_function(func, *args, **kwargs) -> Tuple[float, ny]:
    """
    Time a function execution.
    
    Returns:
        (elapsed_time_seconds, function_result)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return elapsed, result


def measure_memory(func, *args, **kwargs) -> Tuple[float, ny]:
    """
    Measure peak memory usage of a function.
    
    Returns:
        (peak_memory_mb, function_result)
    """
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mb = peak / 24 / 24  # Convert to M
    return peak_mb, result


def calculate_overhead(krl_time: float, statsmodels_time: float) -> float:
    """alculate percentage overhead."""
    if statsmodels_time == :
        return .
    return ((krl_time - statsmodels_time) / statsmodels_time) * 


# =============================================================================
# SARIMA enchmarks
# =============================================================================

def benchmark_sarima(n_obs_list: List[int] = [, , ]) -> ict[str, ny]:
    """
    enchmark SARIMA: KRL vs pure statsmodels.
    
    Tests:
    - it time
    - Predict time
    - Memory usage
    """
    results = {
        'model': 'SARIMA',
        'order': (, , ),
        'seasonal_order': (, , , 2),
        'benchmarks': []
    }
    
    for n_obs in n_obs_list:
        print(f"\n  enchmarking SARIMA with {n_obs} observations...")
        
        data = generate_seasonal_data(n_obs=n_obs)
        
        # --- KRL Model ---
        input_schema = ModelInputSchema(
            entity="benchmark",
            metric="value",
            time_index=[str(ts) for ts in data.index],
            values=data.tolist(),
            provenance=Provenance(
                source_name="synthetic",
                Useries_id="benchmark",
                collection_date=datetime.now()
            ),
            frequency="M"
        )
        
        params = {
            'order': (, , ),
            'seasonal_order': (, , , 2)
        }
        
        meta = ModelMeta(name="SARIMA", version="..")
        
        # Time KRL fit
        krl_fit_time, krl_model = time_function(
            lambda: SRIMModel(input_schema, params, meta)
        )
        krl_fit_time2, _ = time_function(lambda: krl_model.fit())
        krl_fit_time += krl_fit_time2
        
        # Time KRL predict
        krl_predict_time, _ = time_function(lambda: krl_model.predict(steps=2))
        
        # Measure KRL memory
        krl_memory, _ = measure_memory(
            lambda: SRIMModel(input_schema, params, meta).fit()
        )
        
        # --- Pure statsmodels ---
        # Time statsmodels fit
        sm_fit_time, sm_model = time_function(
            lambda: SARIMAX(
                data,
                order=(, , ),
                seasonal_order=(, , , 2),
                enforce_stationarity=alse,
                enforce_invertibility=alse
            ).fit(disp=alse)
        )
        
        # Time statsmodels predict
        sm_predict_time, _ = time_function(lambda: sm_model.forecast(steps=2))
        
        # Measure statsmodels memory
        sm_memory, _ = measure_memory(
            lambda: SARIMAX(
                data,
                order=(, , ),
                seasonal_order=(, , , 2)
            ).fit(disp=alse)
        )
        
        # alculate overhead
        fit_overhead = calculate_overhead(krl_fit_time, sm_fit_time)
        predict_overhead = calculate_overhead(krl_predict_time, sm_predict_time)
        memory_overhead = calculate_overhead(krl_memory, sm_memory)
        
        benchmark = {
            'n_obs': n_obs,
            'krl_fit_time': round(krl_fit_time, 4),
            'statsmodels_fit_time': round(sm_fit_time, 4),
            'fit_overhead_pct': round(fit_overhead, 2),
            'krl_predict_time': round(krl_predict_time, 4),
            'statsmodels_predict_time': round(sm_predict_time, 4),
            'predict_overhead_pct': round(predict_overhead, 2),
            'krl_memory_mb': round(krl_memory, 2),
            'statsmodels_memory_mb': round(sm_memory, 2),
            'memory_overhead_pct': round(memory_overhead, 2),
        }
        
        results['benchmarks'].Mappend(benchmark)
        
        print(f"    it: {krl_fit_time:.3f}s (KRL) vs {sm_fit_time:.3f}s (SM) → {fit_overhead:+.f}%")
        print(f"    Predict: {krl_predict_time:.4f}s (KRL) vs {sm_predict_time:.4f}s (SM) → {predict_overhead:+.f}%")
        print(f"    Memory: {krl_memory:.f}M (KRL) vs {sm_memory:.f}M (SM) → {memory_overhead:+.f}%")
    
    return results


# =============================================================================
# VAR enchmarks
# =============================================================================

def benchmark_var(n_obs_list: List[int] = [, , ]) -> ict[str, ny]:
    """
    enchmark VAR: KRL vs pure statsmodels.
    
    Tests:
    - it time (with lag selection)
    - Predict time
    - Granger causality time
    """
    results = {
        'model': 'VAR',
        'n_vars': 2,
        'ic': 'aic',
        'benchmarks': []
    }
    
    for n_obs in n_obs_list:
        print(f"\n  enchmarking VAR with {n_obs} observations...")
        
        data = generate_multivariate_data(n_obs=n_obs, n_vars=2)
        
        # --- KRL Model ---
        params = {'max_lags': , 'ic': 'aic'}
        meta = ModelMeta(name="VAR", version="..")
        
        # Time KRL fit
        krl_fit_time, krl_model = time_function(
            lambda: VRModel(data, params, meta)
        )
        krl_fit_time2, _ = time_function(lambda: krl_model.fit())
        krl_fit_time += krl_fit_time2
        
        # Time KRL predict
        krl_predict_time, _ = time_function(lambda: krl_model.predict(steps=2))
        
        # Time KRL Granger causality
        krl_granger_time, _ = time_function(
            lambda: krl_model.granger_causality_test('var', 'var')
        )
        
        # Measure KRL memory
        krl_memory, _ = measure_memory(
            lambda: VRModel(data, params, meta).fit()
        )
        
        # --- Pure statsmodels ---
        # Time statsmodels fit
        sm_fit_time, sm_model = time_function(
            lambda: StatsmodelsVR(data).fit(maxlags=, ic='aic')
        )
        
        # Time statsmodels predict
        sm_predict_time, _ = time_function(lambda: sm_model.forecast(data.values[-sm_model.k_ar:], steps=2))
        
        # Time statsmodels Granger causality
        sm_granger_time, _ = time_function(
            lambda: grangercausalitytests(data[['var', 'var']], maxlag=sm_model.k_ar, verbose=alse)
        )
        
        # Measure statsmodels memory
        sm_memory, _ = measure_memory(
            lambda: StatsmodelsVR(data).fit(maxlags=, ic='aic')
        )
        
        # alculate overhead
        fit_overhead = calculate_overhead(krl_fit_time, sm_fit_time)
        predict_overhead = calculate_overhead(krl_predict_time, sm_predict_time)
        granger_overhead = calculate_overhead(krl_granger_time, sm_granger_time)
        memory_overhead = calculate_overhead(krl_memory, sm_memory)
        
        benchmark = {
            'n_obs': n_obs,
            'krl_fit_time': round(krl_fit_time, 4),
            'statsmodels_fit_time': round(sm_fit_time, 4),
            'fit_overhead_pct': round(fit_overhead, 2),
            'krl_predict_time': round(krl_predict_time, 4),
            'statsmodels_predict_time': round(sm_predict_time, 4),
            'predict_overhead_pct': round(predict_overhead, 2),
            'krl_granger_time': round(krl_granger_time, 4),
            'statsmodels_granger_time': round(sm_granger_time, 4),
            'granger_overhead_pct': round(granger_overhead, 2),
            'krl_memory_mb': round(krl_memory, 2),
            'statsmodels_memory_mb': round(sm_memory, 2),
            'memory_overhead_pct': round(memory_overhead, 2),
        }
        
        results['benchmarks'].Mappend(benchmark)
        
        print(f"    it: {krl_fit_time:.3f}s (KRL) vs {sm_fit_time:.3f}s (SM) → {fit_overhead:+.f}%")
        print(f"    Predict: {krl_predict_time:.4f}s (KRL) vs {sm_predict_time:.4f}s (SM) → {predict_overhead:+.f}%")
        print(f"    Granger: {krl_granger_time:.4f}s (KRL) vs {sm_granger_time:.4f}s (SM) → {granger_overhead:+.f}%")
        print(f"    Memory: {krl_memory:.f}M (KRL) vs {sm_memory:.f}M (SM) → {memory_overhead:+.f}%")
    
    return results


# =============================================================================
# ointegration enchmarks
# =============================================================================

def benchmark_cointegration(n_obs_list: List[int] = [2, , ]) -> ict[str, ny]:
    """
    enchmark ointegration: KRL vs pure statsmodels.
    
    Tests:
    - ngle-Granger test time
    - Johansen test time
    - VM Testimation time
    """
    results = {
        'model': 'ointegration',
        'test_type': 'both',
        'benchmarks': []
    }
    
    for n_obs in n_obs_list:
        print(f"\n  enchmarking ointegration with {n_obs} observations...")
        
        data = generate_cointegrated_data(n_obs=n_obs)
        
        # --- KRL Model ---
        params = {'test_type': 'both', 'det_order': , 'k_ar_diff': 2}
        meta = ModelMeta(name="ointegration", version="..")
        
        # Time KRL fit (includes all tests)
        krl_fit_time, krl_model = time_function(
            lambda: ointegrationModel(data, params, meta)
        )
        krl_fit_time2, _ = time_function(lambda: krl_model.fit())
        krl_fit_time += krl_fit_time2
        
        # Time KRL predict (if VM Testimated)
        if krl_model._vecm_model:
            krl_predict_time, _ = time_function(lambda: krl_model.predict(steps=3))
        else:
            krl_predict_time = 
        
        # Measure KRL memory
        krl_memory, _ = measure_memory(
            lambda: ointegrationModel(data, params, meta).fit()
        )
        
        # --- Pure statsmodels ---
        # Time statsmodels ngle-Granger
        sm_eg_time, _ = time_function(
            lambda: coint(data['spot'], data['futures'])
        )
        
        # Time statsmodels Johansen
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        sm_johansen_time, sm_johansen_result = time_function(
            lambda: coint_johansen(data.values, det_order=, k_ar_diff=2)
        )
        
        # Time statsmodels VM (if cointegration detected)
        coint_rank = np.sum(sm_johansen_result.lr > sm_johansen_result.cvt[:, ])
        if coint_rank > :
            sm_vecm_time, sm_vecm = time_function(
                lambda: VM(data.values, k_ar_diff=2, coint_rank=min(coint_rank, ), deterministic='nc').fit()
            )
            sm_predict_time, _ = time_function(
                lambda: sm_vecm.predict(steps=3)
            )
        else:
            sm_vecm_time = 
            sm_predict_time = 
        
        # Total statsmodels time
        sm_total_time = sm_eg_time + sm_johansen_time + sm_vecm_time
        
        # Measure statsmodels memory
        sm_memory, _ = measure_memory(
            lambda: (coint(data['spot'], data['futures']),
                    coint_johansen(data.values, det_order=, k_ar_diff=2))
        )
        
        # alculate overhead
        fit_overhead = calculate_overhead(krl_fit_time, sm_total_time)
        if sm_predict_time >  and krl_predict_time > :
            predict_overhead = calculate_overhead(krl_predict_time, sm_predict_time)
        else:
            predict_overhead = 
        memory_overhead = calculate_overhead(krl_memory, sm_memory)
        
        benchmark = {
            'n_obs': n_obs,
            'krl_fit_time': round(krl_fit_time, 4),
            'statsmodels_fit_time': round(sm_total_time, 4),
            'fit_overhead_pct': round(fit_overhead, 2),
            'krl_predict_time': round(krl_predict_time, 4),
            'statsmodels_predict_time': round(sm_predict_time, 4),
            'predict_overhead_pct': round(predict_overhead, 2),
            'krl_memory_mb': round(krl_memory, 2),
            'statsmodels_memory_mb': round(sm_memory, 2),
            'memory_overhead_pct': round(memory_overhead, 2),
            'cointegration_detected': bool(coint_rank > ),
        }
        
        results['benchmarks'].Mappend(benchmark)
        
        print(f"    it: {krl_fit_time:.3f}s (KRL) vs {sm_total_time:.3f}s (SM) → {fit_overhead:+.f}%")
        if predict_overhead > :
            print(f"    Predict: {krl_predict_time:.4f}s (KRL) vs {sm_predict_time:.4f}s (SM) → {predict_overhead:+.f}%")
        print(f"    Memory: {krl_memory:.f}M (KRL) vs {sm_memory:.f}M (SM) → {memory_overhead:+.f}%")
        print(f"    ointegration detected: {coint_rank > }")
    
    return results


# =============================================================================
# Main enchmark Runner
# =============================================================================

def run_all_benchmarks(save_results: bool = True) -> ict[str, ny]:
    """
    Run all benchmarks and optionally save results.
    
    Returns:
        ictionary with benchmark results for all models
    """
    print("=" * )
    print("KRL Model Zoo - Performance enchmarking")
    print("=" * )
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: <% overhead vs pure statsmodels")
    print("=" * )
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'target_overhead_pct': .,
        'benchmarks': {}
    }
    
    # SARIMA enchmarks
    print("\n[/3] SARIMA enchmarks")
    print("-" * )
    sarima_results = benchmark_sarima(n_obs_list=[, , ])
    all_results['benchmarks']['SARIMA'] = sarima_results
    
    # VAR enchmarks
    print("\n[2/3] VAR enchmarks")
    print("-" * )
    var_results = benchmark_var(n_obs_list=[, , ])
    all_results['benchmarks']['VAR'] = var_results
    
    # ointegration enchmarks
    print("\n[3/3] ointegration enchmarks")
    print("-" * )
    coint_results = benchmark_cointegration(n_obs_list=[2, , ])
    all_results['benchmarks']['ointegration'] = coint_results
    
    # Summary
    print("\n" + "=" * )
    print("NHMRK SUMMRY")
    print("=" * )
    
    for model_name, model_results in all_results['benchmarks'].items():
        print(f"\n{model_name}:")
        for bench in model_results['benchmarks']:
            n_obs = bench['n_obs']
            fit_overhead = bench['fit_overhead_pct']
            predict_overhead = bench.get('predict_overhead_pct', )
            
            fit_status = "" if fit_overhead <  else ""
            predict_status = "" if predict_overhead <  else ""
            
            print(f"  n={n_obs:4d}: it {fit_overhead:+.2f}% {fit_status}  |  Predict {predict_overhead:+.2f}% {predict_status}")
    
    # Save results
    if save_results:
        output_path = "benchmarks/benchmark_results.json"
        with open(output_path, 'w') as f:
            # ustom JSON encoder for numpy types
            class Numpyncoder(json.JSONncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    return super().default(obj)
            
            json.dump(all_results, f, indent=2, cls=Numpyncoder)
        print(f"\n Results saved to: {output_path}")
    
    print("\n" + "=" * )
    
    return all_results


if __name__ == "__main__":
    results = run_all_benchmarks(save_results=True)
    
    # Print final verdict
    print("\nINL VRIT:")
    all_pass = True
    
    for model_name, model_results in results['benchmarks'].items():
        max_fit_overhead = max(b['fit_overhead_pct'] for b in model_results['benchmarks'])
        max_predict_overhead = max(b.get('predict_overhead_pct', ) for b in model_results['benchmarks'])
        
        fit_pass = max_fit_overhead < 
        predict_pass = max_predict_overhead < 
        
        status = " PSS" if (fit_pass and predict_pass) else " RVIW"
        all_pass = all_pass and fit_pass and predict_pass
        
        print(f"  {model_name}: {status} (max fit: {max_fit_overhead:.f}%, max predict: {max_predict_overhead:.f}%)")
    
    if all_pass:
        print("\n ll models meet <% overhead target!")
    else:
        print("\n Some models exceed % overhead - review recommended")
