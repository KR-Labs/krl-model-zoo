"""
Performance enchmarking Suite for KRL Model Zoo

This script benchmarks all implemented models across multiple dimensions:
. itting time vs dataset size
2. Memory usage during fitting
3. Prediction time
4. onvergence rates
. Scalability analysis

Models tested:
- GRH(,)
- GRH(,)
- GJR-GRH(,)
- Kalman ilter
- Local Level Model

ataset sizes: , , , 2,  observations
"""

import numpy as np
import pandas as pd
import time
import tracemalloc
from typing import ict, List, Tuple, ny
import json
from datetime import datetime
import warnings

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore')

# Import models
from krl_models.volatility import GRHModel, GRHModel, GJRGRHModel
from krl_models.state_space import Kalmanilter, LocalLevelModel
from krl_core.model_input_schema import ModelInputSchema
from krl_core.base_model import ModelMeta


class Performanceenchmark:
    """Performance benchmarking suite for time series models"""
    
    def __init__(self, dataset_sizes: List[int] = None, n_runs: int = ):
        """
        Initialize benchmarking suite
        
        Parameters
        ----------
        dataset_sizes : list of int
            Sizes of datasets to test
        n_runs : int
            Number of runs per configuration for averaging
        """
        self.dataset_sizes = dataset_sizes or [, , , 2, ]
        self.n_runs = n_runs
        self.results = {
            'garch': [],
            'egarch': [],
            'gjr_garch': [],
            'kalman_filter': [],
            'local_level': []
        }
        
    def generate_garch_data(self, T: int, omega: float = ., 
                           alpha: float = ., beta: float = .) -> pd.atarame:
        """Generate synthetic GRH(,) data"""
        np.random.seed(42)
        
        returns = np.zeros(T)
        sigma2 = np.zeros(T)
        sigma2[] = omega / ( - alpha - beta)
        
        for t in range(, T):
            sigma2[t] = omega + alpha * returns[t-]**2 + beta * sigma2[t-]
            returns[t] = np.sqrt(sigma2[t]) * np.random.randn()
        
        dates = pd.date_range(start='22--', periods=T, freq='')
        return pd.atarame({'returns': returns}, index=dates)
    
    def generate_kalman_data(self, T: int) -> pd.atarame:
        """Generate synthetic state space data"""
        np.random.seed(42)
        
        # True states
        true_state = np.zeros(T)
        observed = np.zeros(T)
        
        # Random walk + observation noise
        for t in range(, T):
            true_state[t] = true_state[t-] + np.random.randn() * .
        
        observed = true_state + np.random.randn(T) * .
        
        dates = pd.date_range(start='22--', periods=T, freq='')
        return pd.atarame({'value': observed}, index=dates)
    
    def benchmark_model(self, model_name: str, model_class: ny, 
                       data: pd.atarame, **model_kwargs) -> ict:
        """
        enchmark a single model configuration
        
        Returns
        -------
        dict
            enchmark results including timing and memory metrics
        """
        results = {
            'model': model_name,
            'n_obs': len(data),
            'fit_times': [],
            'predict_times': [],
            'memory_peak_mb': [],
            'converged': []
        }
        
        for run in range(self.n_runs):
            try:
                # Start memory tracking
                tracemalloc.start()
                
                # Measure fitting time
                start_time = time.perf_counter()
                
                if model_name in ['kalman_filter', 'local_level']:
                    # State space models
                    model = model_class(**model_kwargs)
                    result = model.fit(data)
                else:
                    # Volatility models - use simplified PI from tests
                    p = model_kwargs.get('p', )
                    q = model_kwargs.get('q', )
                    o = model_kwargs.get('o', None)
                    
                    if o is not None:
                        model = model_class(p=p, o=o, q=q)
                    else:
                        model = model_class(p=p, q=q)
                    
                    result = model.fit(data)
                
                fit_time = time.perf_counter() - start_time
                results['fit_times'].append(fit_time)
                
                # heck convergence
                converged = result.success if hasattr(result, 'success') else True
                results['converged'].append(converged)
                
                # Measure prediction time
                start_time = time.perf_counter()
                forecast = model.predict(steps=)
                predict_time = time.perf_counter() - start_time
                results['predict_times'].append(predict_time)
                
                # Get peak memory usage
                current, peak = tracemalloc.get_traced_memory()
                results['memory_peak_mb'].append(peak / 24 / 24)  # onvert to M
                
                tracemalloc.stop()
                
            except xception as e:
                print(f"    rror in {model_name} (run {run+}): {str(e)[:]}")
                tracemalloc.stop()
                continue
        
        # alculate statistics
        if results['fit_times']:
            results['fit_time_mean'] = np.mean(results['fit_times'])
            results['fit_time_std'] = np.std(results['fit_times'])
            results['predict_time_mean'] = np.mean(results['predict_times'])
            results['predict_time_std'] = np.std(results['predict_times'])
            results['memory_mean_mb'] = np.mean(results['memory_peak_mb'])
            results['convergence_rate'] = np.mean(results['converged'])
        else:
            results['fit_time_mean'] = np.nan
            results['fit_time_std'] = np.nan
            results['predict_time_mean'] = np.nan
            results['predict_time_std'] = np.nan
            results['memory_mean_mb'] = np.nan
            results['convergence_rate'] = .
        
        return results
    
    def run_volatility_benchmarks(self):
        """Run benchmarks for all volatility models"""
        print("\n" + "="*)
        print("VOLTILITY MOLS NHMRKING")
        print("="*)
        
        for size in self.dataset_sizes:
            print(f"\n ataset size: {size} observations")
            
            # Generate data
            data = self.generate_garch_data(size)
            
            # GRH(,)
            print("  Testing GRH(,)...", end=" ")
            result = self.benchmark_model(
                'garch', 
                GRHModel, 
                data,
                p=, 
                q=
            )
            self.results['garch'].append(result)
            print(f" ({result['fit_time_mean']:.3f}s)")
            
            # GRH(,)
            print("  Testing GRH(,)...", end=" ")
            result = self.benchmark_model(
                'egarch',
                GRHModel,
                data,
                p=,
                o=, 
                q=
            )
            self.results['egarch'].append(result)
            print(f" ({result['fit_time_mean']:.3f}s)")
            
            # GJR-GRH(,)
            print("  Testing GJR-GRH(,)...", end=" ")
            result = self.benchmark_model(
                'gjr_garch',
                GJRGRHModel,
                data,
                p=,
                o=,
                q=
            )
            self.results['gjr_garch'].append(result)
            print(f" ({result['fit_time_mean']:.3f}s)")
    
    def run_state_space_benchmarks(self):
        """Run benchmarks for state space models"""
        print("\n" + "="*)
        print("STT SP MOLS NHMRKING")
        print("="*)
        
        for size in self.dataset_sizes:
            print(f"\n ataset size: {size} observations")
            
            # Generate data
            data = self.generate_kalman_data(size)
            
            # Kalman ilter
            print("  Testing Kalman ilter...", end=" ")
            
            # Setup Kalman ilter matrices
             = np.array([[., .], [., .]])
            H = np.array([[., .]])
            Q = np.array([[., .], [., .]])
            R = np.array([[.]])
            x = np.array([., .])
            P = np.array([[., .], [., .]])
            
            result = self.benchmark_model(
                'kalman_filter',
                Kalmanilter,
                data,
                n_states=2,
                n_obs=,
                =, H=H, Q=Q, R=R,
                x=x, P=P
            )
            self.results['kalman_filter'].append(result)
            print(f" ({result['fit_time_mean']:.3f}s)")
            
            # Local Level Model
            print("  Testing Local Level Model...", end=" ")
            result = self.benchmark_model(
                'local_level',
                LocalLevelModel,
                data,
                estimate_params=True
            )
            self.results['local_level'].append(result)
            print(f" ({result['fit_time_mean']:.3f}s)")
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report"""
        report = []
        report.append("\n" + "="*)
        report.append("PRORMN NHMRKING SUMMRY RPORT")
        report.append("="*)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"ataset sizes tested: {self.dataset_sizes}")
        report.append(f"Runs per configuration: {self.n_runs}")
        
        # Summary statistics by model
        report.append("\n" + "-"*)
        report.append("ITTING TIM OMPRISON (seconds)")
        report.append("-"*)
        report.append(f"\n{'Model':<2} {'':<} {'':<} {'':<} {'2':<} {'':<}")
        report.append("-"*)
        
        for model_name in self.results.keys():
            if not self.results[model_name]:
                continue
            
            times = [r['fit_time_mean'] for r in self.results[model_name]]
            row = f"{model_name:<2}"
            for t in times:
                if not np.isnan(t):
                    row += f"{t:<.4f}"
                else:
                    row += f"{'N/':<}"
            report.append(row)
        
        # Memory usage comparison
        report.append("\n" + "-"*)
        report.append("MMORY USG OMPRISON (M)")
        report.append("-"*)
        report.append(f"\n{'Model':<2} {'':<} {'':<} {'':<} {'2':<} {'':<}")
        report.append("-"*)
        
        for model_name in self.results.keys():
            if not self.results[model_name]:
                continue
            
            memory = [r['memory_mean_mb'] for r in self.results[model_name]]
            row = f"{model_name:<2}"
            for m in memory:
                if not np.isnan(m):
                    row += f"{m:<.2f}"
                else:
                    row += f"{'N/':<}"
            report.append(row)
        
        # Scalability analysis
        report.append("\n" + "-"*)
        report.append("SLILITY NLYSIS")
        report.append("-"*)
        
        for model_name in self.results.keys():
            if not self.results[model_name] or len(self.results[model_name]) < 2:
                continue
            
            report.append(f"\n{model_name.upper()}:")
            
            times = [r['fit_time_mean'] for r in self.results[model_name] if not np.isnan(r['fit_time_mean'])]
            sizes = [r['n_obs'] for r in self.results[model_name] if not np.isnan(r['fit_time_mean'])]
            
            if len(times) >= 2:
                # stimate complexity (log-log regression)
                log_sizes = np.log(sizes)
                log_times = np.log(times)
                
                # Simple linear regression
                n = len(log_sizes)
                sum_x = np.sum(log_sizes)
                sum_y = np.sum(log_times)
                sum_xy = np.sum(log_sizes * log_times)
                sum_x2 = np.sum(log_sizes**2)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
                
                report.append(f"  Time complexity: O(n^{slope:.2f})")
                
                # Speed comparison
                speedup_x = times[-] / times[]
                size_ratio = sizes[-] / sizes[]
                report.append(f"  {sizes[]} → {sizes[-]} obs: {speedup_x:.2f}x slower ({size_ratio:.f}x data)")
                
                # onvergence rate
                conv_rate = np.mean([r['convergence_rate'] for r in self.results[model_name]])
                report.append(f"  onvergence rate: {conv_rate*:.f}%")
        
        # Prediction time comparison
        report.append("\n" + "-"*)
        report.append("PRITION TIM OMPRISON (-step ahead, seconds)")
        report.append("-"*)
        report.append(f"\n{'Model':<2} {'Mean':<} {'Std':<} {'Min':<} {'Max':<}")
        report.append("-"*)
        
        for model_name in self.results.keys():
            if not self.results[model_name]:
                continue
            
            pred_times = [r['predict_time_mean'] for r in self.results[model_name] if not np.isnan(r['predict_time_mean'])]
            
            if pred_times:
                row = f"{model_name:<2}"
                row += f"{np.mean(pred_times):<.f}"
                row += f"{np.std(pred_times):<.f}"
                row += f"{np.min(pred_times):<.f}"
                row += f"{np.max(pred_times):<.f}"
                report.append(row)
        
        # Performance recommendations
        report.append("\n" + "-"*)
        report.append("PRORMN ROMMNTIONS")
        report.append("-"*)
        
        # ind fastest model
        avg_times = {}
        for model_name in self.results.keys():
            if self.results[model_name]:
                times = [r['fit_time_mean'] for r in self.results[model_name] if not np.isnan(r['fit_time_mean'])]
                if times:
                    avg_times[model_name] = np.mean(times)
        
        if avg_times:
            fastest = min(avg_times, key=avg_times.get)
            slowest = max(avg_times, key=avg_times.get)
            
            report.append(f"\n   astest model: {fastest} (avg: {avg_times[fastest]:.4f}s)")
            report.append(f"   Slowest model: {slowest} (avg: {avg_times[slowest]:.4f}s)")
            report.append(f"   Speed difference: {avg_times[slowest]/avg_times[fastest]:.2f}x")
        
        # Memory recommendations
        avg_memory = {}
        for model_name in self.results.keys():
            if self.results[model_name]:
                memory = [r['memory_mean_mb'] for r in self.results[model_name] if not np.isnan(r['memory_mean_mb'])]
                if memory:
                    avg_memory[model_name] = np.mean(memory)
        
        if avg_memory:
            most_efficient = min(avg_memory, key=avg_memory.get)
            least_efficient = max(avg_memory, key=avg_memory.get)
            
            report.append(f"\n   Most memory efficient: {most_efficient} (avg: {avg_memory[most_efficient]:.2f} M)")
            report.append(f"   Least memory efficient: {least_efficient} (avg: {avg_memory[least_efficient]:.2f} M)")
        
        # General recommendations
        report.append("\n   GNRL ROMMNTIONS:")
        report.append("     • or large datasets (> obs), prefer simpler models")
        report.append("     • State space models scale better than volatility models")
        report.append("     • onsider computational budget when choosing model complexity")
        report.append("     • Use parameter caching for repeated predictions")
        report.append("     • Monitor convergence rates for production systems")
        
        report.append("\n" + "="*)
        report.append("N O RPORT")
        report.append("="* + "\n")
        
        return "\n".join(report)
    
    def save_results(self, filepath: str = 'benchmark_results.json'):
        """Save detailed results to JSON file"""
        results_dict = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'dataset_sizes': self.dataset_sizes,
                'n_runs': self.n_runs
            },
            'results': {}
        }
        
        for model_name, results in self.results.items():
            results_dict['results'][model_name] = results
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"\n etailed results saved to: {filepath}")


def main():
    """Run complete benchmark suite"""
    print("\n" + "="*)
    print("KRL MOL ZOO - PRORMN NHMRKING SUIT")
    print("="*)
    print("\nThis benchmark will test all models across multiple dataset sizes.")
    print("ach configuration will be run  times for statistical reliability.")
    print("\nModels to benchmark:")
    print("  • GRH(,)")
    print("  • GRH(,)")
    print("  • GJR-GRH(,)")
    print("  • Kalman ilter")
    print("  • Local Level Model")
    print("\nataset sizes: , , , 2,  observations")
    print("\nMetrics collected:")
    print("  • itting time")
    print("  • Prediction time")
    print("  • Memory usage")
    print("  • onvergence rates")
    print("  • Scalability characteristics")
    
    input("\nPress nter to start benchmarking...")
    
    # Initialize benchmark suite
    benchmark = Performanceenchmark(
        dataset_sizes=[, , , 2, ],
        n_runs=
    )
    
    # Run benchmarks
    start_time = time.time()
    
    benchmark.run_volatility_benchmarks()
    benchmark.run_state_space_benchmarks()
    
    total_time = time.time() - start_time
    
    # Generate and display report
    report = benchmark.generate_summary_report()
    print(report)
    
    print(f"\n  Total benchmarking time: {total_time:.2f} seconds")
    
    # Save results
    benchmark.save_results('benchmarks/benchmark_results.json')
    
    # Save report
    with open('benchmarks/PRORMN_RPORT.md', 'w') as f:
        f.write(report)
    
    print(" Performance report saved to: benchmarks/PRORMN_RPORT.md")
    
    print("\n" + "="*)
    print("NHMRKING OMPLT!")
    print("="* + "\n")


if __name__ == '__main__':
    main()
