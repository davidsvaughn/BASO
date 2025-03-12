import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from claude_gpr3 import DiverseAdaptiveSampler as SklearnSampler
from gpytorch_gpr import DiverseAdaptiveSampler as GPyTorchSampler

def run_benchmark(data_file, n_samples=200, n_runs=3, use_gpu=True):
    """
    Run benchmark comparing scikit-learn GP with GPyTorch implementation.
    
    Parameters:
    -----------
    data_file : str
        Path to data file
    n_samples : int
        Number of samples to collect in each run
    n_runs : int
        Number of benchmark runs to average
    use_gpu : bool
        Whether to use GPU for GPyTorch (if available)
    """
    # Configuration for both samplers
    config = {
        'acquisition_strategy': 'ucb', 
        'kappa': 2.0,
        'initial_samples': 30,
        'init_strategy': 'latin_hypercube',
        'diversity_weight': 0.5,
        'force_exploration_every': 10,
    }
    
    # For storing timing results
    sklearn_times = []
    gpytorch_times = []
    sklearn_errors = []
    gpytorch_errors = []
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")
        seed = np.random.randint(1000)
        
        # Run scikit-learn sampler
        print("Running scikit-learn sampler...")
        start_time = time.time()
        sklearn_sampler = SklearnSampler(data_file, random_seed=seed)
        sklearn_sampler.run_sampling(
            n_samples=n_samples,
            verbose=False,
            **config
        )
        sklearn_time = time.time() - start_time
        sklearn_times.append(sklearn_time)
        sklearn_errors.append(sklearn_sampler.errors[-1])
        print(f"scikit-learn time: {sklearn_time:.2f}s, final error: {sklearn_sampler.errors[-1]:.4f}")
        
        # Run GPyTorch sampler
        print("Running GPyTorch sampler...")
        start_time = time.time()
        gpytorch_sampler = GPyTorchSampler(data_file, random_seed=seed, use_gpu=use_gpu)
        gpytorch_sampler.run_sampling(
            n_samples=n_samples,
            verbose=False,
            training_iterations=100,  # GPyTorch specific param
            **config
        )
        gpytorch_time = time.time() - start_time
        gpytorch_times.append(gpytorch_time)
        gpytorch_errors.append(gpytorch_sampler.errors[-1])
        print(f"GPyTorch time: {gpytorch_time:.2f}s, final error: {gpytorch_sampler.errors[-1]:.4f}")
        
        # Force GPU memory cleanup if used
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate averages
    sklearn_avg_time = np.mean(sklearn_times)
    gpytorch_avg_time = np.mean(gpytorch_times)
    sklearn_avg_error = np.mean(sklearn_errors)
    gpytorch_avg_error = np.mean(gpytorch_errors)
    
    speedup = sklearn_avg_time / gpytorch_avg_time
    
    # Print results
    print("\n===== Benchmark Results =====")
    print(f"Average over {n_runs} runs with {n_samples} samples each:")
    print(f"scikit-learn: {sklearn_avg_time:.2f}s, error: {sklearn_avg_error:.4f}")
    print(f"GPyTorch:     {gpytorch_avg_time:.2f}s, error: {gpytorch_avg_error:.4f}")
    print(f"Speedup:      {speedup:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Time comparison
    plt.subplot(1, 2, 1)
    plt.bar(['scikit-learn', 'GPyTorch'], [sklearn_avg_time, gpytorch_avg_time])
    plt.title('Average Execution Time (seconds)')
    plt.ylabel('Time (s)')
    
    # Error comparison
    plt.subplot(1, 2, 2)
    plt.bar(['scikit-learn', 'GPyTorch'], [sklearn_avg_error, gpytorch_avg_error])
    plt.title('Final Error')
    plt.ylabel('Normalized Error')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    plt.show()
    
    return {
        'sklearn_time': sklearn_avg_time,
        'gpytorch_time': gpytorch_avg_time,
        'sklearn_error': sklearn_avg_error,
        'gpytorch_error': gpytorch_avg_error,
        'speedup': speedup
    }

if __name__ == "__main__":
    data_file = "data/phi4-math-4claude.txt"
    
    # Check if GPU is available
    use_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
    print(f"Using device: {device_name}")
    
    # Run benchmark
    results = run_benchmark(
        data_file=data_file,
        n_samples=200,  # Total samples per run
        n_runs=3,       # Number of runs to average
        use_gpu=use_gpu  # Use GPU for GPyTorch if available
    )
