"""Synthetic data experiments for DDAM"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from ddam import DDAM, perturb_gaussian, sample_commuting_gaussians
from ddam.metrics import wasserstein_distance_gaussian


def convergence_experiment(N: int = 1000, 
                          d: int = 50,
                          beta_values: List[float] = [0.1, 1.0, 10.0],
                          n_trials: int = 100,
                          seed: int = 42) -> Dict[str, Any]:
    """
    Run convergence experiments with different temperature parameters.
    
    Args:
        N: Number of stored patterns
        d: Dimension
        beta_values: List of temperature parameters to test
        n_trials: Number of trials per configuration
        seed: Random seed
        
    Returns:
        Dictionary with results
    """
    np.random.seed(seed)
    
    # Generate patterns
    patterns = sample_commuting_gaussians(N, d, radius=np.sqrt(2*d))
    
    results = {beta: [] for beta in beta_values}
    
    for beta in beta_values:
        print(f"Testing beta = {beta}")
        memory = DDAM(patterns, beta=beta)
        
        # Test retrieval for subset of patterns
        test_indices = np.random.choice(N, min(n_trials, N), replace=False)
        
        for idx in test_indices:
            # Perturb pattern
            perturbation_radius = 1 / np.sqrt(beta * N)
            query = perturb_gaussian(patterns[idx], perturbation_radius)
            
            # Retrieve
            retrieved = memory.retrieve(query, max_iterations=10)
            
            # Measure error
            error = wasserstein_distance_gaussian(patterns[idx], retrieved)
            results[beta].append(error)
    
    return results


def plot_convergence_results(results: Dict[str, List[float]]):
    """
    Plot convergence results from experiments.
    
    Args:
        results: Dictionary mapping beta values to error lists
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    beta_values = sorted(results.keys())
    errors_mean = [np.mean(results[beta]) for beta in beta_values]
    errors_std = [np.std(results[beta]) for beta in beta_values]
    
    ax.errorbar(beta_values, errors_mean, yerr=errors_std, 
                marker='o', capsize=5, capthick=2)
    
    ax.set_xlabel('Temperature Parameter β', fontsize=12)
    ax.set_ylabel('Retrieval Error (W₂ distance)', fontsize=12)
    ax.set_title('Retrieval Performance vs Temperature', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def capacity_experiment(d_values: List[int] = [10, 20, 30, 40, 50],
                       beta: float = 1.0,
                       seed: int = 42) -> Dict[str, Any]:
    """
    Test storage capacity for different dimensions.
    
    Args:
        d_values: List of dimensions to test
        beta: Temperature parameter
        seed: Random seed
        
    Returns:
        Dictionary with capacity results
    """
    np.random.seed(seed)
    results = {}
    
    for d in d_values:
        print(f"Testing dimension d = {d}")
        
        # Theoretical capacity
        alpha = 1 - 2 * np.log(1.1)  # Using lambda_max/lambda_min = 1.1
        theoretical_N = int(np.sqrt(0.5) * np.exp(d * alpha**2 / 16))
        
        # Test with fraction of theoretical capacity
        for fraction in [0.1, 0.5, 0.9]:
            N = int(fraction * theoretical_N)
            
            # Generate patterns
            patterns = sample_commuting_gaussians(N, d, radius=np.sqrt(2*d))
            memory = DDAM(patterns, beta=beta)
            
            # Test retrieval success
            success_count = 0
            n_tests = min(100, N)
            
            for i in range(n_tests):
                perturbation_radius = 1 / np.sqrt(beta * N)
                query = perturb_gaussian(patterns[i], perturbation_radius)
                retrieved = memory.retrieve(query, max_iterations=10)
                
                # Check if retrieved correct pattern
                retrieved_idx = memory.find_nearest_pattern(retrieved)
                if retrieved_idx == i:
                    success_count += 1
            
            success_rate = success_count / n_tests
            results[(d, fraction)] = success_rate
    
    return results


if __name__ == "__main__":
    # Run convergence experiment
    print("Running convergence experiment...")
    conv_results = convergence_experiment()
    plot_convergence_results(conv_results)
    
    # Run capacity experiment  
    print("\nRunning capacity experiment...")
    cap_results = capacity_experiment()
    
    print("\nCapacity Results:")
    for (d, fraction), success_rate in cap_results.items():
        print(f"d={d}, fraction={fraction:.1f}: {success_rate:.2%} success rate")
