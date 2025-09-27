"""Real-world experiments with Gaussian word embeddings"""

import numpy as np
from typing import List, Tuple, Dict
from ddam import DDAM, perturb_gaussian
from ddam.metrics import wasserstein_distance_gaussian


def load_word_embeddings(filepath: str = None, 
                        n_words: int = 10000,
                        d: int = 50) -> Tuple[List[str], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Load or generate Gaussian word embeddings.
    
    Args:
        filepath: Path to embeddings file (if None, generates synthetic)
        n_words: Number of words
        d: Embedding dimension
        
    Returns:
        List of words and their Gaussian embeddings
    """
    if filepath is None:
        # Generate synthetic word embeddings for demonstration
        print("Generating synthetic word embeddings...")
        words = [f"word_{i}" for i in range(n_words)]
        
        embeddings = []
        for i in range(n_words):
            # Random mean
            mean = np.random.randn(d) * 0.1
            
            # Spherical covariance with random scale
            sigma = np.random.uniform(0.01, 0.1)
            cov = sigma**2 * np.eye(d)
            
            embeddings.append((mean, cov))
    else:
        # Load from file (implement based on your data format)
        raise NotImplementedError("Implement loading from your data format")
    
    return words, embeddings


def word_retrieval_experiment(words: List[str],
                             embeddings: List[Tuple[np.ndarray, np.ndarray]],
                             beta_values: List[float] = [1, 10, 50],
                             n_test_words: int = 5,
                             seed: int = 42) -> Dict[str, Any]:
    """
    Test word retrieval with different temperature parameters.
    
    Args:
        words: List of words
        embeddings: Gaussian embeddings
        beta_values: Temperature parameters to test
        n_test_words: Number of words to test
        seed: Random seed
        
    Returns:
        Dictionary with retrieval results
    """
    np.random.seed(seed)
    
    # Select test words
    N = len(words)
    test_indices = np.random.choice(N, n_test_words, replace=False)
    test_words = [words[i] for i in test_indices]
    
    results = {beta: {} for beta in beta_values}
    
    for beta in beta_values:
        print(f"\nTesting beta = {beta}")
        memory = DDAM(embeddings, beta=beta)
        
        for idx in test_indices:
            word = words[idx]
            original_embedding = embeddings[idx]
            
            # Perturb embedding
            perturbation_radius = 1 / np.sqrt(beta * N)
            query = perturb_gaussian(original_embedding, perturbation_radius)
            
            # Track retrieval over iterations
            trajectory = []
            current = query
            
            for iteration in range(10):
                # Find nearest word
                distances = [
                    wasserstein_distance_gaussian(current, emb)
                    for emb in embeddings
                ]
                nearest_idx = np.argmin(distances)
                nearest_word = words[nearest_idx]
                trajectory.append(nearest_word)
                
                # Apply one retrieval step
                from ddam.algorithms import phi_operator
                current = phi_operator(current, embeddings, beta)
            
            results[beta][word] = {
                'trajectory': trajectory,
                'success': trajectory[-1] == word,
                'converged_to': trajectory[-1]
            }
    
    return results


def print_retrieval_results(results: Dict[str, Any]):
    """
    Print word retrieval results in a formatted table.
    
    Args:
        results: Retrieval results dictionary
    """
    for beta, beta_results in results.items():
        print(f"\n{'='*50}")
        print(f"Beta = {beta}")
        print(f"{'='*50}")
        
        success_count = sum(1 for r in beta_results.values() if r['success'])
        total = len(beta_results)
        
        print(f"Success rate: {success_count}/{total} ({100*success_count/total:.1f}%)")
        print("\nWord trajectories:")
        
        for word, result in beta_results.items():
            trajectory = result['trajectory']
            status = "✓" if result['success'] else "✗"
            print(f"{status} {word}: {' → '.join(trajectory[:5])}...")


if __name__ == "__main__":
    # Load or generate embeddings
    words, embeddings = load_word_embeddings(n_words=1000, d=50)
    
    # Run retrieval experiment
    print("Running word retrieval experiment...")
    results = word_retrieval_experiment(
        words, embeddings,
        beta_values=[1, 10, 50],
        n_test_words=10
    )
    
    # Print results
    print_retrieval_results(results)
