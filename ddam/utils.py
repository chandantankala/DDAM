"""Utility functions for sampling and data generation"""

import numpy as np
from typing import List, Tuple, Optional


def sample_gaussian_sphere(N: int, d: int, radius: float,
                          lambda_min: float = 1.0, 
                          lambda_max: float = 1.1,
                          seed: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Sample N Gaussian distributions on a Wasserstein sphere.
    
    Args:
        N: Number of samples
        d: Dimension
        radius: Wasserstein sphere radius
        lambda_min: Minimum eigenvalue bound
        lambda_max: Maximum eigenvalue bound
        seed: Random seed
        
    Returns:
        List of (mean, covariance) tuples
    """
    if seed is not None:
        np.random.seed(seed)
    
    patterns = []
    
    for _ in range(N):
        # Sample mean on sphere
        mean = np.random.randn(d)
        mean = mean / np.linalg.norm(mean) * np.sqrt(radius**2 / 2)
        
        # Sample covariance with bounded eigenvalues
        # Generate random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        
        # Sample eigenvalues
        eigenvalues = np.random.uniform(lambda_min, lambda_max, d)
        # Normalize to have correct trace
        eigenvalues = eigenvalues * (radius**2 / 2) / np.sum(eigenvalues)
        
        # Construct covariance
        cov = Q @ np.diag(eigenvalues) @ Q.T
        
        patterns.append((mean, cov))
    
    return patterns


def sample_commuting_gaussians(N: int, d: int, radius: float,
                              lambda_min: float = 1.0,
                              lambda_max: float = 1.1,
                              seed: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Sample N Gaussian distributions with commuting covariance matrices.
    
    Algorithm 2 from the paper.
    
    Args:
        N: Number of samples
        d: Dimension
        radius: Wasserstein sphere radius
        lambda_min: Minimum eigenvalue
        lambda_max: Maximum eigenvalue
        seed: Random seed
        
    Returns:
        List of (mean, covariance) tuples with commuting covariances
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Fix common orthogonal matrix for all covariances
    U, _ = np.linalg.qr(np.random.randn(d, d))
    
    patterns = []
    target_sum = (radius**2) / 2
    
    for _ in range(N):
        # Sample eigenvalues that sum to target
        valid = False
        while not valid:
            # Sample d-1 eigenvalues
            lambdas = np.random.uniform(lambda_min, lambda_max, d-1)
            # Compute last eigenvalue to hit target sum
            lambda_d = target_sum - np.sum(lambdas)
            
            if lambda_min <= lambda_d <= lambda_max:
                lambdas = np.append(lambdas, lambda_d)
                valid = True
        
        # Random permutation to avoid bias
        np.random.shuffle(lambdas)
        
        # Construct covariance with shared eigenbasis
        cov = U @ np.diag(lambdas) @ U.T
        
        # Sample mean on sphere
        mean = np.random.randn(d)
        mean = mean / np.linalg.norm(mean) * np.sqrt(radius**2 / 2)
        
        patterns.append((mean, cov))
    
    return patterns


def perturb_gaussian(gaussian: Tuple[np.ndarray, np.ndarray],
                    perturbation_radius: float,
                    seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perturb a Gaussian distribution by a given Wasserstein distance.
    
    Args:
        gaussian: Original Gaussian as (mean, covariance)
        perturbation_radius: Wasserstein distance for perturbation
        seed: Random seed
        
    Returns:
        Perturbed Gaussian
    """
    if seed is not None:
        np.random.seed(seed)
    
    mean, cov = gaussian
    d = mean.shape[0]
    
    # Split perturbation budget between mean and covariance
    mean_budget = perturbation_radius**2 / 2
    cov_budget = perturbation_radius**2 / 2
    
    # Perturb mean
    mean_perturbation = np.random.randn(d)
    mean_perturbation = mean_perturbation / np.linalg.norm(mean_perturbation)
    mean_perturbation *= np.sqrt(mean_budget)
    new_mean = mean + mean_perturbation
    
    # Perturb covariance (maintain positive definiteness)
    # Generate random positive semi-definite perturbation
    W = np.random.randn(d, d)
    perturbation = W @ W.T
    # Scale to match budget
    perturbation = perturbation * (cov_budget / np.trace(perturbation))
    
    new_cov = cov + 0.1 * perturbation  # Small perturbation to maintain stability
    
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(new_cov))
    if min_eig < 1e-10:
        new_cov += (1e-10 - min_eig) * np.eye(d)
    
    return new_mean, new_cov


def generate_synthetic_data(N: int = 100, d: int = 50,
                           commuting: bool = True,
                           seed: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate synthetic Gaussian patterns for experiments.
    
    Args:
        N: Number of patterns
        d: Dimension
        commuting: If True, generate commuting covariances
        seed: Random seed
        
    Returns:
        List of Gaussian patterns
    """
    radius = np.sqrt(2 * d)  # Standard choice from paper
    
    if commuting:
        return sample_commuting_gaussians(N, d, radius, seed=seed)
    else:
        return sample_gaussian_sphere(N, d, radius, seed=seed)
