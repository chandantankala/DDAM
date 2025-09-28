"""Wasserstein distance metrics for Gaussian distributions"""

import numpy as np
from scipy.linalg import sqrtm
from typing import Tuple


def wasserstein_distance_gaussian(gaussian1: Tuple[np.ndarray, np.ndarray],
                                 gaussian2: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Compute 2-Wasserstein distance between two Gaussian distributions.
    
    W₂²(N(μ₁,Σ₁), N(μ₂,Σ₂)) = ||μ₁-μ₂||² + tr(Σ₁ + Σ₂ - 2(Σ₁^(1/2) Σ₂ Σ₁^(1/2))^(1/2))
    
    Args:
        gaussian1: First Gaussian as (mean, covariance) tuple
        gaussian2: Second Gaussian as (mean, covariance) tuple
        
    Returns:
        2-Wasserstein distance
    """
    m1, S1 = gaussian1
    m2, S2 = gaussian2
    
    # Mean difference term
    mean_diff = np.linalg.norm(m1 - m2)**2
    
    # Covariance term (Bures metric)
    S1_sqrt = sqrtm(S1)
    
    # Compute (S1^(1/2) S2 S1^(1/2))^(1/2)
    try:
        cross_term = sqrtm(S1_sqrt @ S2 @ S1_sqrt)
        cov_term = np.trace(S1 + S2 - 2 * cross_term)
    except:
        # Fallback for numerical issues
        cov_term = np.trace(S1 + S2)
    
    # Ensure non-negative due to numerical errors
    distance_squared = max(0, mean_diff + cov_term)
    
    return np.sqrt(distance_squared)


def bures_wasserstein_distance(gaussian1: Tuple[np.ndarray, np.ndarray],
                              gaussian2: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Alias for wasserstein_distance_gaussian using Bures-Wasserstein terminology.
    """
    return wasserstein_distance_gaussian(gaussian1, gaussian2)


def l2_inner_product_gaussian(gaussian1: Tuple[np.ndarray, np.ndarray],
                             gaussian2: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Compute L² inner product between two Gaussian distributions.
    
    <N(μ₁,Σ₁), N(μ₂,Σ₂)>_L² = (2π)^(-d/2) |Σ₁+Σ₂|^(-1/2) exp(-1/2 (μ₁-μ₂)ᵀ(Σ₁+Σ₂)⁻¹(μ₁-μ₂))
    
    Args:
        gaussian1: First Gaussian
        gaussian2: Second Gaussian
        
    Returns:
        L² inner product value
    """
    m1, S1 = gaussian1
    m2, S2 = gaussian2
    d = m1.shape[0]
    
    S_sum = S1 + S2
    S_sum_inv = np.linalg.inv(S_sum)
    
    mean_diff = m1 - m2
    quadratic_term = mean_diff.T @ S_sum_inv @ mean_diff
    
    det_term = np.linalg.det(S_sum)
    
    inner_product = (2 * np.pi)**(-d/2) * det_term**(-0.5) * np.exp(-0.5 * quadratic_term)
    
    return inner_product
