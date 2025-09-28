"""Core algorithms for DDAM"""

import numpy as np
from typing import List, Tuple
from scipy.linalg import sqrtm
from .metrics import wasserstein_distance_gaussian


def phi_operator(query: Tuple[np.ndarray, np.ndarray],
                patterns: List[Tuple[np.ndarray, np.ndarray]], 
                beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    One step of DAM update (Phi operator) - Algorithm 1 from the paper.
    
    Computes the weighted barycentric transport of the query distribution
    towards the stored patterns.
    
    Args:
        query: Current state as (mean, covariance) tuple
        patterns: List of stored Gaussian patterns
        beta: Temperature parameter
        
    Returns:
        Updated state as (mean, covariance) tuple
    """
    m, omega = query
    N = len(patterns)
    d = m.shape[0]
    
    # Step 1: Compute Wasserstein distances to all stored patterns
    distances_squared = []
    for mu_i, sigma_i in patterns:
        mean_diff = np.linalg.norm(mu_i - m)**2
        
        # Compute covariance term
        sigma_sqrt = sqrtm(sigma_i)
        omega_sqrt = sqrtm(omega)
        
        # Handle numerical issues with matrix square root
        try:
            cross_term = sqrtm(sigma_sqrt @ omega @ sigma_sqrt)
            cov_term = np.trace(sigma_i + omega - 2 * cross_term)
        except:
            # Fallback for numerical stability
            cov_term = np.trace(sigma_i + omega)
        
        distances_squared.append(mean_diff + cov_term)
    
    distances_squared = np.array(distances_squared)
    
    # Step 2: Compute softmax weights
    # Use numerical stability trick
    max_val = np.max(-beta * distances_squared)
    exp_values = np.exp(-beta * distances_squared - max_val)
    weights = exp_values / np.sum(exp_values)
    
    # Step 3: Compute transport map coefficients
    A_matrices = []
    for mu_i, sigma_i in patterns:
        # Compute Aᵢ = Σᵢ^(1/2) (Σᵢ^(1/2) Ω Σᵢ^(1/2))^(-1/2) Σᵢ^(1/2)
        sigma_sqrt = sqrtm(sigma_i)
        
        try:
            # Compute (Σᵢ^(1/2) Ω Σᵢ^(1/2))^(1/2)
            middle_sqrt = sqrtm(sigma_sqrt @ omega @ sigma_sqrt)
            # Compute its inverse
            middle_sqrt_inv = np.linalg.inv(middle_sqrt)
            # Compute Aᵢ
            A_i = sigma_sqrt @ middle_sqrt_inv @ sigma_sqrt
        except:
            # Fallback to identity if numerical issues
            A_i = np.eye(d)
        
        A_matrices.append(A_i)
    
    # Step 4: Update mean and covariance
    # New mean is weighted average of pattern means
    m_new = sum(w * mu_i for w, (mu_i, _) in zip(weights, patterns))
    
    # Compute weighted average of transport matrices
    A_tilde = sum(w * A_i for w, A_i in zip(weights, A_matrices))
    
    # New covariance
    omega_new = A_tilde @ omega @ A_tilde.T
    
    # Ensure symmetry and positive definiteness
    omega_new = (omega_new + omega_new.T) / 2
    
    # Add small regularization if needed for numerical stability
    min_eig = np.min(np.linalg.eigvalsh(omega_new))
    if min_eig < 1e-10:
        omega_new += (1e-10 - min_eig) * np.eye(d)
    
    return m_new, omega_new


def retrieve(query: Tuple[np.ndarray, np.ndarray],
            patterns: List[Tuple[np.ndarray, np.ndarray]],
            beta: float,
            max_iterations: int = 10,
            tolerance: float = 1e-6) -> Tuple[Tuple[np.ndarray, np.ndarray], int]:
    """
    Iterative retrieval algorithm.
    
    Args:
        query: Initial query Gaussian
        patterns: List of stored patterns
        beta: Temperature parameter
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Retrieved Gaussian and number of iterations
    """
    current = query
    
    for iteration in range(max_iterations):
        next_state = phi_operator(current, patterns, beta)
        
        # Check convergence
        distance = wasserstein_distance_gaussian(current, next_state)
        
        if distance < tolerance:
            return next_state, iteration + 1
        
        current = next_state
    
    return current, max_iterations
