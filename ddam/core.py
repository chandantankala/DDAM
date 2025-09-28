"""Core DDAM implementation"""

import numpy as np
from typing import List, Tuple, Optional
from .algorithms import phi_operator
from .metrics import wasserstein_distance_gaussian


class DDAM:
    """
    Dense Associative Memory on Bures-Wasserstein Space
    
    This class implements a dense associative memory that stores and retrieves
    Gaussian distributions using the Bures-Wasserstein geometry.
    
    Attributes:
        patterns: List of stored Gaussian patterns (mean, covariance) tuples
        beta: Temperature parameter controlling sharpness of retrieval
        N: Number of stored patterns
    """
    
    def __init__(self, patterns: List[Tuple[np.ndarray, np.ndarray]], 
                 beta: float = 1.0):
        """
        Initialize DDAM with stored Gaussian patterns.
        
        Args:
            patterns: List of (mean, covariance) tuples representing Gaussian distributions
            beta: Temperature parameter (higher values create sharper energy basins)
        """
        self.patterns = patterns
        self.beta = beta
        self.N = len(patterns)
        self.d = patterns[0][0].shape[0] if patterns else 0
        
        # Validate inputs
        self._validate_patterns()
    
    def _validate_patterns(self):
        """Validate that all patterns have consistent dimensions."""
        if not self.patterns:
            return
            
        d = self.d
        for i, (mean, cov) in enumerate(self.patterns):
            if mean.shape != (d,):
                raise ValueError(f"Pattern {i}: mean has shape {mean.shape}, expected ({d},)")
            if cov.shape != (d, d):
                raise ValueError(f"Pattern {i}: covariance has shape {cov.shape}, expected ({d}, {d})")
            # Check positive definiteness
            eigvals = np.linalg.eigvalsh(cov)
            if np.min(eigvals) <= 0:
                raise ValueError(f"Pattern {i}: covariance matrix is not positive definite")
    
    def energy(self, query: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Compute the log-sum-exp energy at a query distribution.
        
        E(ξ) = -(1/β) log Σᵢ exp(-β W₂²(Xᵢ, ξ))
        
        Args:
            query: Query Gaussian as (mean, covariance) tuple
            
        Returns:
            Energy value at query distribution
        """
        distances_squared = [
            wasserstein_distance_gaussian(pattern, query)**2 
            for pattern in self.patterns
        ]
        
        # Compute log-sum-exp with numerical stability
        max_val = max([-self.beta * d for d in distances_squared])
        log_sum = np.log(sum([
            np.exp(-self.beta * d - max_val) 
            for d in distances_squared
        ]))
        
        return -(1/self.beta) * (log_sum + max_val)
    
    def compute_weights(self, query: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute softmax weights for each stored pattern given a query.
        
        wᵢ(ξ) = exp(-β W₂²(Xᵢ, ξ)) / Σⱼ exp(-β W₂²(Xⱼ, ξ))
        
        Args:
            query: Query Gaussian as (mean, covariance) tuple
            
        Returns:
            Array of weights for each stored pattern
        """
        distances_squared = np.array([
            wasserstein_distance_gaussian(pattern, query)**2 
            for pattern in self.patterns
        ])
        
        # Softmax with numerical stability
        exp_values = np.exp(-self.beta * distances_squared + 
                           np.max(-self.beta * distances_squared))
        weights = exp_values / np.sum(exp_values)
        
        return weights
    
    def retrieve(self, query: Tuple[np.ndarray, np.ndarray], 
                max_iterations: int = 10,
                tolerance: float = 1e-6,
                verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve stored pattern from a query distribution.
        
        Iteratively applies the Phi operator until convergence or max iterations.
        
        Args:
            query: Initial query Gaussian as (mean, covariance) tuple
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance in Wasserstein distance
            verbose: If True, print convergence information
            
        Returns:
            Retrieved Gaussian distribution as (mean, covariance) tuple
        """
        current = query
        
        for iteration in range(max_iterations):
            # Apply Phi operator
            next_state = phi_operator(current, self.patterns, self.beta)
            
            # Check convergence
            distance = wasserstein_distance_gaussian(current, next_state)
            
            if verbose:
                print(f"Iteration {iteration + 1}: W2 distance = {distance:.6f}")
            
            if distance < tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                return next_state
            
            current = next_state
        
        if verbose:
            print(f"Maximum iterations ({max_iterations}) reached")
        
        return current
    
    def find_nearest_pattern(self, query: Tuple[np.ndarray, np.ndarray]) -> int:
        """
        Find the index of the nearest stored pattern to the query.
        
        Args:
            query: Query Gaussian as (mean, covariance) tuple
            
        Returns:
            Index of nearest stored pattern
        """
        distances = [
            wasserstein_distance_gaussian(pattern, query) 
            for pattern in self.patterns
        ]
        return np.argmin(distances)
