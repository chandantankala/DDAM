# Dense Associative Memory on the Bures-Wasserstein Space (DDAM)

This repository contains the implementation of **Dense Associative Memory on the Bures-Wasserstein Space**, extending classical dense associative memories from vector spaces to probability distributions.

## 🎯 Key Features

- **Distributional Storage**: Store and retrieve entire probability distributions (Gaussian) rather than point estimates
- **Exponential Capacity**: Proven storage capacity of O(exp(d)) distributions in d-dimensional space
- **Wasserstein Geometry**: Operates in the Bures-Wasserstein space with proper geometric structure
- **Robust Retrieval**: Theoretical guarantees on retrieval fidelity under Wasserstein perturbations

## 📚 Background

Traditional associative memories store and retrieve vectors. Our framework extends this to **probability distributions**, enabling:
- Uncertainty-aware pattern storage
- Richer representation of complex data
- Natural handling of distributional data (e.g., Gaussian embeddings)

The model uses a log-sum-exp energy functional.

## 🚀 Installation

### Clone the repository
```bash
git clone https://github.com/chandantankala/DDAM.git
cd DDAM

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Quick Start
from ddam import DDAM, sample_commuting_gaussians, perturb_gaussian
import numpy as np

# Generate Gaussian patterns on Wasserstein sphere
N = 100  # Number of patterns
d = 50   # Dimension
patterns = sample_commuting_gaussians(N, d, radius=np.sqrt(2*d))

# Initialize DDAM
memory = DDAM(patterns, beta=1.0)

# Retrieve from noisy query
query = perturb_gaussian(patterns[0], perturbation_radius=0.1)
retrieved = memory.retrieve(query, max_iterations=10, verbose=True)

# Synthetic data experiments
from experiments.synthetic import convergence_experiment, plot_convergence_results

# Test convergence with different temperatures
results = convergence_experiment(N=1000, d=50, beta_values=[0.1, 1.0, 10.0])
plot_convergence_results(results)

# Word embeddings
from experiments.word_embeddings import word_retrieval_experiment

# Test on Gaussian word embeddings
results = word_retrieval_experiment(words, embeddings, beta_values=[1, 10, 50])
