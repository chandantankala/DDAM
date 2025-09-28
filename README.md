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

The model uses a log-sum-exp energy functional:
