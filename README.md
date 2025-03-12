# BASO
Bayesian Adaptive Sampling for Optimization

# GPyTorch-Accelerated Adaptive Sampling

This repository contains implementations of Gaussian Process-based adaptive sampling for efficiently finding the best checkpoint in machine learning models:

1. `claude_gpr3.py`: Original implementation using scikit-learn's GaussianProcessRegressor
2. `gpytorch_gpr.py`: Accelerated implementation using GPyTorch

## Benefits of GPyTorch Implementation

The GPyTorch implementation offers several advantages:

- **GPU Acceleration**: Leverages PyTorch's GPU support for faster training and inference
- **Scalability**: Better handling of larger datasets through modern GP techniques
- **Advanced Kernels**: More flexible kernel compositions and automatic relevance determination
- **Performance**: Significant speedup (typically 5-20x) over the scikit-learn implementation

## Requirements
