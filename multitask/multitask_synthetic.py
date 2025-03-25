import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def generate_synthetic_checkpoint_data(X, n_rows, n_cols, random_state=0):
    """
    Generate synthetic checkpoint performance data similar to the original dataset.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Original data matrix where rows are model checkpoints and 
        columns are performance on different benchmarks
    n_rows : int
        Number of rows for the synthetic dataset
    n_cols : int
        Number of columns for the synthetic dataset
        
    Returns:
    --------
    D : numpy.ndarray
        Synthetic dataset with similar statistical properties to X
    """
    # Step 1: Compute column-wise statistics from the original data
    orig_means = np.mean(X, axis=0)
    orig_stds = np.std(X, axis=0)
    orig_cov = np.cov(X.T)
    
    # step 1.5: Normalize the data
    # X = (X - orig_means) / orig_stds
    
    # Step 2: Learn the temporal trends in the data using Gaussian Process
    orig_rows = np.arange(X.shape[0]).reshape(-1, 1)
    new_rows = np.linspace(0, X.shape[0]-1, n_rows).reshape(-1, 1)
    
    # Determine how many PCs to use
    pca = PCA()
    pca.fit(X)
    explained_var_ratio = pca.explained_variance_ratio_
    n_components = np.sum(np.cumsum(explained_var_ratio) <= 0.95) + 1
    n_components = min(n_components, min(X.shape[0], X.shape[1]))
    
    # Step 3: Compress the data using PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Step 4: Model each principal component with a Gaussian Process
    # kernels = [RBF() + WhiteKernel(noise_level=0.1) for _ in range(n_components)]
    kernels = [RBF(length_scale=X.shape[0]/4, length_scale_bounds=(1e-6, 1e6)) + 
               WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)) for _ in range(n_components)]
    gp_models = []
    
    for i in range(n_components):
        gpr = GaussianProcessRegressor(kernel=kernels[i], normalize_y=True, n_restarts_optimizer=10)
        gpr.fit(orig_rows, X_pca[:, i])
        gp_models.append(gpr)
    
    # Step 5: Generate synthetic PC values using the fitted GPs
    synthetic_pcs = np.zeros((n_rows, n_components))
    for i in range(n_components):
        synthetic_pcs[:, i] = gp_models[i].predict(new_rows)
    
    # Step 6: Back-project to the original space
    D_temp = pca.inverse_transform(synthetic_pcs)
    
    # Step 7: If we need to change the number of columns, do another PCA + scaling
    if n_cols != X.shape[1]:
        # Compute covariance of the generated data
        D_cov = np.cov(D_temp.T)
        
        # Generate random data with matching covariance structure
        target_cov = np.identity(n_cols)
        target_cov[:min(D_cov.shape[0], n_cols), :min(D_cov.shape[1], n_cols)] = D_cov[:min(D_cov.shape[0], n_cols), :min(D_cov.shape[1], n_cols)]
        
        # Add a small epsilon to the diagonal to ensure positive definiteness
        epsilon = 1e-6
        np.fill_diagonal(target_cov, np.diag(target_cov) + epsilon)
        
        # Generate normal random data and shape it to have our target covariance
        rng = np.random.RandomState(random_state)
        D_random = rng.randn(n_rows, n_cols)
        
        # Compute the Cholesky decomposition of our target covariance matrix
        cholesky = np.linalg.cholesky(target_cov)
        
        # Transform to the desired covariance structure
        D = D_random @ cholesky.T
        
        # Scale to match the mean and variance patterns
        time_indices = np.linspace(0, X.shape[0]-1, n_rows)
        for j in range(n_cols):
            # Estimate a mean curve using original data
            if j < X.shape[1]:
                kernel = RBF(length_scale=X.shape[0]/4, length_scale_bounds=(1e-6, 1e6)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
                gpr = GaussianProcessRegressor(kernel=kernel)
                gpr.fit(orig_rows, X[:, j])
                mean_curve = gpr.predict(new_rows)
                
                # Apply the temporal pattern
                D[:, j] = D[:, j] * orig_stds[min(j, orig_stds.size-1)] + mean_curve
            else:
                # For extra columns, use patterns from randomly selected columns
                _rand_col = np.random.randint(0, X.shape[1])
                rand_col = rng.choice(X.shape[1])
                
                kernel = RBF(length_scale=X.shape[0]/4, length_scale_bounds=(1e-6, 1e6)) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
                gpr = GaussianProcessRegressor(kernel=kernel)
                gpr.fit(orig_rows, X[:, rand_col])
                mean_curve = gpr.predict(new_rows)
                
                # Apply the temporal pattern with some random variation
                D[:, j] = D[:, j] * orig_stds[rand_col] + mean_curve
    else:
        # If keeping the same number of columns, just ensure we match the original statistics
        D = D_temp
        
        # Scale columns to match original means and variances
        for j in range(n_cols):
            D[:, j] = (D[:, j] - np.mean(D[:, j])) / np.std(D[:, j]) * orig_stds[j] + orig_means[j]
    
    return D