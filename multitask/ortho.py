import numpy as np
import matplotlib.pyplot as plt

def orthogonal_sampling_2d(n_samples, strength=2):
    """
    Generate a 2D orthogonal sampling pattern, which is a special type of
    Latin Hypercube Sample with additional orthogonality properties.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate. For orthogonal sampling, n_samples 
        should ideally be a perfect square.
    strength : int, optional
        Orthogonal array strength (default: 2).
        
    Returns:
    --------
    samples : ndarray, shape (n_samples, 2)
        Array of sample points in the unit square [0,1)×[0,1).
    """
    # For orthogonal sampling, we need n_samples to be a perfect square
    n = int(np.ceil(np.sqrt(n_samples)))
    n_squared = n * n
    
    # Create Latin Hypercube pattern first
    # Generate n divisions in each dimension
    perms = [np.random.permutation(n) for _ in range(2)]
    
    # Initialize the samples array
    samples = np.zeros((n_squared, 2))
    
    # Construct an orthogonal array of strength 2
    # For 2D, we can use a simple construction based on modular arithmetic
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            # This ensures orthogonality for strength=2
            samples[idx, 0] = perms[0][i]
            samples[idx, 1] = perms[1][(i + j) % n]
    
    # Scale to [0,1) and add jitter
    for d in range(2):
        samples[:, d] = (samples[:, d] + np.random.uniform(0, 1, n_squared)) / n
        
    # Truncate to the requested number of samples if needed
    if n_squared > n_samples:
        samples = samples[:n_samples]
    
    return samples

def latin_hypercube_sampling_2d(n_samples):
    """
    Generate a 2D Latin Hypercube sampling pattern.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate.
        
    Returns:
    --------
    samples : ndarray, shape (n_samples, 2)
        Array of sample points in the unit square [0,1)×[0,1).
    """
    # Initialize the samples array
    samples = np.zeros((n_samples, 2))
    
    # For each dimension, generate a Latin Hypercube sample
    for d in range(2):
        perm = np.random.permutation(n_samples)
        samples[:, d] = (perm + np.random.uniform(0, 1, n_samples)) / n_samples
        
    return samples

def random_sampling_2d(n_samples):
    """
    Generate random samples in 2D.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate.
        
    Returns:
    --------
    samples : ndarray, shape (n_samples, 2)
        Array of random sample points in the unit square [0,1)×[0,1).
    """
    return np.random.random((n_samples, 2))

def plot_samples(samples, title="2D Sampling", show_grid=True, n=None):
    """
    Plot the 2D sample points.
    
    Parameters:
    -----------
    samples : ndarray, shape (n_samples, 2)
        Array of sample points.
    title : str, optional
        Plot title.
    show_grid : bool, optional
        Whether to show the grid divisions.
    n : int, optional
        Number of grid divisions (if None, calculated from samples).
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.7)
    
    if show_grid:
        if n is None:
            n = int(np.ceil(np.sqrt(len(samples))))
        
        # Draw grid lines
        for i in range(1, n):
            plt.axhline(i/n, color='gray', alpha=0.3, linestyle='--')
            plt.axvline(i/n, color='gray', alpha=0.3, linestyle='--')
    
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def calculate_discrepancy(samples):
    """
    Calculate the centered L2-discrepancy, a measure of uniformity.
    Lower values indicate more uniform coverage.
    
    Parameters:
    -----------
    samples : ndarray, shape (n_samples, 2)
        Array of sample points.
        
    Returns:
    --------
    discrepancy : float
        The centered L2-discrepancy value.
    """
    n = len(samples)
    sum_term1 = 0
    for i in range(n):
        xi = samples[i]
        prod = 1
        for k in range(2):
            prod *= (1 + 0.5 * abs(xi[k] - 0.5) - 0.5 * abs(xi[k] - 0.5)**2)
        sum_term1 += prod
    
    sum_term2 = 0
    for i in range(n):
        for j in range(n):
            xi = samples[i]
            xj = samples[j]
            prod = 1
            for k in range(2):
                prod *= (1 + 0.5 * abs(xi[k] - 0.5) + 0.5 * abs(xj[k] - 0.5) 
                        - 0.5 * abs(xi[k] - xj[k]))
            sum_term2 += prod
    
    term1 = (13/12)**2
    term2 = -2/n * sum_term1
    term3 = 1/(n**2) * sum_term2
    
    return np.sqrt(term1 + term2 + term3)

# Example usage
if __name__ == "__main__":
    # Set the number of samples
    n_samples = 25
    
    # Generate samples using different methods
    oa_samples = orthogonal_sampling_2d(n_samples)
    lhs_samples = latin_hypercube_sampling_2d(n_samples)
    random_samples = random_sampling_2d(n_samples)
    
    # Plot the samples
    n_grid = int(np.ceil(np.sqrt(n_samples)))
    plot_samples(oa_samples, title=f"Orthogonal Sampling (n={n_samples})", n=n_grid)
    plot_samples(lhs_samples, title=f"Latin Hypercube Sampling (n={n_samples})", n=n_grid)
    plot_samples(random_samples, title=f"Random Sampling (n={n_samples})", n=n_grid)
    
    # Calculate and print discrepancy (lower is better)
    oa_disc = calculate_discrepancy(oa_samples)
    lhs_disc = calculate_discrepancy(lhs_samples)
    random_disc = calculate_discrepancy(random_samples)
    
    print(f"Discrepancy measures (lower is better):")
    print(f"Orthogonal Sampling: {oa_disc:.6f}")
    print(f"Latin Hypercube Sampling: {lhs_disc:.6f}")
    print(f"Random Sampling: {random_disc:.6f}")