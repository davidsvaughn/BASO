import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def boolean_sample_kde(X, Y, bandwidth=0.05, num_points=1000):
    """
    Perform kernel density estimation for boolean sampled points.
    
    Parameters:
    -----------
    X : np.ndarray
        1D array of x coordinates
    Y : np.ndarray
        1D boolean array where True indicates a point was sampled
    bandwidth : float
        The bandwidth of the Gaussian kernel (controls smoothness)
    num_points : int
        Number of points to evaluate the density function
    
    Returns:
    --------
    x_eval : np.ndarray
        Points where density is evaluated
    density : np.ndarray
        Estimated density at each point in x_eval
    """
    # Extract the x coordinates where Y is True (sampled points)
    sampled_points = X[Y]
    
    if len(sampled_points) == 0:
        raise ValueError("No sampled points found (all Y values are False)")
    
    # Create a KDE using the sampled points
    kde = gaussian_kde(sampled_points, bw_method=bandwidth)
    
    # Create evaluation points spanning the range of X
    x_min, x_max = X.min(), X.max()
    x_eval = np.linspace(x_min, x_max, num_points)
    
    # Evaluate the density at these points
    density = kde(x_eval)
    
    return x_eval, density

# Example usage with visualization
def plot_kde_example(X, Y, bandwidth=0.05):
    """
    Plot the original data points and the estimated density.
    """
    # Compute KDE
    x_eval, density = boolean_sample_kde(X, Y, bandwidth)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot original data
    ax1.scatter(X[Y], np.ones_like(X[Y])*0.5, color='blue', label='Sampled Points')
    ax1.scatter(X[~Y], np.zeros_like(X[~Y])-0.5, color='red', label='Non-sampled Points')
    ax1.set_title('Original Data')
    ax1.set_ylabel('Sampled (1) / Not Sampled (0)')
    ax1.legend()
    ax1.set_yticks([-0.5, 0.5])
    ax1.set_yticklabels(['Not Sampled', 'Sampled'])
    
    # Plot KDE
    ax2.plot(x_eval, density, 'g-', label=f'KDE (bandwidth={bandwidth})')
    ax2.fill_between(x_eval, density, alpha=0.3)
    ax2.set_title('Kernel Density Estimation')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Example with custom bandwidth comparison
def compare_bandwidths(X, Y, bandwidths=[0.01, 0.05, 0.1, 0.2]):
    """
    Compare KDE with different bandwidth parameters.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot sampled points
    ax.scatter(X[Y], np.zeros_like(X[Y]), color='black', marker='|', s=30, label='Sampled Points')
    
    # Plot densities with different bandwidths
    for bw in bandwidths:
        x_eval, density = boolean_sample_kde(X, Y, bandwidth=bw)
        ax.plot(x_eval, density, label=f'Bandwidth = {bw}')
    
    ax.set_title('Comparison of KDE with Different Bandwidths')
    ax.set_xlabel('X')
    ax.set_ylabel('Density')
    ax.legend()
    
    return fig

# Manual implementation (without scipy) for better understanding
def manual_gaussian_kde(X, Y, bandwidth=0.05, num_points=1000):
    """
    Manually implement Gaussian KDE to illustrate the concept.
    """
    # Extract sampled points
    sampled_points = X[Y]
    
    # Create evaluation points
    x_min, x_max = X.min(), X.max()
    x_eval = np.linspace(x_min, x_max, num_points)
    
    # Initialize density array
    density = np.zeros(num_points)
    
    # For each sampled point, add a Gaussian centered at that point
    for point in sampled_points:
        # Gaussian function: exp(-0.5 * ((x - point)/bandwidth)^2)
        gaussian = np.exp(-0.5 * ((x_eval - point) / bandwidth) ** 2)
        density += gaussian
    
    # Normalize by number of points and bandwidth
    density = density / (len(sampled_points) * bandwidth * np.sqrt(2 * np.pi))
    
    return x_eval, density