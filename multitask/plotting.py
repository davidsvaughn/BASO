import sys, os
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import logging
from datetime import datetime
from glob import glob
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def kernel_regression(y, x=None, bandwidth=1.0, kernel='epanechnikov'):
    """
    Perform non-parametric kernel regression on 1D data.
    
    Parameters:
    -----------
    x : array-like
        The independent variable values of the training data.
    y : array-like
        The dependent variable values of the training data.
    x_new : array-like
        The points at which to evaluate the regression function.
    bandwidth : float, default=1.0
        The bandwidth parameter controlling the smoothness of the regression.
    kernel : str, default='gaussian'
        The kernel function to use. Options: 'gaussian', 'epanechnikov', 'uniform'.
        
    Returns:
    --------
    y_pred : array-like
        The predicted values at x_new.
    """
    x = np.asarray(x) if x is not None else np.arange(len(y))
    y = np.asarray(y)
    x_new = x
    
    
    # Define kernel functions
    def gaussian_kernel(u):
        return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    
    def epanechnikov_kernel(u):
        return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
    
    def uniform_kernel(u):
        return np.where(np.abs(u) <= 1, 0.5, 0)
    
    # Select the kernel function
    if kernel == 'gaussian':
        kernel_func = gaussian_kernel
    elif kernel == 'epanechnikov':
        kernel_func = epanechnikov_kernel
    elif kernel == 'uniform':
        kernel_func = uniform_kernel
    else:
        raise ValueError("Kernel must be 'gaussian', 'epanechnikov', or 'uniform'")
    
    # Perform kernel regression
    y_pred = np.zeros_like(x_new, dtype=float)
    
    for i, x_i in enumerate(x_new):
        # Calculate distances
        distances = (x_i - x) / bandwidth
        
        # Calculate weights
        weights = kernel_func(distances)
        
        # Normalize weights
        if np.sum(weights) != 0:
            weights = weights / np.sum(weights)
        
        # Calculate prediction
        y_pred[i] = np.sum(weights * y)
    
    return y_pred

def smooth_spline(values, smooth_factor=3):
    """
    Apply spline-based smoothing to a 1D array.
    
    Args:
        values (array-like): The input array to smooth
        smooth_factor (float): Controls the smoothness. Higher values = smoother curve.
                              This parameter is used differently than in gaussian_filter1d.
                              
    Returns:
        numpy.ndarray: The smoothed array
    """
    # Create x coordinates (evenly spaced indices)
    x = np.arange(len(values))
    
    # Create a univariate spline
    # The s parameter controls the smoothness - higher values = smoother curve
    # We scale the smooth_factor to make it roughly comparable to gaussian_filter1d's sigma
    s = len(values) * smooth_factor * 0.1
    
    # Fit the spline (handling potential errors)
    try:
        spline = UnivariateSpline(x, values, s=s)
        # Evaluate the spline at the original x coordinates
        smoothed_values = spline(x)
        return smoothed_values
    except Exception as e:
        print(f"Spline smoothing failed: {e}. Falling back to original values.")
        return values

# Set seaborn style for better visualization
sns.set_theme(style="whitegrid")




def plot_multiple_columns(test_indices=[1, 2, 3, 4, 17, 18, 22, 28, 37, 40, 66], filename='phi4-math-4claude.txt', 
                         smooth_factor=3, smooth_method='gaussian', show_original=False):
    """
    Create a single plot showing multiple TEST columns with different colors.
    Each column is represented by a smooth line (no dots) with a different color.
    The columns are normalized to range between 0.5 and 0.8, and smoothed for better visualization.
    
    Args:
        test_indices (list or int): Either a list of TEST column indices to plot (e.g., [1, 2, 3] for TEST_1, TEST_2, TEST_3)
                                   or an integer N to randomly select N columns
        filename (str): Name of the data file to use
        smooth_factor (float): Smoothing factor (sigma for Gaussian, smoothness for spline)
        smooth_method (str): Smoothing method to use ('gaussian' or 'spline')
        show_original (bool): If True, also plots the original unsmoothed values as dashed lines
        
    Returns:
        None: Saves the plot to the plots directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    plots_dir = os.path.join(current_dir, 'plots')
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    file_path = os.path.join(data_dir, filename)
    
    # Read the first line to get column names
    with open(file_path, 'r') as f:
        header_line = f.readline().strip()
    
    # Fix the header by adding a tab between CHECKPOINT and TEST_AVERAGE if needed
    if 'CHECKPOINTTEST_AVERAGE' in header_line:
        header_line = header_line.replace('CHECKPOINTTEST_AVERAGE', 'CHECKPOINT\tTEST_AVERAGE')
        
    # Split the header by tabs to get column names
    column_names = header_line.split('\t')
    
    # Read the data with the corrected column names
    df = pd.read_csv(file_path, delimiter='\t', names=column_names, skiprows=1)
    
    # Extract checkpoint numbers for x-axis
    X_feats = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get all column names and identify TEST columns
    all_columns = df.columns.tolist()
    all_test_columns = [col for col in all_columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']
    
    # Handle case where test_indices is an integer (randomly select N columns)
    if isinstance(test_indices, int):
        n_columns = min(test_indices, len(all_test_columns))  # Ensure we don't try to select more columns than available
        # Set a seed for reproducibility, but allow different runs to get different random selections
        random.seed(datetime.now().timestamp())
        selected_columns = random.sample(all_test_columns, n_columns)
        print(f"Randomly selected {n_columns} columns: {', '.join(selected_columns)}")
        
        # Collect data for the randomly selected columns
        data_to_plot = {}
        plotted_columns = []
        for col_name in selected_columns:
            data_to_plot[col_name] = df[col_name].values
            plotted_columns.append(col_name)
    else:
        # Handle case where test_indices is a list (use specified indices)
        data_to_plot = {}
        plotted_columns = []
        for test_idx in test_indices:
            col_name = f'TEST_{test_idx}'
            if col_name in df.columns:
                data_to_plot[col_name] = df[col_name].values
                plotted_columns.append(col_name)
            else:
                print(f"Warning: Column {col_name} not found in the data file.")
    
    # Normalize all columns to range between 0.5 and 0.8, with jitter for better visibility
    normalized_data = {}
    original_data = {}  # Store original data for optional plotting
    # Set a random seed based on column names for consistent jitter across runs
    np.random.seed(42)
    
    for col_name, values in data_to_plot.items():
        # Generate a unique seed for each column based on its name
        col_seed = sum(ord(c) for c in col_name)
        np.random.seed(col_seed)
        
        # Create jittered min and max values for normalization
        # This gives each curve a slightly different range for better visibility
        min_jitter = np.random.uniform(-0.05, 0.05)  # Small jitter for min value
        max_jitter = np.random.uniform(-0.05, 0.05)  # Small jitter for max value
        
        # Ensure min_value is always less than max_value by at least 0.15
        min_value = 0.5 + min_jitter
        max_value = 0.8 + max_jitter
        
        # Adjust if the range gets too small
        if max_value - min_value < 0.15:
            max_value = min_value + 0.15
        
        # Apply smoothing based on selected method
        if smooth_method.lower() == 'gaussian':
            smoothed_values = gaussian_filter1d(values, sigma=smooth_factor)
        elif smooth_method.lower() == 'spline':
            smoothed_values = smooth_spline(values, smooth_factor=smooth_factor)
        elif smooth_method.lower() == 'kernel':
            smoothed_values = kernel_regression(values, bandwidth=smooth_factor, kernel='epanechnikov')
        else:
            print(f"Warning: Unknown smoothing method '{smooth_method}'. Using Gaussian smoothing.")
            smoothed_values = gaussian_filter1d(values, sigma=smooth_factor)
        
        # Normalize both smoothed and original values to the jittered range
        # For smoothed values
        if np.max(smoothed_values) != np.min(smoothed_values):  # Avoid division by zero
            normalized_values = min_value + (max_value - min_value) * (smoothed_values - np.min(smoothed_values)) / (np.max(smoothed_values) - np.min(smoothed_values))
        else:
            normalized_values = np.full_like(smoothed_values, (min_value + max_value) / 2)  # Default to middle of range if all values are the same
        
        normalized_data[col_name] = normalized_values
        
        # For original values (if show_original is True)
        if show_original:
            if np.max(values) != np.min(values):  # Avoid division by zero
                # Use the same normalization range as the smoothed values for consistency
                original_normalized = min_value + (max_value - min_value) * (values - np.min(values)) / (np.max(values) - np.min(values))
            else:
                original_normalized = np.full_like(values, (min_value + max_value) / 2)
                
            original_data[col_name] = original_normalized
    
    # Reset random seed
    np.random.seed(None)
    
    # Create a custom color palette with more distinct colors
    palette = sns.color_palette("husl", len(normalized_data))
    
    # Plot each normalized and smoothed column with a different color
    for i, (col_name, values) in enumerate(normalized_data.items()):
        # Plot smoothed values with solid lines
        plt.plot(X_feats, values, linewidth=2.5, label=col_name, color=palette[i])
        
        # If show_original is True, also plot original values with dashed lines of the same color
        if show_original and col_name in original_data:
            plt.plot(X_feats, original_data[col_name], linestyle='--', linewidth=1.5, color=palette[i], alpha=0.7)
    
    # Set plot properties with seaborn styling
    plt.xlabel('Checkpoint', fontsize=14)
    plt.ylabel('Performance', fontsize=14)
    plt.title(f'Performance on Multiple Benchmarks', fontsize=16)
    # plt.legend(loc='best', frameon=True, framealpha=0.7)
    
    # Calculate the actual min and max values across all normalized data after jitter
    all_min = min(np.min(values) for values in normalized_data.values())
    all_max = max(np.max(values) for values in normalized_data.values())
    
    # Add a small margin to the limits
    margin = 0.02
    plt.ylim(all_min - margin, all_max + margin)
    
    # Add tight layout
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f'multiple_columns_plot_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved: {plot_path}")



def create_test_plots():
    """
    Create plots showing checkpoints on the x-axis and TEST column values on the y-axis.
    One plot is generated for each TEST column.
    All plots are saved in the multitask/plots directory.
    """
    fn = 'phi4-math-4claude.txt'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    plots_dir = os.path.join(current_dir, 'plots')
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # load data
    df = pd.read_csv(os.path.join(data_dir, fn), delimiter='\t')

    # Extract checkpoint numbers
    X_feats = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values
     
    # Identify test columns (excluding average)
    test_cols = [col for col in df.columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']
    Y_test = df[test_cols].values
    n, m = Y_test.shape
    
    # Create a plot for each TEST column
    for i, col_name in enumerate(test_cols):
        plt.figure(figsize=(10, 6))
        plt.plot(X_feats, Y_test[:, i], marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Checkpoint', fontsize=12)
        plt.ylabel(f'{col_name} Score', fontsize=12)
        plt.title(f'Performance of {col_name} Across Checkpoints', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(plots_dir, f'{col_name}_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Plot saved: {plot_path}")

def plot_clustered_columns(filename='phi4-math-4claude.txt', 
                          bandwidth=15, 
                          n_clusters=4, 
                          min_cluster_size=3,
                          corr_min = 0.95,
                          ):
    """
    Create a plot showing clustered TEST columns based on correlation between
    Epanechnikov kernel regression smoothed curves, using K-means clustering.
    
    Args:
        filename (str): Name of the data file to use
        bandwidth (float): Bandwidth parameter for Epanechnikov kernel regression
        n_clusters (int): Number of clusters for K-means
        min_cluster_size (int): Minimum number of members per cluster to display
        
    Returns:
        None: Saves the plot to the plots directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    plots_dir = os.path.join(current_dir, 'plots')
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    file_path = os.path.join(data_dir, filename)
    
    # Read the first line to get column names
    with open(file_path, 'r') as f:
        header_line = f.readline().strip()
    
    # Fix the header by adding a tab between CHECKPOINT and TEST_AVERAGE if needed
    if 'CHECKPOINTTEST_AVERAGE' in header_line:
        header_line = header_line.replace('CHECKPOINTTEST_AVERAGE', 'CHECKPOINT\tTEST_AVERAGE')
        
    # Split the header by tabs to get column names
    column_names = header_line.split('\t')
    
    # Read the data with the corrected column names
    df = pd.read_csv(file_path, delimiter='\t', names=column_names, skiprows=1)
    
    # Extract checkpoint numbers for x-axis
    X_feats = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values
    
    # Get all TEST columns (excluding average)
    all_columns = df.columns.tolist()
    all_test_columns = [col for col in all_columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']
    
    print(f"Processing {len(all_test_columns)} TEST columns...")
    
    # Apply Epanechnikov kernel regression to ALL columns
    smoothed_data = {}
    normalized_data = {}
    
    # Set a fixed range for normalization (no jitter needed for clustering)
    min_value = 0.5
    max_value = 0.8
    
    for col_name in all_test_columns:
        values = df[col_name].values
        
        # Apply Epanechnikov kernel regression
        smoothed_values = kernel_regression(values, bandwidth=bandwidth, kernel='epanechnikov')
        
        # Normalize to range [0.5, 0.8]
        if np.max(smoothed_values) != np.min(smoothed_values):  # Avoid division by zero
            normalized_values = min_value + (max_value - min_value) * (smoothed_values - np.min(smoothed_values)) / (np.max(smoothed_values) - np.min(smoothed_values))
        else:
            normalized_values = np.full_like(smoothed_values, (min_value + max_value) / 2)
            
        smoothed_data[col_name] = smoothed_values
        normalized_data[col_name] = normalized_values
    
    # Create a matrix of the smoothed data for correlation calculation
    smoothed_matrix = np.array([smoothed_data[col] for col in all_test_columns])
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(smoothed_matrix)
    
    # set all correlations below the threshold to 0
    corr_matrix[corr_matrix < corr_min] = 0
    
    # Convert correlation to distance (1 - correlation)
    # Higher correlation = smaller distance
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # Apply K-means clustering to the distance matrix
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    # Count members in each cluster
    cluster_counts = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_counts:
            cluster_counts[label] = []
        cluster_counts[label].append(all_test_columns[i])
    
    # Filter clusters with at least min_cluster_size members
    valid_clusters = {label: members for label, members in cluster_counts.items() 
                     if len(members) >= min_cluster_size}
    
    print(f"Found {len(valid_clusters)} clusters with at least {min_cluster_size} members:")
    for label, members in valid_clusters.items():
        print(f"  Cluster {label}: {len(members)} members")
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create a color palette for clusters
    cluster_palette = sns.color_palette("husl", len(valid_clusters))
    
    # Plot each cluster with its own color
    for i, (cluster_label, members) in enumerate(valid_clusters.items()):
        cluster_color = cluster_palette[i]
        
        # Plot each member of the cluster
        for j, col_name in enumerate(members):
            # Use the same color for all members of the cluster, with slight alpha variations
            alpha = 0.7 + 0.3 * (j / len(members))  # Vary alpha between 0.7 and 1.0
            plt.plot(X_feats, normalized_data[col_name], linewidth=2, 
                    color=cluster_color, alpha=alpha, 
                    label=f"{col_name} (Cluster {cluster_label})")
    
    # Set plot properties with seaborn styling
    plt.xlabel('Checkpoint', fontsize=14)
    plt.ylabel('Normalized Performance', fontsize=14)
    plt.title(f'Clustered Performance Curves (Epanechnikov Kernel, {n_clusters} clusters)', fontsize=16)
    
    # Add legend with cluster grouping
    handles, labels = plt.gca().get_legend_handles_labels()
    by_cluster = {}
    for handle, label in zip(handles, labels):
        cluster = label.split('(Cluster ')[1].split(')')[0]
        if cluster not in by_cluster:
            by_cluster[cluster] = []
        by_cluster[cluster].append((handle, label))
    
    # Create a legend with cluster grouping
    # legend_handles = []
    # legend_labels = []
    # for cluster, items in by_cluster.items():
    #     for handle, label in items:
    #         legend_handles.append(handle)
    #         legend_labels.append(label)
    #     # Add a separator between clusters (empty entry)
    #     if cluster != list(by_cluster.keys())[-1]:  # If not the last cluster
    #         legend_handles.append(plt.Line2D([0], [0], color='white'))
    #         legend_labels.append('')
    
    # plt.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), 
    #           fontsize=10, frameon=True, framealpha=0.7)
    
    # Calculate the actual min and max values across all normalized data
    all_min = min(np.min(values) for values in normalized_data.values())
    all_max = max(np.max(values) for values in normalized_data.values())
    
    # Add a small margin to the limits
    margin = 0.02
    plt.ylim(all_min - margin, all_max + margin)
    
    # Add tight layout
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f'clustered_columns_plot_{timestamp}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    # create_test_plots()
    
    # Example using default Gaussian smoothing
    # plot_multiple_columns(7)
    
    # Example using spline smoothing with original values shown
    # plot_multiple_columns(5, smooth_method='spline', smooth_factor=10, show_original=True)
    
    # plot_multiple_columns(5, smooth_method='kernel', smooth_factor=15, show_original=True)
    
    # Example using the new clustered columns function
    plot_clustered_columns(bandwidth=15, n_clusters=4, min_cluster_size=3)
    
    # Uncomment to compare both methods
    # plot_multiple_columns(7, smooth_method='gaussian', smooth_factor=3)
