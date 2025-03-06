import numpy as np
import matplotlib.pyplot as plt

def import_tab_data(filename):
    """
    Import tabular data from a tab-delimited file into a 2D numpy array.
    
    Args:
        filename (str): Path to the tab-delimited file
        
    Returns:
        np.ndarray: 2D array containing the imported data
    """
    try:
        # Load data using numpy's loadtxt function with tab delimiter
        data = np.loadtxt(filename, delimiter='\t')
        return data
    except Exception as e:
        print(f"Error importing data: {e}")
        return None
    
# plot each column of X on the same chart
def plot_cols(X, title='scores', prefix='item'):
    for i in range(X.shape[1]):
        plt.plot(X[:, i], label=f'{prefix}{i+1}')
    plt.legend()
    plt.title(title)
    plt.show()

# plot the histogram of the row(axis=1) / column(axis=0) -wise statistics
def stat_hist(X, f='mean', prefix='', axis=0, bins=10):
    func = np.mean if f == 'mean' else np.std
    y = func(X, axis=axis)
    plt.hist(y, bins=bins, color='skyblue', edgecolor='black')
    plt.title(f'{prefix}{f} (axis={axis}) Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    
# plot the histogram of the col-wise correlation coefficients
def corr_hist(X, name='column', bins=10):
    corr_matrix = np.corrcoef(X, rowvar=False)
    corr_values = corr_matrix[np.triu_indices(X.shape[1], k=1)]
    plt.hist(corr_values, bins=bins, color='skyblue', edgecolor='black')
    plt.title(f'{name}-wise Correlation Coefficients')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.show()

# Example usage with llama-siam-1.txt
if __name__ == "__main__":
    file_path = 'llama-siam-1.txt'  # Update path if needed
    X = import_tab_data(file_path)
    if X is not None:
        print(f"Successfully imported data with shape: {X.shape}")
        print(f"First few rows:\n{X[:5]}")  # Display first 5 rows
        
    # fishers transformation
    # X = np.arctanh(X)
    # X = 0.5 * np.log((1 + X) / (1 - X))
        
    # plot each column of X on the same chart
    plot_cols(X, prefix='item', title='scores: ' + file_path.split('/')[-1].removesuffix('.txt'))
    
    # plot the histogram of the col-wise stds
    stat_hist(X, f='std', prefix='per-item score ', axis=0, bins=8)
    
    # # plot the histogram of the row-wise means
    # stat_hist(X, f='mean', prefix='per-model score ', axis=1, bins=20)
    
    # plot the histogram of the col-wise correlation coefficients
    corr_hist(X, name='item', bins=10)
    
    # plot the histogram of the row-wise correlation coefficients
    corr_hist(X.T, name='checkpoint', bins=30)
    