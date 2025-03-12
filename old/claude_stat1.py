import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from ruptures import Pelt  # Optional: for advanced change point detection
import warnings
warnings.filterwarnings('ignore')

# pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def load_data(file_path):
    """Load and preprocess the ML checkpoint performance data"""
    df = pd.read_csv(file_path, sep='\t')
    
    # Extract checkpoint number from checkpoint name
    df['checkpoint_num'] = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1]))
    
    # Sort by checkpoint number to ensure chronological order
    df = df.sort_values('checkpoint_num')
    
    return df

def analyze_performance_trend(df):
    """Analyze and visualize the overall performance trend"""
    x = df['checkpoint_num'].values.reshape(-1, 1)
    y = df['TEST_AVERAGE'].values
    
    # Create figure for overall trend
    plt.figure()
    plt.plot(x, y, 'o', alpha=0.7, label='Checkpoint Performance')
    
    # Find checkpoint with max performance
    max_idx = np.argmax(y)
    max_checkpoint = df.iloc[max_idx]['CHECKPOINT']
    max_performance = y[max_idx]
    max_checkpoint_num = x[max_idx][0]
    
    plt.axvline(x=max_checkpoint_num, color='red', linestyle='--', 
                label=f'Max at {max_checkpoint} ({max_performance:.4f})')
    
    plt.xlabel('Checkpoint Number')
    plt.ylabel('Average Test Performance')
    plt.title('ML Model Performance Across Checkpoints')
    plt.legend()
    plt.tight_layout()
    
    return max_checkpoint, max_checkpoint_num, max_performance

def polynomial_regression_analysis(df, degrees=[2, 3, 4]):
    """Fit polynomial regression models of various degrees"""
    x = df['checkpoint_num'].values.reshape(-1, 1)
    y = df['TEST_AVERAGE'].values
    
    plt.figure()
    plt.plot(x, y, 'o', label='Data', alpha=0.7)
    
    models = {}
    
    for degree in degrees:
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(x)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        # Store model and predictions
        models[degree] = {
            'model': model,
            'y_pred': y_pred,
            'poly': poly
        }
        
        # Plot model
        plt.plot(x, y_pred, label=f'Polynomial (degree={degree})')
        
        # For quadratic model, calculate vertex
        if degree == 2:
            a, b, c = model.coef_[2], model.coef_[1], model.intercept_
            vertex_x = -b / (2 * a)
            vertex_y = model.predict(poly.transform([[vertex_x]]))[0]
            
            plt.axvline(x=vertex_x, color='green', linestyle='--', 
                       label=f'Quadratic Peak: {vertex_x:.0f}')
            
            print(f"Quadratic model vertex (peak/trough): checkpoint {vertex_x:.0f}, score {vertex_y:.6f}")
            
            if a < 0:
                print("Quadratic model suggests performance peaks then declines (∩ shape)")
            else:
                print("Quadratic model suggests performance declines then increases (∪ shape)")
    
    plt.xlabel('Checkpoint Number')
    plt.ylabel('Average Test Performance')
    plt.title('Polynomial Regression Models of Performance Trend')
    plt.legend()
    plt.tight_layout()
    
    return models

def loess_smoothing_analysis(df, frac=0.3):
    """Apply LOESS smoothing to detect trend"""
    x = df['checkpoint_num'].values
    y = df['TEST_AVERAGE'].values
    
    # Apply LOESS smoothing
    smoothed = lowess(y, x, frac=frac)
    df_smoothed = pd.DataFrame({'checkpoint_num': smoothed[:, 0], 'smoothed_avg': smoothed[:, 1]})
    
    # Plot original data and smoothed curve
    plt.figure()
    plt.plot(x, y, 'o', alpha=0.5, label='Original Data')
    plt.plot(df_smoothed['checkpoint_num'], df_smoothed['smoothed_avg'], 'r-', 
             linewidth=2, label=f'LOESS Smoothed (frac={frac})')
    
    # Find the peak in the smoothed curve
    peak_idx = df_smoothed['smoothed_avg'].idxmax()
    peak_checkpoint_num = df_smoothed.iloc[peak_idx]['checkpoint_num']
    peak_performance = df_smoothed.iloc[peak_idx]['smoothed_avg']
    
    plt.axvline(x=peak_checkpoint_num, color='green', linestyle='--', 
               label=f'Peak at checkpoint {peak_checkpoint_num:.0f}')
    
    plt.xlabel('Checkpoint Number')
    plt.ylabel('Average Test Performance')
    plt.title('LOESS Smoothing of Performance Trend')
    plt.legend()
    plt.tight_layout()
    
    print(f"Smoothed curve peak at checkpoint: {peak_checkpoint_num:.0f} with score {peak_performance:.6f}")
    
    return df_smoothed, peak_checkpoint_num, peak_performance

def gaussian_process_regression(df):
    """Apply Gaussian Process Regression"""
    x = df['checkpoint_num'].values.reshape(-1, 1)
    y = df['TEST_AVERAGE'].values
    
    # Scale input for better numerical stability
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Define different kernels to try
    kernels = {
        "RBF": 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5),
        "Matern": 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-5)
    }
    
    plt.figure()
    plt.plot(x, y, 'o', label='Data', alpha=0.7)
    
    results = {}
    
    for name, kernel in kernels.items():
        # Fit GP model
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        gpr.fit(x_scaled, y)
        
        # Create points for prediction
        x_pred = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        x_pred_scaled = scaler.transform(x_pred)
        
        # Get predictions and standard deviations
        y_pred, std = gpr.predict(x_pred_scaled, return_std=True)
        
        results[name] = {
            'model': gpr,
            'x_pred': x_pred,
            'y_pred': y_pred,
            'std': std
        }
        
        # Plot predictions with confidence intervals
        plt.plot(x_pred, y_pred, label=f'{name} Mean')
        plt.fill_between(
            x_pred.ravel(), 
            y_pred - 1.96 * std, 
            y_pred + 1.96 * std, 
            alpha=0.2, 
            label=f'{name} 95% CI'
        )
        
        # Find the peak in the GPR curve
        peak_idx = np.argmax(y_pred)
        peak_x = x_pred[peak_idx][0]
        peak_y = y_pred[peak_idx]
        print(f"{name} GPR peak at checkpoint: {peak_x:.0f} with predicted score {peak_y:.6f}")
    
    plt.xlabel('Checkpoint Number')
    plt.ylabel('Average Test Performance')
    plt.title('Gaussian Process Regression Models of Performance Trend')
    plt.legend()
    plt.tight_layout()
    
    return results

def mixed_effects_analysis(df):
    """Basic mixed-effects model analysis"""
    # Reshape data to long format for mixed-effects modeling
    test_cols = [col for col in df.columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']
    
    df_long = pd.melt(
        df, 
        id_vars=['CHECKPOINT', 'checkpoint_num'], 
        value_vars=test_cols,
        var_name='test_id', 
        value_name='performance'
    )
    
    # Create a simple mixed-effects model with quadratic trend
    try:
        formula = 'performance ~ checkpoint_num + I(checkpoint_num**2)'
        md = smf.mixedlm(formula, df_long, groups=df_long['test_id'])
        mdf = md.fit()
        
        print("\nMixed Effects Model Summary:")
        print(mdf.summary().tables[1])  # Print only the coefficients table for brevity
        
        # Extract coefficients and calculate the peak
        beta_0 = mdf.params['Intercept']
        beta_1 = mdf.params['checkpoint_num']
        beta_2 = mdf.params['I(checkpoint_num**2)']
        
        # Calculate vertex
        peak_x_mixed = -beta_1 / (2 * beta_2)
        peak_y_mixed = beta_0 + beta_1 * peak_x_mixed + beta_2 * peak_x_mixed**2
        
        print(f"Mixed-effects model predicts peak at checkpoint: {peak_x_mixed:.0f} with score {peak_y_mixed:.6f}")
        
        return mdf, peak_x_mixed, peak_y_mixed
    except:
        print("Mixed-effects model failed to converge or encountered other issues.")
        return None, None, None

def piecewise_linear_regression(df):
    """Piecewise linear regression for change point detection"""
    x = df['checkpoint_num'].values
    y = df['TEST_AVERAGE'].values
    
    def piecewise_linear(x, x0, y0, k1, k2):
        """Piecewise linear function with two segments"""
        return np.where(x < x0, y0 + k1 * (x - x0), y0 + k2 * (x - x0))
    
    def piecewise_linear_residuals(params, x, y):
        """Calculate residuals for least squares optimization"""
        x0, y0, k1, k2 = params
        return y - piecewise_linear(x, x0, y0, k1, k2)
    
    # Initial guess: changepoint in the middle, mean performance, and small slopes
    initial_params = [np.median(x), np.mean(y), 0.0001, -0.0001]
    
    # Fit the model
    result = minimize(
        lambda params: np.sum(piecewise_linear_residuals(params, x, y)**2),
        initial_params,
        method='Nelder-Mead'
    )
    
    changepoint, y_at_change, slope1, slope2 = result.x
    
    print("\nChange-Point Analysis Results:")
    print(f"Change point detected at checkpoint number: {changepoint:.0f}")
    print(f"Performance at change point: {y_at_change:.6f}")
    print(f"Slope before change point: {slope1:.6f}")
    print(f"Slope after change point: {slope2:.6f}")
    
    # Plot the piecewise linear model
    plt.figure()
    plt.plot(x, y, 'o', alpha=0.5, label='Original Data')
    
    # Sort x for plotting continuous line
    x_sorted = np.sort(x)
    y_pwl = piecewise_linear(x_sorted, changepoint, y_at_change, slope1, slope2)
    
    plt.plot(x_sorted, y_pwl, 'r-', linewidth=2, label='Piecewise Linear Model')
    plt.axvline(x=changepoint, color='g', linestyle='--', label=f'Change Point at {changepoint:.0f}')
    
    plt.xlabel('Checkpoint Number')
    plt.ylabel('Average Test Performance')
    plt.title('Change-Point Analysis of Performance Trend')
    plt.legend()
    plt.tight_layout()
    
    return changepoint, y_at_change, slope1, slope2

def analyze_test_correlations(df):
    """Analyze correlations between different test sets"""
    test_cols = [col for col in df.columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']
    
    # Calculate correlation matrix
    corr_matrix = df[test_cols].corr()
    
    # Plot heatmap of correlations
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
    
    with sns.axes_style("white"):
        sns.heatmap(corr_matrix, mask=mask, center=0, cmap="coolwarm",
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Correlation Matrix Between Test Sets')
    plt.tight_layout()
    
    # Get summary statistics on correlations
    corr_values = corr_matrix.values
    # Get upper triangle excluding diagonal
    mask = np.triu_indices_from(corr_values, k=1)
    corrs = corr_values[mask]
    
    print("\nStatistics on test set correlations:")
    print(f"Mean correlation: {np.mean(corrs):.4f}")
    print(f"Median correlation: {np.median(corrs):.4f}")
    print(f"Min correlation: {np.min(corrs):.4f}")
    print(f"Max correlation: {np.max(corrs):.4f}")
    print(f"Standard deviation: {np.std(corrs):.4f}")
    
    # See what percentage of correlations are above certain thresholds
    thresholds = [0.7, 0.8, 0.9]
    for threshold in thresholds:
        pct = (corrs > threshold).mean() * 100
        print(f"Percentage of correlations > {threshold}: {pct:.2f}%")
    
    return corr_matrix, corrs

def advanced_change_point_detection(df, model="l2", min_size=3):
    """
    Use the ruptures package for advanced change point detection
    Requires: pip install ruptures
    """
    try:
        x = df['checkpoint_num'].values
        y = df['TEST_AVERAGE'].values
        signal = y.reshape(-1, 1)  # Reshape for ruptures
        
        # Initialize detection algorithm
        algo = Pelt(model=model, min_size=min_size).fit(signal)
        
        # Find optimal number of change points
        result = algo.predict(pen=0.1)  # Penalty parameter, adjust as needed
        
        # Plot results
        plt.figure()
        plt.plot(x, y, 'o-', alpha=0.5, label='Performance')
        
        for cp in result[:-1]:  # exclude the last change point which is the signal end
            plt.axvline(x=x[cp], color='r', linestyle='--')
        
        # Highlight segments
        for idx, (start, end) in enumerate(zip([0] + result[:-1], result)):
            plt.axvspan(x[start], x[end-1], alpha=0.1, color=f'C{idx+1}')
        
        plt.xlabel('Checkpoint Number')
        plt.ylabel('Average Test Performance')
        plt.title('Advanced Change Point Detection')
        plt.legend()
        plt.tight_layout()
        
        print("\nAdvanced Change Point Detection:")
        print(f"Detected {len(result)-1} change points at checkpoints: {[x[cp] for cp in result[:-1]]}")
        
        return result
    except ImportError:
        print("Note: Advanced change point detection requires the 'ruptures' package.")
        print("Install with: pip install ruptures")
        return None

def comprehensive_analysis(file_path):
    """Run all analyses in sequence"""
    print("Loading data...")
    df = load_data(file_path)
    print(f"Loaded data with {df.shape[0]} checkpoints and {df.shape[1]} columns")
    
    print("\n==== Overall Performance Trend ====")
    max_checkpoint, max_checkpoint_num, max_performance = analyze_performance_trend(df)
    print(f"Checkpoint with highest average performance: {max_checkpoint} (#{max_checkpoint_num}) with score {max_performance:.6f}")
    
    print("\n==== Polynomial Regression Analysis ====")
    poly_models = polynomial_regression_analysis(df)
    
    print("\n==== LOESS Smoothing Analysis ====")
    df_smoothed, peak_cp_loess, peak_perf_loess = loess_smoothing_analysis(df)
    
    print("\n==== Gaussian Process Regression ====")
    gpr_results = gaussian_process_regression(df)
    
    print("\n==== Mixed Effects Analysis ====")
    mixed_model, peak_cp_mixed, peak_perf_mixed = mixed_effects_analysis(df)
    
    print("\n==== Change Point Analysis ====")
    changepoint, y_at_change, slope1, slope2 = piecewise_linear_regression(df)
    
    print("\n==== Test Correlation Analysis ====")
    corr_matrix, corrs = analyze_test_correlations(df)
    
    try:
        print("\n==== Advanced Change Point Detection ====")
        change_points = advanced_change_point_detection(df)
    except:
        print("Advanced change point detection skipped")
    
    print("\n==== Summary of Results ====")
    print(f"Raw data max performance: Checkpoint {max_checkpoint_num} with score {max_performance:.6f}")
    print(f"Quadratic model peak: Checkpoint {-poly_models[2]['model'].coef_[1] / (2 * poly_models[2]['model'].coef_[2]):.0f}")
    print(f"LOESS smoothing peak: Checkpoint {peak_cp_loess:.0f} with score {peak_perf_loess:.6f}")
    print(f"Piecewise linear change point: Checkpoint {changepoint:.0f}")
    
    if peak_cp_mixed is not None:
        print(f"Mixed-effects model peak: Checkpoint {peak_cp_mixed:.0f} with score {peak_perf_mixed:.6f}")
    
    plt.show()
    
    return df

if __name__ == "__main__":
    # Example usage
    file_path = "data/phi4-math-4claude.txt"  # Path to your data file
    df = comprehensive_analysis(file_path)