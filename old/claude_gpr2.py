import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, DotProduct
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import norm
import time
import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warnings

class AdaptiveSampler:
    """
    Adaptive sequential sampling to efficiently estimate peak checkpoint performance
    using Gaussian Process Regression and Bayesian Optimization techniques.
    """
    
    def __init__(self, data_file, random_seed=42):
        """
        Initialize the adaptive sampler.
        
        Parameters:
        -----------
        data_file : str
            Path to the TSV file containing checkpoint test results
        random_seed : int
            Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Load the full dataset (this would be unknown in a real scenario)
        self.full_data = pd.read_csv(data_file, sep='\t')
        
        # Extract checkpoint numbers
        self.full_data['checkpoint_num'] = self.full_data['CHECKPOINT'].apply(
            lambda x: int(x.split('-')[1])
        )
        
        # Identify test columns (excluding average)
        self.test_cols = [col for col in self.full_data.columns 
                          if col.startswith('TEST_') and col != 'TEST_AVERAGE']
        
        # Create matrix of all test results (checkpoints × tests)
        self.all_results = self.full_data[self.test_cols].values
        
        # Initialize mask for sampled entries (0 = unsampled, 1 = sampled)
        self.sampled_mask = np.zeros_like(self.all_results, dtype=bool)
        
        # Initialize sampled data structures
        self.X_sampled = []  # Will hold (checkpoint_num, test_idx) pairs
        self.y_sampled = []  # Will hold corresponding test results
        
        # Create feature scalers and encoders
        self.checkpoint_scaler = StandardScaler()
        self.test_encoder = OneHotEncoder(sparse_output=False)
        
        # Precompute scaled checkpoint numbers
        checkpoint_nums = self.full_data['checkpoint_num'].values.reshape(-1, 1)
        self.checkpoint_scaler.fit(checkpoint_nums)
        
        # Precompute encoded test indices
        test_indices = np.arange(len(self.test_cols)).reshape(-1, 1)
        self.test_encoder.fit(test_indices)
        
        # Setup GP model with a suitable kernel
        # This combines RBF kernel for checkpoint (continuous) with a more appropriate kernel for test indices
        self.kernel = (
            ConstantKernel(1.0) * 
            Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(0.1, 100.0)) + 
            WhiteKernel(noise_level=0.01)
        )
        
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,  # Small alpha to avoid numerical issues
            normalize_y=True,
            n_restarts_optimizer=5, # Reduced for speed
            random_state=random_seed
        )
        
        # Performance metrics
        self.true_best_checkpoint = None
        self.estimated_best_checkpoints = []
        self.sampling_fractions = []
        self.errors = []
    
    def _encode_features(self, X):
        """
        Encode features appropriately for GP model.
        X should be a list or array of [checkpoint_num, test_idx] pairs.
        """
        X = np.array(X)
        checkpoint_nums = X[:, 0].reshape(-1, 1)
        test_indices = X[:, 1].reshape(-1, 1).astype(int)
        
        # Scale checkpoint numbers
        scaled_checkpoints = self.checkpoint_scaler.transform(checkpoint_nums)
        
        # One-hot encode test indices
        encoded_tests = self.test_encoder.transform(test_indices)
        
        # Return concatenated features
        return np.hstack([scaled_checkpoints, encoded_tests])
    
    def get_true_best_checkpoint(self):
        """Calculate the true best checkpoint based on all data (for validation)"""
        avg_performance = self.full_data['TEST_AVERAGE'].values
        best_idx = np.argmax(avg_performance)
        self.true_best_checkpoint = self.full_data.iloc[best_idx]['checkpoint_num']
        return self.true_best_checkpoint
    
    def initialize_sampling(self, n_initial=10, strategy='random'):
        """
        Initialize sampling with a few samples to start the process.
        
        Parameters:
        -----------
        n_initial : int
            Number of initial samples
        strategy : str
            Sampling strategy: 'random', 'grid', or 'extremes'
        """
        n_checkpoints = len(self.full_data)
        n_tests = len(self.test_cols)
        
        if strategy == 'random':
            # Random sampling
            checkpoint_indices = np.random.choice(n_checkpoints, n_initial, replace=True)
            test_indices = np.random.choice(n_tests, n_initial, replace=True)
            
        elif strategy == 'grid':
            # Grid sampling of checkpoints and tests
            checkpoint_step = max(1, n_checkpoints // int(np.sqrt(n_initial)))
            test_step = max(1, n_tests // int(np.sqrt(n_initial)))
            
            checkpoint_indices = []
            test_indices = []
            
            for i in range(0, n_checkpoints, checkpoint_step):
                for j in range(0, n_tests, test_step):
                    if len(checkpoint_indices) < n_initial:
                        checkpoint_indices.append(i)
                        test_indices.append(j)
            
        elif strategy == 'extremes':
            # Sample from beginning, middle, and end of checkpoints
            # and a variety of tests
            checkpoint_indices = [0, n_checkpoints//2, n_checkpoints-1] * (n_initial // 3 + 1)
            test_indices = list(range(min(n_tests, n_initial)))
            
            # Ensure we have the right number
            while len(checkpoint_indices) < n_initial:
                checkpoint_indices.append(np.random.randint(0, n_checkpoints))
                test_indices.append(np.random.randint(0, n_tests))
                
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
        
        # Collect samples
        for cp_idx, test_idx in zip(checkpoint_indices[:n_initial], test_indices[:n_initial]):
            self._add_sample(cp_idx, test_idx)
    
    def _add_sample(self, checkpoint_idx, test_idx):
        """
        Add a single sample to our collection.
        
        Parameters:
        -----------
        checkpoint_idx : int
            Index of the checkpoint in the dataframe
        test_idx : int
            Index of the test column
        """
        # Mark as sampled in the mask
        self.sampled_mask[checkpoint_idx, test_idx] = True
        
        # Get the result value
        result = self.all_results[checkpoint_idx, test_idx]
        
        # Get the checkpoint number
        checkpoint_num = self.full_data.iloc[checkpoint_idx]['checkpoint_num']
        
        # Store the sample
        self.X_sampled.append([checkpoint_num, test_idx])
        self.y_sampled.append(result)
    
    def update_model(self):
        """Update the Gaussian Process model with current samples"""
        if len(self.y_sampled) < 5:
            print("Warning: Very few samples available. Model may be unreliable.")
            return None
            
        # Convert to numpy arrays
        X = np.array(self.X_sampled)
        y = np.array(self.y_sampled)
        
        # Encode features
        X_encoded = self._encode_features(X)
        
        try:
            # Fit GP model
            self.gp.fit(X_encoded, y)
            return self.gp
        except Exception as e:
            print(f"Error fitting GP model: {str(e)}")
            return None
    
    def predict_performance(self, checkpoint_nums=None):
        """
        Predict mean performance for each checkpoint across all tests.
        
        Parameters:
        -----------
        checkpoint_nums : array-like or None
            Checkpoint numbers to predict for. If None, use all checkpoints.
            
        Returns:
        --------
        DataFrame with predictions for each checkpoint
        """
        if checkpoint_nums is None:
            checkpoint_nums = self.full_data['checkpoint_num'].unique()
        
        results_list = []
        
        # For each checkpoint, predict for all tests
        for cp_num in checkpoint_nums:
            test_predictions = []
            test_uncertainties = []
            
            # Create features for all tests for this checkpoint
            features = [[cp_num, test_idx] for test_idx in range(len(self.test_cols))]
            
            # Encode features
            X_encoded = self._encode_features(features)
            
            # Predict with GP
            try:
                y_pred, y_std = self.gp.predict(X_encoded, return_std=True)
                
                # Average over all tests
                avg_pred = np.mean(y_pred)
                avg_std = np.mean(y_std)
                
                results_list.append({
                    'checkpoint_num': cp_num,
                    'predicted_value': avg_pred,
                    'uncertainty': avg_std
                })
            except Exception as e:
                print(f"Error predicting for checkpoint {cp_num}: {str(e)}")
                # Use fallback values
                results_list.append({
                    'checkpoint_num': cp_num,
                    'predicted_value': 0.0,
                    'uncertainty': 1.0
                })
        
        # Convert to DataFrame
        return pd.DataFrame(results_list)
    
    def get_best_checkpoint(self):
        """Get the current estimate of the best checkpoint"""
        predictions = self.predict_performance()
        if predictions.empty:
            # If predictions failed, return most sampled checkpoint
            checkpoint_counts = np.sum(self.sampled_mask, axis=1)
            best_idx = np.argmax(checkpoint_counts)
            return self.full_data.iloc[best_idx]['checkpoint_num']
            
        best_idx = predictions['predicted_value'].idxmax()
        return int(predictions.iloc[best_idx]['checkpoint_num'])
    
    def acquisition_function(self, strategy='ucb', kappa=2.0):
        """
        Calculate acquisition values for all unsampled points.
        
        Parameters:
        -----------
        strategy : str
            Acquisition strategy: 'ucb', 'ei', or 'thompson'
        kappa : float
            Exploration weight for UCB
            
        Returns:
        --------
        Tuple of (checkpoint_idx, test_idx) for the next sample
        """
        # Get indices of unsampled points
        unsampled_indices = np.where(~self.sampled_mask)
        checkpoint_indices = unsampled_indices[0]
        test_indices = unsampled_indices[1]
        
        # If no unsampled points left, return random point
        if len(checkpoint_indices) == 0:
            return (np.random.randint(0, len(self.full_data)),
                    np.random.randint(0, len(self.test_cols)))
        
        # Create feature matrix for unsampled points
        X_unsampled = []
        for cp_idx, test_idx in zip(checkpoint_indices, test_indices):
            checkpoint_num = self.full_data.iloc[cp_idx]['checkpoint_num']
            X_unsampled.append([checkpoint_num, test_idx])
        
        # Encode features
        X_unsampled_encoded = self._encode_features(X_unsampled)
        
        # Predict with GP
        try:
            mu, sigma = self.gp.predict(X_unsampled_encoded, return_std=True)
        except Exception as e:
            print(f"Error in acquisition function: {str(e)}")
            # If prediction fails, return random point
            rand_idx = np.random.randint(0, len(checkpoint_indices))
            return checkpoint_indices[rand_idx], test_indices[rand_idx]
        
        if strategy == 'ucb':
            # Upper Confidence Bound
            acquisition_values = mu + kappa * sigma
            
        elif strategy == 'ei':
            # Expected Improvement
            current_best = np.max(self.y_sampled) if self.y_sampled else 0
            imp = mu - current_best
            Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma > 0)
            acquisition_values = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            
        elif strategy == 'thompson':
            # Thompson Sampling
            acquisition_values = np.random.normal(mu, sigma)
            
        else:
            raise ValueError(f"Unknown acquisition strategy: {strategy}")
        
        # Find point with highest acquisition value
        best_idx = np.argmax(acquisition_values)
        next_checkpoint_idx = checkpoint_indices[best_idx]
        next_test_idx = test_indices[best_idx]
        
        return next_checkpoint_idx, next_test_idx
    
    def sample_next_point(self, strategy='ucb', kappa=2.0):
        """
        Sample the next point based on the acquisition function.
        
        Parameters:
        -----------
        strategy : str
            Acquisition strategy: 'ucb', 'ei', or 'thompson'
        kappa : float
            Exploration weight for UCB
            
        Returns:
        --------
        Tuple of (checkpoint_idx, test_idx) that was sampled
        """
        # Update the model first
        if self.update_model() is None:
            # If model update failed, sample randomly
            next_cp_idx = np.random.randint(0, len(self.full_data))
            next_test_idx = np.random.randint(0, len(self.test_cols))
        else:
            # Get next point from acquisition function
            next_cp_idx, next_test_idx = self.acquisition_function(strategy, kappa)
        
        # Add the sample
        self._add_sample(next_cp_idx, next_test_idx)
        
        return next_cp_idx, next_test_idx
    
    def run_sampling(self, n_samples, acquisition_strategy='ucb', kappa=2.0,
                    initial_samples=10, init_strategy='grid', verbose=True):
        """
        Run the complete adaptive sampling procedure.
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples to collect
        acquisition_strategy : str
            Acquisition strategy: 'ucb', 'ei', or 'thompson'
        kappa : float
            Exploration weight for UCB
        initial_samples : int
            Number of initial samples
        init_strategy : str
            Initial sampling strategy
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        Results after sampling
        """
        # Calculate true best checkpoint (for comparison)
        true_best = self.get_true_best_checkpoint()
        
        if verbose:
            print(f"True best checkpoint: {true_best}")
            print(f"Initializing with {initial_samples} samples using {init_strategy} strategy...")
            
        # Initialize
        self.initialize_sampling(n_initial=initial_samples, strategy=init_strategy)
        
        # Track performance
        total_samples = self.all_results.size
        checkpoint_range = self.full_data['checkpoint_num'].max() - self.full_data['checkpoint_num'].min()
        
        # Get initial estimate
        self.update_model()
        current_best = self.get_best_checkpoint()
        error = abs(current_best - true_best) / checkpoint_range
        
        self.estimated_best_checkpoints.append(current_best)
        self.sampling_fractions.append(len(self.y_sampled) / total_samples)
        self.errors.append(error)
        
        if verbose:
            print(f"Initial estimate: Checkpoint {current_best}, Error: {error:.4f}")
        
        # Collect remaining samples adaptively
        remaining_samples = n_samples - initial_samples
        
        for i in range(remaining_samples):
            start_time = time.time()
            
            # Sample next point
            next_cp_idx, next_test_idx = self.sample_next_point(
                strategy=acquisition_strategy, kappa=kappa
            )
            
            # Update performance metrics
            current_best = self.get_best_checkpoint()
            error = abs(current_best - true_best) / checkpoint_range
            
            self.estimated_best_checkpoints.append(current_best)
            self.sampling_fractions.append(len(self.y_sampled) / total_samples)
            self.errors.append(error)
            
            elapsed = time.time() - start_time
            
            if verbose and (i+1) % 10 == 0:
                print(f"Sample {i+1+initial_samples}/{n_samples}: "
                      f"Added ({self.full_data.iloc[next_cp_idx]['CHECKPOINT']}, "
                      f"{self.test_cols[next_test_idx]})")
                print(f"Current estimate: Checkpoint {current_best}, Error: {error:.4f}")
                print(f"Sampling fraction: {self.sampling_fractions[-1]:.4f}, "
                      f"Time: {elapsed:.2f}s")
            
        if verbose:
            print(f"\nFinal estimate after {n_samples} samples: Checkpoint {current_best}")
            print(f"Final error: {error:.4f}")
            print(f"Total sampling fraction: {len(self.y_sampled) / total_samples:.4f}")
        
        return self.predict_performance()
    
    def plot_results(self, figsize=(12, 10), save_path=None):
        """
        Plot results of the adaptive sampling procedure.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str or None
            Path to save the figure, if provided
        """
        if not self.estimated_best_checkpoints:
            print("No sampling results to plot. Run sampling first.")
            return
            
        fig, axs = plt.subplots(3, 1, figsize=figsize)
        
        # Plot 1: Estimated best checkpoint vs. samples
        axs[0].plot(
            range(len(self.estimated_best_checkpoints)), 
            self.estimated_best_checkpoints, 
            'o-'
        )
        axs[0].axhline(
            y=self.true_best_checkpoint, 
            color='r', 
            linestyle='--', 
            label=f'True Best ({self.true_best_checkpoint})'
        )
        axs[0].set_xlabel('Number of Samples')
        axs[0].set_ylabel('Estimated Best Checkpoint')
        axs[0].set_title('Convergence to Best Checkpoint')
        axs[0].legend()
        
        # Plot 2: Error vs. sampling fraction
        axs[1].plot(self.sampling_fractions, self.errors, 'o-')
        axs[1].set_xlabel('Sampling Fraction')
        axs[1].set_ylabel('Normalized Error')
        axs[1].set_title('Error vs. Sampling Fraction')
        axs[1].set_yscale('log')
        
        # Plot 3: Sampling distribution
        sns.heatmap(
            self.sampled_mask, 
            cmap='YlGnBu', 
            cbar_kws={'label': 'Sampled'}, 
            ax=axs[2]
        )
        axs[2].set_xlabel('Test Index')
        axs[2].set_ylabel('Checkpoint Index')
        axs[2].set_title('Sampling Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Additional plot: Final predictions
        predictions = self.predict_performance()
        
        plt.figure(figsize=(10, 5))
        
        # Plot predictions with uncertainty
        plt.errorbar(
            predictions['checkpoint_num'],
            predictions['predicted_value'],
            yerr=predictions['uncertainty'],
            fmt='o-',
            alpha=0.7,
            capsize=3
        )
        
        # Highlight the best checkpoint
        best_idx = predictions['predicted_value'].idxmax()
        best_cp = predictions.iloc[best_idx]['checkpoint_num']
        best_score = predictions.iloc[best_idx]['predicted_value']
        
        plt.axvline(x=best_cp, color='r', linestyle='--', 
                    label=f'Estimated Best ({best_cp})')
        plt.axvline(x=self.true_best_checkpoint, color='g', linestyle='--', 
                   label=f'True Best ({self.true_best_checkpoint})')
        
        plt.xlabel('Checkpoint Number')
        plt.ylabel('Predicted Performance')
        plt.title('Final Performance Predictions')
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            base, ext = os.path.splitext(save_path)
            plt.savefig(f"{base}_predictions{ext}", dpi=300, bbox_inches='tight')
        
        plt.show()

    def compare_acquisition_strategies(self, n_samples=100, initial_samples=10, 
                                     strategies=['ucb', 'ei', 'thompson'],
                                     n_trials=3, figsize=(10, 6), save_path=None):
        """
        Compare different acquisition strategies.
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples per trial
        initial_samples : int
            Number of initial samples
        strategies : list
            List of acquisition strategies to compare
        n_trials : int
            Number of trials per strategy
        figsize : tuple
            Figure size
        save_path : str or None
            Path to save the figure, if provided
        """
        results = {strategy: {'errors': [], 'fractions': []} for strategy in strategies}
        
        for strategy in strategies:
            print(f"\nEvaluating strategy: {strategy}")
            
            for trial in range(n_trials):
                print(f"Trial {trial+1}/{n_trials}")
                
                # Reset sampler
                self.__init__(data_file=self.full_data, random_seed=self.random_seed + trial)
                
                # Run sampling
                self.run_sampling(
                    n_samples=n_samples, 
                    acquisition_strategy=strategy,
                    initial_samples=initial_samples,
                    verbose=False
                )
                
                # Store results
                results[strategy]['errors'].append(self.errors)
                results[strategy]['fractions'].append(self.sampling_fractions)
        
        # Plotting
        plt.figure(figsize=figsize)
        
        for strategy in strategies:
            # Average errors across trials
            avg_errors = np.mean([errors for errors in results[strategy]['errors']], axis=0)
            avg_fractions = np.mean([fractions for fractions in results[strategy]['fractions']], axis=0)
            
            plt.plot(avg_fractions, avg_errors, 'o-', label=strategy)
        
        plt.xlabel('Sampling Fraction')
        plt.ylabel('Normalized Error')
        plt.title('Comparison of Acquisition Strategies')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return results

# Example usage
if __name__ == "__main__":
    # Create the adaptive sampler
    sampler = AdaptiveSampler("data/phi4-math-4claude.txt")
    
    # Run sampling with 100 samples
    predictions = sampler.run_sampling(
        n_samples=100,
        acquisition_strategy='ucb',
        kappa=2.0,
        initial_samples=20,  # Increased for better initial model
        init_strategy='grid',
        verbose=True
    )
    
    # Plot results
    sampler.plot_results()