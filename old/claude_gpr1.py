import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import time
from sklearn.gaussian_process.kernels import CompoundKernel, ConstantKernel, RBF, WhiteKernel

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
        
        # Feature scalers
        self.checkpoint_scaler = StandardScaler()
        self.test_idx_scaler = StandardScaler()
        
        # Precompute scaled checkpoint numbers
        checkpoint_nums = self.full_data['checkpoint_num'].values.reshape(-1, 1)
        self.checkpoint_scaler.fit(checkpoint_nums)
        
        # Precompute scaled test indices
        test_indices = np.arange(len(self.test_cols)).reshape(-1, 1)
        self.test_idx_scaler.fit(test_indices)
        
        #---------------------------------------------------------------------------------
        # ORIGINAL CODE
        # # Set up GP model with a suitable kernel
        # # Matern kernel is good for non-smooth functions
        # # self.kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0], nu=2.5) + WhiteKernel(noise_level=0.01)
        
        # # Increase the lower bound for length-scale optimization
        # self.kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0, 5.0], nu=2.5, 
        #                                            length_scale_bounds=[(0.1, 100), (1.0, 100)]) + WhiteKernel(noise_level=0.01)
        
        # self.gp = GaussianProcessRegressor(
        #     kernel=self.kernel,
        #     alpha=1e-10,  # Avoid numerical issues
        #     normalize_y=True,
        #     n_restarts_optimizer=10,
        #     random_state=random_seed
        # )
        
        #---------------------------------------------------------------------------------
        # NEW CODE
        # Create separate kernels for each dimension
        checkpoint_kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        test_kernel = ConstantKernel(1.0) * RBF(length_scale=10.0)  # Larger initial length-scale

        # Combine kernels with dimension-specific operations
        self.kernel = CompoundKernel([
            checkpoint_kernel * RBF(length_scale=[1.0, 1e5]),  # Ignore test dimension
            test_kernel * RBF(length_scale=[1e5, 1.0])         # Ignore checkpoint dimension
        ]) + WhiteKernel(noise_level=0.01)

        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_seed
        )
        
        #---------------------------------------------------------------------------------
        
        # Performance metrics
        self.true_best_checkpoint = None
        self.estimated_best_checkpoints = []
        self.sampling_fractions = []
        self.errors = []
        
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
            checkpoint_indices = [0, n_checkpoints//2, n_checkpoints-1] * (n_initial // 3)
            test_indices = list(range(min(n_tests, n_initial)))
            
            # Ensure we have the right number
            while len(checkpoint_indices) < n_initial:
                checkpoint_indices.append(np.random.randint(0, n_checkpoints))
                test_indices.append(np.random.randint(0, n_tests))
                
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
        
        # Collect samples
        for cp_idx, test_idx in zip(checkpoint_indices, test_indices):
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
        # Convert to numpy arrays
        X = np.array(self.X_sampled)
        y = np.array(self.y_sampled)
        
        # Scale features
        X_scaled = np.column_stack([
            self.checkpoint_scaler.transform(X[:, 0].reshape(-1, 1)).flatten(),
            self.test_idx_scaler.transform(X[:, 1].reshape(-1, 1)).flatten()
        ])
        
        # Fit GP model
        self.gp.fit(X_scaled, y)
        
        return self.gp
    
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
        
        # Create grid of checkpoint_num and test_idx combinations
        checkpoints_grid = []
        checkpoint_ids = []
        
        for cp_num in checkpoint_nums:
            for test_idx in range(len(self.test_cols)):
                checkpoints_grid.append([cp_num, test_idx])
                checkpoint_ids.append(cp_num)
        
        # Scale features
        X_grid = np.array(checkpoints_grid)
        X_grid_scaled = np.column_stack([
            self.checkpoint_scaler.transform(X_grid[:, 0].reshape(-1, 1)).flatten(),
            self.test_idx_scaler.transform(X_grid[:, 1].reshape(-1, 1)).flatten()
        ])
        
        # Predict with GP
        y_pred, y_std = self.gp.predict(X_grid_scaled, return_std=True)
        
        # Organize predictions by checkpoint
        results = pd.DataFrame({
            'checkpoint_num': checkpoint_ids,
            'predicted_value': y_pred,
            'uncertainty': y_std
        })
        
        # Group by checkpoint and calculate mean performance
        checkpoint_predictions = results.groupby('checkpoint_num').agg({
            'predicted_value': 'mean',
            'uncertainty': 'mean'
        }).reset_index()
        
        return checkpoint_predictions
    
    def get_best_checkpoint(self):
        """Get the current estimate of the best checkpoint"""
        predictions = self.predict_performance()
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
        
        X_unsampled = np.array(X_unsampled)
        
        # Scale features
        X_unsampled_scaled = np.column_stack([
            self.checkpoint_scaler.transform(X_unsampled[:, 0].reshape(-1, 1)).flatten(),
            self.test_idx_scaler.transform(X_unsampled[:, 1].reshape(-1, 1)).flatten()
        ])
        
        # Predict with GP
        mu, sigma = self.gp.predict(X_unsampled_scaled, return_std=True)
        
        if strategy == 'ucb':
            # Upper Confidence Bound
            acquisition_values = mu + kappa * sigma
            
        elif strategy == 'ei':
            # Expected Improvement
            current_best = np.max(self.y_sampled)
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
        self.update_model()
        
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
        checkpoint_counts = np.sum(self.sampled_mask, axis=1)
        test_counts = np.sum(self.sampled_mask, axis=0)
        
        # Create heatmap of sampled points
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
        acquisition_strategy='ucb', # 'ucb', 'ei', 'thompson'
        kappa=2.0,
        initial_samples=10,
        init_strategy='grid',
        verbose=True
    )
    
    # Plot results
    sampler.plot_results()
    
    # Compare acquisition strategies
    # sampler.compare_acquisition_strategies(
    #     n_samples=100,
    #     initial_samples=10,
    #     strategies=['ucb', 'ei', 'thompson'],
    #     n_trials=3
    # )