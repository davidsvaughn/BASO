import numpy as np
import numpy_indexed as npi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import norm
import time, sys
import warnings
import torch
warnings.filterwarnings('ignore')  # Suppress convergence warnings

from time_tracker import TimeTracker
tracker = TimeTracker()

class DiverseAdaptiveSampler:
    """
    Adaptive sequential sampling with diversity enforcement to efficiently
    estimate peak checkpoint performance using Gaussian Process Regression.
    """
    
    def __init__(self, data_file, random_seed=42):
        """Initialize the adaptive sampler with diversity constraints."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Load the full dataset (this would be unknown in a real scenario)
        self.full_data = pd.read_csv(data_file, sep='\t')
        
        # remove last row
        # self.full_data = self.full_data[:-1]
        
        # Extract checkpoint numbers
        self.full_data['checkpoint_num'] = self.full_data['CHECKPOINT'].apply(
            lambda x: int(x.split('-')[1])
        )
        
        # Identify test columns (excluding average)
        self.test_cols = [col for col in self.full_data.columns 
                          if col.startswith('TEST_') and col != 'TEST_AVERAGE']
        
        # Create matrix of all test results (checkpoints × tests)
        self.all_results = self.full_data[self.test_cols].values
        self.avg_results = self.all_results.mean(axis=1)
        
        # Initialize mask for sampled entries (0 = unsampled, 1 = sampled)
        self.sampled_mask = np.zeros_like(self.all_results, dtype=bool)
        
        # Initialize sampled data structures
        self.X_sampled = []  # Will hold (checkpoint_num, test_idx) pairs
        self.y_sampled = []  # Will hold corresponding test results
        
        # Track sampling counts for checkpoints and tests
        self.n_checkpoints = len(self.full_data)
        self.n_tests = len(self.test_cols)
        self.checkpoint_counts = np.zeros(self.n_checkpoints)
        self.test_counts = np.zeros(self.n_tests)
        
        # Create feature scalers and encoders
        self.checkpoint_scaler = StandardScaler()
        self.test_encoder = OneHotEncoder(sparse_output=False)
        
        # Precompute scaled checkpoint numbers
        checkpoint_nums = self.full_data['checkpoint_num'].values.reshape(-1, 1)
        self.checkpoint_scaler.fit(checkpoint_nums)
        
        # Precompute encoded test indices
        test_indices = np.arange(len(self.test_cols)).reshape(-1, 1)
        self.test_encoder.fit(test_indices)
        
        # precompute encoded features for all possible combinations
        checkpoint_nums = self.full_data['checkpoint_num'].unique()
        self.all_features = np.array([[cp_num, test_idx] for cp_num in checkpoint_nums for test_idx in range(len(self.test_cols))])
        self.all_X_encoded = self._encode_features(self.all_features)
        
        #-------------------------
        # original...
        # Setup GP model with a suitable kernel
        # self.kernel = (
        #     ConstantKernel(1.0) * 
        #     Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(0.1, 100.0)) + 
        #     WhiteKernel(noise_level=0.01)
        # )
        
        #-------------------------
        # from chatGPT...
        # kernel = C(1.0, (1e-3, 1e3)) \
        # * RBF(length_scale=[10.0, 5.0], length_scale_bounds=(1e-2, 1e2)) \
        # + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
        #-------------------------
        
        self.kernel = (
            ConstantKernel(1.0) * 
            RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + 
            WhiteKernel(noise_level=0.01)
            # WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))
        )
        
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_seed
        )
        
        # Performance metrics
        self.true_best_checkpoint = None
        self.estimated_best_checkpoints = []
        self.sampling_fractions = []
        self.errors = []
        
        # Diversity settings
        self.diversity_weight = 0.5  # Balance between utility and diversity
        self.force_exploration_every = 5  # Force exploration every N samples
        self.current_sample_count = 0
    
    def plot_results(self, figsize=(14, 12), save_path=None):
        """Plot results of the adaptive sampling procedure."""
        if not self.estimated_best_checkpoints:
            print("No sampling results to plot. Run sampling first.")
            return
            
        fig, axs = plt.subplots(4, 1, figsize=figsize)
        
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
        
        # Plot 4: Checkpoint and test sampling histograms
        cp_hist = np.sum(self.sampled_mask, axis=1)
        test_hist = np.sum(self.sampled_mask, axis=0)
        
        # Plot as two subplots side by side
        ax4_left = axs[3].inset_axes([0.05, 0.1, 0.4, 0.8])
        ax4_right = axs[3].inset_axes([0.55, 0.1, 0.4, 0.8])
        
        ax4_left.bar(range(len(cp_hist)), cp_hist)
        ax4_left.set_xlabel('Checkpoint Index')
        ax4_left.set_ylabel('Samples')
        ax4_left.set_title('Checkpoint Sampling Distribution')
        
        ax4_right.bar(range(len(test_hist)), test_hist)
        ax4_right.set_xlabel('Test Index')
        ax4_right.set_ylabel('Samples')
        ax4_right.set_title('Test Sampling Distribution')
        
        axs[3].set_title('Sampling Histograms')
        axs[3].axis('off')
        
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
            import os
            base, ext = os.path.splitext(save_path)
            plt.savefig(f"{base}_predictions{ext}", dpi=300, bbox_inches='tight')
        
        plt.show()
        
    @tracker
    def _encode_features(self, X):
        """Encode features appropriately for GP model."""
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
        # avg_performance = self.full_data['TEST_AVERAGE'].values
        best_idx = np.argmax(self.avg_results)
        self.true_best_checkpoint = self.full_data.iloc[best_idx]['checkpoint_num']
        self.true_best_performance = self.avg_results[best_idx]
        return self.true_best_checkpoint, self.true_best_performance
    
    def initialize_sampling(self, n_initial=10, strategy='grid'):
        """Initialize sampling with a few samples to start the process."""
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
            checkpoint_indices = [0, n_checkpoints//4, n_checkpoints//2, 3*n_checkpoints//4, n_checkpoints-1]
            checkpoint_indices = checkpoint_indices * (n_initial // 5 + 1)
            
            test_indices = [0, n_tests//4, n_tests//2, 3*n_tests//4, n_tests-1]
            test_indices = test_indices * (n_initial // 5 + 1)
            
            # Ensure we have the right number
            while len(checkpoint_indices) < n_initial:
                checkpoint_indices.append(np.random.randint(0, n_checkpoints))
                test_indices.append(np.random.randint(0, n_tests))
                
        elif strategy == 'latin_hypercube':
            # Latin hypercube sampling for better space coverage
            from scipy.stats.qmc import LatinHypercube
            
            sampler = LatinHypercube(d=2, seed=self.random_seed)
            samples = sampler.random(n=n_initial)
            
            checkpoint_indices = (samples[:, 0] * (n_checkpoints - 1)).astype(int)
            test_indices = (samples[:, 1] * (n_tests - 1)).astype(int)
        
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
        
        # Collect samples
        for cp_idx, test_idx in zip(checkpoint_indices[:n_initial], test_indices[:n_initial]):
            self._add_sample(cp_idx, test_idx)
    
    @tracker
    def _add_sample(self, checkpoint_idx, test_idx):
        """Add a single sample to our collection."""
        # Mark as sampled in the mask
        self.sampled_mask[checkpoint_idx, test_idx] = True
        
        # Get the result value
        result = self.all_results[checkpoint_idx, test_idx]
        
        # Get the checkpoint number
        checkpoint_num = self.full_data.iloc[checkpoint_idx]['checkpoint_num']
        
        # Store the sample
        self.X_sampled.append([checkpoint_num, test_idx])
        self.y_sampled.append(result)
        
        # Update counts for diversity tracking
        self.checkpoint_counts[checkpoint_idx] += 1
        self.test_counts[test_idx] += 1
        
        self.current_sample_count += 1
    
    @tracker
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
    
    
    @tracker
    def predict_performance(self, checkpoint_nums=None):
        """Predict mean performance for each checkpoint across all tests."""
        if checkpoint_nums is None:
            checkpoint_nums = self.full_data['checkpoint_num'].unique()
        else:
            print('Huh?')
            sys.exit()
        # features = np.array([[cp_num, test_idx] for cp_num in checkpoint_nums for test_idx in range(len(self.test_cols))])
        # X_encoded = self._encode_features(features)
        
        # precomputed...
        features = self.all_features
        X_encoded = self.all_X_encoded
        
        # Predict with GP
        y_pred, y_std = self.gp.predict(X_encoded, return_std=True)
        cp_nums, avg_preds = npi.group_by(features[:, 0]).mean(y_pred)
        _, avg_stds = npi.group_by(features[:, 0]).mean(y_std)
        
        results_list = []
        for cp_num, avg_pred, avg_std in zip(cp_nums, avg_preds, avg_stds):
            results_list.append({
                'checkpoint_num': cp_num,
                'predicted_value': avg_pred,
                'uncertainty': avg_std
            })
            
        # Convert to DataFrame
        return pd.DataFrame(results_list)
        
    
    @tracker
    def predict_performance_OLD(self, checkpoint_nums=None):
        """Predict mean performance for each checkpoint across all tests."""
        if checkpoint_nums is None:
            checkpoint_nums = self.full_data['checkpoint_num'].unique()
        
        results_list = []
        
        features = [[cp_num, test_idx] for cp_num in checkpoint_nums for test_idx in range(len(self.test_cols))]
        
        # For each checkpoint, predict for all tests
        for cp_num in checkpoint_nums:
            
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
    
    @tracker
    def get_best_checkpoint(self):
        """Get the current estimate of the best checkpoint"""
        predictions = self.predict_performance()
        if predictions.empty:
            # If predictions failed, return most sampled checkpoint
            checkpoint_counts = np.sum(self.sampled_mask, axis=1)
            best_idx = np.argmax(checkpoint_counts)
            return self.full_data.iloc[best_idx]['checkpoint_num']
            
        best_idx = predictions['predicted_value'].idxmax()
        best_perf = self.avg_results[best_idx] # TRUE avg perf of the checkpoint
        # best_perf = predictions.iloc[best_idx]['predicted_value'] # PREDICTED avg perf of the checkpoint
        return int(predictions.iloc[best_idx]['checkpoint_num']), best_perf
    
    @tracker
    def compute_diversity_score(self, checkpoint_idx, test_idx):
        """
        Compute a diversity score for a candidate point.
        Higher is better (more diverse).
        """
        # Get current max counts
        max_checkpoint_count = max(1, np.max(self.checkpoint_counts))
        max_test_count = max(1, np.max(self.test_counts))
        
        # Normalize counts
        norm_checkpoint_count = self.checkpoint_counts[checkpoint_idx] / max_checkpoint_count
        norm_test_count = self.test_counts[test_idx] / max_test_count
        
        # Compute inverse diversity score (lower is more diverse)
        inverse_diversity = norm_checkpoint_count + norm_test_count
        
        # Return diversity score (higher is more diverse)
        return 1.0 - (inverse_diversity / 2.0)
    
    @tracker
    def compute_diversity_scores(self, checkpoint_indices, test_indices):
        """
        Compute a diversity score for a candidate point.
        Higher is better (more diverse).
        """
        # Get current max counts
        max_checkpoint_count = max(1, np.max(self.checkpoint_counts))
        max_test_count = max(1, np.max(self.test_counts))
        
        # Normalize counts
        norm_checkpoint_counts = self.checkpoint_counts[checkpoint_indices] / max_checkpoint_count
        norm_test_counts = self.test_counts[test_indices] / max_test_count
        
        # Compute inverse diversity score (lower is more diverse)
        inverse_diversity = norm_checkpoint_counts + norm_test_counts
        
        # Return diversity score (higher is more diverse)
        return 1.0 - (inverse_diversity / 2.0)
    
    @tracker
    def force_exploration(self):
        """
        Force exploration by sampling from underrepresented 
        checkpoints and tests.
        """
        # Find least sampled checkpoint and test
        least_sampled_checkpoint = np.argmin(self.checkpoint_counts)
        least_sampled_test = np.argmin(self.test_counts)
        
        # If there are unsampled combinations, prioritize those
        potential_points = []
        
        # Add combinations with the least sampled checkpoint
        for test_idx in range(self.n_tests):
            if not self.sampled_mask[least_sampled_checkpoint, test_idx]:
                potential_points.append((least_sampled_checkpoint, test_idx))
        
        # Add combinations with the least sampled test
        for cp_idx in range(self.n_checkpoints):
            if not self.sampled_mask[cp_idx, least_sampled_test]:
                potential_points.append((cp_idx, least_sampled_test))
        
        # If there are potential points, choose one randomly
        if potential_points:
            return potential_points[np.random.randint(len(potential_points))]
        
        # If all combinations are sampled, use standard acquisition function
        return self.acquisition_function()
    
    @tracker
    def acquisition_function(self, strategy='ucb', kappa=2.0):
        """
        Calculate acquisition values for all unsampled points with diversity bonus.
        
        Returns:
        --------
        Tuple of (best_point, all_candidate_points)
        where best_point is (checkpoint_idx, test_idx)
        and all_candidate_points is a list of (checkpoint_idx, test_idx, acq_value)
        """
        # Get indices of unsampled points
        unsampled_indices = np.where(~self.sampled_mask)
        checkpoint_indices = unsampled_indices[0]
        test_indices = unsampled_indices[1]
        
        # If no unsampled points left, return random point
        if len(checkpoint_indices) == 0:
            random_point = (np.random.randint(0, self.n_checkpoints),
                           np.random.randint(0, self.n_tests))
            return random_point, [(random_point[0], random_point[1], 0.0)]
        
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
            random_point = (checkpoint_indices[rand_idx], test_indices[rand_idx])
            return random_point, [(random_point[0], random_point[1], 0.0)]
        
        # Calculate base acquisition values
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
        
        #-------------------------------------------------------------------
        diversity_scores = self.compute_diversity_scores(checkpoint_indices, test_indices)
        combined_scores = (1 - self.diversity_weight) * acquisition_values + self.diversity_weight * diversity_scores
        best_idx = np.argmax(combined_scores)
        return (checkpoint_indices[best_idx], test_indices[best_idx])
        
        #-------------------------------------------------------------------
        # Calculate diversity scores and combined acquisition values
        combined_values = []
        for i, (cp_idx, test_idx) in enumerate(zip(checkpoint_indices, test_indices)):
            diversity_score = self.compute_diversity_score(cp_idx, test_idx) if self.diversity_weight > 0 else 0
            
            combined_value = (1 - self.diversity_weight) * acquisition_values[i] + \
                            self.diversity_weight * diversity_score
            
            combined_values.append((cp_idx, test_idx, combined_value))

        # Sort by combined value (descending)
        combined_values.sort(key=lambda x: x[2], reverse=True)
        
        # Return best point and all candidates (for analysis)
        best_point = (combined_values[0][0], combined_values[0][1])
        return best_point#, combined_values
    
    @tracker 
    def sample_next_point(self, strategy='ucb', kappa=2.0):
        """Sample the next point based on the acquisition function with diversity."""
        
        # Update the model first
        if self.update_model() is None:
            # If model update failed, sample randomly
            next_cp_idx = np.random.randint(0, self.n_checkpoints)
            next_test_idx = np.random.randint(0, self.n_tests)
        else:
            # Check if we should force exploration
            if self.current_sample_count % self.force_exploration_every == 0:
                next_cp_idx, next_test_idx = self.force_exploration()
            else:
                k = kappa
                # sample kappa from normal distribution
                k = np.random.normal(k, 1.0, 1).clip(0.5, 3.5)
                # Get next point from acquisition function
                (next_cp_idx, next_test_idx)  = self.acquisition_function(strategy, kappa=k) # kappa=2.0
        
        # Add the sample
        self._add_sample(next_cp_idx, next_test_idx)
        
        return next_cp_idx, next_test_idx
    
    def estimate_optimality_confidence(self, epsilon=0.01):
        """
        Estimate the probability that the current best checkpoint's performance
        is within epsilon of the true best checkpoint's performance.
        """
        # Get current predictions for all checkpoints
        predictions = self.predict_performance()
        
        # Current best checkpoint and its predicted performance
        current_best_idx = predictions['predicted_value'].idxmax()
        current_best_cp = predictions.iloc[current_best_idx]['checkpoint_num']
        current_best_perf = predictions.iloc[current_best_idx]['predicted_value']
        
        # Calculate probability that NO other checkpoint outperforms current best by more than epsilon
        probability_within_epsilon = 1.0
        
        for idx, row in predictions.iterrows():
            if row['checkpoint_num'] == current_best_cp:
                continue
                
            # Performance difference distribution is normally distributed
            # with μ_diff = μ_current_best - μ_other
            # and σ²_diff = σ²_current_best + σ²_other
            
            mean_diff = current_best_perf - row['predicted_value']
            std_diff = np.sqrt(predictions.iloc[current_best_idx]['uncertainty']**2 + 
                            row['uncertainty']**2)
            
            # Probability that another checkpoint is better by more than epsilon
            # P(other_perf > current_best_perf + epsilon)
            prob_outperform = 1 - norm.cdf(epsilon / std_diff)
            
            # Update overall probability (assuming independence)
            probability_within_epsilon *= (1 - prob_outperform)
        
        return probability_within_epsilon

    def estimate_optimality_confidence_mc_OLD(self, epsilon=0.01, n_samples=1000):
        """
        Use Monte Carlo sampling to estimate the probability that the current best
        checkpoint is within epsilon of the true best.
        """
        # Get all unique checkpoint numbers
        checkpoint_nums = self.full_data['checkpoint_num'].unique()
        
        # Create features for all checkpoints (averaging across tests)
        features = []
        for cp_num in checkpoint_nums:
            features.extend([[cp_num, test_idx] for test_idx in range(self.n_tests)])
        
        # Encode features
        X_encoded = self._encode_features(features)
        
        # Get mean and covariance matrix from GP model
        mu, cov = self.gp.predict(X_encoded, return_cov=True)
        
        # Reshape to organize by checkpoint
        mu = mu.reshape(len(checkpoint_nums), self.n_tests)
        
        # Draw samples from multivariate normal distribution
        samples = np.random.multivariate_normal(mu.flatten(), cov, size=n_samples)
        samples = samples.reshape(n_samples, len(checkpoint_nums), self.n_tests)
        
        # Calculate average performance per checkpoint for each sample
        avg_performances = np.mean(samples, axis=2)
        
        # Current best estimate
        current_best_idx,_ = self.get_best_checkpoint()
        current_best_idx = np.where(checkpoint_nums == current_best_idx)[0][0]
        
        # Count samples where current best is within epsilon of true best
        within_epsilon_count = 0
        for i in range(n_samples):
            sample_best_idx = np.argmax(avg_performances[i])
            if (avg_performances[i, sample_best_idx] - 
                avg_performances[i, current_best_idx]) <= epsilon:
                within_epsilon_count += 1
        
        return within_epsilon_count / n_samples

    def estimate_optimality_confidence_mc(self, epsilon=[0.05, 0.01], n_samples=1000):
        """
        GPU-accelerated Monte Carlo sampling to estimate the probability that 
        the current best checkpoint is within epsilon of the true best.
        """
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Get all unique checkpoint numbers
        checkpoint_nums = self.full_data['checkpoint_num'].unique()
        
        # Create features for all checkpoints (averaging across tests)
        features = []
        for cp_num in checkpoint_nums:
            features.extend([[cp_num, test_idx] for test_idx in range(self.n_tests)])
        
        # Encode features
        X_encoded = self._encode_features(features)
        
        # Get mean and covariance matrix from GP model
        mu, cov = self.gp.predict(X_encoded, return_cov=True)
        
        # Move to GPU
        mu_tensor = torch.tensor(mu.flatten(), dtype=torch.float32).to(device)
        cov_tensor = torch.tensor(cov, dtype=torch.float32).to(device)
        
        # Compute Cholesky decomposition (L is lower triangular)
        # catch positive semidef errors
        try:
            L = torch.linalg.cholesky(cov_tensor)
        except Exception as e:
            print(f"Cholesky decomposition error: {e}")
            return None
        
        # Generate standard normal samples
        z = torch.randn(n_samples, mu_tensor.shape[0], device=device)
        
        # Transform to desired distribution: samples = mean + L·z
        samples_tensor = mu_tensor + torch.matmul(z, L.T)
        
        # Move back to CPU and convert to numpy
        samples = samples_tensor.cpu().numpy()
        
        # Reshape to organize by checkpoint and test
        samples = samples.reshape(n_samples, len(checkpoint_nums), self.n_tests)
        
        # Continue with the same analysis as before
        avg_performances = np.mean(samples, axis=2)
        
        # Current best estimate
        # current_best_idx = np.where(checkpoint_nums == self.get_best_checkpoint())[0][0]
        current_best_idx,_ = self.get_best_checkpoint()
        current_best_idx = np.where(checkpoint_nums == current_best_idx)[0][0]
        
        # Count samples where current best is within epsilon of true best
        sample_best_indices = np.argmax(avg_performances, axis=1)
        diffs = avg_performances[np.arange(n_samples), sample_best_indices] - avg_performances[:, current_best_idx]
        diffs /= avg_performances[:, current_best_idx]
        
        # epsilon is a list of values
        within_epsilon_count = np.sum(diffs[:, None] <= np.array(epsilon), axis=0)
        return within_epsilon_count / n_samples
    
        # within_epsilon_count = np.mean(np.abs(diffs) <= epsilon, axis=1)
        # return within_epsilon_count
    
        within_epsilon_count = 0
        diffs = []
        for i in range(n_samples):
            sample_best_idx = np.argmax(avg_performances[i])
            diffs.append(avg_performances[i, sample_best_idx] - avg_performances[i, current_best_idx])
            if (avg_performances[i, sample_best_idx] - 
                avg_performances[i, current_best_idx]) <= epsilon:
                within_epsilon_count += 1
        return within_epsilon_count / n_samples
    
    def run_sampling(self, n_samples, acquisition_strategy='ucb', kappa=2.0,
                   initial_samples=10, init_strategy='grid', diversity_weight=0.5,
                   force_exploration_every=5, verbose=True):
        """
        Run the complete adaptive sampling procedure with diversity constraints.
        
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
        diversity_weight : float
            Weight for diversity (0.0 to 1.0)
        force_exploration_every : int
            Force exploration every N samples
        verbose : bool
            Whether to print progress
        """
        # Set diversity parameters
        self.diversity_weight = diversity_weight
        self.force_exploration_every = force_exploration_every
        
        # Calculate true best checkpoint (for comparison)
        true_best_checkpt, true_best_perf = self.get_true_best_checkpoint()
        
        if verbose:
            print(f"True best checkpoint: {true_best_checkpt}")
            print(f"Initializing with {initial_samples} samples using {init_strategy} strategy...")
            
        # Initialize
        self.initialize_sampling(n_initial=initial_samples, strategy=init_strategy)
        
        # Track performance
        total_samples = self.all_results.size
        checkpoint_range = self.full_data['checkpoint_num'].max() - self.full_data['checkpoint_num'].min()
        
        # Get initial estimate
        self.update_model()
        current_best_checkpt, current_best_perf = self.get_best_checkpoint()
        # error = abs(current_best_checkpt - true_best_checkpt) / checkpoint_range
        error = abs(current_best_perf - true_best_perf) / true_best_perf
        
        self.estimated_best_checkpoints.append(current_best_checkpt)
        self.sampling_fractions.append(len(self.y_sampled) / total_samples)
        self.errors.append(error)
        
        if verbose:
            print(f"Initial estimate: Checkpoint {current_best_checkpt}, Error: {error:.4f}")
        
        # Collect remaining samples adaptively
        remaining_samples = n_samples - initial_samples
        
        try:
            for i in range(remaining_samples):
                start_time = time.time()
                
                # Sample next point
                next_cp_idx, next_test_idx = self.sample_next_point(
                    strategy=acquisition_strategy, kappa=kappa
                )
                
                # Update performance metrics
                current_best_checkpt, current_best_perf = self.get_best_checkpoint()
                # error = abs(current_best_checkpt - true_best_checkpt) / checkpoint_range
                error = abs(current_best_perf - true_best_perf) / true_best_perf
                
                self.estimated_best_checkpoints.append(current_best_checkpt)
                self.sampling_fractions.append(len(self.y_sampled) / total_samples)
                self.errors.append(error)
                
                elapsed = time.time() - start_time
                
                if verbose and (i+1) % 10 == 0:
                    print()
                    # print('\n' + '-'*100)
                    print(f"Sample {i+1+initial_samples}/{n_samples}: ")
                    #   f"Added ({self.full_data.iloc[next_cp_idx]['CHECKPOINT']}, "
                    #   f"{self.test_cols[next_test_idx]})")
                    print(f"Curr Best:\tCheckpoint {current_best_checkpt}, Error: {error:.4f}")
                    print(f"Sparsity:\t{100*self.sampling_fractions[-1]:.2f}%, "
                    f"Time: {elapsed:.2f}s")
                    
                    # Print diversity metrics
                    cp_hist = np.bincount(np.where(self.sampled_mask)[0])
                    test_hist = np.bincount(np.where(self.sampled_mask)[1])
                    print(f"Diversity:\t{len(cp_hist)}/{self.n_checkpoints} checkpoints, " 
                    f"{len(test_hist)}/{self.n_tests} tests sampled")
                    print(f"Max samples:\tper-checkpoint: {np.max(cp_hist)}, "
                    f"per-test: {np.max(test_hist)}")
                    print()
                    
                    #-------------------------------------------------------------------
                    # Print optimality confidence
                    # eps = [0.005, 0.001, 0.0005, 0.0001]
                    # # conf1 = self.estimate_optimality_confidence(epsilon=eps)
                    # # print(f"Confidence:\t{100*conf1:.4f}%")
                    # conf2 = self.estimate_optimality_confidence_mc(epsilon=eps, n_samples=100000)
                    # if conf2 is not None:
                    #     for e, c in zip(eps, conf2):
                    #         print(f"Confidence:\t{100*c:.2f}% within {e}")
                    #-------------------------------------------------------------------
                            
                else:
                    print(f"Curr Best:\tCheckpoint {current_best_checkpt}, Error: {error:.4f}")
                    
        except KeyboardInterrupt:
            print("\nSampling interrupted by user. Using current best estimate.")
            
        if verbose:
            print(f"\nFinal estimate after {n_samples} samples: Checkpoint {current_best_checkpt}")
            print(f"Final error: {error:.4f}")
            print(f"Total sampling fraction: {len(self.y_sampled) / total_samples:.4f}")
        
        return self.predict_performance()

# Example usage
if __name__ == "__main__":
    
    rand_seed = np.random.randint(1000)
    rand_seed = 939 # 418  622  669??
    print(f"Random seed: {rand_seed}")
    
    # Create the adaptive sampler with diversity
    sampler = DiverseAdaptiveSampler("data/phi4-math-4claude.txt",
                                     random_seed=rand_seed,
                                     )
    
    # Run sampling with 100 samples and strong diversity enforcement
    predictions = sampler.run_sampling(
        n_samples=1000,
        acquisition_strategy='ucb', # ucb, ei, thompson
        kappa=2.0,
        initial_samples=50,
        init_strategy='latin_hypercube', # random, grid, extremes, latin_hypercube
        diversity_weight=0.5,       # preference for diversity
        force_exploration_every=10,  # force exploration frequently
        verbose=True
    )
    
    # Print timing information
    print("\n=== Performance Profiling ===")
    tracker.print_stats(sort_by="fraction")  # Sort by percentage of runtime
    
    # Plot results
    sampler.plot_results()
    
    # BEST: acquisition_strategy=thompson, diversity_weight=0.5, force_exploration_every=10