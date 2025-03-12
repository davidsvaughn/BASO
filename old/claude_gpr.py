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

def check_break(reset=False):
    if reset:
        try:
            with open('break.txt', 'w') as f:
                f.write('')
            return True
        except:
            pass
        return False
    signals = ['yes', 'true', 'break']
    try:
        with open('break.txt', 'r') as f:
            s = f.read().lower().strip()
        for signal in signals:
            if signal in s:
                return True
    except:
        pass
    return False
            
class DiverseAdaptiveSampler:
    """
    Adaptive sequential sampling with diversity enforcement to efficiently
    estimate peak checkpoint performance using Gaussian Process Regression.
    """
    
    def __init__(self, data_file, test_subset=None, random_seed=42):
        """Initialize the adaptive sampler with diversity constraints."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Load the full dataset (this would be unknown in a real scenario)
        self.full_data = pd.read_csv(data_file, sep='\t')
        
        # remove last rows
        self.full_data = self.full_data[:-20]
        
        # Extract checkpoint numbers
        self.full_data['checkpoint_num'] = self.full_data['CHECKPOINT'].apply(
            lambda x: int(x.split('-')[1])
        )
        
        # Identify test columns (excluding average)
        self.test_cols = [col for col in self.full_data.columns 
                          if col.startswith('TEST_') and col != 'TEST_AVERAGE']
        
        # remove random fraction of test columns
        # test_subset = 0.75
        if test_subset is not None and test_subset > 0:
            if test_subset < 1:
                test_subset = int(len(self.test_cols)*test_subset)
            self.test_cols = np.random.choice(self.test_cols, test_subset, replace=False)
        
        # Create matrix of all test results (checkpoints × tests)
        self.all_results = self.full_data[self.test_cols].values
        
        #------------------------------------------------------------------
        # best_idx = self.all_results.mean(axis=1).argmax()
        # print(f"Best checkpoint BEFORE: {self.full_data.iloc[best_idx]['checkpoint_num']}")
        # #------------------------------------------------------------------
        # # divide each column by its max value
        # self.all_results = self.all_results / np.max(self.all_results, axis=0)
        # #------------------------------------------------------------------
        # best_idx = self.all_results.mean(axis=1).argmax()
        # print(f"Best checkpoint AFTER: {self.full_data.iloc[best_idx]['checkpoint_num']}")
        #------------------------------------------------------------------
        
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
        self.checkpoint_nums = self.full_data['checkpoint_num'].unique()
        self.all_features = np.array([[cp_num, test_idx] for cp_num in self.checkpoint_nums for test_idx in range(len(self.test_cols))])
        self.all_X_encoded = self._encode_features(self.all_features)
        
        # record acquisition function values
        self.acq_values = [(0, 0, 0, 0)]
        self.eigs = np.array([0, 0, 0])
        
        self.log = {}
        
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
        
    def print_log(self, min_count=3):
        for k, v in self.log.items():
            print('-'*25  + f"\n{k}:")
            v = np.array(v)[:,:2].astype(int)   
            
            # test counts
            unique, counts = np.unique(v[:, 1], return_counts=True)
            print(f"\ntest counts:")
            for u, c in zip(unique, counts):
                if c >= min_count:
                    print(f"\t{u}\t{c}")
                
            # checkpoint counts
            unique, counts = np.unique(v[:, 0], return_counts=True)
            print(f"\ncheckpoint counts:")
            for u, c in zip(unique, counts):
                if c >= min_count:
                    print(f"\t{u}\t{c}")

            # # print all values
            # v = v[np.lexsort((v[:, 0], v[:, 1]))]
            # print(f"{k}:\n{v}\n")
        print('-'*25)
        print()
    
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
        
    # @tracker
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
            
            # select random subset of tests size of n_initial/5
            test_indices = np.random.choice(n_tests, n_initial // 5 + 1, replace=False)
            # repeat the test indices to match the number of checkpoints and sort
            test_indices = sorted(np.tile(test_indices, 5))
            
            # test_indices = [0, n_tests//4, n_tests//2, 3*n_tests//4, n_tests-1]
            # test_indices = test_indices * (n_initial // 5 + 1)
            # test_indices.sort()
            
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
    
    # @tracker
    def _add_sample(self, checkpoint_idx, test_idx):
        """Add a single sample to our collection."""
        
        # print(f"Adding sample: Checkpoint {checkpoint_idx}, Test {test_idx}")
        if self.sampled_mask[checkpoint_idx, test_idx]:
            print(f"Adding sample: Checkpoint {checkpoint_idx}, Test {test_idx}")
            print("Warning: Sample already exists. Should not be trying this sample!")
            print()
        
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
    
    # @tracker
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
    
    
    # @tracker
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
    
    # @tracker
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
        
        #-------------------------------------------------------------------
        best_score = predictions.iloc[best_idx]['predicted_value'] # PREDICTED avg perf of the checkpoint
        best_uncertainty = predictions.iloc[best_idx]['uncertainty']
        # Maximum possible score from any checkpoint (95% confidence)
        max_possible = np.max(predictions['predicted_value'] + 1.96 * predictions['uncertainty'])
        self.expected_regret = (max_possible - best_score)/max_possible
        self.avg_uncertainty = np.mean(predictions['uncertainty'])
        
        #-------------------------------------------------------------------
        return int(predictions.iloc[best_idx]['checkpoint_num']), best_perf
    
    # @tracker
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
    
    # @tracker
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
    
    # function to compute single diversity score quantifying global diversity of sampled points so far
    def compute_global_diversity_score(self):
        # Get current max counts
        max_checkpoint_count = max(1, np.max(self.checkpoint_counts))
        max_test_count = max(1, np.max(self.test_counts))
        
        # get all sampled indices 
        checkpoint_indices, test_indices = np.where(self.sampled_mask)
        
        # Normalize counts
        norm_checkpoint_counts = self.checkpoint_counts[checkpoint_indices] / max_checkpoint_count
        norm_test_counts = self.test_counts[test_indices] / max_test_count
        
        # Compute inverse diversity score (lower is more diverse)
        test_diversity = np.mean(norm_test_counts)
        checkpoint_diversity = np.mean(norm_checkpoint_counts)
        
        diversity = min(test_diversity, checkpoint_diversity)
        return diversity
        
        # # Return diversity score (higher is more diverse)
        # return 1.0 - inverse_diversity / 2.0
    
    # @tracker
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
    
    def compute_ucb(self, mu, sigma, kappa=2.0):
        # kappa = np.random.normal(kappa, 1.0, 1).clip(0.5, 3.5)
        return mu + kappa * sigma
    
    def compute_thompson(self, mu, sigma, kappa=1.0):
        return np.random.normal(mu, kappa*sigma)
    
    def compute_ei(self, mu, sigma):
        current_best = np.max(self.y_sampled) if self.y_sampled else 0
        imp = mu - current_best
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma > 0)
        acquisition_values = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return acquisition_values
    
    def compute_pi(self, mu, sigma):
        current_best = np.max(self.y_sampled) if self.y_sampled else 0
        imp = mu - current_best
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma > 0)
        acquisition_values = norm.cdf(Z)
        return acquisition_values
    
    def batch_expected_information_gain(self, checkpoint_indices, test_indices, X_candidates_encoded):
        """
        Compute the expected information gain for all unsampled points in batch.
        
        Parameters:
        -----------
        max_points : int
            Maximum number of points to consider (for computational efficiency)
        
        Returns:
        --------
        checkpoint_indices, test_indices, info_gains : arrays
            The indices and corresponding information gain values
        """
        # # Get all unsampled points
        # unsampled_indices = np.where(~self.sampled_mask)
        # checkpoint_indices = unsampled_indices[0]
        # test_indices = unsampled_indices[1]
        
        # # # If too many points, randomly subsample for efficiency
        # # if len(checkpoint_indices) > max_points:
        # #     idx = np.random.choice(len(checkpoint_indices), max_points, replace=False)
        # #     checkpoint_indices = checkpoint_indices[idx]
        # #     test_indices = test_indices[idx]
        
        # # Create feature vectors for all candidates
        # X_candidates = np.array([
        #     [self.full_data.iloc[cp_idx]['checkpoint_num'], test_idx]
        #     for cp_idx, test_idx in zip(checkpoint_indices, test_indices)
        # ])
        # X_candidates_encoded = self._encode_features(X_candidates)
        
        # Get current predictions to identify competing best checkpoints
        predictions = self.predict_performance()
        
        # Find checkpoints whose upper confidence bound overlaps with current best
        best_idx = predictions['predicted_value'].idxmax()
        best_lower = predictions.iloc[best_idx]['predicted_value'] - 2 * predictions.iloc[best_idx]['uncertainty']
        
        potential_best = predictions[
            predictions['predicted_value'] + 2 * predictions['uncertainty'] > best_lower
        ]
        
        # If no uncertainty about which is best, return zeros
        if len(potential_best) <= 1:
            return checkpoint_indices, test_indices, np.zeros(len(checkpoint_indices))
        
        # Calculate probabilities of each checkpoint being the best
        predicted_values = potential_best['predicted_value'].values
        rel_values = predicted_values - np.max(predicted_values)
        probs = np.exp(np.clip(rel_values, -700, 0))  # Prevent overflow
        probs = probs / np.sum(probs)
        
        # Process by test index for efficiency (we can reuse calculations)
        unique_test_indices = np.unique(test_indices)
        info_gains = np.zeros(len(checkpoint_indices))
        
        for t_idx in unique_test_indices:
            # Get candidates with this test index
            candidate_mask = test_indices == t_idx
            candidates_test_encoded = X_candidates_encoded[candidate_mask]
            
            # Create features for potential best checkpoints
            X_potential_encoded = self._encode_features(np.array([
                [cp_num, t_idx] for cp_num in potential_best['checkpoint_num'].values
            ]))
            
            # Calculate covariance matrices efficiently
            K_s = self.gp.kernel_(X_potential_encoded, candidates_test_encoded)
            K_ss_diag = np.diag(self.gp.kernel_(candidates_test_encoded, candidates_test_encoded)) + self.gp.alpha
            
            if torch.cuda.is_available():
                K_s_torch = torch.tensor(K_s, device='cuda')
                K_ss_diag_torch = torch.tensor(K_ss_diag, device='cuda')
                probs_torch = torch.tensor(probs, device='cuda').reshape(-1, 1)
                
                # Calculate on GPU
                var_reduction = (K_s_torch**2) / K_ss_diag_torch
                test_info_gains = torch.sum(var_reduction * probs_torch, dim=0).cpu().numpy()
            else:
                # Calculate variance reduction (using broadcasting)
                var_reduction = (K_s**2) / K_ss_diag
                # Calculate expected information gain as weighted sum
                test_info_gains = np.sum(var_reduction * probs.reshape(-1, 1), axis=0)
            
            # Store in main array
            info_gains[candidate_mask] = test_info_gains
        
        return info_gains #checkpoint_indices, test_indices, info_gains
    
    # @tracker
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
            acquisition_values = self.compute_ucb(mu, sigma, kappa)
            # acquisition_values = mu + kappa * sigma
            
        elif strategy == 'ei':
            # Expected Improvement
            acquisition_values = self.compute_ei(mu, sigma)
            # current_best = np.max(self.y_sampled) if self.y_sampled else 0
            # imp = mu - current_best
            # Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma > 0)
            # acquisition_values = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            
        elif strategy == 'pi':
            # Probability of Improvement
            acquisition_values = self.compute_pi(mu, sigma)
            # current_best = np.max(self.y_sampled) if self.y_sampled else 0
            # imp = mu - current_best
            # Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma > 0)
            # acquisition_values = norm.cdf(Z)
            
        elif strategy == 'thompson':
            # Thompson Sampling
            acquisition_values = self.compute_thompson(mu, sigma, kappa)
            # acquisition_values = np.random.normal(mu, sigma)
            
        elif strategy == 'eig':
            # Expected Information Gain
            acquisition_values = self.batch_expected_information_gain(checkpoint_indices, test_indices, X_unsampled_encoded)
            
        else:
            raise ValueError(f"Unknown acquisition strategy: {strategy}")
        
        #-------------------------------------------------------------------
        # compute all acquisition optimum values
        # ucb = acquisition_values.max() if strategy == 'ucb' else self.compute_ucb(mu, sigma, kappa).max()
        # ei = acquisition_values.max() if strategy == 'ei' else self.compute_ei(mu, sigma).max()
        # pi = acquisition_values.max() if strategy == 'pi' else self.compute_pi(mu, sigma).max()
        # th = acquisition_values.max() if strategy == 'thompson' else self.compute_thompson(mu, sigma, kappa).max()
        # eig = acquisition_values.max() if strategy == 'eig' else self.batch_expected_information_gain(checkpoint_indices, test_indices, X_unsampled_encoded).max()
        # self.acq_values.append((ucb, ei, pi, th, eig))
        
        # self.eigs = eigs = self.batch_expected_information_gain(checkpoint_indices, test_indices, X_unsampled_encoded)
        # ucbs = acquisition_values if strategy == 'ucb' else self.compute_ucb(mu, sigma, kappa)
        # eis = acquisition_values if strategy == 'ei' else self.compute_ei(mu, sigma)
        # ths = acquisition_values if strategy == 'thompson' else self.compute_thompson(mu, sigma, kappa)
        
        # print(f"UCB: {ucbs.mean():0.4f}, EI: {eis.mean():0.4f}, Thompson: {ths.mean():0.4f}, EIG: {self.eigs.mean():0.4f}")
        
        #-------------------------------------------------------------------
        
        diversity_score = self.compute_global_diversity_score()
        # increase diversity weight the more diversity_score is below 0.5
        # if diversity_score < 0.5:
        #     self.diversity_weight /= self.diversity_decay
        # else:
        #     self.diversity_weight *= self.diversity_decay  # Reduce diversity weight over time
        
        # self.diversity_weight *= self.diversity_decay
        self.diversity_weight = np.mean([self.diversity_weight, 1-diversity_score])
        self.diversity_weight *= self.diversity_decay
        
        print(f"Diversity:\tscore: {diversity_score:0.4f}\tweight: {self.diversity_weight:0.4f}")
        
        # get diversity score for each candidate point
        diversity_scores = self.compute_diversity_scores(checkpoint_indices, test_indices)
        
        #--------------------------------------------------------------
        # normalize diversity scores and acquisition values ???
        diversity_scores = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min())
        acquisition_values = (acquisition_values - acquisition_values.min()) / (acquisition_values.max() - acquisition_values.min())
        #--------------------------------------------------------------
        
        # compute combined scores
        combined_scores = (1 - self.diversity_weight) * acquisition_values + self.diversity_weight * diversity_scores
        best_idx = np.argmax(combined_scores)
        
        #--------------------------------------------------------------
        # log points
        if strategy not in self.log:
            self.log[strategy] = []
        self.log[strategy].append((checkpoint_indices[best_idx], test_indices[best_idx], combined_scores[best_idx]))
        
        # # compute z-score for mi compared to mu, and si compared to sigma
        # zmi = (mu[best_idx] - mu.mean())/mu.std()
        # zsi = (sigma[best_idx] - sigma.mean())/sigma.std()
        # print(f"z-mu: {zmi:0.4f}\tz-sig: {zsi:0.4f}")
        
        # if test_indices[best_idx] == 24:
        #     print(best_idx)
        #     mi = mu[best_idx]
        #     si = sigma[best_idx]
            
        #     # # compute z-score for mi compared to mu, and si compared to sigma
        #     # zmi = (mu[best_idx] - mu.mean())/mu.std()
        #     # zsi = (sigma[best_idx] - sigma.mean())/sigma.std()
        #     # print(f"zmi: {zmi:0.4f}, zsi: {zsi:0.4f}")
        #     # # print(f"mu: {mu.mean():0.4f}, sigma: {sigma.mean():0.4f}")
        #     # # print(f"mi: {mi:0.4f}, si: {si:0.4f}")
            
        #     # self.compute_ucb(mu, sigma, kappa)
        #     # self.compute_ei(mu, sigma)
        #     # self.compute_thompson(mu, sigma, kappa)
        #     print()
            
        #--------------------------------------------------------------
        
        return (checkpoint_indices[best_idx], test_indices[best_idx])
    
    
    # @tracker 
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
                next_cp_idx, next_test_idx  = self.acquisition_function(strategy, kappa=kappa) # kappa=2.0
        
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

    def estimate_optimality_confidence_mc(self, epsilon=[0.05, 0.01], n_samples=1000):
        """
        GPU-accelerated Monte Carlo sampling to estimate the probability that 
        the current best checkpoint is within epsilon of the true best.
        """
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Get mean and covariance matrix from GP model
        mu, cov = self.gp.predict(self.all_X_encoded, return_cov=True)
        
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
        samples = samples.reshape(n_samples, len(self.checkpoint_nums), self.n_tests)
        
        # Continue with the same analysis as before
        avg_performances = np.mean(samples, axis=2)
        
        # Current best estimate
        # current_best_idx = np.where(checkpoint_nums == self.get_best_checkpoint())[0][0]
        current_best_idx,_ = self.get_best_checkpoint()
        current_best_idx = np.where(self.checkpoint_nums == current_best_idx)[0][0]
        
        # Count samples where current best is within epsilon of true best
        sample_best_indices = np.argmax(avg_performances, axis=1)
        diffs = avg_performances[np.arange(n_samples), sample_best_indices] - avg_performances[:, current_best_idx]
        diffs /= avg_performances[:, current_best_idx]
        
        # epsilon is a list of values
        within_epsilon_count = np.sum(diffs[:, None] <= np.array(epsilon), axis=0)
        return within_epsilon_count / n_samples
    
    
    def run_sampling(self, n_samples, 
                     acquisition_strategy='ucb', 
                     kappa=2.0,
                     initial_samples=10, 
                     init_strategy='grid', 
                     diversity_weight=0.5,
                     diversity_decay=0.9,
                     stopping_criteria=None,
                     force_exploration_every=100000000, 
                     verbose=True):
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
        self.diversity_decay = diversity_decay
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
        # checkpoint_range = self.full_data['checkpoint_num'].max() - self.full_data['checkpoint_num'].min()
        
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
        
        # reset the break flag
        check_break(reset=True)
        
        try:
            for i in range(remaining_samples):
                start_time = time.time()
                
                acq_strat = acquisition_strategy if isinstance(acquisition_strategy, str) else acquisition_strategy[i % len(acquisition_strategy)]
                
                # Sample next point
                next_cp_idx, next_test_idx = self.sample_next_point(
                    strategy=acq_strat, 
                    kappa=kappa
                )
                
                # Update performance metrics
                current_best_checkpt, current_best_perf = self.get_best_checkpoint()

                error = abs(current_best_perf - true_best_perf) / true_best_perf
                
                self.estimated_best_checkpoints.append(current_best_checkpt)
                self.sampling_fractions.append(len(self.y_sampled) / total_samples)
                self.errors.append(error)
                
                elapsed = time.time() - start_time
                
                # acq_vals = np.array(self.acq_values[-1]).round(8)
                # ei = acq_vals[1]
                # er = self.expected_regret
                # au = self.avg_uncertainty
                
                if verbose and (i+1) % 10 == 0:
                    print()
                    # print('\n' + '-'*100)
                    
                    #-------------------------------------------------------------------
                    # print sampled point counts...
                    # self.print_log()
                    #-------------------------------------------------------------------
                    print(f"Sample {i+1+initial_samples}/{n_samples}: ")

                    print(f"Curr Best:\tCheckpoint {current_best_checkpt}, Error: {error:.2g}")
                    # print(f"Curr Best:\tCheckpoint {current_best_checkpt}\tError: {error:.4f}\tEIG_avg: {self.eigs.mean():.4f}")
                    # print(f"Curr Best:\tCheckpoint {current_best_checkpt}\tError: {error:.4f}\tEI: {acq_vals[1]}")
                    # print(f"Curr Best:\tCheckpoint {current_best_checkpt}, Error: {error:.4f}, Acq:\tUCB: {acq_vals[0]}, EI: {acq_vals[1]}, PI: {acq_vals[2]}, Th: {acq_vals[3]}, ER: {er:.4f}, AU: {au:.4f}")
                    
                    sparsity = self.sampling_fractions[-1]
                    print(f"Sparsity:\t{100*self.sampling_fractions[-1]:.2g}%, "
                    f"Time: {elapsed:.2f}s")
                    
                    # Print diversity metrics
                    cp_hist = np.bincount(np.where(self.sampled_mask)[0])
                    test_hist = np.bincount(np.where(self.sampled_mask)[1])
                    print(f"Diversity:\t{len(cp_hist)}/{self.n_checkpoints} checkpoints, " 
                    f"{len(test_hist)}/{self.n_tests} tests sampled")
                    print(f"Max samples:\tper-checkpoint: {np.max(cp_hist)}, "
                    f"per-test: {np.max(test_hist)}")
                    # print diversity weight
                    # print(f"Diversity Weight: {self.diversity_weight:.2g}")
                    print()
                    
                    #-------------------------------------------------------------------
                    if stopping_criteria is not None:
                        if 'sparsity' in stopping_criteria:
                            if sparsity > stopping_criteria['sparsity']:
                                break
                        # if 'sparsity' in stopping_criteria:
                        #     if sparsity > stopping_criteria['sparsity']:
                        #         break
                        

                    if check_break():
                        break
                    #-------------------------------------------------------------------
                                               
                else:
                    print(f"Curr Best:\tCheckpoint {current_best_checkpt}, Error: {error:.4f}")
                    # print(f"Curr Best:\tCheckpoint {current_best_checkpt}\tError: {error:.4f}\tEIG_avg: {self.eigs.mean():.4f}")
                    # print(f"Curr Best:\tCheckpoint {current_best_checkpt}\tError: {error:.4f}\tEI: {acq_vals[1]}")
                    # print(f"Curr Best:\tCheckpoint {current_best_checkpt}, Error: {error:.4f}, Acq:\tUCB: {acq_vals[0]}, EI: {acq_vals[1]}, PI: {acq_vals[2]}, Th: {acq_vals[3]}, ER: {er:.4f}, AU: {au:.4f}")
                    
                # if ei < 0.002 and i > 10:
                #     break
                    
        except KeyboardInterrupt:
            print("\nSampling interrupted by user. Using current best estimate.")
            
        if verbose:
            print(f"\nFinal estimate after {n_samples} samples: Checkpoint {current_best_checkpt}")
            print(f"Final error: {error:.4f}")
            print(f"Total sampling fraction: {len(self.y_sampled) / total_samples:.4f}")
        
        return self.predict_performance()

# Example usage
if __name__ == "__main__":
    fn = 'data/phi4-math-4claude.txt'
    # fn = 'data/phi4-bw-4claude.txt'
    
    rand_seed = np.random.randint(1000)
    # rand_seed = 438 # 445  608  295  939  418  622  669??
    print(f"Random seed: {rand_seed}")
    
    stopping_criteria = {
        'sparsity': 0.05,
    }
    
    # Create the adaptive sampler with diversity
    sampler = DiverseAdaptiveSampler(fn,
                                     test_subset=0.75, 
                                     random_seed=rand_seed)
    
    # Run sampling with 100 samples and strong diversity enforcement
    predictions = sampler.run_sampling(
        n_samples=1000,
        # acquisition_strategy='thompson', # ucb, ei, pi, thompson, eig
        # acquisition_strategy=['eig', 'ucb', 'ei', 'thompson'],
        # acquisition_strategy=['ucb', 'thompson'],
        acquisition_strategy='ucb',
        kappa=4,
        
        initial_samples=50,
        init_strategy='latin_hypercube', # random, grid, extremes, latin_hypercube
        diversity_weight=0.9,       # preference for diversity
        diversity_decay=0.98,        # decay rate for diversity
        # force_exploration_every=10,  # force exploration frequently
        stopping_criteria=stopping_criteria,
        verbose=True
    )
    
    # print points chosen by each strategy
    sampler.print_log()
    
    # Print timing information
    # print("\n=== Performance Profiling ===")
    # tracker.print_stats(sort_by="fraction")  # Sort by percentage of runtime
    
    
    # Plot results
    sampler.plot_results()
    
    # BEST: acquisition_strategy=thompson, diversity_weight=0.5, force_exploration_every=10