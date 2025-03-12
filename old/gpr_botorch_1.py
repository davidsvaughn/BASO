import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gpytorch
import botorch

from botorch.models import SingleTaskGP
# from botorch.fit import fit_gpytorch_model
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement, qExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf_mixed
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import norm
import time
import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warnings

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DiverseAdaptiveSampler:
    """
    Adaptive sequential sampling with diversity enforcement to efficiently
    estimate peak checkpoint performance using BoTorch for acceleration.
    """
    
    def __init__(self, data_file, random_seed=42, use_gpu=True):
        """Initialize the adaptive sampler with diversity constraints."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Set device
        self.device = device if use_gpu else torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Load the full dataset (this would be unknown in a real scenario)
        self.full_data = pd.read_csv(data_file, sep='\t')
        
        # remove last row
        self.full_data = self.full_data[:-1]
        
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
        
        # Setup BoTorch model - will be initialized later
        self.model = None
        self.mll = None
        
        # Performance metrics
        self.true_best_checkpoint = None
        self.estimated_best_checkpoints = []
        self.sampling_fractions = []
        self.errors = []
        
        # Diversity settings
        self.diversity_weight = 0.5  # Balance between utility and diversity
        self.force_exploration_every = 5  # Force exploration every N samples
        self.current_sample_count = 0
        
        # For parallel sampling
        self.mc_samples = 512  # Number of Monte Carlo samples for q-acquisition functions
        self.q = 1  # Default batch size for sampling (can be increased for parallel sampling)
    
    def _encode_features(self, X):
        """Encode features appropriately for the model."""
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
            
        elif strategy == 'sobol':
            # Use BoTorch's Sobol sequence for better quasi-random sampling
            from botorch.utils.sampling import draw_sobol_samples
            
            # Create bounds for the sampling space (normalized 0-1)
            bounds = torch.zeros(2, 2, device=self.device)
            bounds[0, 1] = n_checkpoints  # Upper bound for first dimension
            bounds[1, 1] = n_tests # Upper bound for second dimension
            
            # Draw Sobol samples
            sobol_samples = draw_sobol_samples(
                bounds=bounds.t(), 
                n=n_initial, 
                q=1, 
                seed=self.random_seed
            ).squeeze(1).cpu().numpy()
            
            # Scale to indices
            # checkpoint_indices = (sobol_samples[:, 0] * (n_checkpoints - 1)).astype(int)
            # test_indices = (sobol_samples[:, 1] * (n_tests - 1)).astype(int)
            checkpoint_indices = sobol_samples[:, 0].astype(int)
            test_indices = sobol_samples[:, 1].astype(int)
        
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
        
        # Collect samples
        for cp_idx, test_idx in zip(checkpoint_indices[:n_initial], test_indices[:n_initial]):
            self._add_sample(cp_idx, test_idx)
    
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
    
    def update_model(self):
        """Update the BoTorch model with current samples."""
        if len(self.y_sampled) < 5:
            print("Warning: Very few samples available. Model may be unreliable.")
            return None
            
        # Convert to numpy arrays
        X_np = np.array(self.X_sampled)
        y_np = np.array(self.y_sampled)
        
        # Encode features
        X_encoded = self._encode_features(X_np)
        
        # try:
        # Convert to PyTorch tensors and move to device
        X_tensor = torch.tensor(X_encoded, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_np, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # Standardize y values for better numerical stability
        y_mean = y_tensor.mean()
        y_std = y_tensor.std()
        y_normalized = (y_tensor - y_mean) / y_std
        
        # Initialize BoTorch model with normalized targets
        self.model = SingleTaskGP(X_tensor, y_normalized)
        self.model = self.model.to(self.device)
        
        # Initialize likelihood and MLL
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.mll = self.mll.to(self.device)
        
        # Fit model using BoTorch's optimized fitting function
        # This automatically handles optimization settings
        fit_gpytorch_mll(self.mll)
        
        # Store normalization constants for later use
        self.y_mean = y_mean
        self.y_std = y_std
        
        return self.model
            
        # except Exception as e:
        #     print(f"Error fitting BoTorch model: {str(e)}")
        #     return None
    
    def predict_performance(self, checkpoint_nums=None):
        """Predict mean performance for each checkpoint across all tests using BoTorch."""
        if checkpoint_nums is None:
            checkpoint_nums = self.full_data['checkpoint_num'].unique()
        
        results_list = []
        
        # Create features for all checkpoint-test combinations
        features = []
        checkpoint_indices = []
        
        for i, cp_num in enumerate(checkpoint_nums):
            for test_idx in range(len(self.test_cols)):
                features.append([cp_num, test_idx])
                checkpoint_indices.append(i)
        
        # Encode features
        X_encoded = self._encode_features(features)
        
        try:
            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X_encoded, dtype=torch.float32).to(self.device)
            
            # Set evaluation mode
            self.model.eval()
            
            # Apply performance optimizations using BoTorch settings
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # Process in batches for memory efficiency
                batch_size = 1024  # Larger batch size for BoTorch
                
                means = []
                variances = []
                
                for i in range(0, X_tensor.size(0), batch_size):
                    X_batch = X_tensor[i:i+batch_size]
                    posterior = self.model(X_batch)
                    
                    # Get mean and variance
                    batch_mean = posterior.mean
                    batch_variance = posterior.variance
                    
                    # Denormalize predictions
                    batch_mean = batch_mean * self.y_std + self.y_mean
                    batch_variance = batch_variance * (self.y_std ** 2)
                    
                    means.append(batch_mean.cpu())
                    variances.append(batch_variance.cpu())
                
                # Combine batches
                all_means = torch.cat(means).numpy()
                all_stds = torch.sqrt(torch.cat(variances)).numpy()
            
            # Group by checkpoint
            checkpoint_groups = {}
            for i, (pred, std, cp_idx) in enumerate(zip(all_means, all_stds, checkpoint_indices)):
                cp_num = checkpoint_nums[cp_idx]
                if cp_num not in checkpoint_groups:
                    checkpoint_groups[cp_num] = {'preds': [], 'stds': []}
                
                checkpoint_groups[cp_num]['preds'].append(pred)
                checkpoint_groups[cp_num]['stds'].append(std)
            
            # Calculate average for each checkpoint
            for cp_num, data in checkpoint_groups.items():
                avg_pred = np.mean(data['preds'])
                avg_std = np.mean(data['stds'])
                
                results_list.append({
                    'checkpoint_num': cp_num,
                    'predicted_value': avg_pred,
                    'uncertainty': avg_std
                })
                
        except Exception as e:
            print(f"Error predicting performance: {str(e)}")
            # Fallback values for all checkpoints
            for cp_num in checkpoint_nums:
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
    
    def create_diversity_penalized_acquisition_function(self, base_acqf):
        """
        Create a wrapper around a base acquisition function that incorporates 
        the diversity penalty directly into the acquisition values.
        """
        original_forward = base_acqf.forward
        
        def diversity_forward(X):
            # Get original acquisition values
            acq_values = original_forward(X)
            
            # Apply diversity penalty to each point
            # Extract indices from the input
            X_np = X.detach().cpu().numpy()
            
            # Scale back from normalized to original space
            # This assumes X contains normalized checkpoint values followed by one-hot test indices
            # We need to convert back to actual indices to assess diversity
            
            # First, get original checkpoint values from scaled values
            scaled_checkpoints = X_np[:, 0].reshape(-1, 1)
            checkpoint_nums = self.checkpoint_scaler.inverse_transform(scaled_checkpoints).flatten()
            
            # Map checkpoint numbers back to indices
            checkpoint_mapping = {num: idx for idx, num in enumerate(self.full_data['checkpoint_num'].values)}
            checkpoint_indices = np.array([checkpoint_mapping.get(num, 0) for num in checkpoint_nums])
            
            # For test indices, get the argmax of one-hot encoding
            one_hot_size = self.test_encoder.categories_[0].size
            test_one_hot = X_np[:, 1:1+one_hot_size]
            test_indices = np.argmax(test_one_hot, axis=1)
            
            # Compute diversity scores
            diversity_scores = np.zeros(X.shape[0])
            for i, (cp_idx, test_idx) in enumerate(zip(checkpoint_indices, test_indices)):
                diversity_scores[i] = self.compute_diversity_score(cp_idx, test_idx)
            
            # Convert to tensor
            diversity_tensor = torch.tensor(diversity_scores, dtype=X.dtype, device=X.device).view_as(acq_values)
            
            # Combine acquisition and diversity (1-diversity_weight for acquisition, diversity_weight for diversity)
            combined_values = (1 - self.diversity_weight) * acq_values + self.diversity_weight * diversity_tensor
            
            return combined_values
        
        # Replace the forward method
        base_acqf.forward = diversity_forward
        return base_acqf
    
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
        return self.acquisition_function()[0]
    
    def get_search_bounds(self):
        """
        Get the bounds for optimization in the transformed feature space.
        """
        # bounds = torch.zeros(2, 2, device=self.device)
        # bounds[0, 1] = self.n_checkpoints  # Upper bound for first dimension
        # bounds[1, 1] = self.n_tests # Upper bound for second dimension
        # return bounds.t()
            
        # Get the min and max values for scaled checkpoints
        checkpoint_nums = self.full_data['checkpoint_num'].values.reshape(-1, 1)
        scaled_checkpoints = self.checkpoint_scaler.transform(checkpoint_nums)
        min_scaled_cp = np.min(scaled_checkpoints)
        max_scaled_cp = np.max(scaled_checkpoints)
        
        # For one-hot encoding, bounds are 0 and 1
        one_hot_size = self.test_encoder.categories_[0].size
        
        # Create bounds tensor of shape (d, 2) where d is the feature dimension
        bounds = torch.zeros((1 + one_hot_size, 2), device=self.device)
        
        # Set checkpoint bounds
        bounds[0, 0] = min_scaled_cp
        bounds[0, 1] = max_scaled_cp
        
        # Set one-hot bounds (all 0 to 1)
        bounds[1:, 1] = 1.0
        
        # return transposed bounds
        # return bounds.t()
        
        return bounds
    
    def acquisition_function(self, strategy='ucb', kappa=2.0):
        """
        Calculate acquisition values for all unsampled points with diversity bonus
        using BoTorch's optimized acquisition functions.
        
        Returns:
        --------
        Tuple of (best_point, all_candidate_points)
        where best_point is (checkpoint_idx, test_idx)
        and all_candidate_points is a list of (checkpoint_idx, test_idx, acq_value)
        """
        # Get indices of unsampled points
        unsampled_indices = np.where(~self.sampled_mask)
        
        # fixed_features_list = [{0: float(cp), 1: float(test)} for cp, test in zip(*unsampled_indices)]
        
        checkpoint_indices = unsampled_indices[0]
        test_indices = unsampled_indices[1]
        
        # If no unsampled points left, return random point
        if len(checkpoint_indices) == 0:
            random_point = (np.random.randint(0, self.n_checkpoints),
                           np.random.randint(0, self.n_tests))
            return random_point, [(random_point[0], random_point[1], 0.0)]
        
        # try:
    
        # Get optimization bounds
        bounds = self.get_search_bounds()
        
        # Create acquisition function
        if strategy == 'ucb':
            # Upper Confidence Bound
            acq_func = UpperConfidenceBound(self.model, beta=kappa**2)
            
        elif strategy == 'ei':
            # Expected Improvement
            best_f = self.model.train_targets.max() * self.y_std + self.y_mean
            acq_func = ExpectedImprovement(self.model, best_f=best_f)
            
        elif strategy == 'qucb':
            # q-Upper Confidence Bound (batch acquisition)
            # sampler = SobolQMCNormalSampler(num_samples=self.mc_samples)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))
            acq_func = qUpperConfidenceBound(self.model, beta=kappa**2, sampler=sampler)
            
        elif strategy == 'qei':
            # q-Expected Improvement (batch acquisition)
            best_f = self.model.train_targets.max() * self.y_std + self.y_mean
            # sampler = SobolQMCNormalSampler(num_samples=self.mc_samples)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))
            acq_func = qExpectedImprovement(self.model, best_f=best_f, sampler=sampler)
            
        elif strategy == 'thompson':
            # Thompson Sampling - implemented manually since BoTorch doesn't have this directly
            # For Thompson sampling, we just sample from the posterior at candidate points
            X_tensor = torch.zeros((len(checkpoint_indices), bounds.shape[0]), device=self.device)
            
            # Create features for all unsampled points
            for i, (cp_idx, test_idx) in enumerate(zip(checkpoint_indices, test_indices)):
                # Get the checkpoint number
                checkpoint_num = self.full_data.iloc[cp_idx]['checkpoint_num']
                
                # Create and encode the feature
                feature = [[checkpoint_num, test_idx]]
                encoded = self._encode_features(feature)
                X_tensor[i] = torch.tensor(encoded, dtype=torch.float32, device=self.device)
            
            # Sample from posterior
            with torch.no_grad():
                posterior = self.model(X_tensor)
                samples = posterior.rsample().view(-1).cpu().numpy()
                
            # Find best point
            best_idx = np.argmax(samples)
            best_point = (checkpoint_indices[best_idx], test_indices[best_idx])
            
            # Create list of all candidate points with acquisition values
            all_candidates = []
            for i, (cp_idx, test_idx, acq_val) in enumerate(zip(checkpoint_indices, test_indices, samples)):
                all_candidates.append((cp_idx, test_idx, acq_val))
            
            # Sort by acquisition value
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            
            return best_point, all_candidates
        
        else:
            raise ValueError(f"Unknown acquisition strategy: {strategy}")
        
        # # Apply diversity penalty if not Thompson sampling
        # if strategy != 'thompson':
        #     acq_func = self.create_diversity_penalized_acquisition_function(acq_func)
        
        # For parallel acquisition (q > 1)
        if strategy.startswith('q') and self.q > 1:
            # Use BoTorch's optimize_acqf for batch acquisition
            candidates, acq_values = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds.t(),
                q=self.q,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 50, "maxiter": 200},
            )
            
            # candidates, acq_values = optimize_acqf_mixed(
            #     acq_function=acq_func,
            #     fixed_features_list=fixed_features_list,
            #     # bounds=bounds,
            #     bounds=None,
            #     q=self.q,
            #     num_restarts=10,
            #     raw_samples=512,
            #     options={"batch_limit": 50, "maxiter": 200},
            # )
            
            
            # Process batch candidates
            all_candidates = []
            for i in range(self.q):
                # Extract candidate
                candidate = candidates[i].cpu().numpy()
                
                # Convert back to checkpoint and test indices
                cp_idx, test_idx = self._convert_candidate_to_indices(candidate)
                
                # Store with acquisition value
                all_candidates.append((cp_idx, test_idx, acq_values[i].item()))
            
            # Sort by acquisition value
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Return best point and all candidates
            best_point = (all_candidates[0][0], all_candidates[0][1])
            return best_point, all_candidates
            
        else:
            # Sequential acquisition (q = 1)
            # Use BoTorch's optimize_acqf for single-point acquisition
            candidate, acq_value = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 50, "maxiter": 200},
            )
            
            # Convert back to checkpoint and test indices
            cp_idx, test_idx = self._convert_candidate_to_indices(candidate[0].cpu().numpy())
            
            # For single point, we still return a list of tuples for consistency
            best_point = (cp_idx, test_idx)
            all_candidates = [(cp_idx, test_idx, acq_value.item())]
            
            # Also include random points from unsampled indices for diversity
            sample_indices = np.random.choice(len(checkpoint_indices), min(10, len(checkpoint_indices)), replace=False)
            for i in sample_indices:
                cp_idx = checkpoint_indices[i]
                test_idx = test_indices[i]
                # Assign a placeholder acquisition value
                all_candidates.append((cp_idx, test_idx, -1.0))
            
            return best_point, all_candidates
        
        # except Exception as e:
        #     print(f"Error in acquisition function: {str(e)}")
        #     # If prediction fails, return random point
        #     rand_idx = np.random.randint(0, len(checkpoint_indices))
        #     random_point = (checkpoint_indices[rand_idx], test_indices[rand_idx])
        #     return random_point, [(random_point[0], random_point[1], 0.0)]
    
    def _convert_candidate_to_indices(self, candidate):
        """
        Convert a candidate point from the optimization space back to checkpoint and test indices.
        
        Parameters:
        -----------
        candidate : numpy.ndarray
            Candidate point from BoTorch optimization
            
        Returns:
        --------
        Tuple of (checkpoint_idx, test_idx)
        """
        # Extract the scaled checkpoint value
        scaled_checkpoint = candidate[0]
        
        # Convert back to original checkpoint number
        checkpoint_num = self.checkpoint_scaler.inverse_transform([[scaled_checkpoint]])[0][0]
        
        # Find the closest checkpoint index
        checkpoint_nums = self.full_data['checkpoint_num'].values
        checkpoint_idx = np.argmin(np.abs(checkpoint_nums - checkpoint_num))
        
        # Extract one-hot encoded test indices
        one_hot_size = self.test_encoder.categories_[0].size
        one_hot_test = candidate[1:1+one_hot_size]
        
        # Find the most likely test index
        test_idx = np.argmax(one_hot_test)
        
        return checkpoint_idx, test_idx
    
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
                # Sample kappa from normal distribution
                k = kappa
                k = np.random.normal(k, 1.0, 1).clip(0.5, 3.5)
                
                # Get next point from acquisition function
                (next_cp_idx, next_test_idx), _ = self.acquisition_function(strategy, k)
        
        # Add the sample
        self._add_sample(next_cp_idx, next_test_idx)
        
        return next_cp_idx, next_test_idx
    
    def sample_batch_points(self, batch_size, strategy='qei', kappa=2.0):
        """
        Sample multiple points in parallel using BoTorch's batch acquisition functions.
        
        Parameters:
        -----------
        batch_size : int
            Number of points to sample in parallel
        strategy : str
            Acquisition strategy: 'qucb' or 'qei'
        kappa : float
            Exploration weight for UCB
            
        Returns:
        --------
        List of (checkpoint_idx, test_idx) tuples
        """
        # Set batch size
        self.q = batch_size
        
        # Update the model first
        if self.update_model() is None:
            # If model update failed, sample randomly
            batch_points = []
            for _ in range(batch_size):
                next_cp_idx = np.random.randint(0, self.n_checkpoints)
                next_test_idx = np.random.randint(0, self.n_tests)
                batch_points.append((next_cp_idx, next_test_idx))
        else:
            # Get batch points from acquisition function
            _, all_candidates = self.acquisition_function(strategy, kappa)
            
            # Take top batch_size candidates
            batch_points = [(cp_idx, test_idx) for cp_idx, test_idx, _ in all_candidates[:batch_size]]
        
        # Add all samples
        for next_cp_idx, next_test_idx in batch_points:
            self._add_sample(next_cp_idx, next_test_idx)
        
        # Reset q to 1 for sequential sampling
        self.q = 1
        
        return batch_points
    
    def run_sampling(self, n_samples, acquisition_strategy='ucb', kappa=2.0,
                   initial_samples=10, init_strategy='sobol', diversity_weight=0.5,
                   force_exploration_every=5, batch_size=1, verbose=True):
        """
        Run the complete adaptive sampling procedure with diversity constraints.
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples to collect
        acquisition_strategy : str
            Acquisition strategy: 'ucb', 'ei', 'qucb', 'qei', or 'thompson'
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
        batch_size : int
            Number of points to sample in each iteration (parallel sampling)
        verbose : bool
            Whether to print progress
        """
        # Set diversity parameters
        self.diversity_weight = diversity_weight
        self.force_exploration_every = force_exploration_every
        
        # Set batch size
        self.q = batch_size
        
        # Adjust acquisition strategy for batch sampling
        if batch_size > 1 and not acquisition_strategy.startswith('q'):
            if acquisition_strategy == 'ucb':
                acquisition_strategy = 'qucb'
            elif acquisition_strategy == 'ei':
                acquisition_strategy = 'qei'
            else:
                print(f"Warning: {acquisition_strategy} does not support batch sampling. Using qei instead.")
                acquisition_strategy = 'qei'
        
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
        remaining_iterations = (remaining_samples + batch_size - 1) // batch_size  # Ceiling division
        
        try:
            for i in range(remaining_iterations):
                start_time = time.time()
                
                # Calculate actual batch size for this iteration
                current_batch_size = min(batch_size, n_samples - len(self.y_sampled))
                
                if current_batch_size > 1:
                    # Batch sampling
                    batch_points = self.sample_batch_points(
                        batch_size=current_batch_size,
                        strategy=acquisition_strategy,
                        kappa=kappa
                    )
                else:
                    # Sequential sampling
                    next_cp_idx, next_test_idx = self.sample_next_point(
                        strategy=acquisition_strategy,
                        kappa=kappa
                    )
                
                # Update performance metrics
                current_best = self.get_best_checkpoint()
                error = abs(current_best - true_best) / checkpoint_range
                
                self.estimated_best_checkpoints.append(current_best)
                self.sampling_fractions.append(len(self.y_sampled) / total_samples)
                self.errors.append(error)
                
                elapsed = time.time() - start_time
                
                if verbose and (i+1) % max(1, 10 // batch_size) == 0:
                    print('\n' + '-'*100)
                    print(f"Iteration {i+1}/{remaining_iterations}, Total samples: {len(self.y_sampled)}/{n_samples}")
                    print(f"Current Best:\tCheckpoint {current_best}, Error: {error:.4f}")
                    print(f"Sparsity:\t{100*self.sampling_fractions[-1]:.2f}%, "
                    f"Time per iteration: {elapsed:.2f}s")
                    
                    # Print diversity metrics
                    cp_hist = np.bincount(np.where(self.sampled_mask)[0])
                    test_hist = np.bincount(np.where(self.sampled_mask)[1])
                    print(f"Diversity:\t{len(cp_hist)}/{self.n_checkpoints} checkpoints, " 
                    f"{len(test_hist)}/{self.n_tests} tests sampled")
                    print(f"Max samples:\tper-checkpoint: {np.max(cp_hist)}, "
                    f"per-test: {np.max(test_hist)}")
                    
        except KeyboardInterrupt:
            print("\nSampling interrupted by user. Using current best estimate.")
            
        if verbose:
            print(f"\nFinal estimate after {len(self.y_sampled)} samples: Checkpoint {current_best}")
            print(f"Final error: {error:.4f}")
            print(f"Total sampling fraction: {len(self.y_sampled) / total_samples:.4f}")
        
        return self.predict_performance()
    
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

# Example usage
if __name__ == "__main__":
    
    rand_seed = np.random.randint(1000)
    print(f"Random seed: {rand_seed}")
    
    # Create the adaptive sampler with diversity and GPU acceleration
    sampler = DiverseAdaptiveSampler("data/phi4-math-4claude.txt",
                                     random_seed=rand_seed,
                                     use_gpu=torch.cuda.is_available())
    
    # Run sampling with 300 samples and strong diversity enforcement
    # Now with batch sampling for faster evaluation
    predictions = sampler.run_sampling(
        n_samples=300,
        acquisition_strategy='qei',  # 'ucb', 'ei', 'qucb', 'qei', 'thompson'
        kappa=2.0,
        initial_samples=50,
        init_strategy='sobol',  # 'random', 'grid', 'extremes', 'latin_hypercube', 'sobol'
        diversity_weight=0.5,    # Strong preference for diversity
        force_exploration_every=10,  # Force exploration frequently
        batch_size=4,  # Sample 4 points in parallel for faster execution
        verbose=True
    )
    
    # Plot results
    sampler.plot_results()