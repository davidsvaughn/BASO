import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import norm
import time
import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warnings

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPyTorchModel(gpytorch.models.ExactGP):
    """
    GPyTorch model for adaptive sampling.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(GPyTorchModel, self).__init__(train_x, train_y, likelihood)
        # Define input dimension from training data
        input_dim = train_x.size(-1)
        
        # Mean module
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Covariance module (equivalent to the kernel in the original code)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)  # Simplified for better performance
        ) #+ gpytorch.kernels.WhiteNoiseKernel()
        
    def forward(self, x):
        """
        Forward pass of the GPyTorch model.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DiverseAdaptiveSampler:
    """
    Adaptive sequential sampling with diversity enforcement to efficiently
    estimate peak checkpoint performance using GPyTorch for acceleration.
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
        
        # Setup GP model - will be initialized later
        self.gp_model = None
        self.likelihood = None
        
        # Performance metrics
        self.true_best_checkpoint = None
        self.estimated_best_checkpoints = []
        self.sampling_fractions = []
        self.errors = []
        
        # Diversity settings
        self.diversity_weight = 0.5  # Balance between utility and diversity
        self.force_exploration_every = 5  # Force exploration every N samples
        self.current_sample_count = 0
    
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
        """Update the Gaussian Process model with current samples using GPyTorch with optimizations"""
        if len(self.y_sampled) < 5:
            print("Warning: Very few samples available. Model may be unreliable.")
            return None
            
        # Convert to numpy arrays
        X = np.array(self.X_sampled)
        y = np.array(self.y_sampled)
        
        # Encode features
        X_encoded = self._encode_features(X)
        
        try:
            # Convert to PyTorch tensors and move to device
            X_tensor = torch.tensor(X_encoded, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            
            # Initialize likelihood and model
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.gp_model = GPyTorchModel(X_tensor, y_tensor, self.likelihood).to(self.device)
            
            # Use the Adam optimizer
            optimizer = torch.optim.Adam([
                {'params': self.gp_model.parameters()},
            ], lr=0.1)
            
            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
            
            # Training loop
            self.gp_model.train()
            self.likelihood.train()
            
            # Use early stopping to avoid overfitting
            patience = 10
            best_loss = float('inf')
            no_improve_count = 0
            
            # Set up mixed precision training if on CUDA
            use_amp = self.device.type == 'cuda'
            scaler = torch.cuda.amp.GradScaler() if use_amp else None
            
            # Optimized training using context managers
            with gpytorch.settings.use_toeplitz(False), \
                 gpytorch.settings.cg_tolerance(0.01), \
                 gpytorch.settings.max_preconditioner_size(15):
                
                for i in range(100):  # Maximum of 100 training iterations
                    optimizer.zero_grad()
                    
                    # Mixed precision training path
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            output = self.gp_model(X_tensor)
                            loss = -mll(output, y_tensor)
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    # Standard training path
                    else:
                        output = self.gp_model(X_tensor)
                        loss = -mll(output, y_tensor)
                        loss.backward()
                        optimizer.step()
                    
                    # Early stopping check
                    current_loss = loss.item()
                    if current_loss < best_loss:
                        best_loss = current_loss
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    if no_improve_count >= patience:
                        break
            
            # Set model to evaluation mode
            self.gp_model.eval()
            self.likelihood.eval()
            
            return self.gp_model
            
        except Exception as e:
            print(f"Error fitting GP model: {str(e)}")
            return None
    
    def predict_performance(self, checkpoint_nums=None):
        """Predict mean performance for each checkpoint across all tests using GPyTorch with optimizations."""
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
            
            # Apply performance optimizations and batch processing
            batch_size = 1000  # Adjust based on memory constraints
            mu_list = []
            sigma_list = []
            
            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.use_toeplitz(False), \
                 gpytorch.settings.max_root_decomposition_size(100):
                
                # Process in batches
                for i in range(0, len(X_tensor), batch_size):
                    X_batch = X_tensor[i:i+batch_size]
                    
                    pred_dist = self.likelihood(self.gp_model(X_batch))
                    mu_batch = pred_dist.mean.cpu().numpy()
                    var_batch = pred_dist.variance.cpu().numpy()
                    sigma_batch = np.sqrt(var_batch)
                    
                    mu_list.append(mu_batch)
                    sigma_list.append(sigma_batch)
            
            # Combine batch results
            mu = np.concatenate(mu_list)
            sigma = np.concatenate(sigma_list)
            
            # Group by checkpoint
            checkpoint_groups = {}
            for i, (pred, std, cp_idx) in enumerate(zip(mu, sigma, checkpoint_indices)):
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
    
    def acquisition_function(self, strategy='ucb', kappa=2.0):
        """
        Calculate acquisition values for all unsampled points with diversity bonus.
        Optimized version using batch processing with GPyTorch.
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
            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X_unsampled_encoded, dtype=torch.float32).to(self.device)
            
            # Use optimized settings for faster prediction
            with torch.no_grad(), \
                 gpytorch.settings.fast_pred_var(), \
                 gpytorch.settings.use_toeplitz(False), \
                 gpytorch.settings.max_root_decomposition_size(100):
                
                # Use batch prediction for better performance
                batch_size = 1000  # Adjust based on memory constraints
                
                mu_list = []
                sigma_list = []
                
                # Process in batches for memory efficiency
                for i in range(0, len(X_tensor), batch_size):
                    X_batch = X_tensor[i:i+batch_size]
                    
                    pred_dist = self.likelihood(self.gp_model(X_batch))
                    mu_batch = pred_dist.mean.cpu().numpy()
                    variance_batch = pred_dist.variance.cpu().numpy()
                    sigma_batch = np.sqrt(variance_batch)
                    
                    mu_list.append(mu_batch)
                    sigma_list.append(sigma_batch)
                
                # Combine batch results
                mu = np.concatenate(mu_list)
                sigma = np.concatenate(sigma_list)
                
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
        
        # Calculate diversity scores and combined acquisition values
        combined_values = []
        
        for i, (cp_idx, test_idx) in enumerate(zip(checkpoint_indices, test_indices)):
            diversity_score = self.compute_diversity_score(cp_idx, test_idx)
            
            # Combine acquisition value with diversity score
            # Higher diversity_weight emphasizes more diverse sampling
            combined_value = (1 - self.diversity_weight) * acquisition_values[i] + \
                            self.diversity_weight * diversity_score
            
            combined_values.append((cp_idx, test_idx, combined_value))
        
        # Sort by combined value (descending)
        combined_values.sort(key=lambda x: x[2], reverse=True)
        
        # Return best point and all candidates (for analysis)
        best_point = (combined_values[0][0], combined_values[0][1])
        return best_point, combined_values
    
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
        
        try:
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
                    print('\n' + '-'*100)
                    print(f"Sample {i+1+initial_samples}/{n_samples}: ")
                    print(f"Curr Best:\tCheckpoint {current_best}, Error: {error:.4f}")
                    print(f"Sparsity:\t{100*self.sampling_fractions[-1]:.2f}%, "
                    f"Time: {elapsed:.2f}s")
                    
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
            print(f"\nFinal estimate after {n_samples} samples: Checkpoint {current_best}")
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
    predictions = sampler.run_sampling(
        n_samples=300,
        acquisition_strategy='ucb',  # ucb, ei, thompson
        kappa=2.0,
        initial_samples=50,
        init_strategy='latin_hypercube',  # random, grid, extremes, latin_hypercube
        diversity_weight=0.5,    # Strong preference for diversity
        force_exploration_every=10,  # Force exploration frequently
        verbose=True
    )
    
    # Plot results
    sampler.plot_results()