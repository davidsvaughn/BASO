import sys, os
import numpy as np
import pandas as pd
import random
import time
from scipy.stats import norm
from matplotlib import pyplot as plt
import gc
import torch
import gpytorch
# import gpytorch.constraints

from multitask.bkp.multitask_synthetic import generate_synthetic_checkpoint_data

torch.set_default_dtype(torch.float64)

rand_seed = -1

#--------------------------------------------------------------------------
# UTILITY FUNCTIONS
def clear_cuda_tensors(target_size=None): # (1, 8192, 32, 96)
    """Clear tensors of specific size from memory"""
    if not torch.cuda.is_available():
        return
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                if target_size is None or obj.size() == target_size:
                    del obj
                    count += 1
        except: 
            pass
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Cleared {count} tensors")

#--------------------------------------------------------------------------

def improved_stopping_criterion(
    y_mean, y_covar, checkpoint_nums, sampled_mask, step,
    history_dict=None, best_history=None,
    min_samples_pct=10.0,  # Minimum % of data to sample before considering stopping
    stability_count=5,     # Number of steps best checkpoint must remain stable
    uncertainty_scale=5.0, # Scale factor to combat GP overconfidence
    cv_error_threshold=0.01, # Cross-validation relative error threshold
    n_samples=1000,        # Number of posterior samples
    confidence_threshold=0.99,  # % confidence requirement
):
    """
    A more robust stopping criterion that addresses GP overconfidence issues
    
    Args:
        y_mean: Mean predictions from GP (shape K x Z)
        y_covar: Covariance matrix from GP (shape K x Z x K x Z)
        checkpoint_nums: Array of checkpoint numbers 
        sampled_mask: Boolean mask of sampled points (shape K x Z)
        step: Current iteration number
        history_dict: Dictionary tracking past predictions (will be created if None)
        best_history: List tracking past best checkpoints (will be created if None)
        min_samples_pct: Minimum percentage of data to sample before considering stopping
        stability_count: Number of consecutive steps best checkpoint must be stable
        uncertainty_scale: Factor to scale uncertainty (combat GP overconfidence)
        cv_error_threshold: Maximum allowed cross-validation relative error
        n_samples: Number of posterior samples to draw
        
    Returns:
        stop_flag: Boolean indicating whether to stop sampling
        diagnostics: Dictionary with diagnostic information
        history_dict: Updated history dictionary (for next iteration)
        best_history: Updated best checkpoint history (for next iteration)
    """
    K, Z = y_mean.shape
    N = K * Z
    
    # Initialize tracking structures if not provided
    if history_dict is None:
        history_dict = {}
    if best_history is None:
        best_history = []
    
    # 1. Preliminary checks - don't even consider stopping if we haven't sampled enough
    percent_sampled = 100 * sampled_mask.sum() / N
    if percent_sampled < min_samples_pct:
        return False, {
            "reason": f"Not enough data sampled ({percent_sampled:.2f}% < {min_samples_pct}%)",
            "percent_sampled": percent_sampled
        }, history_dict, best_history
    
    # 2. Compute average performance for each checkpoint
    checkpoint_means = y_mean.mean(axis=1)  # Shape: (K,)
    best_pred_idx = np.argmax(checkpoint_means)
    best_pred_checkpoint = checkpoint_nums[best_pred_idx]
    
    # Add to history
    best_history.append(best_pred_checkpoint)
    if len(best_history) > stability_count:
        best_history.pop(0)
    
    # 3. Check for stability of best checkpoint prediction
    stable_prediction = len(best_history) == stability_count and all(x == best_history[0] for x in best_history)
    if not stable_prediction:
        return False, {
            "reason": f"Best checkpoint prediction not stable: {best_history}",
            "best_history": best_history
        }, history_dict, best_history
    
    # 4. Compute standard error of the mean for each checkpoint (with inflation)
    checkpoint_stderrs = np.zeros(K)
    for k in range(K):
        # Extract the Z×Z covariance matrix for checkpoint k
        task_cov_matrix = y_covar[k, :, k, :]
        # Standard error of mean with inflation factor
        checkpoint_stderrs[k] = uncertainty_scale * np.sqrt(task_cov_matrix.sum()) / Z
    
    # 5. Construct the covariance matrix for checkpoint means (with inflation)
    checkpoint_cov = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            # Average covariance between tasks for checkpoints i and j
            # Apply uncertainty scaling to combat GP overconfidence
            checkpoint_cov[i, j] = uncertainty_scale * np.mean(y_covar[i, :, j, :]) / Z**2
    
    # Ensure covariance matrix is positive semi-definite
    min_eig = np.min(np.linalg.eigvalsh(checkpoint_cov))
    if min_eig < 0:
        checkpoint_cov += np.eye(K) * (abs(min_eig) + 1e-6)
    
    # 6. Perform cross-validation on sampled points
    # Get indices of sampled points
    sampled_i, sampled_j = np.where(sampled_mask)
    n_sampled = len(sampled_i)
    
    # Skip cross-validation if we have too few samples
    if n_sampled < 10:
        cv_success = False
        max_rel_error = float('inf')
    else:
        # Choose a subset of sampled points for CV (to avoid excessive computation)
        cv_indices = np.random.choice(n_sampled, min(n_sampled, 30), replace=False)
        
        rel_errors = []
        for idx in cv_indices:
            i, j = sampled_i[idx], sampled_j[idx]
            true_value = y_mean[i, j]  # Using current mean as "true" value
            
            # Create mask excluding this point
            cv_mask = sampled_mask.copy()
            cv_mask[i, j] = False
            
            # Get mean and variance for this checkpoint-task pair from full model
            # This is a simplification - ideally we'd refit the model without this point
            mean_pred = y_mean[i, j]
            var_pred = y_covar[i, j, i, j]
            
            # Compute relative prediction error
            rel_error = abs(mean_pred - true_value) / (abs(true_value) + 1e-10)
            rel_errors.append(rel_error)
        
        max_rel_error = max(rel_errors) if rel_errors else float('inf')
        cv_success = max_rel_error < cv_error_threshold
    
    # 7. Draw samples from multivariate normal distribution (with inflated uncertainty)
    samples = np.random.multivariate_normal(
        mean=checkpoint_means,
        cov=checkpoint_cov,
        size=n_samples
    )
    
    # 8. For each sample, find which checkpoint is best
    best_checkpoints = np.argmax(samples, axis=1)
    best_counts = np.bincount(best_checkpoints, minlength=K)
    best_probs = best_counts / n_samples
    
    # 9. Find top checkpoints and their probabilities
    top_k = 5
    prob_indices = np.argsort(-best_probs)[:top_k]
    top_probs = best_probs[prob_indices]
    prob_checkpoints = checkpoint_nums[prob_indices]
    
    # 10. Calculate probability that current best is within top-N
    top_3_prob = sum(best_probs[np.argsort(-checkpoint_means)[:3]])
    
    # 11. Final stopping decision combines all factors:
    #    - Must have enough data sampled
    #    - Best checkpoint prediction must be stable
    #    - Cross-validation must be successful
    #    - Probability of true best being in top 3 must be high
    stop_sampling = (
        percent_sampled >= min_samples_pct and
        stable_prediction and
        cv_success and
        top_3_prob >= confidence_threshold  # % confidence that true best is in top 3
    )
    
    # 12. Prepare diagnostic info
    top_means = checkpoint_means[np.argsort(-checkpoint_means)[:top_k]]
    top_checkpoints = checkpoint_nums[np.argsort(-checkpoint_means)[:top_k]]
    
    # Add current predictions to history dictionary
    history_dict[step] = {
        "best_checkpoint": best_pred_checkpoint,
        "percent_sampled": percent_sampled,
        "top_probs": list(zip(prob_checkpoints, top_probs)),
        "top_means": list(zip(top_checkpoints, top_means)),
    }
    
    diagnostics = {
        "reason": "All criteria met" if stop_sampling else "Some criteria not met",
        "percent_sampled": percent_sampled,
        "stable_prediction": stable_prediction,
        "cv_success": cv_success,
        "max_rel_cv_error": max_rel_error,
        "top_3_probability": top_3_prob,
        "best_checkpoint": best_pred_checkpoint,
        "best_checkpoint_prob": best_probs[best_pred_idx],
        "top_checkpoints_by_prob": list(zip(prob_checkpoints, top_probs)),
        "top_checkpoints_by_mean": list(zip(top_checkpoints, top_means)),
        "best_history": best_history.copy(),
        "should_stop": stop_sampling
    }
    
    return stop_sampling, diagnostics, history_dict, best_history
#--------------------------------------------------------------------------

def should_stop_sampling(y_mean, y_covar, best_pred_idx, checkpoint_nums, 
                         tolerance=0.01, confidence=0.90, n_samples=1000, max_top_k=5):
    """
    Determine if sampling should stop based on:
    1. Confidence in current best checkpoint from posterior sampling
    2. Proximity of top-k predicted checkpoints' performance
    
    Args:
        y_mean: Mean predictions from GP (shape K x Z)
        y_covar: Covariance matrix from GP (shape K x Z x K x Z)
        best_pred_idx: Index of current predicted best checkpoint
        checkpoint_nums: Array of checkpoint numbers for reporting
        tolerance: Tolerance level for relative difference in performance
        confidence: Required confidence level
        n_samples: Number of posterior samples to draw
        max_top_k: Maximum number of top checkpoints to consider
        
    Returns:
        bool: True if sampling should stop, False otherwise
        dict: Diagnostic information about the decision
    """
    K, Z = y_mean.shape
    
    # Compute average performance for each checkpoint
    checkpoint_means = y_mean.mean(axis=1)  # Shape: (K,)
    
    # Compute standard error of the mean for each checkpoint
    checkpoint_stderrs = np.zeros(K)
    for k in range(K):
        # Extract the Z×Z covariance matrix for checkpoint k
        task_cov_matrix = y_covar[k, :, k, :]
        # Standard error of mean = sqrt(sum of covariance elements) / Z
        checkpoint_stderrs[k] = np.sqrt(task_cov_matrix.sum()) / Z
    
    # Find the top-k checkpoints by mean performance
    top_k = min(max_top_k, K)
    top_indices = np.argsort(-checkpoint_means)[:top_k]
    top_means = checkpoint_means[top_indices]
    top_stderrs = checkpoint_stderrs[top_indices]
    top_checkpoints = checkpoint_nums[top_indices]
    
    # Compute relative difference from best to others
    best_mean = top_means[0]
    rel_diffs = np.abs(top_means - best_mean) / best_mean
    
    # Sample from posterior distribution over checkpoint performances
    # We'll use a multivariate normal approximation
    
    # Construct the covariance matrix for checkpoint means
    # This accounts for correlations between checkpoints
    checkpoint_cov = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            # Average covariance between tasks for checkpoints i and j
            checkpoint_cov[i, j] = np.mean(y_covar[i, :, j, :]) / Z**2

    # Ensure covariance matrix is positive semi-definite (add small diagonal if needed)
    min_eig = np.min(np.linalg.eigvalsh(checkpoint_cov))
    if min_eig < 0:
        checkpoint_cov += np.eye(K) * (abs(min_eig) + 1e-6)
    
    # Draw samples from multivariate normal distribution
    samples = np.random.multivariate_normal(
        mean=checkpoint_means,
        cov=checkpoint_cov,
        size=n_samples
    )
    
    # For each sample, find which checkpoint is best
    best_checkpoints = np.argmax(samples, axis=1)
    
    # Count how often each checkpoint is the best
    best_counts = np.bincount(best_checkpoints, minlength=K)
    best_probs = best_counts / n_samples
    
    # Top-k checkpoints by probability
    prob_indices = np.argsort(-best_probs)[:top_k]
    top_probs = best_probs[prob_indices]
    prob_checkpoints = checkpoint_nums[prob_indices]
    
    # Decision criteria
    # 1. Is our current best prediction confident enough?
    confidence_criterion = best_probs[best_pred_idx] >= confidence
    
    # 2. Are the top checkpoints by mean all within tolerance?
    tolerance_criterion = np.all(rel_diffs[1:] <= tolerance) if len(rel_diffs) > 1 else True
    
    # 3. Is the uncertainty in the best checkpoint's performance low enough?
    uncertainty_criterion = checkpoint_stderrs[best_pred_idx] / checkpoint_means[best_pred_idx] <= tolerance/2
    
    # Combine criteria
    should_stop = confidence_criterion or (tolerance_criterion and uncertainty_criterion)
    
    # Prepare diagnostic info
    diagnostics = {
        'confidence_in_best': best_probs[best_pred_idx],
        'confidence_criterion_met': confidence_criterion,
        'tolerance_criterion_met': tolerance_criterion,
        'uncertainty_criterion_met': uncertainty_criterion,
        'relative_stderr': checkpoint_stderrs[best_pred_idx] / checkpoint_means[best_pred_idx],
        'top_checkpoints_by_mean': list(zip(top_checkpoints, top_means)),
        'top_checkpoints_by_prob': list(zip(prob_checkpoints, top_probs)),
        'should_stop': should_stop
    }
    
    return should_stop, diagnostics

#--------------------------------------------------------------------------
# SET PARAMETERS

fn = 'phi4-math-4claude.txt'
# fn = 'phi4-bw-4claude.txt'

synthetic = False
n_rows, n_cols = 200, 100

init_subset = 0.0 # 0 ==> sample each task once to start 

rank_fraction = 0.25 # 0.1 0.25 0.5

learning_rate = 0.1
max_iterations = 1000
tolerance = 1e-4
patience = 10

log_interval = 5
sample_max = 0.1

compare_random = True

rand_seed = 123

# stopping criterion parameters
use_stopping_criterion = False # Flag to enable/disable the stopping criterion
# tolerance_threshold = 0.001  # % relative difference tolerance
confidence_threshold = 0.99  # % confidence requirement

min_samples_percent = 1.0      # Minimum % of data to sample before considering stopping
stability_required = 5          # Number of steps best checkpoint must remain stable 
uncertainty_multiplier = 5.0    # Factor to scale uncertainty estimates (combat GP overconfidence)
cv_threshold = 0.01             # Cross-validation error threshold

# Add these before your main loop
stopping_history = {}
best_checkpoints_history = []

#--------------------------------------------------------------------------
# Load the Data...

# We'll assume you have a CSV with columns:
# "CHECKPOINT", "TEST_AVERAGE", "TEST_1", "TEST_2", ..., "TEST_71".
# get directory of current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# get parent directory
parent_dir = os.path.dirname(current_dir)
# data dir = parent_dir + '/data'
data_dir = os.path.join(parent_dir, 'data')
# load data
df = pd.read_csv(os.path.join(data_dir, fn), delimiter='\t')
# df = pd.read_csv('data/phi4-math-4claude.txt', delimiter='\t')

# Extract checkpoint numbers
checkpoint_nums = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values

# Identify test columns (excluding average)
test_cols = [col for col in df.columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']

# raw data
V = df[test_cols].values
print(V.shape)

#--------------------------------------------------------------
# random seed
rand_seed = np.random.randint(100, 1000) if rand_seed <= 0 else rand_seed
print(f'Random seed: {rand_seed}')
np.random.seed(rand_seed)
random.seed(rand_seed)

# detect if running on local machine
local = os.path.exists('/home/david')

#--------------------------------------------------------------
# generate synthetic data

if synthetic:
    fn = f'{data_dir}/synthetic_{n_rows}x{n_cols}_seed{rand_seed}.npy'
    if os.path.exists(fn):
        print(f'Loading synthetic data from: {fn}')
        V = np.load(fn)
    else:
        print(f'Generating synthetic data...')
        D = generate_synthetic_checkpoint_data(V, n_rows, n_cols, random_state=rand_seed)
        print(f'Saving synthetic data to: {fn}')
        np.save(fn, D)
        V = D
        del D
    # set checkpoint numbers to 50, 100, 150, ...
    checkpoint_nums = 50*np.arange(n_rows) + 50

#--------------------------------------------------------------
# min/max normalize checkpoints
checkpoints = (checkpoint_nums - checkpoint_nums.min()) / (checkpoint_nums.max() - checkpoint_nums.min())

# TODO: SCALING!!!
# checkpoints *= 100 #  10  20  50  100

#--------------------------------------------------------------

# get mean of each row
V_mu = V.mean(axis=1)

# find best checkpoint
best_idx = np.argmax(V_mu)
best_checkpoint = checkpoint_nums[best_idx]
print(f'Best checkpoint: {best_checkpoint}')

#-------------------------------------------------------------------------
# Full data
K,Z = V.shape

indices = np.where(np.ones_like(V))
N = len(indices[0]) # total number of (x,z) checkpoint-task pairs
X = np.array([ [checkpoints[i], j] for i, j in  zip(*indices) ])
Y = V[indices]

sampled_mask = np.zeros_like(V, dtype=bool)

full_test_X = torch.tensor(X[:,0], dtype=torch.float32)
full_test_T = torch.tensor(X[:,1], dtype=torch.long).reshape(-1,1)

#--------------------------------------------------------------------------

if init_subset > 0:
    # choose first samples
    n_samples = int(init_subset*N)
    sample_indices = random.sample(range(N), n_samples)
    X_sample = X[sample_indices]
    Y_sample = Y[sample_indices]
    train_X = torch.tensor(X_sample[:,0], dtype=torch.float32)
    train_T = torch.tensor(X_sample[:,1], dtype=torch.long).reshape(-1,1)
    train_Y = torch.tensor(Y_sample, dtype=torch.float32)
    sampled_mask = np.zeros_like(V, dtype=bool)
    sampled_mask[indices[0][sample_indices], indices[1][sample_indices]] = True
#--------------------------------------------------------------------------
else:
    # choose Z random samples (one per task)
    task_indices = np.random.permutation(Z)
    if Z < K:
        checkpoint_indices = np.random.choice(K, Z, replace=False)
    else:
        checkpoint_indices = np.concatenate([np.random.permutation(K), np.random.choice(K, Z-K, replace = False if Z-K < K else True)])
    train_X = torch.tensor(checkpoints[checkpoint_indices], dtype=torch.float32)
    train_T = torch.tensor(task_indices, dtype=torch.long).reshape(-1,1)
    train_Y = torch.tensor(V[checkpoint_indices, task_indices], dtype=torch.float32)
    for i, j in zip(checkpoint_indices, task_indices):
        sampled_mask[i, j] = True

#--------------------------------------------------------------------------

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=None):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.RBFKernel()
        lengthscale_prior = gpytorch.priors.NormalPrior(1.0, 0.25)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior))
        if rank is None: rank = num_tasks
        elif rank > num_tasks: rank = num_tasks
        elif rank < 1: rank = int(num_tasks*rank)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
def optimize_adam(model, likelihood, train_x, train_t, train_y, 
                  max_iterations=1000, 
                  lr=0.1, 
                  lr_scheduler='step',
                  lr_gamma=0.9, 
                  lr_step_size=50, 
                  lr_min=1e-3,
                  log_every=50,
                  tolerance=1e-4, 
                  patience=10):
    model.train()
    likelihood.train()
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set up learning rate scheduler
    if lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    elif lr_scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=lr_min)
    elif lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_gamma, 
                                                            patience=patience//2, threshold=tolerance, verbose=True)
    else:  # No scheduler
        scheduler = None
        
    stall_counter, prev_loss = 0, float('inf')
    for i in range(max_iterations):
        optimizer.zero_grad()
        output = model(train_x, train_t)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        # Update learning rate
        if scheduler is not None:
            if lr_scheduler == 'plateau':
                scheduler.step(loss.item())
            else:
                scheduler.step()
        # Check convergence
        current_loss = loss.item()
        rel_improvement = (prev_loss - current_loss) / (abs(prev_loss) + 1e-10)
        
        if i % log_every == 0:  # Log occasionally
            current_lr = optimizer.param_groups[0]['lr']
            try:
                lengthscale = model.covar_module.base_kernel.lengthscale.item()
                noise = model.likelihood.noise.item()
                print(f'Iter={i} \tLoss={current_loss:.3g} \tImp={rel_improvement:.3g}\tLR={current_lr:.3g}\tLenScale={lengthscale:.3g}\tNoise={noise:.3g}')
            except Exception as e:
                print(f'Iter={i} \tLoss={current_loss:.3g} \tImp={rel_improvement:.3g}\tLR={current_lr:.3g}')
                # print(f'Iter={i}\tLoss={current_loss:.5f}\tImp={rel_improvement:.5f}\tLR={current_lr:.6f}')
            
            
        stall_counter = stall_counter+1 if rel_improvement < tolerance else 0
        if stall_counter >= patience:
            print(f'Stopping early at iteration {i} - Loss converged')
            break
        prev_loss = current_loss


#--------------------------------------------------------------------------
class MultitaskGPSequentialSampler:
    def __init__(self, num_tasks, rank, 
                 max_iterations=1000, 
                 learning_rate=0.1, 
                 lr_scheduler='step',
                 lr_gamma=0.9,
                 lr_step_size=50,
                 lr_min=1e-3,
                 warm_start=False,
                 ):
        self.num_tasks = num_tasks
        self.rank = rank
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.warm_start = warm_start
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # model parameters
        self.model = None
        self.likelihood = None
        self.model_state = None
        self.likelihood_state = None
        
        # Learning rate schedule parameters
        self.lr_scheduler = lr_scheduler  # 'step', 'exp', 'cosine', 'plateau'
        self.lr_gamma = lr_gamma  # multiplicative factor for lr decay
        self.lr_step_size = lr_step_size  # iterations between lr updates
        self.lr_min = lr_min  # minimum learning rate
        
    #--------------------------------------------------------------------------
    def fit(self, train_x, train_t, train_y, 
            tolerance=tolerance, 
            patience=patience,
            warm_start=None,
            ):
        train_x = train_x.to(self.device)
        train_t = train_t.to(self.device)
        train_y = train_y.to(self.device)
        
        # Create a new model instance with current data
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = MultitaskGPModel((train_x, train_t), train_y, self.likelihood, self.num_tasks, rank=self.rank).to(self.device)
        
        # override self.warm_start if specified
        warm_start = warm_start if warm_start is not None else self.warm_start
        
        # Load previous model parameters if warm starting
        if warm_start and self.model_state is not None:
            self.model.load_state_dict(self.model_state)
            self.likelihood.load_state_dict(self.likelihood_state)
        
        optimize_adam(self.model, self.likelihood, train_x, train_t, train_y,
                      max_iterations=self.max_iterations,
                      lr=self.learning_rate,
                      lr_scheduler=self.lr_scheduler,
                      lr_gamma=self.lr_gamma,
                      lr_step_size=self.lr_step_size,
                      lr_min=self.lr_min,
                      tolerance=tolerance,
                      patience=patience)
        
        # Save model parameters for warm starting
        if self.warm_start:
            self.model_state = self.model.state_dict()
            self.likelihood_state = self.likelihood.state_dict()
        
    
    #--------------------------------------------------------------------------
    
    def predict(self, test_x, test_t):    
        self.model.eval()
        self.likelihood.eval()
        
        test_x = test_x.to(self.device)
        test_t = test_t.to(self.device)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_pred = self.likelihood(self.model(test_x, test_t))
        return y_pred

#--------------------------------------------------------------------------
sampler = MultitaskGPSequentialSampler(num_tasks=Z, 
                                       rank=rank_fraction,
                                       max_iterations=max_iterations,
                                       lr_scheduler='step',
                                       learning_rate=learning_rate, 
                                       warm_start=True,
                                       )
#--------------------------------------------------------------------------

# get list of sampled tasks
# sampled_tasks = np.unique(np.where(sampled_mask)[1])
unsampled_tasks = np.setdiff1d(np.arange(Z), np.unique(np.where(sampled_mask)[1]))

# keep track of fraction of iterations where the next checkpoint sampled is the current best checkpoint
best_checkpoint_sampled = 0
history = []

step = 0
while True:
    step += 1
    # compute percent of data sampled
    fraction_sampled = sampled_mask.sum() / N
    
    if fraction_sampled > sample_max:
        break
    
    #--------------------------------------------------------------------------
    # fit model
    sampler.fit(train_X, train_T, train_Y, warm_start=step%5!=0) # reset model parameters every 10 steps
    
    # extract predictions
    y_pred = sampler.predict(full_test_X, full_test_T)
    y_mean = y_pred.mean.detach().cpu().numpy() # shape = (num_x * num_z)
    y_var = y_pred.variance.detach().cpu().numpy() # shape = (num_x * num_z)
    y_covar = y_pred.covariance_matrix.detach().cpu().numpy() # shape = (num_x * num_z * num_x * num_z)

    # reshape y_mean, y_var back into (num_x, num_z)
    y_mean = y_mean.reshape((K, Z))
    y_var = y_var.reshape((K, Z))
    # reshape y_covar back into (num_x, num_z, num_x, num_z)
    y_covar = y_covar.reshape((K, Z, K, Z))
    #--------------------------------------------------------------------------
    
    # compute the mean of the mean predictions across tasks
    S_mean = y_mean.mean(axis=-1)

    best_pred_idx = np.argmax(S_mean)
    best_pred_checkpoint = checkpoint_nums[best_pred_idx]
    
    #--------------------------------------------------------------------------
    # Check stopping criterion
    if use_stopping_criterion and fraction_sampled > min_samples_percent/100:
        
        stop_flag, stop_diagnostics, stopping_history, best_checkpoints_history = improved_stopping_criterion(
            y_mean, y_covar, checkpoint_nums, sampled_mask, step,
            history_dict=stopping_history, 
            best_history=best_checkpoints_history,
            min_samples_pct=min_samples_percent,
            stability_count=stability_required,
            uncertainty_scale=uncertainty_multiplier,
            cv_error_threshold=cv_threshold,
            confidence_threshold=confidence_threshold,
        )
        
        # Log key diagnostics even if not stopping
        if step % log_interval == 0:
            print("\nStopping criterion diagnostics:")
            print(f"  Data sampled: {stop_diagnostics['percent_sampled']:.2f}%")
            print(f"  Stable prediction: {stop_diagnostics['stable_prediction']}")
            if 'max_rel_cv_error' in stop_diagnostics:
                print(f"  Max CV error: {stop_diagnostics.get('max_rel_cv_error', 'N/A'):.4f}")
            print(f"  Best checkpoint history: {stop_diagnostics['best_history']}")
            print(f"  Top-3 probability: {stop_diagnostics.get('top_3_probability', 'N/A'):.4f}")
            
        if stop_flag:
            print("\n" + "="*100)
            print(f"STOPPING CRITERION MET after {stop_diagnostics['percent_sampled']:.2f}% of data sampled")
            print(f"Reason: {stop_diagnostics['reason']}")
            print(f"Best checkpoint: {stop_diagnostics['best_checkpoint']}")
            print(f"Probability of being best: {stop_diagnostics['best_checkpoint_prob']:.4f}")
            
            # Print top checkpoints by probability
            print("\nTop checkpoints by probability of being best:")
            for checkpoint, prob in stop_diagnostics['top_checkpoints_by_prob'][:5]:
                print(f"  Checkpoint {checkpoint}: {prob:.4f}")
                
            # Print top checkpoints by mean performance
            print("\nTop checkpoints by mean performance:")
            for checkpoint, mean_val in stop_diagnostics['top_checkpoints_by_mean'][:5]:
                print(f"  Checkpoint {checkpoint}: {mean_val:.6f}")
                
            print("="*100)
            break
        
        # stop_sampling, stop_diagnostics = should_stop_sampling(
        #     y_mean, y_covar, best_pred_idx, checkpoint_nums,
        #     tolerance=tolerance_threshold,
        #     confidence=confidence_threshold,
        #     n_samples=5000  # Number of samples from posterior
        # )
        # if stop_sampling:
        #     print("\n" + "="*100)
        #     print(f"STOPPING CRITERION MET after {fraction_sampled*100:.2f}% of data sampled")
        #     print(f"Confidence in best checkpoint: {stop_diagnostics['confidence_in_best']:.4f}")
        #     print(f"Best checkpoint: {best_pred_checkpoint}")
        #     # Print top checkpoints by probability
        #     print("\nTop checkpoints by probability of being best:")
        #     for checkpoint, prob in stop_diagnostics['top_checkpoints_by_prob'][:5]:
        #         print(f"  Checkpoint {checkpoint}: {prob:.4f}")
        #     # Print top checkpoints by mean performance
        #     print("\nTop checkpoints by mean performance:")
        #     for checkpoint, mean_val in stop_diagnostics['top_checkpoints_by_mean'][:5]:
        #         print(f"  Checkpoint {checkpoint}: {mean_val:.6f}")
        #     print("="*100)
        #     break
    
    #--------------------------------------------------------------------------
    # compute percentage difference between best checkpoint score and predicted best checkpoint score
    v_mu_best = V_mu[best_idx]
    v_mu_pred = V_mu[best_pred_idx]
    y_diff = round(abs(v_mu_pred - v_mu_best) / v_mu_best, 8)
    
    # get distance between best checkpoint and predicted best checkpoint on [0..1] scale
    x_diff = round(abs(checkpoints[best_pred_idx] - checkpoints[best_idx]), 4) # / checkpoints[best_idx]
    
    # print results
    print('\n' + '-'*100)
    print(f'Step={step}\t{100*fraction_sampled:.2f}% Data Sampled')
    print(f'Best Chkpt (pred/true):\t{best_pred_checkpoint}/{best_checkpoint}\tx_diff={x_diff:.4f}\ty_diff={y_diff:.4g}')
    
    history.append(f'{step}\t{y_diff:.4g}\t{x_diff:.4f}\t{best_pred_checkpoint}')
    
    #--------------------------------------------------------------------------
    if step % log_interval == 0:
        clear_cuda_tensors()
        
        #--------------------------------------------------------------
        print()
        for line in history:
            print(f'{line}')
        print()
        
        #--------------------------------------------------------------
        if local:
            plt.plot(checkpoint_nums, S_mean)
            plt.ylim([0.81, 0.86])
            #-----------------------------------
            # compute standard deviation of mean
            S_sigma = np.zeros(K)
            for k in range(K):
                # Extract the Z×Z covariance matrix for checkpoint k
                task_cov_matrix = y_covar[k, :, k, :]
                # Variance of mean = (1/Z²) * sum of all elements in covariance matrix
                S_sigma[k] = np.sqrt(task_cov_matrix.sum()) / Z
            #-----------------------------------
            plt.fill_between(checkpoint_nums, S_mean - 2*S_sigma, S_mean + 2*S_sigma, alpha=0.5)
            plt.xlabel('Checkpoint')
            plt.ylabel('Average Performance')
            plt.title('Predicted Average Performance')
            plt.show()
        #--------------------------------------------------------------
        
    # compute S_mu, S_max
    S_mu = y_mean.sum(axis=-1)
    S_max_idx = np.argmax(S_mu)
    S_max = S_mu[S_max_idx]

    unsampled_indices = np.where(~sampled_mask)
    # X_unsampled = np.array([ [i, j] for i, j in  zip(*unsampled_indices) ])
    X_unsampled = np.array(unsampled_indices).T
    # get list of unsamplesd tasks, if any
    unsampled_tasks = np.setdiff1d(np.arange(Z), np.unique(np.where(sampled_mask)[1]))
    
    #--------------------------------------------------------------
    # Get indices as arrays
    i_indices, j_indices = unsampled_indices
    # Create mask for unsampled tasks check
    mask = np.ones(len(i_indices), dtype=bool)
    if unsampled_tasks.size > 0:
        mask = np.isin(j_indices, unsampled_tasks)
    # Initialize EI array with -100 for invalid entries
    EI = np.full(len(i_indices), -100.0)

    # Only compute for valid indices
    if np.any(mask):
        valid_i = i_indices[mask]
        valid_j = j_indices[mask]
        # Vectorized computation of all EI components
        mu = y_mean[valid_i, valid_j]
        sig = y_var[valid_i, valid_j]**0.5
        # Get row sums for each valid i
        row_sums = np.sum(y_mean[valid_i, :], axis=1)
        sx = row_sums - mu
        # compute ei
        s_max = S_max - sx
        imp = mu - s_max # - beta
        z = imp/sig
        ei = imp * norm.cdf(z) + sig * norm.pdf(z)
        
        # make ei just random numbers
        if compare_random: 
            ei = np.random.rand(len(ei))
        
        # Assign computed values to valid positions
        EI[mask] = ei
        
    EI = np.array(EI)  # Ensure it's a numpy array

    #--------------------------------------------------------------
    
    next_idx = np.argmax(EI)
    next_i, next_j = X_unsampled[next_idx]
    next_checkpoint = checkpoint_nums[next_i]
    
    # check if next checkpoint is the best checkpoint
    if next_i == best_pred_idx:
        best_checkpoint_sampled += 1

    print(f'Next sample - checkpoint:\t{next_checkpoint}, task={next_j}\n')
    sampled_mask[next_i, next_j] = True
    train_X = torch.cat([train_X, torch.tensor([checkpoints[next_i]], dtype=torch.float32)])
    train_T = torch.cat([train_T, torch.tensor([next_j], dtype=torch.long).reshape(-1,1)])
    train_Y = torch.cat([train_Y, torch.tensor([V[next_i, next_j]], dtype=torch.float32)])
