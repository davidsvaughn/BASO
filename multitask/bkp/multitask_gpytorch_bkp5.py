import sys, os
import gpytorch.constraints
import numpy as np
import pandas as pd
import random
import time
from scipy.stats import norm
from matplotlib import pyplot as plt
import gc
import torch
import gpytorch
# from botorch import fit_gpytorch_model

#--------------------------------------------------------------

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

# ---------------------------------------------------------------------
fn = 'phi4-math-4claude.txt'
# fn = 'phi4-bw-4claude.txt'

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

#--------------------------------------------------------------
# min/max normalize checkpoints
checkpoints = (checkpoint_nums - checkpoint_nums.min()) / (checkpoint_nums.max() - checkpoint_nums.min())

# TODO: SCALING!!!
checkpoints *= 20 #  10  20  50  100
#--------------------------------------------------------------

V = df[test_cols].values
print(V.shape)
V_mu = V.mean(axis=1)

# find best checkpoint
best_idx = np.argmax(V_mu)
best_checkpoint = checkpoint_nums[best_idx]
print(f'Best checkpoint: {best_checkpoint}')

#-------------------------------------------------------------------------
# Full data
indices = np.where(np.ones_like(V))
N = len(indices[0]) # total number of (x,z) checkpoint-task pairs
X = np.array([ [checkpoints[i], j] for i, j in  zip(*indices) ])
Y = V[indices]
Z = len(test_cols) # number of tasks/validation sets
K = len(checkpoints) # number of checkpoints

sampled_mask = np.zeros_like(V, dtype=bool)

full_test_X = torch.tensor(X[:,0], dtype=torch.float32)
full_test_T = torch.tensor(X[:,1], dtype=torch.long).reshape(-1,1)

#--------------------------------------------------------------------------
# sample subset of data

rand_seed = np.random.randint(100, 1000)
# rand_seed = 737 # 260 737 ###     605 286 111
print(f'Random seed: {rand_seed}')
np.random.seed(rand_seed)
random.seed(rand_seed)

#--------------------------------------------------------------------------
# set parameters

init_subset = 0.05

rank_fraction = 0.25 # 0.1 0.25 0.5
# rank_fraction = 0.1

learning_rate = 0.1
max_iterations = 1000
tolerance = 1e-4
patience = 10

beta = 0.0

log_interval = 5
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
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        # https://docs.gpytorch.ai/en/v1.12/examples/00_Basic_Usage/Hyperparameters.html#Priors
        # lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior,))
        #--------------------------------------------------------------------------
        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        if rank is None: rank = num_tasks
        elif rank > num_tasks: rank = num_tasks
        elif rank < 1: rank = int(num_tasks*rank)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)
        # self.task_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank))

    def forward(self, x, i):
        # x, i = inputs
        mean_x = self.mean_module(x)
        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

#--------------------------------------------------------------------------
class MultitaskGPSequentialSampler:
    def __init__(self, num_tasks, rank, 
                 max_iterations=100, 
                 learning_rate=0.1, 
                 beta=0.05,
                 lr_scheduler='step',
                 lr_gamma=0.9,
                 lr_step_size=50,
                 lr_min=1e-3,
                 ):
        self.num_tasks = num_tasks
        self.rank = rank
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.beta = beta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learning rate schedule parameters
        self.lr_scheduler = lr_scheduler  # 'step', 'exp', 'cosine', 'plateau'
        self.lr_gamma = lr_gamma  # multiplicative factor for lr decay
        self.lr_step_size = lr_step_size  # iterations between lr updates
        self.lr_min = lr_min  # minimum learning rate
        
    #--------------------------------------------------------------------------
    def fit(self, train_x, train_t, train_y, tolerance=tolerance, patience=patience):
        train_x = train_x.to(self.device)
        train_t = train_t.to(self.device)
        train_y = train_y.to(self.device)
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = MultitaskGPModel((train_x, train_t), train_y, self.likelihood, self.num_tasks, rank=self.rank).to(self.device)
        self.likelihood.train()
        self.model.train()
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Set up learning rate scheduler
        if self.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        elif self.lr_scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)
        elif self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iterations, eta_min=self.lr_min)
        elif self.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.lr_gamma, 
                                                                patience=patience//2, threshold=tolerance, verbose=True)
        else:  # No scheduler
            scheduler = None
        
        # For tracking convergence
        prev_loss = float('inf')
        stall_counter = 0
        
        for i in range(self.max_iterations):
            optimizer.zero_grad()
            output = self.model(train_x, train_t)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                if self.lr_scheduler == 'plateau':
                    scheduler.step(loss.item())
                else:
                    scheduler.step()
            
            # Check convergence
            current_loss = loss.item()
            rel_improvement = (prev_loss - current_loss) / (abs(prev_loss) + 1e-10)
            
            if i % 50 == 0:  # Log occasionally
                current_lr = optimizer.param_groups[0]['lr']
                # print(f'Iter {i} - Loss: {current_loss:.5f}, Improvement: {rel_improvement:.5f}')
                print(f'Iter {i} - Loss: {current_loss:.5f}, Improvement: {rel_improvement:.5f}, LR: {current_lr:.6f}')
                
            if rel_improvement < tolerance:
                stall_counter += 1
            else:
                stall_counter = 0
                
            if stall_counter >= patience:
                print(f'Stopping early at iteration {i} - Loss converged')
                break
                
            prev_loss = current_loss
    
    #--------------------------------------------------------------------------
    
    def predict(self, test_x, test_t):    
        self.model.eval()
        self.likelihood.eval()
        
        test_x = test_x.to(self.device)
        test_t = test_t.to(self.device)
        
        with torch.no_grad():#, gpytorch.settings.fast_pred_var():
            y_pred = self.likelihood(self.model(test_x, test_t))
        return y_pred

#--------------------------------------------------------------------------
sampler = MultitaskGPSequentialSampler(num_tasks=Z, 
                                       rank=rank_fraction,
                                       max_iterations=max_iterations, 
                                       learning_rate=learning_rate, beta=beta)
#--------------------------------------------------------------------------

# get list of sampled tasks
# sampled_tasks = np.unique(np.where(sampled_mask)[1])
unsampled_tasks = np.setdiff1d(np.arange(Z), np.unique(np.where(sampled_mask)[1]))

# keep track of fraction of iterations where the next checkpoint sampled is the current best checkpoint
best_checkpoint_sampled = 0

step = 0
while True:
    step += 1
    # compute percent of data sampled
    fraction_sampled = sampled_mask.sum() / N
    
    if fraction_sampled >= 0.2:
        break

    sampler.fit(train_X, train_T, train_Y)

    y_pred = sampler.predict(full_test_X, full_test_T)
    y_mean = y_pred.mean.detach().cpu().numpy() # shape = (num_x * num_z)
    y_var = y_pred.variance.detach().cpu().numpy() # shape = (num_x * num_z)
    y_covar = y_pred.covariance_matrix.detach().cpu().numpy() # shape = (num_x * num_z * num_x * num_z)

    # reshape y_mean, y_var back into (num_x, num_z)
    y_mean = y_mean.reshape((K, Z))
    y_var = y_var.reshape((K, Z))
    
    # reshape y_covar back into (num_x, num_z, num_x, num_z)
    y_covar = y_covar.reshape((K, Z, K, Z))
    
    # compute the mean of the mean predictions across tasks
    S_mean = y_mean.mean(axis=-1)

    best_pred_idx = np.argmax(S_mean)
    best_pred_checkpoint = checkpoint_nums[best_pred_idx]

    # print(f'\nBest checkpoint: {best_checkpoint}')
    print(f'\nStep {step}: % sampled={100*fraction_sampled:.2f}%')
    print(f'\tbest predicted checkpoint:\t{best_pred_checkpoint} (best checkpoint: {best_checkpoint})')
    
    if step % log_interval == 0:
        clear_cuda_tensors()
        plt.plot(checkpoints, S_mean)
        # set y axis bounds
        plt.ylim([0.81, 0.86])
        
        #--------------------------------------------------------------
        S_sigma = np.zeros(K)
        for k in range(K):
            # Extract the Z×Z covariance matrix for checkpoint k
            task_cov_matrix = y_covar[k, :, k, :]
            # Variance of mean = (1/Z²) * sum of all elements in covariance matrix
            S_sigma[k] = np.sqrt(task_cov_matrix.sum()) / Z
        #--------------------------------------------------------------
        
        plt.fill_between(checkpoints, S_mean - 2*S_sigma, S_mean + 2*S_sigma, alpha=0.5)
        plt.xlabel('Checkpoint')
        plt.ylabel('Average Performance')
        plt.title('Predicted Average Performance')
        plt.show()
        
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
    # start = time.time()
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
        
        s_max = S_max - sx
        imp = mu - s_max - beta
        z = imp/sig
        ei = imp * norm.cdf(z) + sig * norm.pdf(z)
        
        # Assign computed values to valid positions
        EI[mask] = ei

    EI = np.array(EI)  # Ensure it's a numpy array
    # end = time.time()
    # print(f'EI computation time: {end-start:.2g} seconds')
    
    # compare to _EI
    # assert np.allclose(EI, _EI, atol=1e-5), f'EI mismatch at step {step}'
    
    #--------------------------------------------------------------
    
    next_idx = np.argmax(EI)
    next_i, next_j = X_unsampled[next_idx]
    next_checkpoint = checkpoint_nums[next_i]
    
    # check if next checkpoint is the best checkpoint
    if next_i == best_pred_idx:
        best_checkpoint_sampled += 1

    print(f'\tnext sample - checkpoint:\t{next_checkpoint}, task={next_j}')
    sampled_mask[next_i, next_j] = True
    train_X = torch.cat([train_X, torch.tensor([checkpoints[next_i]], dtype=torch.float32)])
    train_T = torch.cat([train_T, torch.tensor([next_j], dtype=torch.long).reshape(-1,1)])
    train_Y = torch.cat([train_Y, torch.tensor([V[next_i, next_j]], dtype=torch.float32)])
