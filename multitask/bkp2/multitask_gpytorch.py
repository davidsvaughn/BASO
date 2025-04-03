import sys, os
import numpy as np
import pandas as pd
import random
import math
from scipy.stats import norm
from matplotlib import pyplot as plt
import gc
import torch
import gpytorch

from utils import clear_cuda_tensors, log_h
from multitask.bkp.multitask_synthetic import generate_synthetic_checkpoint_data

from surprise import initialize_surprise_tracker, surprise_based_stopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
rand_seed = -1

#--------------------------------------------------------------------------
# SET PARAMETERS

fn = 'phi4-math-4claude.txt'
# fn = 'phi4-bw-4claude.txt'

compare_random = False

synthetic = False
n_rows, n_cols = 100, 100

# rand_seed = 333

init_subset = 0.0 # 0 ==> sample each task once to start 

rank_fraction = 0.1 # 0.1 0.25 0.5

learning_rate = 0.1
max_iterations = 1000
tolerance = 1e-4
patience = 5

sample_max = 0.075 # TODO : or max_num_points_sampled

log_interval = 5
warm_start_interval = -1
use_logei = True

history_win = 20

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

# create history filename
history_fn = f'{data_dir}/history.txt'
counter = 0
while os.path.exists(history_fn):
    counter += 1
    history_fn = f'{data_dir}/history_{counter}.txt'
print(f'History File: {history_fn}')

#-------------------------------------------------------------------------
# generate synthetic data (?)

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

#-------------------------------------------------------------------------
# Full data
K,Z = V.shape
sampled_mask = np.zeros_like(V, dtype=bool)
indices = np.where(np.ones_like(V))

# min/max normalize checkpoints
checkpoints = (checkpoint_nums - checkpoint_nums.min()) / (checkpoint_nums.max() - checkpoint_nums.min())

N = len(indices[0]) # total number of (x,z) checkpoint-task pairs (N = K*Z)
X = np.array([ [checkpoints[i], j] for i, j in  zip(*indices) ])
Y = V[indices]

# Full test data grid (all-checkpoints X all-tasks)
full_test_X = torch.tensor(X[:,0], dtype=torch.float32).to(device)
full_test_T = torch.tensor(X[:,1], dtype=torch.long).reshape(-1,1).to(device)

# get mean at each checkpoint
V_mu = V.mean(axis=1)

# find best checkpoint
best_idx = np.argmax(V_mu)
best_checkpoint = checkpoint_nums[best_idx]
print(f'True Best checkpoint: {best_checkpoint}')

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
        lengthscale_prior = gpytorch.priors.NormalPrior(1.0, 0.5)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior))
        
        if rank is None: rank = num_tasks
        elif rank > num_tasks: rank = num_tasks
        elif rank < 1: rank = max(int(num_tasks*rank), 5)
        
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
                  lr_gamma=0.95, 
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
        
        #-------------------------------------------------------------------------
        # Update learning rate
        if scheduler is not None:
            if lr_scheduler == 'plateau':
                scheduler.step(loss.item())
            else:
                scheduler.step()
            # Enforce minimum learning rate for all scheduler types
            if lr_scheduler in ['step', 'exp']:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'], lr_min)
        #-------------------------------------------------------------------------
             
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
                 lr_gamma=0.95,
                 lr_step_size=50,
                 lr_min=1e-2,
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
        warm_start = (warm_start and self.warm_start) if warm_start is not None else self.warm_start
        
        # Load previous model parameters if warm starting
        if warm_start and self.model_state is not None:
            self.model.load_state_dict(self.model_state)
            self.likelihood.load_state_dict(self.likelihood_state)
            patience = patience // 2  # reduce patience for warm starting
            # tolerance = tolerance / 10  # reduce tolerance for warm starting
        
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
        
        # check if test data is on the same device as the model
        if test_x.device != self.device:
            test_x = test_x.to(self.device)
        if test_t.device != self.device:
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
                                       warm_start=warm_start_interval>0,
                                       )
#--------------------------------------------------------------------------

# get list of sampled tasks
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
    sampler.fit(train_X, train_T, train_Y, warm_start=step%warm_start_interval!=0) # reset model parameters every 10 steps
    
    # extract predictions
    y_pred = sampler.predict(full_test_X, full_test_T)
    y_mean = y_pred.mean.detach().cpu().numpy() # shape = (num_x * num_z)
    y_var = y_pred.variance.detach().cpu().numpy() # shape = (num_x * num_z)
    y_covar = y_pred.covariance_matrix.detach().cpu().numpy() # shape = (num_x * num_z * num_x * num_z)

    # reshape y_mean, y_var back into (num_x, num_z)
    y_mean = y_mean.reshape((K, Z))
    y_var = y_var.reshape((K, Z))
    y_covar = y_covar.reshape((K, Z, K, Z)) # reshape y_covar back into (num_x, num_z, num_x, num_z)
    
    #--------------------------------------------------------------------------
    
    # compute the mean of the mean predictions across tasks
    S_mean = y_mean.mean(axis=-1)
    best_pred_idx = np.argmax(S_mean)
    best_pred_checkpoint = checkpoint_nums[best_pred_idx]
    
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
    
    #--------------------------------------------------------------------------
    # history.append([step, best_pred_checkpoint, x_diff, y_diff])
    
    # if len(history) > history_win:
    #     history_x = np.array(history).astype(np.float32)
    #     yd = history_x[:,-1]
    #     yd_ma = np.zeros_like(yd)
    #     for i in range(len(yd)):
    #         start = max(0, i - history_win + 1)
    #         yd_ma[i] = np.mean(yd[start:i+1])
    #     history_x = np.hstack([history_x, yd_ma.reshape(-1,1)])
    #     np.savetxt(history_fn, history_x, fmt='%d\t%d\t%.4g\t%.4g\t%.4g')
    
    if K*Z > 10000: clear_cuda_tensors()
    
    #--------------------------------------------------------------------------
    if step % log_interval == 0:
        
        #--------------------------------------------------------------
        if K*Z <= 10000: clear_cuda_tensors()
        
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
    
    #--------------------------------------------------------------
    # Compute Expected Improvement
    
    # compute S_mu, S_max
    S_mu = y_mean.sum(axis=-1)
    S_max_idx = np.argmax(S_mu)
    S_max = S_mu[S_max_idx]

    unsampled_indices = np.where(~sampled_mask)
    X_unsampled = np.array(unsampled_indices).T
    
    # get list of unsampled tasks, if any
    unsampled_tasks = np.setdiff1d(np.arange(Z), np.unique(np.where(sampled_mask)[1]))
    
    # Get indices as arrays
    i_indices, j_indices = unsampled_indices
    
    # Create mask for unsampled tasks check
    mask = np.ones(len(i_indices), dtype=bool)
    if unsampled_tasks.size > 0:
        mask = np.isin(j_indices, unsampled_tasks)
        
    # Initialize EI array with -inf for invalid entries
    EI = np.full(len(i_indices), -math.inf)
    
    # make EI vector just random numbers instead
    if compare_random: 
        ei = np.random.rand(mask.shape[0])
    
    #----------------------------------------------------------------
    else:
        # EI computation
        valid_i, valid_j = i_indices[mask], j_indices[mask]
        
        # Vectorized computation of all EI components
        mu = y_mean[valid_i, valid_j]
        sig = y_var[valid_i, valid_j]**0.5
        
        # Get row sums for each valid i
        row_sums = np.sum(y_mean[valid_i, :], axis=1)
        sx = row_sums - mu
        
        # improvement vector, z-scores
        s_max = S_max - sx
        imp = mu - s_max # - beta
        z = imp/sig

        if use_logei:
            # logEI computation (for stability)
            logh = log_h(torch.tensor(z, dtype=torch.float64)).numpy()
            ei = np.log(sig) + logh
            ei_values = np.exp(ei)
        else:
            # EI computation
            ei = imp * norm.cdf(z) + sig * norm.pdf(z)
            ei_values = ei
        
    #----------------------------------------------------------------
    
    # Assign computed values to valid positions
    EI[mask] = ei
    EI = np.array(EI)
    
    next_idx = np.argmax(EI)
    next_i, next_j = X_unsampled[next_idx]
    next_checkpoint = checkpoint_nums[next_i]
    max_ei = np.max(ei_values)
    
    # check if next checkpoint is the best checkpoint
    if next_i == best_pred_idx:
        best_checkpoint_sampled += 1

    print(f'Next sample - checkpoint:\t{next_checkpoint}, task={next_j}\n')
    
    sampled_mask[next_i, next_j] = True
    train_X = torch.cat([train_X, torch.tensor([checkpoints[next_i]], dtype=torch.float32)])
    train_T = torch.cat([train_T, torch.tensor([next_j], dtype=torch.long).reshape(-1,1)])
    train_Y = torch.cat([train_Y, torch.tensor([V[next_i, next_j]], dtype=torch.float32)])
    
    #----------------------------------------------------------------
    
    # TODO
    max_ei = np.max(logh)
    
    history.append([step, best_pred_checkpoint, max_ei, y_diff])
    
    if len(history) > history_win:
        history_x = np.array(history).astype(np.float32)
        yd = history_x[:,-1]
        yd_ma = np.zeros_like(yd)
        for i in range(len(yd)):
            start = max(0, i - history_win + 1)
            yd_ma[i] = np.mean(yd[start:i+1])
        history_x = np.hstack([history_x, yd_ma.reshape(-1,1)])
        np.savetxt(history_fn, history_x, fmt='%d\t%d\t%.4g\t%.4g\t%.4g')
