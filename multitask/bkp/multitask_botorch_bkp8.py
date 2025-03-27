import sys, os
import gpytorch.constraints
import numpy as np
import pandas as pd
import random
from scipy.stats import norm
from matplotlib import pyplot as plt
import torch
import gpytorch
import math

from botorch.models import MultiTaskGP

from utils import task_standardize, inv_task_standardize, to_numpy, degree_metric, log_h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
rand_seed = -1
rank_frac = -1

#--------------------------------------------------------------------------
# SET PARAMETERS

fn = 'phi4-math-4claude.txt'
# fn = 'phi4-bw-4claude.txt'

# rand_seed = 777

task_sample = 1 # select random subset of tasks
init_tpc    = 0.05 # number of random tasks per checkpoint (tpc) to initially sample
             # -> if fraction, tpc++ tpc until that fraction of tasks are sampled
max_sample = 0.25 # stop BO sampling after this fraction of points are sampled

# MLE estimation
max_iterations = 1000
learning_rate = 0.1
rank_frac = 0.4

# logging intervals
log_loop = 5
log_fit = 50

use_logei = True

# compare_random = False
# synthetic = False
# n_rows, n_cols = 100, 100

#-------------------------------------------------------------------------
# random seed
rand_seed = np.random.randint(100, 1000) if rand_seed <= 0 else rand_seed
print(f'Random seed: {rand_seed}')
np.random.seed(rand_seed)
random.seed(rand_seed)

# detect if running on local machine
local = os.path.exists('/home/david')

#--------------------------------------------------------------------------
# Load the Data...

# We'll assume you have a CSV with columns:
# "CHECKPOINT", "TEST_AVERAGE", "TEST_1", "TEST_2", ..., "TEST_71".
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')

# load data
df = pd.read_csv(os.path.join(data_dir, fn), delimiter='\t')

# Extract checkpoint numbers
checkpoint_nums = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values
 
# Identify test columns (excluding average)
test_cols = [col for col in df.columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']
V = df[test_cols].values
del test_cols
K,Z = V.shape

# sample subset of tasks (possibly)
if task_sample>0 and task_sample!=1:
    if task_sample < 1:
        task_sample = int(task_sample * Z)
    idx = np.random.choice(range(Z), task_sample, replace=False)
    V = V[:, idx]
    K,Z = V.shape

#-------------------------------------------------------------------------

def init_samples(K, Z, init_tpc):
    if init_tpc >= 1:
        m = init_tpc * max(K,Z)
    else:
        m = int(max(init_tpc * K * Z , max(K,Z)))
    
    tasks = list(range(Z))
    checkpoints = list(range(K))
    chk_tasks = [[] for _ in range(K)]
    n = 0
    while True:
        # select a random checkpoint
        k = random.choice(checkpoints)
        try:
            # select random task not already selected for this checkpoint
            t = random.choice([tt for tt in tasks if tt not in chk_tasks[k]])
        except:
            continue # no task satisfies above condition... retry
        chk_tasks[k].append(t)
        n += 1
        if n >= m:
            break
        tasks.remove(t)
        checkpoints.remove(k)
        if len(tasks) == 0:
            tasks = list(range(Z)) # reset task list
        if len(checkpoints) == 0:
            checkpoints = list(range(K))
    random.shuffle(chk_tasks)
    
    # convert to x,y indices
    x,y = [],[]
    for i, tasks in enumerate(chk_tasks):
        for j in tasks:
            x.append(i)
            y.append(j)
    return np.array(x), np.array(y)
            
#--------------------------------------------------------------------------     
# init dataset constructs
sampled_mask = np.zeros_like(V, dtype=bool)
full_indices = np.where(np.ones_like(V))

# min/max normalize checkpoints to unit interval (cube)
checkpoints = (checkpoint_nums - checkpoint_nums.min()) / (checkpoint_nums.max() - checkpoint_nums.min())

# find best checkpoint
best_idx = np.argmax(V.mean(axis=1))
best_checkpoint = checkpoint_nums[best_idx]
print(f'TRUE BEST CHECKPOINT: {best_checkpoint}')

# subsample data
x_idx, t_idx = init_samples(K, Z, init_tpc)

train_X = torch.tensor([ [checkpoints[i], j] for i, j in  zip(x_idx, t_idx) ], dtype=torch.float64)
train_Y = torch.tensor(V[x_idx, t_idx], dtype=torch.float64).unsqueeze(-1)

for i, j in zip(x_idx, t_idx):
    sampled_mask[i, j] = True

#--------------------------------------------------------------------------
# standardize V, train_Y

# get FULL DATASET stats from V instead --> CHEATING!!!!!!
# task_means, task_stds = V.mean(axis=0), V.std(axis=0)

# train_Y, (t_mu, t_sig) = task_standardize(train_Y, train_X)
    
# standardize V
# V = (V - t_mu) / t_sig
# v_mu = V.mean(axis=1)

#--------------------------------------------------------------------------

# build useful tensors and arrays
full_X = torch.tensor([ [checkpoints[i], j] for i, j in  zip(*full_indices) ], dtype=torch.float64)
test_X = np.array(checkpoints)
test_Y = np.array(V)

#--------------------------------------------------------------------------

class BotorchSampler:
    def __init__(self, full_x, sampled_mask,
                 lr=0.1, 
                 max_iterations=1000,
                 max_sample=0.25, 
                 rank_frac=0.5,
                 log_interval=50):   
        self.sampled_mask = sampled_mask
        self.lr = lr
        self.max_iterations = max_iterations
        self.max_sample = max_sample
        self.rank_frac = rank_frac
        self.log_interval = log_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.z = int(full_x[:,1].max().item() + 1)
        self.k, self.z = sampled_mask.shape
        self.full_x = full_x.to(self.device)
        
    def fit(self, train_x, train_y):
        k,z = self.k, self.z
        
        # standardize train_y
        train_y, (t_mu, t_sig) = task_standardize(train_y, train_x)
        self.t_mu, self.t_sig = t_mu, t_sig
        
        # compute rank
        rank = int(self.rank_frac * z) if self.rank_frac > 0 else None
        self.model = MultiTaskGP(train_x, train_y, task_feature=-1, rank=rank).to(self.device)
        train_x, train_y = train_x.to(self.device), train_y.to(self.device)
        
        # Set the model and likelihood to training mode
        self.model.train()
        self.model.likelihood.train() # need this ???
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # degree stopping criterion
        degree_history = []
        window_size = 3  # Size of moving average window
        rise_patience = 3  # Number of consecutive rises to trigger stopping
        consecutive_rises = 0
        degree_check_interval = 5  # Check degree every n iterations
        min_iterations = 50  # Minimum iterations before allowing early stopping
        
        # train loop
        for i in range(self.max_iterations):
            optimizer.zero_grad()
            output = self.model(train_x)
            
            try:
                loss = -mll(output, self.model.train_targets)
            except Exception as e:
                print(f'Early stopping at iteration {i+1}: ERROR in loss calculation...')
                print(f'ERROR:{i}: {e}')
                break
            
            loss.backward()
            
            if i % degree_check_interval == 0:

                degree = degree_metric(self.model, self.full_x)
                degree_history.append(degree)
                
                # Only check for early stopping after collecting enough data points
                if len(degree_history) >= window_size + 1 and i >= min_iterations:
                    # Calculate current and previous moving averages
                    current_avg = sum(degree_history[-window_size:]) / window_size
                    prev_avg = sum(degree_history[-(window_size+1):-1]) / window_size
                    
                    # Check if the moving average is rising
                    if current_avg > prev_avg * 1.005:  # Small threshold to avoid stopping due to tiny fluctuations
                        consecutive_rises += 1
                        if consecutive_rises >= rise_patience:
                            print(f'Early stopping at iteration {i+1}: degree metric rising for {consecutive_rises} consecutive checks')
                            break
                    else:
                        consecutive_rises = 0
                
            if i % self.log_interval == 0:
                print(f'Iter {i}/{self.max_iterations} - Loss: {loss.item():.3f}, avg_degree: {degree:.3f}')

            optimizer.step()
            
    def predict(self, x=None):
        if x is None:
            x = self.full_x
        self.model.eval()
        self.model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(x))
            
        y_mean = predictions.mean
        lower, upper = predictions.confidence_region()
        y_var = predictions.variance
        y_covar = predictions.covariance_matrix
        
        k,z = self.k, self.z
            
        y_pred = to_numpy(y_mean.reshape(k, z))
        lower = to_numpy(lower.reshape(k, z))
        upper = to_numpy(upper.reshape(k, z))
        y_var = to_numpy(y_var.reshape(k, z))
        y_covar = to_numpy(y_covar.reshape((k, z, k, z)))
        
        #---------------------------------------------------------------------
        # inverse standardize y_pred, lower, upper, y_var, y_covar
        y_pred = y_pred * self.t_sig + self.t_mu
        lower = lower * self.t_sig + self.t_mu
        upper = upper * self.t_sig + self.t_mu
        
        y_var = y_var * self.t_sig**2
        y_sig = np.sqrt(y_var)
        y_mean = y_pred.mean(axis=1)
        y_covar = y_covar * self.t_sig**2
        y_sigma = np.array([np.sqrt(y_covar[i,:,i].sum()) for i in range(k)]) / z
        
        #---------------------------------------------------------------------
        return y_pred, y_sig, lower, upper, y_mean, y_sigma

#--------------------------------------------------------------------------

def plot_all(train_X, train_Y, test_X, test_Y, y_pred, y_sig, lower, upper, y_mean, y_sigma, max_fig=None):
    K,Z = test_Y.shape
    
    test_Y_mu = test_Y.mean(axis=1)
    
    plt.figure(figsize=(15, 10))
    plt.plot(y_mean, 'b')
    plt.fill_between(range(K), y_mean - 2*y_sigma, y_mean + 2*y_sigma, alpha=0.5)
    plt.plot(test_Y_mu, 'r')
    plt.show()
    
    for i in range(Z):
        if max_fig is not None and i > max_fig:
            break
        
        plt.figure(figsize=(15, 10))
        
        # Plot full data
        plt.plot(test_X, test_Y[:, i], 'k*')
        
        # Plot training data as red circles
        # find indices of train_X where 2nd column is i
        idx = np.where(to_numpy(train_X)[:,1] == i)
        plt.plot(to_numpy(train_X[idx][:,0]), to_numpy(train_Y[idx]), 'ro')
        
        #--------------------------------------------------------------
        # sanity check!!! standardization...
        xx = to_numpy(train_X[idx][:,0])
        yy = to_numpy(train_Y[idx])
        # check that xx,yy are in test_X, test_Y[:, i]
        for x, y in zip(xx, yy):
            for xxx,yyy in zip(test_X, test_Y[:, i]):
                if abs(x-xxx) < 1e-6 and abs(y-yyy) < 1e-6:
                    break
            else:
                print('ERROR: train point not in test data')
        #--------------------------------------------------------------
        
        # Plot predictive means as blue line
        plt.plot(test_X, y_pred[:, i], 'b')
        
        # compute lower,upper with y_var
        win = 2 * y_sig[:, i]
        plt.fill_between(test_X, y_pred[:, i] - win, y_pred[:, i] + win, alpha=0.5)
        
        # Shade between the lower and upper confidence bounds, different shading color
        # plt.fill_between(test_X, lower[:, i], upper[:, i], alpha=0.5, color='orange')
        
        # set y limits??
        mean_y = test_Y[:, i].mean()
        # plt.ylim([mean_y-win, mean_y+win])
        
        plt.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.title(f'Task {i}')
        # set figsize
        plt.show()

#--------------------------------------------------------------------------
# Fit model

sampler = BotorchSampler(full_X, sampled_mask, 
                         lr=learning_rate, 
                         max_iterations=max_iterations, 
                         max_sample=max_sample, 
                         rank_frac=rank_frac,
                         log_interval=log_fit)

sampler.fit(train_X, train_Y)
y_pred, y_sig, lower, upper, y_mean, y_sigma = sampler.predict()

plot_all(train_X, train_Y, test_X, test_Y, y_pred, y_sig, lower, upper, y_mean, y_sigma, max_fig=3)

#--------------------------------------------------------------------------
# Expected Improvement

S_mu = y_pred.sum(axis=-1)
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

# EI computation
valid_i, valid_j = i_indices[mask], j_indices[mask]

# Vectorized computation of all EI components
mu = y_pred[valid_i, valid_j]
sig = y_sig[valid_i, valid_j]

# Get row sums for each valid i
row_sums = np.sum(y_pred[valid_i, :], axis=1)
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
    # normal EI
    ei = imp * norm.cdf(z) + sig * norm.pdf(z)
    ei_values = ei

# Assign computed values to valid positions
EI[mask] = ei
EI = np.array(EI)

next_idx = np.argmax(EI)
next_i, next_j = X_unsampled[next_idx]
next_checkpoint = checkpoint_nums[next_i]
max_ei = np.max(ei_values)

print(f'Next checkpoint: {next_checkpoint} with EI: {max_ei:.3f}')
print()