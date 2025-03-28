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

# traceback
import traceback

from botorch.models import MultiTaskGP

from gpytorch.priors import LKJCovariancePrior, GammaPrior, SmoothedBoxPrior

from utils import task_standardize, inv_task_standardize, to_numpy, degree_metric, log_h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
rand_seed = -1
rank_frac = -1

#--------------------------------------------------------------------------
# SET PARAMETERS

fn = 'phi4-math-4claude.txt'
# fn = 'phi4-bw-4claude.txt'

rand_seed = 985

task_sample = 1 # select random subset of tasks
init_tpc    = 0.02 # number of random tasks per checkpoint (tpc) to initially sample
             # -> if fraction, tpc++ tpc until that fraction of tasks are sampled
max_sample = 0.25 # stop BO sampling after this fraction of points are sampled

# MLE estimation
max_iterations = 1000
learning_rate = 0.1
rank_frac = 0.5 # 0.25

# lkj prior
eta = 0.25

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
#--------------------------------------------------------------------------

# build useful tensors and arrays
full_X = torch.tensor([ [checkpoints[i], j] for i, j in  zip(*full_indices) ], dtype=torch.float64)
test_X = np.array(checkpoints)
test_Y = np.array(V)

#--------------------------------------------------------------------------

class BotorchSampler:
    def __init__(self, full_x, sampled_mask, 
                 train_X, train_Y, 
                 test_X, test_Y,
                 lr=0.1, 
                 max_iterations=1000,
                 max_sample=0.25, 
                 rank_frac=0.5,
                 eta=0.25,
                 log_interval=50):   
        self.sampled_mask = sampled_mask
        
        self.train_X = train_X
        self.train_Y = train_Y
        self.x_vals = test_X # original checkpoint numbers
        self.test_X = (test_X - test_X.min()) / (test_X.max() - test_X.min()) # min/max normalized
        self.test_Y = test_Y # == V
        
        self.lr = lr
        self.max_iterations = max_iterations
        self.max_sample = max_sample
        self.rank_frac = rank_frac
        self.eta = eta
        self.log_interval = log_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k, self.z = sampled_mask.shape
        self.full_x = full_x.to(self.device)
        self.y_pred = None
        
    def fit(self, train_x=None, train_y=None):
        if train_x is None:
            train_x = self.train_X
        if train_y is None:
            train_y = self.train_Y
        k,z = self.k, self.z
        
        # standardize train_y
        train_y, (t_mu, t_sig) = task_standardize(train_y, train_x)
        self.t_mu, self.t_sig = t_mu, t_sig
        
        # compute rank
        rank = int(self.rank_frac * z) if self.rank_frac > 0 else None
        
        #---------------------------------------------------------------------
        # define task_covar_prior
        
        task_covar_prior = LKJCovariancePrior(n=z, 
                                              eta=torch.tensor(self.eta).to(self.device),
                                              sd_prior=SmoothedBoxPrior(math.exp(-6), math.exp(1.25), 0.05))
        
        # see: https://archive.botorch.org/v/0.9.2/api/_modules/botorch/models/multitask.html
        # sd_prior = GammaPrior(1.0, 1).to(self.device) # GammaPrior(1.0, 0.15)
        # sd_prior._event_shape = torch.Size([z])
        # task_covar_prior = LKJCovariancePrior(n=z, eta=torch.tensor(self.eta).to(self.device),
        #                                       sd_prior=sd_prior.to(self.device)).to(self.device)
        #---------------------------------------------------------------------
        # Initialize multitask model
        
        self.model = MultiTaskGP(train_x, train_y, task_feature=-1, 
                                 rank=rank,
                                 task_covar_prior=task_covar_prior.to(self.device),
                                 ).to(self.device)

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
            
            loss = -mll(output, self.model.train_targets)
            # try:
            #     loss = -mll(output, self.model.train_targets)
            # except Exception as e:
            #     print(f'Early stopping at iteration {i+1}: ERROR in loss calculation...')
            #     print(f'ERROR:{i}: {e}')
            #     print(traceback.format_exc())
            #     break
            
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
        k,z = self.k, self.z
        if x is None:
            x = self.full_x
        self.model.eval()
        self.model.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(x))
        
        # retrieve
        y_mean = predictions.mean
        y_var = predictions.variance
        y_covar = predictions.covariance_matrix
        
        # reshape
        y_pred = to_numpy(y_mean.reshape(k, z))
        y_var = to_numpy(y_var.reshape(k, z))
        y_covar = to_numpy(y_covar.reshape((k, z, k, z)))
        
        # inverse standardize...
        y_pred = y_pred * self.t_sig + self.t_mu
        y_var = y_var * self.t_sig**2
        y_sig = np.sqrt(y_var)
        y_mean = y_pred.mean(axis=1)
        y_covar = y_covar * self.t_sig**2
        y_sigma = np.array([np.sqrt(y_covar[i,:,i].sum()) for i in range(k)]) / z
        
        # lower, upper = predictions.confidence_region()
        # lower = to_numpy(lower.reshape(k, z))
        # upper = to_numpy(upper.reshape(k, z))
        # lower = lower * self.t_sig + self.t_mu
        # upper = upper * self.t_sig + self.t_mu
        self.y_pred = y_pred
        self.y_mean = y_mean
        self.y_sig = y_sig
        self.y_sigma = y_sigma
        return y_pred, y_sig, y_mean, y_sigma
    
    def plot_task(self, j):
        # train_X = self.train_X
        # train_Y = self.train_Y
        # test_X = self.test_X
        # test_Y = self.test_Y
        # y_pred = self.y_pred
        # y_sig = self.y_sig
        plt.figure(figsize=(15, 10))
        
        # Plot all data as black stars
        plt.plot(self.test_X, self.test_Y[:, j], 'k*')
        
        # Plot training (observed) data as red circles
        idx = np.where(to_numpy(self.train_X)[:,1] == j)
        plt.plot(to_numpy(self.train_X[idx][:,0]), to_numpy(self.train_Y[idx]), 'ro')
        
        # Plot predictive means as blue line
        plt.plot(self.test_X, self.y_pred[:, j], 'b')
        
        # cconfidences
        win = 2 * self.y_sig[:, j]
        plt.fill_between(self.test_X, self.y_pred[:, j] - win, self.y_pred[:, j] + win, alpha=0.5)
        
        plt.legend(['Unobserved', 'Observed', 'Posterior Mean', 'Confidence'])
        plt.title(f'Task {j}')
        plt.show()
        
    def plot_posterior_mean(self):
        K,Z = self.test_Y.shape
        test_Y_mu = self.test_Y.mean(axis=1)
        plt.figure(figsize=(15, 10))
        plt.plot(self.y_mean, 'b')
        plt.fill_between(range(K), self.y_mean - 2*self.y_sigma, self.y_mean + 2*self.y_sigma, alpha=0.5)
        plt.plot(test_Y_mu, 'r')
        plt.show()
    
    # Plotting function
    def plot_all(self, max_fig=None):
        train_X = self.train_X
        train_Y = self.train_Y
        test_X = self.test_X
        test_Y = self.test_Y
        y_pred = self.y_pred
        y_sig = self.y_sig
        y_mean = self.y_mean
        y_sigma = self.y_sigma
        
        K,Z = test_Y.shape
        test_Y_mu = test_Y.mean(axis=1)
        
        plt.figure(figsize=(15, 10))
        plt.plot(y_mean, 'b')
        plt.fill_between(range(K), y_mean - 2*y_sigma, y_mean + 2*y_sigma, alpha=0.5)
        plt.plot(test_Y_mu, 'r')
        plt.show()
        
        for j in range(Z):
            if max_fig is not None and j > max_fig:
                break
            plt.figure(figsize=(15, 10))
            
            # Plot all data as black stars
            plt.plot(test_X, test_Y[:, j], 'k*')
            
            # Plot training (observed) data as red circles
            idx = np.where(to_numpy(train_X)[:,1] == j)
            plt.plot(to_numpy(train_X[idx][:,0]), to_numpy(train_Y[idx]), 'ro')
            
            # Plot predictive means as blue line
            plt.plot(test_X, y_pred[:, j], 'b')
            
            # cconfidences
            win = 2 * y_sig[:, j]
            plt.fill_between(test_X, y_pred[:, j] - win, y_pred[:, j] + win, alpha=0.5)
            
            plt.legend(['Unobserved', 'Observed', 'Posterior Mean', 'Confidence'])
            plt.title(f'Task {j}')
            plt.show()
    
    # compute expected improvement
    def expected_improvement(self):
        S_mu = self.y_pred.sum(axis=-1)
        S_max_idx = np.argmax(S_mu)
        S_max = S_mu[S_max_idx]
        
        # get unsampled indices
        unsampled_indices = np.where(~self.sampled_mask)

        # get list of unsampled tasks, if any
        unsampled_tasks = np.setdiff1d(np.arange(Z), np.unique(np.where(self.sampled_mask)[1]))
        i_indices, j_indices = unsampled_indices

        # Create mask for unsampled tasks check
        mask = np.ones(len(i_indices), dtype=bool)
        if unsampled_tasks.size > 0:
            mask = np.isin(j_indices, unsampled_tasks)
            
        # Initialize EI array with -inf for invalid entries
        EI = np.full(len(i_indices), -math.inf)

        # Vectorized computation of all EI components
        valid_i, valid_j = i_indices[mask], j_indices[mask]
        mu = self.y_pred[valid_i, valid_j]
        sig = self.y_sig[valid_i, valid_j]

        # Get row sums for each valid i
        row_sums = np.sum(self.y_pred[valid_i, :], axis=1)
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
        max_ei = np.max(ei_values)
        return EI, max_ei
    
    # use EI to choose next sample
    def sample_next(self):
        unsampled_indices = np.where(~self.sampled_mask)
        x_unsampled = np.array(unsampled_indices).T
        
        exp_imp, max_ei = self.expected_improvement()
        next_idx = np.argmax(exp_imp)
        next_i, next_j = x_unsampled[next_idx]
        next_checkpoint = self.x_vals[next_i]
        print(f'Next Sample: checkpoint-{next_checkpoint}, task-{next_j}, (EI={max_ei:.3f})')
        
        self.plot_task(next_j)
        
        self.sampled_mask[next_i, next_j] = True
        self.train_X = torch.cat([self.train_X, torch.tensor([ [self.test_X[next_i], next_j] ], dtype=torch.float64)])
        self.train_Y = torch.cat([self.train_Y, torch.tensor([self.test_Y[next_i, next_j]], dtype=torch.float64).unsqueeze(-1)])
        
        return next_i, next_j

#--------------------------------------------------------------------------
# Fit model

sampler = BotorchSampler(full_X, sampled_mask,
                         train_X=train_X, train_Y=train_Y, 
                         test_X=checkpoint_nums, test_Y=test_Y,
                         lr=learning_rate, 
                         max_iterations=max_iterations, 
                         max_sample=max_sample, 
                         rank_frac=rank_frac,
                         eta=eta,
                         log_interval=log_fit)

#--------------------------------------------------------------------------

sampler.fit()
sampler.predict()
sampler.plot_all(max_fig=10)

#--------------------------------------------------------------------------

while True:
    _,next_task = sampler.sample_next()
    sampler.fit()
    sampler.predict()
    sampler.plot_task(next_task)
    sampler.plot_posterior_mean()

#--------------------------------------------------------------------------