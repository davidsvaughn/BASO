import sys, os
import gpytorch.constraints
import numpy as np
import pandas as pd
import random
from scipy.stats import norm
from matplotlib import pyplot as plt
import torch
import gpytorch
import logging
from datetime import datetime
import math
from math import ceil
from glob import glob
import shutil
import traceback

from botorch.models import MultiTaskGP
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.objective import PosteriorTransform
from gpytorch.priors import LKJCovariancePrior, SmoothedBoxPrior

from utils import task_standardize, display_fig, inv_task_standardize, inspect_matrix
from utils import to_numpy, degree_metric, log_h, clear_cuda_tensors, curvature_metric

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
rand_seed = -1
rank_frac = -1

#--------------------------------------------------------------------------
# SET PARAMETERS

fn = 'phi4-math-4claude.txt'
# fn = 'phi4-bw-4claude.txt'

# rand_seed = 2951

# number of random obs-per-task (opt) to initially sample
# if <1, then fraction of total obs
init_obs    = 2
# init_obs    = 0.05

# stop BO sampling after this fraction of points are sampled
max_sample = 0.06

# MLE estimation
learning_rate = 0.1
max_iterations = 1000
max_attempts = 20

# metric stopping criteria
use_curvature = True
metric_check_interval = 5  # Check degree every n iterations
window_size = 3  # Size of moving average window
min_iterations = 50  # Minimum iterations before allowing early stopping
rise_patience = 5  # Number of consecutive rises to trigger stopping

# rank_frac = 0.5 # 0.25
rank_frac = 0.25

# lkj prior
eta = 0.25
eta_gamma = 0.9

# logging intervals
log_fit = 50
log_loop = 5

use_cuda = True

# TODO: redefine EI with mean (not sum) ???????????????????????????????????????????
use_ei = True
use_logei = True
ei_beta = 0.5
# beta will be ei_f of its start value when ei_t of all points have been sampled
ei_f, ei_t = 0.1, 0.04
# ei_gamma = 0.9925


ucb_lambda = 3
ucb_gamma = 0.995

# select random subset of tasks
task_sample = 1

# compare_random = False
# synthetic = False
# n_rows, n_cols = 100, 100

#-------------------------------------------------------------------------
# random seed
rand_seed = np.random.randint(1000, 10000) if rand_seed <= 0 else rand_seed
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
run_base = os.path.join(parent_dir, 'runs') if local else '/mnt/llm-train/baso/runs'

#--------------------------------------------------------------------------
# remove all empty run directories
for d in glob(os.path.join(run_base, f'run_*')):
    if len(os.listdir(d)) == 0:
        os.rmdir(d)

# create a new run directory
run_id = f'run_{rand_seed}'
n = len(glob(os.path.join(run_base, f'{run_id}*')))
run_dir = os.path.join(run_base, f'{run_id}_{n}' if n>0 else run_id)
while os.path.exists(run_dir): run_dir += 'a'
os.makedirs(run_dir, exist_ok=False)

# copy current file to run directory (save parameters, etc)
src = os.path.join(current_dir, os.path.basename(__file__))
dst = os.path.join(run_dir, os.path.basename(__file__))
# shutil.copy(src, dst)
# execute command using os
os.system(f'cp {src} {dst}')

#-----------------------------------------------------------------------
# setup logging
# log_file = os.path.join(run_dir, 'log.txt')
log_file = os.path.join(run_dir, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will print to console too
    ]
)
logging.info('-'*80)
logging.info(f'Run directory: {run_dir}')
logging.info(f'Random seed: {rand_seed}')

'''
Usage:
    logging.info('Info message')
    logging.warning('Warning message')
    logging.error('Error message')
    logging.debug('Debug message')  # Only shows if level=logging.DEBUG
    logging.exception('Exception message')  # Logs the stack trace
'''
#--------------------------------------------------------------------------
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
    
# ei_gamma
ei_gamma = np.exp(np.log(ei_f) / (ei_t*K*Z - 2*Z))
logging.info('FYI: ei_gamma: %s', ei_gamma)

#-------------------------------------------------------------------------

def init_samples(K, Z, init_obs):
    if init_obs >= 1:
        if init_obs < 2:
            init_obs = 2
            logging.info('FYI: increasing init_obs to 2 (minimum 2 obs/task allowed)')
        m = ceil(init_obs * Z)
    else:
        min_frac = 2/K # == 2*Z/(K*Z)
        if init_obs < min_frac:
            m = 2*Z
            logging.info(f'FYI: increasing init_obs to {min_frac:.4g} (minimum 2 obs/task allowed)')
        else:
            m = max(2*Z, ceil(init_obs * K * Z))
    logging.info(f'FYI: initializing sampler with {m} observations ( ~{m/(K*Z):.4g} of all obs, ~{m/Z:.4g} obs/task )\n')
    logging.info('-'*80)
    
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
best_y_mean = V.mean(axis=1)[best_idx]
# logging.info(f'TRUE BEST CHECKPOINT:\t{best_checkpoint}\tY={best_y_mean:.4f}')

# subsample data
x_idx, t_idx = init_samples(K, Z, init_obs)

train_X = torch.tensor([ [checkpoints[i], j] for i, j in  zip(x_idx, t_idx) ], dtype=torch.float64)
train_Y = torch.tensor(V[x_idx, t_idx], dtype=torch.float64).unsqueeze(-1)

for i, j in zip(x_idx, t_idx):
    sampled_mask[i, j] = True

#--------------------------------------------------------------------------

# build useful tensors and arrays
full_X = torch.tensor([ [checkpoints[i], j] for i, j in  zip(*full_indices) ], dtype=torch.float64)
full_Y = torch.tensor(V[full_indices], dtype=torch.float64).unsqueeze(-1)
test_X = np.array(checkpoints)
test_Y = np.array(V)

#--------------------------------------------------------------------------
# Full Data Regression Model

def full_model(train_x, train_y, k, z, rank_frac, eta=None):
    # k = test_x.shape[0]
    # z = train_y.reshape([k, -1]).shape[-1]
    train_y, (t_mu, t_sig) = task_standardize(train_y, train_x)
    rank = int(rank_frac * z) if rank_frac > 0 else None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if eta is None:
        task_covar_prior = None
    else:
        task_covar_prior = LKJCovariancePrior(n=z, 
                                            eta=torch.tensor(eta).to(device),
                                            sd_prior=SmoothedBoxPrior(math.exp(-6), math.exp(1.25), 0.05)).to(device)
    model = MultiTaskGP(train_x, train_y, task_feature=-1, 
                        rank=rank,
                        task_covar_prior=task_covar_prior,
                        ).to(device)

    train_x, train_y = train_x.to(device), train_y.to(device)
    
    # Set the model and likelihood to training mode
    model.train()
    model.likelihood.train() # need this ???
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    metric_history = []
    consecutive_rises = 0
    
    max_iter = max_iterations//2
    
    for i in range(max_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, model.train_targets)
        loss.backward()
        #-------------------------------------------------------
        if i % (metric_check_interval//2) == 0:
            if use_curvature:
                metric = curvature_metric(model, train_x, verbose=False)
            else:
                metric = degree_metric(model, train_x)
            metric_history.append(metric)
            
            # print degree and curve (debugging)
            # logging.info(f'Iter {i}/{max_iter} - Loss: {loss.item():.3f}, avg_degree: {degree:.3f}, curvature: {curve:.3f}')
            
            # Only check for early stopping after collecting enough data points
            if len(metric_history) >= window_size + 1 and i >= min_iterations//2:
                # Calculate current and previous moving averages
                current_avg = sum(metric_history[-window_size:]) / window_size
                prev_avg = sum(metric_history[-(window_size+1):-1]) / window_size
                # Check if the moving average is rising
                if current_avg > prev_avg * 1.004:  # Small threshold to avoid stopping due to tiny fluctuations
                    consecutive_rises += 1
                    if consecutive_rises >= rise_patience//2:
                        logging.info(f'FYI: Early stopping at iteration {i+1}: metric rising for {consecutive_rises} consecutive checks')
                        break
                else:
                    consecutive_rises = 0
                    
        if i % (log_fit//10) == 0:
            logging.info(f'ITER-{i}/{max_iter} - Loss: {loss.item():.3f}, metric: {metric:.3f}')
        #-------------------------------------------------------
        optimizer.step()
    #---- end train loop --------------------------------------------------
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model.likelihood(model(train_x))
        
    #----------------------------------------------------------------------
    y_mean = predictions.mean
    y_var = predictions.variance
    y_covar = predictions.covariance_matrix
    
    # reshape
    y_pred = to_numpy(y_mean.reshape(k, z))
    y_var = to_numpy(y_var.reshape(k, z))
    y_covar = to_numpy(y_covar.reshape((k, z, k, z)))
    
    # inverse standardize...
    y_pred = y_pred * t_sig + t_mu
    y_var = y_var * t_sig**2
    y_sig = np.sqrt(y_var)
    y_mean = y_pred.mean(axis=1)
    y_covar = y_covar * t_sig**2
    y_sigma = np.array([np.sqrt(y_covar[i,:,i].sum()) for i in range(k)]) / z
    
    # current max estimate
    i = np.argmax(y_mean)
    regression_y_max = y_mean[i] # current y_mean estimate at peak_idx of y_mean estimate
    j = np.argmax(V.mean(axis=1))
    # true_y_max = V.mean(axis=1)[j]
    
    x_vals = checkpoint_nums
    regression_best_checkpoint = x_vals[i]
    true_best_checkpoint = x_vals[j]
    # regression_err = err = abs(regression_y_max - true_y_max)/true_y_max
    
    # logging.info(f'REG BEST CHECKPOINT:\t{regression_best_checkpoint}\tY={regression_y_max:.4f}')
    #------------------------------------------------------------------
    # plot GP regression
    # test_Y_mu = test_Y.mean(axis=1)
    # plt.figure(figsize=(15, 10))
    # plt.plot(x_vals, y_mean, 'b')
    # plt.fill_between(x_vals, y_mean - 2*y_sigma, y_mean + 2*y_sigma, alpha=0.5)
    # plt.plot(x_vals, test_Y_mu, 'r')
    # plt.legend(['Posterior Mean', 'Confidence', 'True Mean'])
    # plt.title(f'Full Regression Model - Rank: {rank} - Regression Best Checkpoint: {regression_best_checkpoint}')
    # # draw vertical line at regression best checkpoint
    # plt.axvline(x=regression_best_checkpoint, color='g', linestyle='--', label='Regression Best Checkpoint')
    # plt.axvline(x=true_best_checkpoint, color='m', linestyle='--', label='True Best Checkpoint')
    # display_fig(run_dir)
    #-------------------------------------------------------------------
    # return regression_best_checkpoint, regression_y_max, y_mean
    return y_mean
#--------------------------------------------------------------------------
# fit full regression model to all data
reference_y = full_model(full_X, full_Y, K, Z, rank_frac=0.5)#, eta=1.0)

# shortcut
# reference_y = V.mean(axis=1)

i = np.argmax(reference_y)
regression_best_checkpoint = checkpoint_nums[i]
regression_y_max = reference_y[i]

logging.info(f'TRU BEST CHECKPOINT:\t{best_checkpoint}\tY={best_y_mean:.4f}')
logging.info(f'REF BEST CHECKPOINT:\t{regression_best_checkpoint}\tY={regression_y_max:.4f}')

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
                 eta_gamma=0.9,
                 ei_beta=0.5,
                 ei_gamma=0.9925,
                 ucb_lambda=3,
                 ucb_gamma=0.995,
                 log_interval=50,
                 max_fit_attempts=10,
                 use_cuda=True,
                 run_dir=None):   
        self.sampled_mask = sampled_mask
        
        self.train_X = train_X
        self.train_Y = train_Y
        self.x_vals = test_X # original checkpoint numbers
        
        self.x_width = test_X.max() - test_X.min()
        self.x_min = test_X.min()
        self.test_X = (test_X - self.x_min ) / self.x_width # min/max normalized
        self.test_Y = test_Y # == V
        
        self.lr = lr
        self.max_iterations = max_iterations
        self.max_sample = max_sample
        self.rank_frac = rank_frac
        self.eta = eta
        self.eta_gamma = eta_gamma
        self.ei_beta = ei_beta
        self.ei_gamma = ei_gamma
        self.ucb_lambda = ucb_lambda
        self.ucb_gamma = ucb_gamma
        self.log_interval = log_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        if use_cuda:
            logging.info(f'Using CUDA: {torch.cuda.is_available()}')
        else:
            logging.info('Using CPU')
        logging.info(f'Using device: {self.device}')
        self.k, self.z = sampled_mask.shape
        self.full_x = full_x.to(self.device)
        self.run_dir = run_dir
        if run_dir is not None:
            plt.ioff()
        self.max_fit_attempts = max_fit_attempts
        self.round = 0
        self.reset()
        
    def reset(self):
        self.num_fit_attempts = -1
    
    @property
    def sample_fraction(self):
        return np.mean(self.sampled_mask)
    
    # repeatedly attempt to fit model
    def fit(self):
        self.round += 1
        clear_cuda_tensors(logging)
        for i in range(self.max_fit_attempts):
            if self._fit():
                return True
            else:
                if i+1 < self.max_fit_attempts:
                    logging.warning('-'*80)
                    logging.warning(f'FAILED... ATTEMPT {i+2}')
        raise Exception('ERROR: Failed to fit model - max_fit_attempts reached')
    
    # fit model inner loop
    def _fit(self):
        self.num_fit_attempts += 1
        train_x = self.train_X
        train_y = self.train_Y
        k,z = self.k, self.z
        
        # standardize train_y
        train_y, (t_mu, t_sig) = task_standardize(train_y, train_x)
        self.t_mu, self.t_sig = t_mu, t_sig
        
        # compute rank
        rank = int(self.rank_frac * z) if self.rank_frac > 0 else None
        eta = self.eta
        
        #---------------------------------------------------------------------
        # if fit is failing...
        if self.num_fit_attempts > 0:
            # rank adjustment...
            if self.rank_frac > 0:
                w = self.num_fit_attempts * (z-rank)//self.max_fit_attempts
                rank = min(z, rank + w)
                logging.info(f'[ROUND-{self.round+1}]\tFYI: rank adjusted to {rank}')
            # eta adjustment... ??
            eta = eta * (self.eta_gamma ** max(0, self.num_fit_attempts - self.max_fit_attempts//2))
            logging.info(f'[ROUND-{self.round+1}]\tFYI: eta adjusted to {eta:.4g}')
                
        #---------------------------------------------------------------------
        # define task_covar_prior (IMPORTANT!!! nothing works without this...)
        # see: https://archive.botorch.org/v/0.9.2/api/_modules/botorch/models/multitask.html
        
        task_covar_prior = LKJCovariancePrior(n=z, 
                                              eta=torch.tensor(eta).to(self.device),
                                              sd_prior=SmoothedBoxPrior(math.exp(-6), math.exp(1.25), 0.05))
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
        metric_history = []
        consecutive_rises = 0
        # metric_check_interval = 5  # Check degree every n iterations
        # window_size = 3  # Size of moving average window
        # min_iterations = 50  # Minimum iterations before allowing early stopping
        # rise_patience = 5  # Number of consecutive rises to trigger stopping
        
        # train loop
        for i in range(self.max_iterations):
            optimizer.zero_grad()
            output = self.model(train_x)

            try:
                loss = -mll(output, self.model.train_targets)
            except Exception as e:
                if self.num_fit_attempts+1 == self.max_fit_attempts:
                    logging.exception("An exception occurred")
                else:
                    logging.error(f'error in loss calculation at iteration {i}:\n{e}')
                return False
            
            loss.backward()
            
            if i % metric_check_interval == 0:

                if use_curvature:
                    metric = curvature_metric(self.model, self.full_x, verbose=False)
                else:
                    metric = degree_metric(self.model, self.full_x)
                metric_history.append(metric)
                
                # print degree and curve (debugging)
                # logging.info(f'Iter {i}/{self.max_iterations} - Loss: {loss.item():.3f}, avg_degree: {degree:.3f}, curvature: {curve:.3f}')
                
                # Only check for early stopping after collecting enough data points
                if len(metric_history) >= window_size + 1 and i >= min_iterations:
                    # Calculate current and previous moving averages
                    current_avg = sum(metric_history[-window_size:]) / window_size
                    prev_avg = sum(metric_history[-(window_size+1):-1]) / window_size
                    # Check if the moving average is rising
                    if current_avg > prev_avg * 1.005:  # Small threshold to avoid stopping due to tiny fluctuations
                        consecutive_rises += 1
                        if consecutive_rises >= rise_patience:
                            logging.info(f'[ROUND-{self.round+1}]\tFYI: Early stopping at iteration {i+1}: metric rising for {consecutive_rises} consecutive checks')
                            break
                    else:
                        consecutive_rises = 0
            if i % self.log_interval == 0:
                logging.info(f'[ROUND-{self.round+1}]\tITER-{i}/{self.max_iterations} - Loss: {loss.item():.3f}, metric: {metric:.3f}')
                # logging.info(f'Iter {i}/{self.max_iterations} - Loss: {loss.item():.3f}, curvature: {curve:.3f}')
                # logging.info(f'Iter {i}/{self.max_iterations} - Loss: {loss.item():.3f}, avg_degree: {degree:.3f}, curvature: {curve:.3f}')
                
            optimizer.step()
            
        #---- end train loop --------------------------------------------------
        self.reset() # self.num_fit_attempts = 0
        return True
    #--------------------------------------------------------------------------
            
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
        
        # current max estimate
        i = np.argmax(y_mean)
        self.current_best_checkpoint = self.x_vals[i]
        # current_y_mean = y_mean[i] # current y_mean estimate at peak_idx of y_mean estimate
        
        # percent difference between current_y_mean and best_y_mean
        current_y_mean = V.mean(axis=1)[i] # true y_mean at peak_idx of y_mean estimate
        current_y_val = reference_y[i] # regression_y at peak_idx of y_mean estimate
        self.current_err = err = abs(current_y_val - regression_y_max)/regression_y_max
        
        logging.info('-'*80)
        logging.info(f'[ROUND-{self.round+1}]\tSTATS\t{self.current_best_checkpoint}\t{current_y_val:.2f}\t{err:.4g}\t{self.sample_fraction:.4f}')
        logging.info(f'[ROUND-{self.round+1}]\tCURRENT BEST\tCHECKPOINT-{self.current_best_checkpoint}\tY_PRED={current_y_val:.4f}\tY_ERR={100*err:.4g}%\t({100*self.sample_fraction:.2f}% sampled)')
        
        self.y_pred = y_pred
        self.y_mean = y_mean
        self.y_sig = y_sig
        self.y_sigma = y_sigma
        return y_pred, y_sig, y_mean, y_sigma
    
    #-------------------------------------------------------------------------
    # Plotting functions
    
    def display(self, fig=None, fn=None):
        display_fig(self.run_dir, fig=fig, fn=fn)
    
    def plot_posterior_mean(self):
        K,Z = self.test_Y.shape
        test_Y_mu = self.test_Y.mean(axis=1)
        plt.figure(figsize=(15, 10))
        plt.plot(self.x_vals, self.y_mean, 'b')
        plt.fill_between(self.x_vals, self.y_mean - 2*self.y_sigma, self.y_mean + 2*self.y_sigma, alpha=0.5)
        plt.plot(self.x_vals, test_Y_mu, 'r')
        
        plt.legend(['Posterior Mean', 'Confidence', 'True Mean'])
        plt.title(f'Round: {self.round} - Current Best Checkpoint: {self.current_best_checkpoint}')
        self.display()
        
        
    def plot_task(self, j, msg='', fvals=None):
        fig, ax1 = plt.subplots(figsize=(15, 10))
        
        # x_grid = self.test_X
        x_grid = self.x_vals
        
        # Plot all data as black stars
        ax1.plot(x_grid, self.test_Y[:, j], 'k*')
        
        # Plot training (observed) data as red circles
        idx = np.where(to_numpy(self.train_X)[:,1] == j)
        xx = to_numpy(self.train_X[idx][:,0])
        
        # inverse normalize x
        xx = xx * self.x_width + self.x_min
        yy = to_numpy(self.train_Y[idx])
        ax1.plot(xx, yy, 'ro')
        
        # Plot predictive means as blue line
        ax1.plot(x_grid, self.y_pred[:, j], 'b')
        
        # confidences
        win = 2 * self.y_sig[:, j]
        ax1.fill_between(x_grid, self.y_pred[:, j] - win, self.y_pred[:, j] + win, alpha=0.5)
        
        # Set up primary y-axis labels
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        if fvals is not None:
            # Create secondary y-axis for fvals
            ax2 = ax1.twinx()
            
            # Plot fvals as green dashed line on secondary axis
            ax2.plot(x_grid, fvals, 'g--')
            ax2.set_ylabel('Acquisition Function Value', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            # # Legend for both axes
            # lines1, labels1 = ax1.get_legend_handles_labels()
            # lines2, labels2 = ax2.get_legend_handles_labels()
            
            # Create custom legend with all elements
            ax1.legend(
                ['Unobserved', 'Observed', 'Posterior Mean', 'Confidence', 'AcqFxn'],
                loc='best'
            )
        else:
            ax1.legend(['Unobserved', 'Observed', 'Posterior Mean', 'Confidence'])
        
        plt.title(f'Round: {self.round} - Task: {j} {msg}')
        plt.tight_layout()  # Adjust layout to make room for the second y-axis label
        self.display()
    
    def plot_all(self, max_fig=None):
        Z = test_Y.shape[-1]
        self.plot_posterior_mean()
        for j in range(Z):
            if max_fig is not None and j > max_fig:
                break
            self.plot_task(j)
            
    #-------------------------------------------------------------------------
    
    # compute expected improvement
    def expected_improvement(self, beta=0):
        S_mu = self.y_pred.sum(axis=-1)
        S_max_idx = np.argmax(S_mu)
        S_max = S_mu[S_max_idx]
        
        best_f =self.y_pred.max()
        LogEI = LogExpectedImprovement(self.model, best_f)
        
        # get unsampled indices
        i_indices, j_indices = np.where(np.ones_like(self.sampled_mask))
        mask = (~self.sampled_mask).reshape(-1)
        
        # Vectorized computation of all EI components
        valid_i, valid_j = i_indices[mask], j_indices[mask]
            
        # Initialize EI array with -inf for invalid entries
        EI = np.full(len(mask), -math.inf)

        # Vectorized computation of all EI components
        mu = self.y_pred[valid_i, valid_j]
        sig = self.y_sig[valid_i, valid_j]

        # Get row sums for each valid i
        row_sums = np.sum(self.y_pred[valid_i, :], axis=1)
        sx = row_sums - mu

        # improvement vector, z-scores
        s_max = S_max - sx
        imp = mu - s_max - beta
        z = imp/sig

        if use_logei:
            # logEI computation (for stability)
            logh = log_h(torch.tensor(z, dtype=torch.float64)).numpy()
            ei = np.log(sig) + logh
        else:
            # normal EI
            ei = imp * norm.cdf(z) + sig * norm.pdf(z)
            # take log
            ei = np.log(ei)

        # Assign computed values to valid positions and return
        EI[mask] = ei
        k = np.argmax(EI)
        return EI, EI[k], i_indices[k], j_indices[k]
    
    def ucb(self):
        S_mu = self.y_pred.sum(axis=-1)
        # S_max_idx = np.argmax(S_mu)
        # S_max = S_mu[S_max_idx]
        
        # get unsampled indices
        i_indices, j_indices = np.where(np.ones_like(self.sampled_mask))
        mask = (~self.sampled_mask).reshape(-1)
        # Initialize EI array with -inf for invalid entries
        UCB = np.full(len(mask), -math.inf)
        # Vectorized computation of all EI components
        valid_i, valid_j = i_indices[mask], j_indices[mask]
        
        mu = self.y_pred[valid_i, valid_j]
        sig = self.y_sig[valid_i, valid_j]

        # Get row sums for each valid i (sum[mu] over all tasks for each checkpoint)
        row_sums = np.sum(self.y_pred[valid_i, :], axis=1)
        row_means = np.mean(self.y_pred[valid_i, :], axis=1)
        
        row_vals = row_sums
        # row_vals = row_means
        
        ucb = row_vals + self.ucb_lambda * sig
        
        # Assign computed values to valid positions and return
        UCB[mask] = ucb
        k = np.argmax(UCB)
        return UCB, UCB[k], i_indices[k], j_indices[k]
    #----------------------------------------------------------------------
    
    
    # choose next sample
    def sample_next(self):
       
        # acquisition function
        if use_ei:
            # expected improvement
            acq_values, max_val, next_i, next_j = self.expected_improvement(beta=self.ei_beta)
            # acq_values_2, _ = self.expected_improvement(beta=0.5) # debugging
            self.ei_beta = self.ei_beta * self.ei_gamma
            logging.info(f'[ROUND-{self.round+1}]\tEI beta: {self.ei_beta:.4g}')
        else:
            # upper confidence bound
            acq_values, max_val, next_i, next_j = self.ucb()
            self.ucb_lambda = self.ucb_lambda * self.ucb_gamma
            logging.info(f'[ROUND-{self.round+1}]\tUCB lambda: {self.ucb_lambda:.4g}')
        
        # convert to original checkpoint numbers
        next_checkpoint = self.x_vals[next_i]
        
        logging.info(f'[ROUND-{self.round+1}]\tNEXT SAMPLE\tCHECKPOINT-{next_checkpoint}\tTASK-{next_j}\t(acq_fxn_max={max_val:.3g})')
        logging.info('='*80)
        
        # plot task (before sampling)
        self.plot_task(next_j, '(before)', acq_values.reshape(self.k, self.z)[:, next_j])
        # self.plot_task(next_j, '(before)', acq_values_2.reshape(self.k, self.z)[:, next_j]) # debugging
        
        # add new sample to training set (observe) and update mask 
        self.train_X = torch.cat([self.train_X, torch.tensor([ [self.test_X[next_i], next_j] ], dtype=torch.float64)])
        self.train_Y = torch.cat([self.train_Y, torch.tensor([self.test_Y[next_i, next_j]], dtype=torch.float64).unsqueeze(-1)])
        self.sampled_mask[next_i, next_j] = True
        
        # find task with most samples
        task_counts = np.sum(self.sampled_mask, axis=0)
        max_task = np.argmax(task_counts)
        max_count = task_counts[max_task]
        logging.info(f'[ROUND-{self.round+1}]\tTASK-{max_task} has most samples: {max_count}')
        
        # return next sample
        return next_i, next_j

#--------------------------------------------------------------------------
# Fit model

sampler = BotorchSampler(full_X, sampled_mask,
                         train_X=train_X, train_Y=train_Y, 
                         test_X=checkpoint_nums, test_Y=test_Y,
                         lr=learning_rate,
                         eta=eta,
                         eta_gamma=eta_gamma,
                         ucb_lambda=ucb_lambda,
                         ucb_gamma=ucb_gamma,
                         max_iterations=max_iterations,
                         max_fit_attempts=max_attempts,
                         max_sample=max_sample, 
                         rank_frac=rank_frac,
                         log_interval=log_fit,
                         use_cuda=use_cuda,
                         run_dir=run_dir)

#--------------------------------------------------------------------------

sampler.fit()
sampler.predict()
# sampler.plot_all(max_fig=10)

while sampler.sample_fraction < max_sample:
    _, next_task = sampler.sample_next()
    sampler.fit()
    sampler.predict()
    sampler.plot_task(next_task, '(after)')
    sampler.plot_posterior_mean()

#--------------------------------------------------------------------------