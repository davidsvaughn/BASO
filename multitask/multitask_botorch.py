import sys, os
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
from scipy.stats import gaussian_kde
from math import ceil
from glob import glob
import traceback
# import shutil

# import partial
from functools import partial

from botorch.models import MultiTaskGP
from gpytorch.priors import LKJCovariancePrior, SmoothedBoxPrior

from utils import task_standardize, display_fig, degree_metric, adict
from utils import to_numpy, log_h, clear_cuda_tensors
from normalize import Transform
from stopping import StoppingCondition, StoppingConditions
from sampling import init_samples

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
rand_seed = -1
rank_frac = -1

#--------------------------------------------------------------------------
# SET PARAMETERS

fn = 'phi4-math-4claude.txt'
# fn = 'phi4-bw-4claude.txt'

# select random subset of tasks
task_sample = 1.0
# task_sample = 0.5

# rand_seed = 2951

# number of random obs-per-task (opt) to initially sample
# if <1, then fraction of total obs
n_obs    = 2
# n_obs    = 0.04

# stop BO sampling after this fraction of points are sampled
max_sample_fraction = 0.1

# MLE estimation
learning_rate = 0.1
min_iterations = 100
max_iterations = 1000
max_retries = 20
use_cuda = True

# lkj prior
eta = 0.25
eta_gamma = 0.9
rank_frac = 0.25

# Expected Improvement parameters
use_ei = True
use_logei = True
ei_beta = 0.5
# beta will be ei_f of its start value when ei_t of all points have been sampled
ei_f, ei_t = 0.2, 0.05
# ei_gamma = 0.9925

# logging intervals
log_interval = 10
verbosity = 1

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
j = len(glob(os.path.join(run_base, f'{run_id}*')))
run_dir = os.path.join(run_base, f'{run_id}_{j}' if j>0 else run_id)
while os.path.exists(run_dir): run_dir += 'a'
os.makedirs(run_dir, exist_ok=False)

# copy current file to run directory (save parameters, etc)
src = os.path.join(current_dir, os.path.basename(__file__))
dst = os.path.join(run_dir, os.path.basename(__file__))
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

def log(msg, verbosity_level=1,
        level=logging.INFO):
    if verbosity_level > verbosity:
        return
    if level == logging.DEBUG:
        logging.debug(msg)
    elif level == logging.INFO:
        logging.info(msg)
    elif level == logging.WARNING:
        logging.warning(msg)
    elif level == logging.ERROR:
        logging.error(msg)
    elif level == logging.CRITICAL:
        logging.critical(msg)
    else:
        raise ValueError(f'Unknown log level: {level}')
    
def warn(msg, verbosity_level=1, level=logging.WARNING):
    log(msg, verbosity_level=verbosity_level, level=level)
    
log('-'*100)
log(f'Run directory: {run_dir}')
log(f'Random seed: {rand_seed}')

#--------------------------------------------------------------------------
# load data
df = pd.read_csv(os.path.join(data_dir, fn), delimiter='\t')

# Extract checkpoint numbers
X_feats = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values
 
# Identify test columns (excluding average)
test_cols = [col for col in df.columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']
Y_test = df[test_cols].values
del test_cols
N,M = Y_test.shape

# sample subset of tasks (possibly)
if task_sample>0 and task_sample!=1:
    if task_sample < 1:
        task_sample = int(task_sample * M)
    idx = np.random.choice(range(M), task_sample, replace=False)
    Y_test = Y_test[:, idx]
    N,M = Y_test.shape
    
# compute ei_gamma
ei_gamma = np.exp(np.log(ei_f) / (ei_t*N*M - 2*M))
log(f'FYI: ei_gamma: {ei_gamma:.4g}')

#--------------------------------------------------------------------------

class BotorchSampler:
    def __init__(self,
                 X_feats,
                 Y_obs,
                 Y_test=None,
                 min_iterations=50,
                 max_iterations=1000,
                 lr=0.1, # learning rate for MLE fit
                 patience=5,
                 max_sample_fraction=0.25, 
                 rank_frac=0.5,
                 eta=0.25,
                 eta_gamma=0.9,
                 ei_beta=0.5,
                 ei_gamma=0.9925,
                 log_interval=10,
                 max_retries=10,
                 verbosity=1,
                 use_cuda=True,
                 run_dir=None): 
        
        #--------------------------------------------------------------------------
        self.X_feats = X_feats # 1D : original input feature space
        self.Y_obs = Y_obs # 2D : observed y values (np.nan everywhere else)
        self.Y_test = Y_test # 2D : ALL y values (optional)
        
        # normalize X_feats to unit interval [0..1]
        self.X_test, self.X_norm = Transform.normalize(X_feats)
        
        # create X_inputs (dx2) that includes task indices
        all_idx = np.where(np.ones_like(Y_obs))
        self.X_inputs = torch.tensor([ [self.X_test[i], j] for i, j in  zip(*all_idx) ], dtype=torch.float64)
        
        # create X_train (dx2) and Y_train (dx1) from observed data
        sample_idx = np.where(self.S)
        self.X_train = torch.tensor([ [self.X_test[i], j] for i, j in  zip(*sample_idx) ], dtype=torch.float64)
        self.Y_train = torch.tensor( Y_obs[sample_idx], dtype=torch.float64 ).unsqueeze(-1)

        #--------------------------------------------------
        self.lr = lr
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.patience = patience
        self.max_sample_fraction = max_sample_fraction
        self.rank_frac = rank_frac
        self.eta = eta
        self.eta_gamma = eta_gamma
        self.ei_beta = ei_beta
        self.ei_gamma = ei_gamma
        self.ei_decay = 1.0
        self.log_interval = log_interval
        self.max_retries = max_retries
        self.verbosity = verbosity
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        
        if use_cuda:
            log(f'Using CUDA: {torch.cuda.is_available()}')
        else:
            log('Using CPU')
        log(f'Using device: {self.device}')
    
        self.X_inputs = self.X_inputs.to(self.device)
        self.n, self.m = Y_obs.shape
        self.run_dir = run_dir
        if run_dir is not None:
            plt.ioff()
        self.round = 0
        self.reset()
        
    def reset(self):
        self.num_retries = -1
    
    @property
    def S(self):
        return ~np.isnan(self.Y_obs)
    
    @property
    def sample_fraction(self):
        return np.mean(self.S)
    
    # repeatedly attempt to fit model
    def fit(self):
        # increment round
        self.round += 1
        
        # reset next sample indices
        self.next_i, self.next_j = None, None
        
        # log current sample fraction
        clear_cuda_tensors(log = partial(log, verbosity_level=2))
        
        # run MLE fit loop
        for i in range(self.max_retries):
            if self._fit():
                return True
            else:
                if i+1 < self.max_retries:
                    warn('-'*100)
                    warn(f'FAILED... ATTEMPT {i+2}')
        raise Exception('ERROR: Failed to fit model - max_retries reached')
    
    # fit model inner loop
    def _fit(self):
        self.num_retries += 1
        x_train = self.X_train
        y_train = self.Y_train
        m = self.S.shape[1]
        
        # standardize y_train
        y_train, self.Y_stand = Transform.standardize(x_train, y_train)
        
        # compute rank
        rank = int(self.rank_frac * m) if self.rank_frac > 0 else None
        
        # init retry-adjusted parameters
        eta = self.eta
        patience = self.patience
        min_iterations = self.min_iterations
        
        #---------------------------------------------------------------------
        # if fit is failing...
        if self.num_retries > 0:
            # rank adjustment...
            if self.rank_frac > 0:
                w = self.num_retries * (m-rank)//self.max_retries
                rank = min(m, rank + w)
                log(f'[ROUND-{self.round+1}]\tFYI: rank adjusted to {rank}', 2)
            # eta adjustment... ??
            eta = eta * (self.eta_gamma ** max(0, self.num_retries - self.max_retries//2))
            log(f'[ROUND-{self.round+1}]\tFYI: eta adjusted to {eta:.4g}', 2)
            
            # patience, min_iterations adjustment...
            patience = max(5, patience - self.num_retries//2)
            min_iterations = max(50, min_iterations - 10*self.num_retries//2)
                
        #---------------------------------------------------------------------
        # define task_covar_prior (IMPORTANT!!! nothing works without this...)
        # see: https://archive.botorch.org/v/0.9.2/api/_modules/botorch/models/multitask.html
        if eta is None:
            task_covar_prior = None
        else:
            task_covar_prior = LKJCovariancePrior(n=m, 
                                                  eta=torch.tensor(eta).to(self.device),
                                                  sd_prior=SmoothedBoxPrior(math.exp(-6), math.exp(1.25), 0.05)).to(self.device)
            
        #---------------------------------------------------------------------
        # Initialize multitask model
        
        self.model = MultiTaskGP(x_train, y_train, task_feature=-1, 
                                 rank=rank,
                                 task_covar_prior=task_covar_prior,
                                 outcome_transform=None,
                                 ).to(self.device)

        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        
        # Set the model and likelihood to training mode
        self.model.train()
        self.model.likelihood.train() # need this ???
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        #---------------------------------------------------------
        # stopping criteria
        
        loss_condition = StoppingCondition(
            value="loss",
            condition="0 < (x[-2] - x[-1])/abs(x[-2]) < t",
            t=0.0005,
            alpha=5,
            min_iterations=min_iterations, # 50
            patience=patience, # 5
            lr_steps=5,
            lr_gamma=0.8,
            optimizer=optimizer,
            logging=logging,
            verbosity=self.verbosity,
            # prefix=f'[ROUND-{self.round+1}]'
        )
        
        # function closure to evaluate degree-based stopping criterion
        deg_stats = adict()
        def max_degree(**kwargs):
            nonlocal deg_stats
            degree_metric(model=self.model, X_inputs=self.X_inputs, m=self.m, ret=deg_stats)
            return deg_stats.max
        
        degree_condition = StoppingCondition(
            value=max_degree,
            condition="x[-1] > t",
            t=3,
            interval=5,
            min_iterations=min_iterations, # 50
            logging=logging,
            verbosity=self.verbosity,
            # prefix=f'[ROUND-{self.round+1}]'
        )
        
        # combine stopping conditions
        stop_conditions = StoppingConditions([loss_condition, degree_condition])
        
        #---------------------------------------------------------
        
        # train loop
        for i in range(self.max_iterations):
            optimizer.zero_grad()
            output = self.model(x_train)
            try:
                loss = -mll(output, self.model.train_targets)
            except Exception as e:
                if self.num_retries+1 == self.max_retries:
                    logging.exception("An exception occurred")
                else:
                    logging.error(f'error in loss calculation at iteration {i}:\n{e}')
                return False
            loss.backward()
        
            if stop_conditions.check(loss=loss.item()):
                break
            
            optimizer.step()
            if i % self.log_interval == 0:
                try: log(f'[ROUND-{self.round+1}]\tITER-{i}/{self.max_iterations}\tLoss: {loss.item():.4g}\tMaxDeg: {deg_stats.max}\tAvgDeg: {deg_stats.avg:.4g}')
                except: log(f'[ROUND-{self.round+1}]\tITER-{i}/{self.max_iterations}\tLoss: {loss.item():.4g}')    
                     
        #---- end train loop --------------------------------------------------
        
        self.reset()
        return True
    #--------------------------------------------------------------------------
            
    def predict(self, x=None):
        n,m = self.n, self.m
        if x is None:
            x = self.X_inputs
        self.model.eval()
        self.model.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.model.likelihood(self.model(x))
        
        # retrieve
        y_mean = predictions.mean
        y_var = predictions.variance
        y_covar = predictions.covariance_matrix
        
        # reshape
        y_pred = to_numpy(y_mean.reshape(n, m))
        y_var = to_numpy(y_var.reshape(n, m))
        y_covar = to_numpy(y_covar.reshape((n, m, n, m)))
        
        # inverse standardize...
        y_pred = self.Y_stand.inv(y_pred)
        sigma = self.Y_stand.params['sigma']
        y_var = y_var * sigma**2
        y_sig = np.sqrt(y_var)
        y_mean = y_pred.mean(axis=1)
        y_covar = y_covar * sigma**2
        y_sigma = np.array([np.sqrt(y_covar[i,:,i].sum()) for i in range(n)]) / m
        
        # current max estimate
        self.current_best_idx = i = np.argmax(y_mean)
        self.current_best_checkpoint = self.X_feats[i]
        # current_y_mean = y_mean[i] # current y_mean estimate at peak_idx of y_mean estimate
        
        #-------------------------------------------------
        
        self.y_pred = y_pred
        self.y_mean = y_mean
        self.y_sig = y_sig
        self.y_sigma = y_sigma
        return y_mean, y_sigma, y_pred, y_sig
    
    def compare(self, y_ref):
        i = np.argmax(y_ref)
        y_ref_max = y_ref[i]
        # ref_best_checkpoint = self.X_feats[i]
        current_y_val = y_ref[self.current_best_idx]
        self.current_err = err = abs(current_y_val - y_ref_max)/y_ref_max
        log('-'*100)
        log(f'[ROUND-{self.round+1}]\tSTATS\t{self.current_best_checkpoint}\t{current_y_val:.4f}\t{err:.4g}\t{self.sample_fraction:.4f}')
        log(f'[ROUND-{self.round+1}]\tCURRENT BEST:\tCHECKPOINT-{self.current_best_checkpoint}\tY_PRED={current_y_val:.4f}\tY_ERR={100*err:.4g}%\t({100*self.sample_fraction:.2f}% sampled)')
        
    #-------------------------------------------------------------------------
    # Plotting functions
    
    def display(self, fig=None, fn=None):
        display_fig(self.run_dir, fig=fig, fn=fn)
    
    def plot_posterior_mean(self):
        plt.figure(figsize=(15, 10))
        plt.plot(self.X_feats, self.y_mean, 'b')
        plt.fill_between(self.X_feats, self.y_mean - 2*self.y_sigma, self.y_mean + 2*self.y_sigma, alpha=0.5)
        
        # true mean of all data (optional?)
        if self.Y_test is not None:
            Y_test_mean = self.Y_test.mean(axis=1)
            plt.plot(self.X_feats, Y_test_mean, 'r')
            plt.legend(['Posterior Mean', 'Confidence', 'True Mean'])
        else:
            plt.legend(['Posterior Mean', 'Confidence'])
            
        plt.title(f'Round: {self.round} - Current Best Checkpoint: {self.current_best_checkpoint}')
        self.display()
        
    def plot_task(self, j, msg='', fvals=None):
        fig, ax1 = plt.subplots(figsize=(15, 10))
        x = self.X_feats
        legend = []
        
        # Plot all data as black stars (optional?)
        if self.Y_test is not None:
            ax1.plot(x, self.Y_test[:, j], 'k*')
            legend.append('Unobserved')
        
        # Plot training (observed) data as red circles
        idx = np.where(to_numpy(self.X_train)[:,1] == j)
        xx = to_numpy(self.X_train[idx][:,0])
        # transform to original feature space
        xx = self.X_norm.inv(xx)
        yy = to_numpy(self.Y_train[idx])
        ax1.plot(xx, yy, 'ro')
        legend.append('Observed')
        
        # Plot predictive means as blue line
        ax1.plot(x, self.y_pred[:, j], 'b')
        legend.append('Posterior Mean')
        
        # confidences
        win = 2 * self.y_sig[:, j]
        ax1.fill_between(x, self.y_pred[:, j] - win, self.y_pred[:, j] + win, alpha=0.5)
        legend.append('Confidence')
        
        # Set up primary y-axis labels
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        if fvals is not None:
            # Create secondary y-axis for fvals
            ax2 = ax1.twinx()
            
            # Plot fvals as green dashed line on secondary axis
            ax2.plot(x, fvals, 'g--')
            ax2.set_ylabel('Acquisition Function Value', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            legend.append('EI')
            
            # Create custom legend with all elements
            # ax1.legend(legend, loc='best')
        # else:
            # ax1.legend(['Unobserved', 'Observed', 'Posterior Mean', 'Confidence'])
        ax1.legend(legend, loc='best')
        
        plt.title(f'Round: {self.round} - Task: {j} {msg}')
        plt.tight_layout()  # Adjust layout to make room for the second y-axis label
        self.display()
    
    def plot_all(self, max_fig=None):
        self.plot_posterior_mean()
        for j in range(self.m):
            if max_fig is not None and j > max_fig:
                break
            self.plot_task(j)
            
    #-------------------------------------------------------------------------
    
    # dampen the acquisition function in highly sampled regions
    def sample_damper(self, decay=1.0, bw_mult=25.0):
        # bandwidth for kde smoother
        bw = bw_mult * decay / self.n
        # x-axis possible sample locations
        X = self.X_test
        # matrix to hold kde estimates
        K = np.ones_like(self.S, dtype=np.float64) * np.nan
        
        # for each task compute kde
        for j in range(K.shape[1]):
            y = self.S[:, j]
            # extract sampled and unsampled points
            x, o = X[y], X[~y]
            # compute kde from sampled points
            kde = gaussian_kde(x, bw_method=bw)
            # evaluate kde at unsampled points
            v = kde(o)
            # normalize kde...
            v = v/np.sum(v)
            # and scale each task by number of sampled points in that task
            v *= len(x)/K.shape[0]
            # assign back to K
            K[~y, j] = v
        return K
    
    # compute expected improvement
    def max_expected_improvement(self, beta=0.5, decay=1.0, debug=True):
        S_mu = self.y_pred.sum(axis=-1)
        S_max_idx = np.argmax(S_mu)
        S_max = S_mu[S_max_idx]
        
        # get unsampled indices
        i_indices, j_indices = np.where(np.ones_like(self.S))
        mask = (~self.S).reshape(-1)
        
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

        if use_logei: # logEI computation (for stability)
            logh = log_h(torch.tensor(z, dtype=torch.float64)).numpy()
            ei = np.log(sig) + logh
        else: # normal EI
            ei = imp * norm.cdf(z) + sig * norm.pdf(z)
            ei = np.log(ei)
            
        # # dampen highly sampled regions
        D = self.sample_damper(decay=decay)
        d = D[valid_i, valid_j]
        
        if debug:
            EI0 = EI.copy()
            EI0[mask] = ei
            
        # shift (so non-negative) -> apply dampening -> unshift
        ei_min = ei.min()
        ei = (ei-ei_min) * (1 - decay * d**0.5) + ei_min

        # Assign computed values to valid positions and return
        EI[mask] = ei
        k = np.argmax(EI)
        next_i, next_j = i_indices[k], j_indices[k]
        
        #-----------------------------------------------------------
        # debug
        if debug:
            self.plot_task(next_j, 'ORIGINAL', EI0.reshape(self.n, self.m)[:, next_j])
            self.plot_task(next_j, '(before)', EI.reshape(self.n, self.m)[:, next_j])
        #------------------------------------------------------------
        
        # decay EI parameters
        self.ei_decay = self.ei_decay * self.ei_gamma
        self.ei_beta = self.ei_beta * self.ei_gamma
        log(f'[ROUND-{self.round+1}]\tFYI: EI beta: {self.ei_beta:.4g}', 2)
        
        #------------------------------------------------------------
        return next_i, next_j
    
    #----------------------------------------------------------------------
    # choose next sample
    
    def get_next_sample_point(self):
        
        # Maximize Expected Improvement acquisition function
        next_i, next_j = self.max_expected_improvement(beta=self.ei_beta, decay=self.ei_decay)
        self.next_i, self.next_j = next_i, next_j
        
        # convert to original X feature space
        next_checkpoint = self.X_feats[next_i]
        log(f'[ROUND-{self.round+1}]\tNEXT SAMPLE:\tCHECKPOINT-{next_checkpoint}\tTASK-{next_j}')
        
        # report task with most samples
        task_counts = np.sum(self.S, axis=0)
        max_task = np.argmax(task_counts)
        max_count = task_counts[max_task]
        log(f'[ROUND-{self.round+1}]\tFYI: TASK-{max_task} has most samples: {max_count}', 2)
        log('='*100)
        
        # return next sample point
        return next_i, next_j
    
    
    def add_next_sample(self, Y=None):
        if Y is None:
            Y = self.Y_test
        if Y is None:
            raise ValueError('Y is None, no data source for next sample')
        
        if self.next_i is None or self.next_j is None:
            # check if Y is scalar value
            if np.isscalar(Y):
                raise ValueError('Y is a scalar value, but next sample point is not chosen yet')
            self.get_next_sample_point()
            
        # get next sample indices
        next_i, next_j = self.next_i, self.next_j
        
        # acquire next observation
        if np.isscalar(Y):
            y = Y
        elif callable(Y):
            # if y is callable function, then call it with next_i, next_j
            y = Y(next_i, next_j)
        else:
            # else, access Y directly with next_i, next_j
            y = Y[next_i][next_j]
            
        # add new sample to training set (observe) and update mask
        self.X_train = torch.cat([self.X_train, torch.tensor([ [self.X_test[next_i], next_j] ], dtype=torch.float64)])
        self.Y_train = torch.cat([self.Y_train, torch.tensor([y], dtype=torch.float64).unsqueeze(-1)])
        self.Y_obs[next_i, next_j] = y
        
        # return next sample point
        return next_i, next_j


#--------------------------------------------------------------------------
# Train regression model on all data for gold standard
Y_test_reg = None

Y_test_reg = Y_test.mean(axis=1) # shortcut (just take the mean across tasks)

if Y_test_reg is None:
    sampler = BotorchSampler(S=np.ones((N, M), dtype=bool),
                            X_feats=X_feats, 
                            Y_test=Y_test,
                            lr=learning_rate,
                            eta=None,
                            max_iterations=1000,
                            min_iterations=100,
                            patience=10,
                            rank_frac=0.5,
                            log_interval=10,
                            use_cuda=use_cuda,
                            run_dir=run_dir)
    # Fit model to full dataset
    sampler.fit()
    Y_test_reg, _,_,_ = sampler.predict()

#--------------------------------------------------------------------------

# find best checkpoint
best_idx = np.argmax(Y_test.mean(axis=1))
best_y_mean = Y_test.mean(axis=1)[best_idx]
best_checkpoint = X_feats[best_idx]

i = np.argmax(Y_test_reg)
regression_best_checkpoint = X_feats[i]
regression_y_max = Y_test_reg[i]

log(f'TRU BEST CHECKPOINT:\t{best_checkpoint}\tY={best_y_mean:.4f}')
log(f'REF BEST CHECKPOINT:\t{regression_best_checkpoint}\tY={regression_y_max:.4f}')

#--------------------------------------------------------------------------

for _ in range(10):
    
    try:
        # Subsample data
        S = init_samples(N, M, n_obs, log=log)
        Y_obs = np.full(S.shape, np.nan)
        Y_obs[S] = Y_test[S]
            
        # Initialize the sampler
        sampler = BotorchSampler(X_feats=X_feats,
                                 Y_obs=Y_obs,
                                 Y_test=Y_test,
                                 lr=learning_rate,
                                 eta=eta,
                                 eta_gamma=eta_gamma,
                                 ei_beta=ei_beta,
                                 ei_gamma=ei_gamma,
                                 max_iterations=max_iterations,
                                 max_retries=max_retries,
                                 verbosity=1,
                                 max_sample_fraction=max_sample_fraction, 
                                 rank_frac=rank_frac,
                                 log_interval=log_interval,
                                 use_cuda=use_cuda,
                                 run_dir=run_dir)

        # Fit model to initial samples
        sampler.fit()
        sampler.predict()
        sampler.compare(Y_test_reg)
        # sampler.plot_all(max_fig=10)

        # Run Bayesian optimization loop
        while sampler.sample_fraction < max_sample_fraction:
            _, next_task = sampler.add_next_sample()
            sampler.fit()
            sampler.predict()
            sampler.compare(Y_test_reg)
            sampler.plot_task(next_task, '(after)')
            sampler.plot_posterior_mean()
            
        break
        
    except Exception as e:
        log(f'ERROR: {e}')
        log(traceback.format_exc())
        pass

#--------------------------------------------------------------------------