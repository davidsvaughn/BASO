import sys, os
import gpytorch.constraints
import numpy as np
import pandas as pd
import random
from scipy.stats import norm
from matplotlib import pyplot as plt
import torch
import gpytorch

from botorch.models import MultiTaskGP

from utils import task_standardize, inv_task_standardize, to_numpy, degree_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
rand_seed = -1
rank_frac = -1

#--------------------------------------------------------------------------
# SET PARAMETERS

fn = 'phi4-math-4claude.txt'
# fn = 'phi4-bw-4claude.txt'

rand_seed = 555

task_sample = 0.5 # select random subset of tasks
init_tpc    = 0.02 # number of random tasks per checkpoint (tpc) to initially sample
             # -> if fraction, tpc++ tpc until that fraction of tasks are sampled
max_sample = 0.25 # stop BO sampling after this fraction of points are sampled

# MLE estimation
max_iterations = 1000
learning_rate = 0.1
rank_frac = 0.5

# logging intervals
log_loop = 5
log_fit = 50


# rank_fraction = 0.1 # 0.1 0.25 0.5
# use_logei = True

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

# get mean at each checkpoint
V_mu = V.mean(axis=1)

# find best checkpoint
best_idx = np.argmax(V_mu)
best_checkpoint = checkpoint_nums[best_idx]
print(f'TRUE BEST CHECKPOINT: {best_checkpoint}')

# subsample data
x_idx, t_idx = init_samples(K, Z, init_tpc)

train_X = torch.tensor([ [checkpoints[i], j] for i, j in  zip(x_idx, t_idx) ], dtype=torch.float64)
train_Y = torch.tensor(V[x_idx, t_idx], dtype=torch.float64).unsqueeze(-1)

for i, j in zip(x_idx, t_idx):
    sampled_mask[i, j] = True

#--------------------------------------------------------------------------
# standardize V?  Y?

# debugging...
# y1, (mu, sig) = task_standardize(train_Y, train_X)
# y2 = inv_task_standardize(y1, train_X, mu, sig)
# assert torch.allclose(train_Y, y2)

# get FULL DATASET stats from V instead --> CHEATING!!!!!!
# task_means, task_stds = V.mean(axis=0), V.std(axis=0)

train_Y, (t_mu, t_sig) = task_standardize(train_Y, train_X)
    
# standardize V
V = (V - t_mu) / t_sig

# # reset train_Y from V (sanity check!!!)
# _train_Y = torch.tensor(V[x_idx, t_idx], dtype=torch.float64).unsqueeze(-1)
# assert torch.allclose(train_Y, _train_Y)

#--------------------------------------------------------------------------

# build data tensors
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
        self.z = int(full_x[:,1].max().item() + 1)
        self.full_x = full_x.to(self.device)
        
    def fit(self, train_x, train_y):
        z = self.z
        rank = int(self.rank_frac * z) if self.rank_frac > 0 else None
        self.model = MultiTaskGP(train_x, train_y, task_feature=-1, rank=rank).to(self.device)
        train_x, train_y = train_x.to(self.device), train_y.to(self.device)
        
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
            mean = predictions.mean
            lower, upper = predictions.confidence_region()
        y_pred = to_numpy(mean.reshape(-1, self.z))
        lower = to_numpy(lower.reshape(-1, self.z))
        upper = to_numpy(upper.reshape(-1, self.z))
        return y_pred, lower, upper
    
#--------------------------------------------------------------------------
# TRAIN MODEL

# rank = int(rank_frac * Z) if rank_frac > 0 else None # Z
# model = MultiTaskGP(train_X, train_Y, task_feature=-1, rank=rank)
# likelihood = model.likelihood

# CUDA = True
# if CUDA and torch.cuda.is_available():
#     model = model.cuda()
#     likelihood = likelihood.cuda()
    
#     train_X, train_Y = train_X.cuda(), train_Y.cuda()
#     full_X = full_X.cuda()

# # Find optimal model hyperparameters
# model.train()
# likelihood.train()

# # "Loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=likelihood, model=model)

# # Use the adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters


# degree_history = []
# window_size = 3  # Size of moving average window
# rise_patience = 3  # Number of consecutive rises to trigger stopping
# consecutive_rises = 0
# check_interval = 5  # Check degree every n iterations
# min_iterations = 50  # Minimum iterations before allowing early stopping

# training_iterations = 1000

# for i in range(training_iterations):
#     optimizer.zero_grad()
#     output = model(train_X)
    
#     try:
#         loss = -mll(output, model.train_targets)
#     except Exception as e:
#         print(f'Early stopping at iteration {i+1}: ERROR in loss calculation...')
#         print(f'ERROR:{i}: {e}')
#         break
    
#     loss.backward()
    
#     if i % check_interval == 0:

#         degree = degree_metric(model, full_X)
#         degree_history.append(degree)
        
#         # Only check for early stopping after collecting enough data points
#         if len(degree_history) >= window_size + 1 and i >= min_iterations:
#             # Calculate current and previous moving averages
#             current_avg = sum(degree_history[-window_size:]) / window_size
#             prev_avg = sum(degree_history[-(window_size+1):-1]) / window_size
            
#             # Check if the moving average is rising
#             if current_avg > prev_avg * 1.005:  # Small threshold to avoid stopping due to tiny fluctuations
#                 consecutive_rises += 1
#                 if consecutive_rises >= rise_patience:
#                     print(f'Early stopping at iteration {i+1}: degree metric rising for {consecutive_rises} consecutive checks')
#                     break
#             else:
#                 consecutive_rises = 0
        
#         if i % (check_interval*5) == 0:
#             print(f'Iter {i}/{training_iterations} - Loss: {loss.item():.3f}, avg_degree: {degree:.3f}')

#     optimizer.step()
    
# #--------------------------------------------------------------------------
# # PREDICT

# # Set into eval mode
# model.eval()
# likelihood.eval()

# # make predictions
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     predictions = likelihood(model(full_X))
#     mean = predictions.mean
#     lower, upper = predictions.confidence_region()
    

# y_pred = to_numpy(mean.reshape(-1, Z))
# lower = to_numpy(lower.reshape(-1, Z))
# upper = to_numpy(upper.reshape(-1, Z))

#--------------------------------------------------------------------------
# BOTORCH SAMPLER

sampler = BotorchSampler(full_X, sampled_mask, 
                         lr=learning_rate, 
                         max_iterations=max_iterations, 
                         max_sample=max_sample, 
                         rank_frac=rank_frac,
                         log_interval=log_fit)
sampler.fit(train_X, train_Y)
y_pred, lower, upper = sampler.predict()

#--------------------------------------------------------------------------
    
win = 0.05
for i in range(Z):
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
    
    # Shade between the lower and upper confidence bounds
    plt.fill_between(test_X, lower[:, i], upper[:, i], alpha=0.5)
    
    # set y limits??
    mean_y = test_Y[:, i].mean()
    # plt.ylim([mean_y-win, mean_y+win])
    
    plt.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.title(f'Task {i}')
    # set figsize
    plt.show()
    