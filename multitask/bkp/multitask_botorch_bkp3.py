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

from utils import clear_cuda_tensors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
rand_seed = -1

#--------------------------------------------------------------------------
# SET PARAMETERS

fn = 'phi4-math-4claude.txt'
# fn = 'phi4-bw-4claude.txt'

# rand_seed = 333

task_sample = 1 # select random subset of tasks
init_tpc = 0.05 # number of random tasks per checkpoint (tpc) to initially sample
             # -> if fraction, tpc++ tpc until that fraction of tasks are sampled
max_sample = 0.25 # stop BO sampling after this fraction of points are sampled

# MLE estimation
max_iterations = 1000
learning_rate = 0.1

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
if task_sample>0:
    if task_sample <= 1:
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
            t = random.choice([t for t in tasks if t not in chk_tasks[k]])
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

# N = len(full_indices[0]) # total number of (x,z) checkpoint-task pairs (N = K*Z)
# X = np.array([ [checkpoints[i], j] for i, j in  zip(*full_indices) ])
# Y = V[full_indices]

# Full test data grid (all-checkpoints X all-tasks)
# full_test_X = torch.tensor(X[:,0], dtype=torch.float64).to(device)
# full_test_T = torch.tensor(X[:,1], dtype=torch.long).reshape(-1,1).to(device)

# get mean at each checkpoint
V_mu = V.mean(axis=1)

# find best checkpoint
best_idx = np.argmax(V_mu)
best_checkpoint = checkpoint_nums[best_idx]
print(f'TRUE BEST CHECKPOINT: {best_checkpoint}')

# sampled_indices = init_samples(K, Z, init_tpc)
x_idx, t_idx = init_samples(K, Z, init_tpc)
print()

# train_X = torch.tensor(checkpoints[x_idx], dtype=torch.float64)
# train_T = torch.tensor(t_idx, dtype=torch.long).reshape(-1,1)

train_X = torch.tensor([ [checkpoints[i], j] for i, j in  zip(x_idx, t_idx) ], dtype=torch.float64)
train_Y = torch.tensor(V[x_idx, t_idx], dtype=torch.float64).unsqueeze(-1)

for i, j in zip(x_idx, t_idx):
    sampled_mask[i, j] = True

# standardize V?  Y?
def task_mu_sigma(train_Y, t_idx):
    task_means, task_stds, ys = [], [], []
    Z = t_idx.max() + 1
    for i in range(Z):
        y = train_Y[t_idx==i].squeeze()
        mu = y.mean()
        task_means.append(mu)
        ys.extend(y-mu)
    sigma = np.std(ys)
    task_stds = np.array([sigma for _ in range(Z)])
    task_means = np.array(task_means)
    return task_means, task_stds
task_means, task_stds = task_mu_sigma(train_Y, t_idx)

# get mu,sig from V instead
task_means, task_stds = V.mean(axis=0), V.std(axis=0)


# standardize train_Y
for i in range(Z):
    train_Y[train_X[:,1]==i] = (train_Y[train_X[:,1]==i] - task_means[i]) / task_stds[i]
    
# standardize V
V = (V - task_means) / task_stds

# reset train_Y from V (sanity check)
_train_Y = torch.tensor(V[x_idx, t_idx], dtype=torch.float64).unsqueeze(-1)
assert torch.allclose(train_Y, _train_Y)

# build full data tensors
full_X = torch.tensor([ [checkpoints[i], j] for i, j in  zip(*full_indices) ], dtype=torch.float64)

test_X = np.array(checkpoints)
test_Y = np.array(V)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# OR real data

# standardize V
v_means = V.mean(axis=0) # column-wise mean
v_sigmas = V.std(axis=0) # column-wise std
# v_sigmas = v_sigmas.mean()
# v_sigmas = V.std()

# V = (V - v_means) / v_sigmas # standardize V by column

# normalize V
# V = (V - V.mean()) / V.std() # globally standardize V
# V = (V - V.mean(axis=0)) / V.std(axis=0) # standardize V by column
# V = (V - V.min()) / (V.max() - V.min()) # [0-1]
# V = 2*(V - V.min()) / (V.max() - V.min()) -1 # [-1,1]

SUBTASK = 1    # select random subset of tasks
SAMPLE  = 0.05 # select random subset of data points

if SUBTASK <= 1:
    SUBTASK = int(SUBTASK * Z)

# np.random.seed(0)
idx = np.random.choice(range(Z), SUBTASK, replace=False)
V = V[:, idx]

train_x = torch.tensor(checkpoint_nums, dtype=torch.float64)
train_y = torch.tensor(V, dtype=torch.float64)
#--------------------------------------------------------------------------

# normalize train_x
train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min()) # min/max
# train_x = 2*train_x-1
# train_x = (train_x - train_x.mean()) / train_x.std() # standardize
train_x = train_x * 1 # mult by scaling factor


RANK = 0.5
numtasks = train_y.size(-1)
if RANK<1:
    RANK = min(int(numtasks*RANK)+1, numtasks)
    

# deep copy train_x and train_y
full_x = train_x.clone()
full_y = train_y.clone()

MODEL_TYPE = 3
#--------------------------------------------------------------------------
# if MODEL_TYPE == 1:
#     # KroneckerMultitaskGPModel
#     likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=numtasks)
#     # model = KroneckerMultitaskGPModel(train_x, train_y, likelihood, numtasks, rank=RANK)
#     full_x = train_x.clone()
# #--------------------------------------------------------------------------
# else:

# HadamardMultitaskGPModel
indices = np.where(np.ones_like(V))
X = np.array([ [train_x[i], j] for i, j in  zip(*indices) ])
train_x = torch.tensor(X[:,0], dtype=torch.float64)
train_t = torch.tensor(X[:,1], dtype=torch.long).reshape(-1,1)
train_y = torch.tensor(train_y[indices], dtype=torch.float64)
full_x = train_x.clone()

# if MODEL_TYPE == 2:
#     likelihood = gpytorch.likelihoods.GaussianLikelihood()
#     # model = HadamardMultitaskGPModel((train_x, train_t), train_y, likelihood, numtasks, rank=RANK, ard_num_dims=V.shape[0])

# else: # BoTorch version !!!

train_x = torch.tensor(X, dtype=torch.float64)
train_y = train_y.unsqueeze(-1)
full_x = train_x.clone()

# random subsample
n = int(SAMPLE * len(train_x))
idx = np.random.choice(range(len(train_x)), n, replace=False)
train_x = train_x[idx]
train_y = train_y[idx]

model = MultiTaskGP(train_x, train_y, task_feature=-1, rank=RANK)
likelihood = model.likelihood


#--------------------------------------------------------------------------

CUDA = True
if CUDA and torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()
    
    train_x, train_y = train_x.cuda(), train_y.cuda()
    full_x, full_y = full_x.cuda(), full_y.cuda()
    
    
    
    # if MODEL_TYPE == 2:
    #     train_t = train_t.cuda()

#--------------------------------------------------------------------------
# function to search all attributes of a parameter recursively until finding an attribute name
def search_attr(obj, attr, default=0):
    if hasattr(obj, attr) and getattr(obj, attr) is not None:
        val = getattr(obj, attr)
        try:
            return val.item()
        except:
            return val
    else:
        for subobj in obj.children():
            res = search_attr(subobj, attr)
            if res is not None:
                return res
    return default

# function to convert tensor to numpy, first to cpu if needed
def to_numpy(x):
    x = x.cpu() if x.is_cuda else x
    return x.numpy() 

from multitask.degree import count_line_curve_intersections

# degree metric
def degree_metric(model, X, verbose=False):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X))
    mean = pred.mean.reshape(-1, numtasks)
    model.train()
    likelihood.train()
    
    degrees = []
    for i in range(numtasks):
        y = to_numpy(mean[:, i])
        x = to_numpy(X[X[:,1]==i][:,0])
        d = count_line_curve_intersections(x, y)
        # plt.plot(x, y)
        # plt.show()
        degrees.append(d)
    avg_degree = np.mean(degrees)
    # show histogram
    if verbose:
        print(f'Average degree: {avg_degree}')
        plt.hist(degrees, bins=np.ptp(degrees)+1)
        plt.show()
    return avg_degree

#--------------------------------------------------------------------------
# Find optimal model hyperparameters
model.train()
likelihood.train()

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=likelihood, model=model)

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters


degree_history = []
window_size = 3  # Size of moving average window
rise_patience = 3  # Number of consecutive rises to trigger stopping
consecutive_rises = 0
check_interval = 5  # Check degree every n iterations
min_iterations = 50  # Minimum iterations before allowing early stopping

training_iterations = 1000

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x, train_t) if MODEL_TYPE==2 else model(train_x)
    loss = -mll(output, model.train_targets) #train_y)
    loss.backward()
    
    if i % check_interval == 0:

        degree = degree_metric(model, full_x)
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
        
        if i % (check_interval*5) == 0:
            print(f'Iter {i}/{training_iterations} - Loss: {loss.item():.3f}, avg_degree: {degree:.3f}')

    optimizer.step()
    
#--------------------------------------------------------------------------

# Set into eval mode
model.eval()
likelihood.eval()

# make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(train_x, train_t)) if MODEL_TYPE==2 else likelihood(model(full_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
    
# if model_type == 2, reshape mean, lower, upper
if MODEL_TYPE >= 2:
    mean = mean.reshape(-1, numtasks)
    lower = lower.reshape(-1, numtasks)
    upper = upper.reshape(-1, numtasks) 
    
win = 0.05
for i in range(numtasks):
    plt.figure(figsize=(15, 10))
    
    # Plot full data
    # plt.plot(to_numpy(full_x), to_numpy(full_y[:, i]), 'k*')
    plt.plot(test_X, test_Y[:, i], 'k*')
    
    # Plot training data as red circles
    # find indices of train_x where 2nd column is i
    idx = np.where(to_numpy(train_x)[:,1] == i)
    plt.plot(to_numpy(train_x[idx][:,0]), to_numpy(train_y[idx]), 'ro')
    
    xx = to_numpy(train_x[idx][:,0])
    yy = to_numpy(train_y[idx])
    # check that xx,yy are in test_X, test_Y[:, i]
    for x, y in zip(xx, yy):
        for xxx,yyy in zip(test_X, test_Y[:, i]):
            if abs(x-xxx) < 1e-6 and abs(y-yyy) < 1e-6:
                break
        else:
            print('ERROR: train point not in test data')
    
    # Plot predictive means as blue line
    plt.plot(test_X, to_numpy(mean[:, i]), 'b')
    
    # Shade between the lower and upper confidence bounds
    plt.fill_between(test_X, to_numpy(lower[:, i]), to_numpy(upper[:, i]), alpha=0.5)
    
    # set y limits??
    mean_y = test_Y[:, i].mean()
    # plt.ylim([mean_y-win, mean_y+win])
    
    plt.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.title(f'Task {i}')
    # set figsize
    plt.show()
    