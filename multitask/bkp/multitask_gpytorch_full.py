import sys, os
import gpytorch.constraints
import numpy as np
import pandas as pd
import random
from scipy.stats import norm
from matplotlib import pyplot as plt
import gc
import torch
import gpytorch

from utils import clear_cuda_tensors

#--------------------------------------------------------------------

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
X = np.array([ [checkpoint_nums[i], j] for i, j in  zip(*indices) ])
Y = V[indices]
Z = len(test_cols) # number of tasks/validation sets

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

init_subset = 0.06

rank_fraction = 0.8

learning_rate = 0.1
max_iterations = 1000
tolerance = 1e-4
patience = 5

beta = 0.1

log_interval = 5
#--------------------------------------------------------------------------

n_samples = int(init_subset*N)
sample_indices = random.sample(range(N), n_samples)
X_sample = X[sample_indices]
Y_sample = Y[sample_indices]

# random sample one task in each checkpoint

# mark sampled data
sampled_mask[indices[0][sample_indices], indices[1][sample_indices]] = True
train_X = torch.tensor(X_sample[:,0], dtype=torch.float32)
train_T = torch.tensor(X_sample[:,1], dtype=torch.long).reshape(-1,1)
train_Y = torch.tensor(Y_sample, dtype=torch.float32)

#--------------------------------------------------------------------------

class KroneckerMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=None):
        super(KroneckerMultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        if rank is None: rank = num_tasks
        elif rank>num_tasks: rank = num_tasks
        elif rank<1: rank = int(num_tasks*rank)
        
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        
        # lengthscale_prior = gpytorch.priors.NormalPrior(0.5, 0.25)
        lengthscale_prior = gpytorch.priors.NormalPrior(5, 0.5)
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior))
        
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(),
            # gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior),
            # gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior)),
            num_tasks=num_tasks, rank=rank,
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class HadamardMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=None, ard_num_dims=None):
        
        super(HadamardMultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.RBFKernel()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        # lengthscale_prior = gpytorch.priors.NormalPrior(5.0, 0.5)
        # self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior)
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior))
        
        lengthscale_constraint=gpytorch.constraints.GreaterThan(10.0)
        self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint=lengthscale_constraint)
        
        # outputscale_constraint=gpytorch.constraints.LessThan(0.21)
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_constraint=lengthscale_constraint),
        #                                                  outputscale_constraint=outputscale_constraint,
        #                                                 )
        
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

#--------------------------------------------------------------------------
# synthetic data
import math
train_x = torch.linspace(0, 1, 100)
train_y = torch.stack([
    torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
], -1)

#--------------------------------------------------------------------------
# OR real data

# normalize V
# V = (V - V.mean()) / V.std() # globally standardize V
V = (V - V.mean(axis=0)) / V.std(axis=0) # standardize V by column
# V = (V - V.min()) / (V.max() - V.min()) # [0-1]
# V = 2*(V - V.min()) / (V.max() - V.min()) -1 # [-1,1]


SUB = 40    # select random subset of tasks
RANK = 0.9

np.random.seed(0)
idx = np.random.choice(range(Z), SUB, replace=False)
V = V[:, idx]

train_x = torch.tensor(checkpoint_nums, dtype=torch.float32)
train_y = torch.tensor(V, dtype=torch.float32)
#--------------------------------------------------------------------------

# normalize train_x
# train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min()) # min/max
# train_x = 2*train_x-1
train_x = (train_x - train_x.mean()) / train_x.std() # standardize
train_x = train_x * 1 # mult by scaling factor

numtasks = train_y.size(-1)
if RANK<1:
    RANK = int(numtasks*RANK)+1
    

# deep copy train_x and train_y
full_x = train_x.clone()
full_y = train_y.clone()

MODEL_TYPE = 2
#--------------------------------------------------------------------------
if MODEL_TYPE == 1:
    # KroneckerMultitaskGPModel
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=numtasks)
    model = KroneckerMultitaskGPModel(train_x, train_y, likelihood, numtasks, rank=RANK)
#--------------------------------------------------------------------------
else:
    # HadamardMultitaskGPModel
    indices = np.where(np.ones_like(V))
    X = np.array([ [train_x[i], j] for i, j in  zip(*indices) ])
    train_x = torch.tensor(X[:,0], dtype=torch.float32)
    train_t = torch.tensor(X[:,1], dtype=torch.long).reshape(-1,1)
    train_y = torch.tensor(train_y[indices], dtype=torch.float32)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = HadamardMultitaskGPModel((train_x, train_t), train_y, likelihood, numtasks, rank=RANK, ard_num_dims=V.shape[0])
#--------------------------------------------------------------------------

CUDA = True
if CUDA and torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()
    train_x, train_y = train_x.cuda(), train_y.cuda()
    full_x, full_y = full_x.cuda(), full_y.cuda()
    if MODEL_TYPE == 2:
        train_t = train_t.cuda()

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

# Find optimal model hyperparameters
model.train()
likelihood.train()

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

training_iterations = 500

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x) if MODEL_TYPE==1 else model(train_x, train_t)
    loss = -mll(output, train_y)
    loss.backward()
    if i % 100 == 0:
        # try to get lengthscale and outputscale
        lenscale = search_attr(model.covar_module, 'lengthscale')
        outscale = search_attr(model.covar_module, 'outputscale')
        # lenscale = model.covar_module.data_covar_module.lengthscale.item()
        # lenscale = model.covar_module.base_kernel.data_covar_module.lengthscale.item()
        # lenscale = model.covar_module.data_covar_module.base_kernel.lengthscale.item()
        # lenscale = model.covar_module.lengthscale.item()
        # lenscale = model.covar_module.base_kernel.lengthscale.item()
            
        # print('Iter %d/%d - Loss: %.3f, lengthscale: %.3f' % (i + 1, training_iterations, loss.item(), lenscale))
        print(f'Iter {i+1}/{training_iterations} - Loss: {loss.item():.3f}, lengthscale: {lenscale:.3f}, outputscale: {outscale:.3f}')
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()

# make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(train_x)) if MODEL_TYPE==1 else likelihood(model(train_x, train_t))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
    
# if model_type == 2, reshape mean, lower, upper
if MODEL_TYPE == 2:
    mean = mean.reshape(-1, numtasks)
    lower = lower.reshape(-1, numtasks)
    upper = upper.reshape(-1, numtasks)

# function to convert tensor to numpy, first to cpu if needed
def to_numpy(x):
    x = x.cpu() if x.is_cuda else x
    return x.numpy()  
    
win = 0.05
for i in range(numtasks):
    plt.figure(figsize=(15, 10))
    
    # Plot training data
    plt.plot(to_numpy(full_x), to_numpy(full_y[:, i]), 'k*')
    
    # Plot predictive means as blue line
    plt.plot(to_numpy(full_x), to_numpy(mean[:, i]), 'b')
    
    # Shade between the lower and upper confidence bounds
    plt.fill_between(to_numpy(full_x), to_numpy(lower[:, i]), to_numpy(upper[:, i]), alpha=0.5)
    
    # set y limits??
    mean_y = to_numpy(full_y[:, i]).mean()
    # plt.ylim([mean_y-win, mean_y+win])
    
    plt.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.title(f'Task {i}')
    # set figsize
    plt.show()
    