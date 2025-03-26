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

# class MultitaskGPModel_old(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, num_tasks, rank=None):
#         super(MultitaskGPModel_old, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         # self.covar_module = gpytorch.kernels.RBFKernel()
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#         # https://docs.gpytorch.ai/en/v1.12/examples/00_Basic_Usage/Hyperparameters.html#Priors
#         # lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
#         # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior,))
#         #--------------------------------------------------------------------------
#         # We learn an IndexKernel for 2 tasks
#         # (so we'll actually learn 2x2=4 tasks with correlations)
#         if rank is None: rank = num_tasks
#         elif rank > num_tasks: rank = num_tasks
#         elif rank < 1: rank = int(num_tasks*rank)
#         # self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)
#         self.task_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank))

#     def forward(self,x,i):
#         mean_x = self.mean_module(x)
#         # Get input-input covariance
#         covar_x = self.covar_module(x)
#         # Get task-task covariance
#         covar_i = self.task_covar_module(i)
#         # Multiply the two together to get the covariance we want
#         covar = covar_x.mul(covar_i)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar)

#--------------------------------------------------------------------------

# lengthscale_constraint=gpytorch.constraints.GreaterThan(1.0)
# lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 1.0)  # Higher alpha/beta ratio = larger values

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=None):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        if rank is None: rank = num_tasks
        elif rank>num_tasks: rank = num_tasks
        elif rank<1: rank = int(num_tasks*rank)
        
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        
        lengthscale_prior = gpytorch.priors.NormalPrior(0.5, 0.1)
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

#--------------------------------------------------------------------------
# test data
import math
train_x = torch.linspace(0, 1, 100)
train_y = torch.stack([
    torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
], -1)

# real data
train_x = torch.tensor(checkpoint_nums, dtype=torch.float32)
train_y = torch.tensor(V, dtype=torch.float32)

# min/max normalize train_x
train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
# mult by 10
train_x = train_x * 5


numtasks = train_y.size(-1)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=numtasks)
model = MultitaskGPModel(train_x, train_y, likelihood, numtasks)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 1000

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        try:
            lenscale = model.covar_module.data_covar_module.lengthscale.item()
        except:
            try:
                lenscale = model.covar_module.base_kernel.data_covar_module.lengthscale.item()
            except:
                lenscale = model.covar_module.data_covar_module.base_kernel.lengthscale.item()
            
        print('Iter %d/%d - Loss: %.3f, lengthscale: %.3f' % (i + 1, training_iterations, loss.item(), lenscale))
    optimizer.step()
    
# print lengthscale of RBF kernel
print(model.covar_module.data_covar_module.lengthscale)

# Set into eval mode
model.eval()
likelihood.eval()

# make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = train_x[::2]
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    # lower, upper = predictions.confidence_region()


for i in range(numtasks):
    plt.figure(figsize=(15, 10))
    # Plot training data
    plt.plot(train_x.detach().numpy(), train_y[:, i].detach().numpy(), 'k*')
    # Plot predictive means as blue line
    plt.plot(test_x.detach().numpy(), mean[:, i].detach().numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    # plt.fill_between(test_x.detach().numpy(), lower[:, i].detach().numpy(), upper[:, i].detach().numpy(), alpha=0.5)
    # plt.ylim([0.7, 0.9])
    plt.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.title(f'Task {i}')
    # set figsize
    plt.show()
    
# dsv
# if numtasks == 2:
#     # Initialize plots
#     f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

#     # Make predictions
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         test_x = torch.linspace(0, 1, 51)
#         predictions = likelihood(model(test_x))
#         mean = predictions.mean
#         lower, upper = predictions.confidence_region()

#     # Plot training data as black stars
#     y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
#     # Predictive mean as blue line
#     y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
#     # Shade in confidence
#     y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
#     y1_ax.set_ylim([-3, 3])
#     y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
#     y1_ax.set_title('Observed Values (Likelihood)')

#     # Plot training data as black stars
#     y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
#     # Predictive mean as blue line
#     y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
#     # Shade in confidence
#     y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
#     y2_ax.set_ylim([-3, 3])
#     y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
#     y2_ax.set_title('Observed Values (Likelihood)')
#     plt.show()
