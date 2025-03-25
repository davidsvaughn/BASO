import sys, os
import numpy as np
import pandas as pd
import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import GPy

import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------
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
df = pd.read_csv(os.path.join(data_dir, 'phi4-math-4claude.txt'), delimiter='\t')
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

#--------------------------------------------------------------------------
# sample subset of data

rand_seed = np.random.randint(100, 1000)
rand_seed = 605 #   605 286 111
print(f'Random seed: {rand_seed}')
np.random.seed(rand_seed)
random.seed(rand_seed)

n = 0.05
n_samples = int(n*N)
sample_indices = random.sample(range(N), n_samples)
X_sample = X[sample_indices]
Y_sample = Y[sample_indices]

# mark sampled data
sampled_mask[indices[0][sample_indices], indices[1][sample_indices]] = True


full_train_X = torch.tensor(X_sample[:,0], dtype=torch.float32)
full_train_T = torch.tensor(X_sample[:,1], dtype=torch.long).reshape(-1,1)
full_train_Y = torch.tensor(Y_sample, dtype=torch.float32)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=None):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        if rank is None:
            rank = num_tasks
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)

    def forward(self,x,i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
    
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = MultitaskGPModel((full_train_X, full_train_T), full_train_Y, likelihood, Z, rank=Z//4)

# set to train mode
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 100

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(full_train_X, full_train_T)
    loss = -mll(output, full_train_Y)
    loss.backward()
    print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.item()))
    optimizer.step()

# Set to eval mode
model.eval()
likelihood.eval()

full_test_X = torch.tensor(X[:,0], dtype=torch.float32)
full_test_T = torch.tensor(X[:,1], dtype=torch.long).reshape(-1,1)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_y = likelihood(model(full_test_X, full_test_T))
    
y_mean = observed_pred_y.mean.detach().numpy()
y_var = observed_pred_y.variance.detach().numpy()

# reshape y_mean back into (num_x, num_z)
y_mean = y_mean.reshape((len(checkpoint_nums), Z))
y_var = y_var.reshape((len(checkpoint_nums), Z))

# find best predicted checkpoint
avg_performance = y_mean.mean(axis=-1)
print(avg_performance)

plt.plot(checkpoint_nums, avg_performance)
plt.show()

best_pred_idx = np.argmax(avg_performance)
best_pred_checkpoint = int(checkpoint_nums[best_pred_idx])

print(f'\nBest checkpoint: {best_checkpoint}')
print(f'Best predicted checkpoint: {best_pred_checkpoint}')

#-------------------------------------------------------------------------

S_mu = y_mean.sum(axis=-1)
# S_var = var_mat.sum(axis=-1)

S_max = S_mu[best_pred_idx]
S_max_idx = best_pred_idx

unsampled_indices = np.where(~sampled_mask)
# X_unsampled = np.array([ [checkpoint_nums[i], j] for i, j in  zip(*unsampled_indices) ])
# X_unsampled = np.array([ [i, j] for i, j in  zip(*unsampled_indices) ])
X_unsampled = np.array(unsampled_indices).T
# Y_unsampled = V[unsampled_indices]

beta = 0.1
EI = []
for i,j in zip(*unsampled_indices):
    mu = y_mean[i,j]
    sig = y_var[i,j]**0.5
    sx = y_mean[i].sum() - mu
    s_max = S_max - sx
    imp = mu - s_max - beta
    z = imp/sig
    ei = imp * norm.cdf(z) + sig * norm.pdf(z)
    EI.append(ei)

EI = np.array(EI)
best_idx = np.argmax(EI)
best_i, best_j = X_unsampled[best_idx]
best_checkpoint = checkpoint_nums[best_i]

print(f'\nBest checkpoint: {best_checkpoint}')

