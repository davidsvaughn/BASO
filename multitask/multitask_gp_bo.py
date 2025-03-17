import sys, os
import numpy as np
import pandas as pd
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import GPy

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
rand_seed = 286 #  286 111
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

# X_list, Y_list: each element of these lists corresponds to one of the Z tasks
# e.g. X_list[i] is the Nx1 array of x-values for task i
#      Y_list[i] is the Nx1 array of observed y-values for task i
Y_list = [Y_sample[X_sample[:,1] == i].reshape(-1,1) for i in range(Z)]
X_list = [X_sample[X_sample[:,1] == i, 0].reshape(-1,1) for i in range(Z)]


#--------------------------------------------------------------------------
method = 1
#-------------------------------------------------------------------------
# method 1 : https://chatgpt.com/g/g-p-67d111a602648191a7250608ce8c7637-gaussianprocessmodelopt/c/67d113be-6634-8011-a028-f2e6fa7926ce

if method == 1:
    # Now define kernel as a product of:
    #   - a standard RBF over x
    #   - a 'Coregionalize' kernel over z
    # This is essentially k_x(x,x') * B[z,z']

    kern_x = GPy.kern.RBF(input_dim=1, active_dims=[0], name='rbf_x')
    icm = GPy.util.multioutput.ICM(
        input_dim=1,       # dimension of the x input
        num_outputs=Z,     # number of tasks
        kernel=kern_x
    )

    m = GPy.models.GPCoregionalizedRegression(
        X_list,        # the continuous input (checkpoint) 
        Y_list,        # output
        kernel=icm
    )

#--------------------------------------------------------------------------
# method 2 : https://chat.deepseek.com/a/chat/s/5bcc33da-ff57-43cc-8749-66d467dfa7db

elif method == 2:

    # Define kernel with low-rank task correlations and task-specific noise
    k_rbf = GPy.kern.RBF(input_dim=1, active_dims=[0], name='rbf_x')
    k_coreg = GPy.kern.Coregionalize(input_dim=1, output_dim=Z, rank=Z//8, active_dims=[1], name='coreg_z')  # Rank-1 task correlations
    k_white = GPy.kern.White(1, active_dims=[1], name='white_z')  # Task-specific noise
    kernel = k_rbf * k_coreg + k_white

    # Create model
    m = GPy.models.GPCoregionalizedRegression(
        X_list=X_list, 
        Y_list=Y_list,
        kernel=kernel
    )

#-------------------------------------------------------------------------

# Fit hyperparameters
m.optimize(messages=True, max_iters=1000)

#-------------------------------------------------------------------------
# Predict average performance at new checkpoint x_star
# We can do: predict each task's performance, then average
# x_star = np.linspace(100, 5000, 50).reshape(-1,1)  # a grid of new checkpoints
x_star = checkpoint_nums.reshape(-1,1)  # a grid of new checkpoints

#-------------------------------------------------------------------------
# works!
# a = np.hstack([x_star, 0*np.ones_like(x_star)])
# Y_metadata = {'output_index': np.array([[0]])}
# mu, var = m.predict(a, Y_metadata=Y_metadata)
#-------------------------------------------------------------------------

mu_list, var_list = [], []
for z_id in range(Z):
    xtest = np.hstack([x_star, z_id*np.ones_like(x_star)])
    Y_metadata = {'output_index': np.array([[z_id]])}
    # GPy typically uses .predict(...) with continuous dims only, 
    # plus an 'output_index' argument for the tasks
    mu, var = m.predict(xtest, Y_metadata=Y_metadata)#, output_indices=z_id)
    mu_list.append(mu)
    var_list.append(var)

mu_mat = np.stack(mu_list, axis=-1) # shape (num_x_star, Z)
var_mat = np.stack(var_list, axis=-1) # shape (num_x_star, Z)
avg_performance = mu_mat.mean(axis=-1)  # shape (num_x_star,)

print(avg_performance)

# find best predicted checkpoint
best_pred_idx = np.argmax(avg_performance)
best_pred_checkpoint = int(x_star[best_pred_idx][0])

print(f'\nBest checkpoint: {best_checkpoint}')
print(f'Best predicted checkpoint: {best_pred_checkpoint}')

# 'avg_performance[i]' is the predicted average across tasks at checkpoint x_star[i]

#--------------------------------------------------------------------------

S_mu = mu_mat.sum(axis=-1)
S_var = var_mat.sum(axis=-1)

unsampled_indices = np.where(~sampled_mask)
X_unsampled = np.array([ [checkpoint_nums[i], j] for i, j in  zip(*unsampled_indices) ])
Y_unsampled = V[unsampled_indices]

print()
