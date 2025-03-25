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
df = pd.read_csv('data/phi4-math-4claude.txt', delimiter='\t')

# Extract checkpoint numbers
checkpoint_nums = df['CHECKPOINT'].apply(lambda x: int(x.split('-')[1])).values   
# Identify test columns (excluding average)
test_cols = [col for col in df.columns if col.startswith('TEST_') and col != 'TEST_AVERAGE']

V = df[test_cols].values
print(V.shape)
V_mu = V.mean(axis=1)

#-------------------------------------------------------------------------
# Full data
indices = np.where(np.ones_like(V))
N = len(indices[0]) # total number of (x,z) checkpoint-task pairs
X = np.array([ [checkpoint_nums[i], j] for i, j in  zip(*indices) ])
Y = V[indices]
Z = len(test_cols) # number of tasks/validation sets

#--------------------------------------------------------------------------
# sample subset of data
n = 0.05
n_samples = int(n*N)
sample_indices = random.sample(range(N), n_samples)
X_sample = X[sample_indices]
Y_sample = Y[sample_indices]

# X_list, Y_list: each element of these lists corresponds to one of the Z tasks
# e.g. X_list[i] is the Nx1 array of x-values for task i
#      Y_list[i] is the Nx1 array of observed y-values for task i
Y_list = [Y_sample[X_sample[:,1] == i].reshape(-1,1) for i in range(Z)]
X_list = [X_sample[X_sample[:,1] == i, 0].reshape(-1,1) for i in range(Z)]
#-------------------------------------------------------------------------

# Now define kernel as a product of:
#   - a standard RBF over x
#   - a 'Coregionalize' kernel over z
# This is essentially k_x(x,x') * B[z,z']

# In GPy, you might do something like:
kern_x = GPy.kern.RBF(input_dim=1, active_dims=[0], name='rbf_x')
icm = GPy.util.multioutput.ICM(
    input_dim=1,       # dimension of the x input
    num_outputs=Z,     # number of tasks
    kernel=kern_x
)
# 'icm' is effectively k_x(x,x') * B[z,z']
#-------------------------------------------------------------------------

# Build GP model
# We pass X[:,0:1] as the continuous part, Y, plus we pass in X[:,1].astype(int) as the 'output_index'
# so the ICM kernel knows which task each row belongs to:
m = GPy.models.GPCoregionalizedRegression(
    X_list,        # the continuous input (checkpoint) 
    Y_list,        # output
    kernel=icm
)
#-------------------------------------------------------------------------

# Fit hyperparameters
m.optimize(messages=True, max_iters=100)

#-------------------------------------------------------------------------
# Predict average performance at new checkpoint x_star
# We can do: predict each task's performance, then average
x_star = np.linspace(100, 5000, 50).reshape(-1,1)  # a grid of new checkpoints

#-------------------------------------------------------------------------
# works!
# a = np.hstack([x_star, 0*np.ones_like(x_star)])
# Y_metadata = {'output_index': np.array([[0]])}
# mu, var = m.predict(a, Y_metadata=Y_metadata)
#-------------------------------------------------------------------------

mu_list = []
for z_id in range(Z):
    xtest = np.hstack([x_star, z_id*np.ones_like(x_star)])
    Y_metadata = {'output_index': np.array([[z_id]])}
    # GPy typically uses .predict(...) with continuous dims only, 
    # plus an 'output_index' argument for the tasks
    mu, var = m.predict(xtest, Y_metadata=Y_metadata)#, output_indices=z_id)
    mu_list.append(mu)

all_preds = np.stack(mu_list, axis=-1) # shape (num_x_star, Z)
avg_performance = all_preds.mean(axis=-1)  # shape (num_x_star,)

print(avg_performance)

# 'avg_performance[i]' is the predicted average across tasks at checkpoint x_star[i]