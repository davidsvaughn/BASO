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

#--------------------------------------------------------------

def clear_cuda_tensors(target_size=None): # (1, 8192, 32, 96)
    """Clear tensors of specific size from memory"""
    
    # check if cuda is being used
    if not torch.cuda.is_available():
        return
    
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                if target_size is None or obj.size() == target_size:
                    del obj
                    count += 1
        except: 
            pass
    
    torch.cuda.empty_cache()
    gc.collect()
    # printmain(f"Cleared {count} tensors")
    print(f"Cleared {count} tensors")

# ---------------------------------------------------------------------
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
# rand_seed = 605 #   605 286 111
print(f'Random seed: {rand_seed}')
np.random.seed(rand_seed)
random.seed(rand_seed)

#--------------------------------------------------------------------------
# set parameters

init_subset = 0.05

rank_fraction = 0.8

learning_rate = 0.1
max_iterations = 1000
tolerance = 1e-4
patience = 5

beta = 0.5

log_interval = 5
#--------------------------------------------------------------------------

n_samples = int(init_subset*N)
sample_indices = random.sample(range(N), n_samples)
X_sample = X[sample_indices]
Y_sample = Y[sample_indices]

# mark sampled data
sampled_mask[indices[0][sample_indices], indices[1][sample_indices]] = True
train_X = torch.tensor(X_sample[:,0], dtype=torch.float32)
train_T = torch.tensor(X_sample[:,1], dtype=torch.long).reshape(-1,1)
train_Y = torch.tensor(Y_sample, dtype=torch.float32)

# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = MultitaskGPModel((train_X, train_T), train_Y, likelihood, Z, rank=rank)
# # Use the adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Includes GaussianLikelihood parameters
# # "Loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=None):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        
        # self.covar_module = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # https://docs.gpytorch.ai/en/v1.12/examples/00_Basic_Usage/Hyperparameters.html#Priors
        
        #--------------------------------------------------------------------------
        # self.covar_module.register_constraint("raw_lengthscale", gpytorch.constraints.positive.Positive(lower_bound=0.5))
        # self.covar_module.register_constraint("raw_outputscale", gpytorch.constraints.Posipositive.Positivetive(lower_bound=0.5))
        # self.mean_module.register_constraint("raw_offset", gpytorch.constraints.Positive(lower_bound=1.0))
        # gpytorch.constraints.positive.Positive(lower_bound=1.0)
        
        #---------
        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        if rank is None:
            rank = num_tasks
        elif rank > num_tasks:
            rank = num_tasks
        elif rank < 1:
            rank = int(num_tasks*rank)
            
        # self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)
        self.task_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank))

    def forward(self,x,i):
        mean_x = self.mean_module(x)
        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

#--------------------------------------------------------------------------
class MultitaskGPSequentialSampler:
    def __init__(self, num_tasks, rank, max_iterations=100, learning_rate=0.1, beta=0.05):
        self.num_tasks = num_tasks
        self.rank = rank
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.beta = beta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #--------------------------------------------------------------------------
    def fit(self, train_x, train_t, train_y, tolerance=tolerance, patience=patience):
        train_x = train_x.to(self.device)
        train_t = train_t.to(self.device)
        train_y = train_y.to(self.device)
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = MultitaskGPModel((train_x, train_t), train_y, self.likelihood, self.num_tasks, rank=self.rank).to(self.device)
        self.model.train()
        self.likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # For tracking convergence
        prev_loss = float('inf')
        stall_counter = 0
        
        for i in range(self.max_iterations):
            optimizer.zero_grad()
            output = self.model(train_x, train_t)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
            # Check convergence
            current_loss = loss.item()
            rel_improvement = (prev_loss - current_loss) / (abs(prev_loss) + 1e-10)
            
            if i % 100 == 0:  # Log occasionally
                print(f'Iter {i} - Loss: {current_loss:.5f}, Improvement: {rel_improvement:.5f}')
                
            if rel_improvement < tolerance:
                stall_counter += 1
            else:
                stall_counter = 0
                
            if stall_counter >= patience:
                print(f'Stopping early at iteration {i} - Loss converged')
                break
                
            prev_loss = current_loss
    
    #--------------------------------------------------------------------------
    
    def predict(self, test_x, test_t):
        test_x = test_x.to(self.device)
        test_t = test_t.to(self.device)
        
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_pred = self.likelihood(self.model(test_x, test_t))
        return y_pred

#--------------------------------------------------------------------------
sampler = MultitaskGPSequentialSampler(num_tasks=Z, 
                                       rank=rank_fraction,
                                       max_iterations=max_iterations, 
                                       learning_rate=learning_rate, beta=beta)
#--------------------------------------------------------------------------

step = 0
while True:
    step += 1
    # compute percent of data sampled
    fraction_sampled = sampled_mask.sum() / N
    
    if fraction_sampled >= 0.2:
        break

    sampler.fit(train_X, train_T, train_Y)

    y_pred = sampler.predict(full_test_X, full_test_T)
    y_mean = y_pred.mean.detach().cpu().numpy()
    y_var = y_pred.variance.detach().cpu().numpy()

    # reshape y_mean back into (num_x, num_z)
    y_mean = y_mean.reshape((len(checkpoint_nums), Z))
    y_var = y_var.reshape((len(checkpoint_nums), Z))
    S_var = y_var.sum(axis=-1)

    # find best predicted checkpoint
    avg_performance = y_mean.mean(axis=-1)
    # print(avg_performance)

    best_pred_idx = np.argmax(avg_performance)
    best_pred_checkpoint = int(checkpoint_nums[best_pred_idx])

    # print(f'\nBest checkpoint: {best_checkpoint}')
    print(f'\nStep {step}: % sampled={100*fraction_sampled:.2f}%')
    print(f'\tbest predicted checkpoint:\t{best_pred_checkpoint} (best checkpoint: {best_checkpoint})')
    
    if step % log_interval == 0:
        clear_cuda_tensors()
        plt.plot(checkpoint_nums, avg_performance)
        # set y axis between .83 and .85
        plt.ylim([0.83, 0.85])
        # show confidence using S_var (sum of variances across tasks)
        S_sigma = S_var**0.5
        # plt.fill_between(checkpoint_nums, avg_performance - S_var, avg_performance + S_var, alpha=0.5)
        # plt.fill_between(checkpoint_nums, avg_performance - S_sigma, avg_performance + S_sigma, alpha=0.5)
        plt.xlabel('Checkpoint')
        plt.ylabel('Average Performance')
        plt.title('Predicted Average Performance')
        plt.show()

    S_mu = y_mean.sum(axis=-1)
    S_max = S_mu[best_pred_idx]
    S_max_idx = best_pred_idx

    unsampled_indices = np.where(~sampled_mask)
    # X_unsampled = np.array([ [i, j] for i, j in  zip(*unsampled_indices) ])
    X_unsampled = np.array(unsampled_indices).T

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
    next_idx = np.argmax(EI)
    next_i, next_j = X_unsampled[next_idx]
    next_checkpoint = checkpoint_nums[next_i]

    print(f'\tnext sample - checkpoint:\t{next_checkpoint}, task={next_j}')
    sampled_mask[next_i, next_j] = True
    train_X = torch.cat([train_X, torch.tensor([checkpoint_nums[next_i]], dtype=torch.float32)])
    train_T = torch.cat([train_T, torch.tensor([next_j], dtype=torch.long).reshape(-1,1)])
    train_Y = torch.cat([train_Y, torch.tensor([V[next_i, next_j]], dtype=torch.float32)])
