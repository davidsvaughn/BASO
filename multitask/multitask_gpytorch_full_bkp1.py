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

lengthscale_constraint=gpytorch.constraints.GreaterThan(1.0)
lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 1.0)  # Higher alpha/beta ratio = larger values

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=None):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        if rank is None: rank = num_tasks
        elif rank>num_tasks: rank = num_tasks
        elif rank<1: rank = int(num_tasks*rank)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(),
            # gpytorch.kernels.RBFKernel(lengthscale_constraint=lengthscale_constraint),
            # gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior),
            # gpytorch.kernels.MaternKernel(),#nu=2.5),
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
train_x = train_x * 10


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

training_iterations = 800

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        print('Iter %d/%d - Loss: %.3f, lengthscale: %.3f' % (i + 1, training_iterations, loss.item(), model.covar_module.data_covar_module.lengthscale.item()))
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

#--------------------------------------------------------------------------
class MultitaskGPSequentialSampler:
    def __init__(self, num_tasks, rank, 
                 max_iterations=100, 
                 learning_rate=0.1, 
                 beta=0.05,
                 lr_scheduler='step',
                 lr_gamma=0.9,
                 lr_step_size=50,
                 lr_min=1e-3,
                 ):
        self.num_tasks = num_tasks
        self.rank = rank
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.beta = beta
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learning rate schedule parameters
        self.lr_scheduler = lr_scheduler  # 'step', 'exp', 'cosine', 'plateau'
        self.lr_gamma = lr_gamma  # multiplicative factor for lr decay
        self.lr_step_size = lr_step_size  # iterations between lr updates
        self.lr_min = lr_min  # minimum learning rate
        
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
        
        # Set up learning rate scheduler
        if self.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        elif self.lr_scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)
        elif self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iterations, eta_min=self.lr_min)
        elif self.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.lr_gamma, 
                                                                patience=patience//2, threshold=tolerance, verbose=True)
        else:  # No scheduler
            scheduler = None
        
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
            
            # Update learning rate
            if scheduler is not None:
                if self.lr_scheduler == 'plateau':
                    scheduler.step(loss.item())
                else:
                    scheduler.step()
            
            # Check convergence
            current_loss = loss.item()
            rel_improvement = (prev_loss - current_loss) / (abs(prev_loss) + 1e-10)
            
            if i % 50 == 0:  # Log occasionally
                current_lr = optimizer.param_groups[0]['lr']
                # print(f'Iter {i} - Loss: {current_loss:.5f}, Improvement: {rel_improvement:.5f}')
                print(f'Iter {i} - Loss: {current_loss:.5f}, Improvement: {rel_improvement:.5f}, LR: {current_lr:.6f}')
                
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
