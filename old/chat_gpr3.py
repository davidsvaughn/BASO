import numpy as np
import pandas as pd
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

# ---------------------------------------------------------------------
# 1. Load the Data
# ---------------------------------------------------------------------
# We'll assume you have a CSV with columns:
# "CHECKPOINT", "TEST_AVERAGE", "TEST_1", "TEST_2", ..., "TEST_71".
df = pd.read_csv('data/phi4-math-4claude.txt', delimiter='\t')

num_checkpoints = df.shape[0]   # e.g. 73
test_cols = [col for col in df.columns if col.startswith("TEST_") and col != "TEST_AVERAGE"]
num_tests = len(test_cols)      # e.g. 71

# We'll build a fast lookup for the 'expensive' value at (checkpoint, test).
# checkpoint, test both 1-indexed for clarity
def get_performance(cp, test):
    row_idx = cp - 1
    col_name = f"TEST_{test}"
    return df.loc[row_idx, col_name]

# ---------------------------------------------------------------------
# 2. Prepare the Domain
# ---------------------------------------------------------------------
# Our domain is all discrete (checkpoint, test) pairs. We'll represent these as 2D integer points.
all_points = []
for cp in range(1, num_checkpoints+1):
    for t in range(1, num_tests+1):
        all_points.append((cp, t))
all_points = np.array(all_points)  # shape (5183, 2) if 73*71

# We'll keep track of which points have been sampled:
sampled_points = []  # list of shape (k, 2)
sampled_values = []  # list of length k

# ---------------------------------------------------------------------
# 3. Initialize Some Random Samples
# ---------------------------------------------------------------------
init_samples = 100  # how many random points to sample as a "warm start"
chosen_init = random.sample(range(len(all_points)), init_samples)
for idx in chosen_init:
    cp, t = all_points[idx]
    y = get_performance(cp, t)
    sampled_points.append([cp, t])
    sampled_values.append(y)

# Convert to numpy arrays
X_sampled = np.array(sampled_points)
y_sampled = np.array(sampled_values)

# ---------------------------------------------------------------------
# 4. Define a GP Surrogate Model for the 2D Domain
# ---------------------------------------------------------------------
# We'll use a "product" RBF kernel by specifying a length_scale in each dimension, e.g. [10, 5].
# WhiteKernel for noise, plus a constant multiplier. Tweak these hyperparams as needed.
kernel = C(1.0, (1e-3, 1e3)) \
        * RBF(length_scale=[10.0, 5.0], length_scale_bounds=(1e-2, 1e2)) \
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))

gp = GaussianProcessRegressor(kernel=kernel, 
                              alpha=0.0, 
                              n_restarts_optimizer=5, 
                              normalize_y=True)

# ---------------------------------------------------------------------
# 5. Bayesian Optimization Loop
# ---------------------------------------------------------------------
budget = 200  # total queries you want to make
used_so_far = init_samples

while used_so_far < budget:
    
    # 5a. Fit (or refit) the GP to the data we have
    gp.fit(X_sampled, y_sampled)
    
    # 5b. Compute acquisition function (UCB in this example) over *unsampled* points
    # We'll skip those that are already sampled
    unsampled_mask = np.ones(len(all_points), dtype=bool)
    unsampled_mask[chosen_init] = False  # but let's make it dynamic
    # Actually, let's just build a set of what's sampled:
    sampled_set = set(tuple(p) for p in X_sampled)
    
    best_ucb = -1e10
    best_point = None
    
    # For efficiency, you might do this sampling in a smaller subset or approximate search
    for idx, (cp, t) in enumerate(all_points):
        if (cp, t) in sampled_set:
            continue
        
        # Evaluate GP posterior mean & stdev at this point
        # Need shape (1,2) for a single prediction
        xquery = np.array([[cp, t]])
        mu, std = gp.predict(xquery, return_std=True)
        # UCB
        alpha = 2.0  # you can adjust this for exploration
        ucb = mu[0] + alpha * std[0]
        
        if ucb > best_ucb:
            best_ucb = ucb
            best_point = (cp, t)
    
    # 5c. "Sample" the best point found by the acquisition
    if best_point is None:
        print("All points sampled or no best found. Stopping.")
        break
    chosen_cp, chosen_t = best_point
    y_new = get_performance(chosen_cp, chosen_t)
    
    # Add new observation
    X_sampled = np.vstack([X_sampled, [chosen_cp, chosen_t]])
    y_sampled = np.concatenate([y_sampled, [y_new]])
    used_so_far += 1

    if used_so_far % 20 == 0:
        print(f"Used {used_so_far} samples...")

# ---------------------------------------------------------------------
# 6. Post-processing: which checkpoint is best?
# ---------------------------------------------------------------------
# If your ultimate goal is to identify the checkpoint with the best average performance:
#   - We can estimate each checkpoint’s average performance by *predicting*
#     the performance for all test indices (1..num_tests) for that checkpoint
#     and then taking the mean of those predicted values.
checkpoint_estimates = []
for cp in range(1, num_checkpoints+1):
    test_grid = np.array([[cp, t] for t in range(1, num_tests+1)])
    mean_preds, _ = gp.predict(test_grid, return_std=True)
    cp_est_mean = np.mean(mean_preds)
    checkpoint_estimates.append(cp_est_mean)

# The best checkpoint index according to our final GP model:
best_cp_idx = np.argmax(checkpoint_estimates)
best_cp = best_cp_idx + 1  # 1-index
print(f"\nEstimated best checkpoint by GP = {best_cp} with predicted avg = {checkpoint_estimates[best_cp_idx]:.4f}")

# Compare to the "true" best from the data's TEST_AVERAGE column (if available)
true_vals = df["TEST_AVERAGE"].values
true_best_idx = np.argmax(true_vals)
print(f"True best checkpoint = {df.loc[true_best_idx, 'CHECKPOINT']}, true avg = {true_vals[true_best_idx]:.4f}")
