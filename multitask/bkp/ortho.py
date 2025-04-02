import numpy as np
import matplotlib.pyplot as plt
import os, sys
import itertools

def orthogonal_sampling_2d(n_samples, strength=2, norm=True, jitter=True):
    """
    Generate a 2D orthogonal sampling pattern, which is a special type of
    Latin Hypercube Sample with additional orthogonality properties.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate. For orthogonal sampling, n_samples 
        should ideally be a perfect square.
    strength : int, optional
        Orthogonal array strength (default: 2).
        
    Returns:
    --------
    samples : ndarray, shape (n_samples, 2)
        Array of sample points in the unit square [0,1)×[0,1).
    """
    # For orthogonal sampling, we need n_samples to be a perfect square
    n = int(np.ceil(np.sqrt(n_samples)))
    n_squared = n * n
    
    # Create Latin Hypercube pattern first
    # Generate n divisions in each dimension
    perms = [np.random.permutation(n) for _ in range(2)]
    
    # Initialize the samples array
    samples = np.zeros((n_squared, 2))
    
    # Construct an orthogonal array of strength 2
    # For 2D, we can use a simple construction based on modular arithmetic
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            # This ensures orthogonality for strength=2
            samples[idx, 0] = perms[0][i]
            samples[idx, 1] = perms[1][(i + j) % n]
    
    # Scale to [0,1) and add jitter
    for d in range(2):
        if jitter:
            samples[:, d] += np.random.uniform(0, 1, n_squared)
        if norm:
            samples[:, d] /= n
        
    # Truncate to the requested number of samples if needed
    if n_squared > n_samples:
        samples = samples[:n_samples]
    
    return samples

def latin_hypercube_sampling_2d(n_samples, norm=True):
    """
    Generate a 2D Latin Hypercube sampling pattern.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate.
        
    Returns:
    --------
    samples : ndarray, shape (n_samples, 2)
        Array of sample points in the unit square [0,1)×[0,1).
    """
    # Initialize the samples array
    samples = np.zeros((n_samples, 2))
    
    # For each dimension, generate a Latin Hypercube sample
    for d in range(2):
        perm = np.random.permutation(n_samples)
        samples[:, d] = (perm + np.random.uniform(0, 1, n_samples))
        if norm:
            samples[:, d] /= n_samples
        
    return samples

def random_sampling_2d(n_samples, norm=False):
    """
    Generate random samples in 2D.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate.
        
    Returns:
    --------
    samples : ndarray, shape (n_samples, 2)
        Array of random sample points in the unit square [0,1)×[0,1).
    """
    x = np.random.random((n_samples, 2))
    return x

def plot_samples(samples, title="2D Sampling", show_grid=True, n=None):
    """
    Plot the 2D sample points.
    
    Parameters:
    -----------
    samples : ndarray, shape (n_samples, 2)
        Array of sample points.
    title : str, optional
        Plot title.
    show_grid : bool, optional
        Whether to show the grid divisions.
    n : int, optional
        Number of grid divisions (if None, calculated from samples).
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.7)
    
    if show_grid:
        if n is None:
            n = int(np.ceil(np.sqrt(len(samples))))
        
        # Draw grid lines
        for i in range(1, n):
            plt.axhline(i/n, color='gray', alpha=0.3, linestyle='--')
            plt.axvline(i/n, color='gray', alpha=0.3, linestyle='--')
    
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def calculate_discrepancy(samples):
    """
    Calculate the centered L2-discrepancy, a measure of uniformity.
    Lower values indicate more uniform coverage.
    
    Parameters:
    -----------
    samples : ndarray, shape (n_samples, 2)
        Array of sample points.
        
    Returns:
    --------
    discrepancy : float
        The centered L2-discrepancy value.
    """
    n = len(samples)
    sum_term1 = 0
    for i in range(n):
        xi = samples[i]
        prod = 1
        for k in range(2):
            prod *= (1 + 0.5 * abs(xi[k] - 0.5) - 0.5 * abs(xi[k] - 0.5)**2)
        sum_term1 += prod
    
    sum_term2 = 0
    for i in range(n):
        for j in range(n):
            xi = samples[i]
            xj = samples[j]
            prod = 1
            for k in range(2):
                prod *= (1 + 0.5 * abs(xi[k] - 0.5) + 0.5 * abs(xj[k] - 0.5) 
                        - 0.5 * abs(xi[k] - xj[k]))
            sum_term2 += prod
    
    term1 = (13/12)**2
    term2 = -2/n * sum_term1
    term3 = 1/(n**2) * sum_term2
    
    return np.sqrt(term1 + term2 + term3)

from math import ceil
import random

def init_samples(K, Z, init_obs):
    if init_obs >= 1:
        if init_obs < 2:
            init_obs = 2
            # log('FYI: increasing init_obs to 2 (minimum 2 obs/task allowed)')
        m = ceil(init_obs * Z)
    else:
        min_frac = 2/K # == 2*Z/(K*Z)
        if init_obs < min_frac:
            m = 2*Z
            # log(f'FYI: increasing init_obs to {min_frac:.4g} (minimum 2 obs/task allowed)')
        else:
            m = max(2*Z, ceil(init_obs * K * Z))
    # log(f'FYI: initializing sampler with {m} observations ( ~{m/(K*Z):.4g} of all obs, ~{m/Z:.4g} obs/task )\n')
    # log('-'*80)
    
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
    x, y = np.array(x), np.array(y)
                    
    sampled_mask = np.zeros((K, Z), dtype=bool)
    for i, j in zip(x, y):
        sampled_mask[i, j] = True
    
    return sampled_mask

def min_abs_diff(X, Y):
    """
    Vectorized implementation to compute minimum absolute differences.
    
    Parameters:
    X (numpy.ndarray): First input vector
    Y (numpy.ndarray): Second input vector
    
    Returns:
    numpy.ndarray: Vector of minimum absolute differences
    """
    # Reshape X and Y for broadcasting
    X_reshaped = X.reshape(-1, 1)
    Y_reshaped = Y.reshape(1, -1)
    
    # Compute all pairwise absolute differences
    abs_diffs = np.abs(X_reshaped - Y_reshaped)
    
    # Find minimum along the Y axis (axis=1)
    Xd = np.min(abs_diffs, axis=1)
    
    return Xd

def init_samples2(K, Z, init_obs):
    if init_obs >= 1:
        if init_obs < 2:
            init_obs = 2
        m = ceil(init_obs * Z)
    else:
        min_frac = 2/K # == 2*Z/(K*Z)
        if init_obs < min_frac:
            m = 2*Z
        else:
            m = max(2*Z, ceil(init_obs * K * Z))
    
    tasks = list(range(Z))
    checkpoints = list(range(K))
    chk_tasks = [[] for _ in range(K)]
    
    n = 0
    while True:
        # select a random task
        t = random.choice(tasks)
        try:
            # get checkpoints that have not been assigned this task
            p0 = np.array([x for x in checkpoints if t not in chk_tasks[x]])
            # get checkpoints that have been assigned this task, if any
            p1 = np.array([x for x in checkpoints if t in chk_tasks[x]])
            if len(p1)==0:
                k = random.choice(p0)
            else:
                w = min_abs_diff(p0, p1)**2
                # w = w / w.sum()
                # w[w<0.5] = 0.001
                # w = w**2
                k = np.random.choice(p0, p=w/w.sum())
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
    x, y = np.array(x), np.array(y)
                    
    sampled_mask = np.zeros((K, Z), dtype=bool)
    for i, j in zip(x, y):
        sampled_mask[i, j] = True
    
    return sampled_mask

def pair_dists(x):
    x = np.array(x)
    d = np.abs(x[:, None] - x[None, :])
    d = d[np.tril_indices(len(x), -1)]
    return d

def all_pair_dists(M):
    X,Y = np.where(M)
    return np.concatenate([ pair_dists(X[Y==j]) for j in range(M.shape[1]) ])

def mean_dist(M):
    return np.mean(all_pair_dists(M))

def min_dist(M):
    return np.min(all_pair_dists(M))

#--------------------------------------------------------------
# x = orthogonal_sampling_2d(25, norm=False, jitter=False)
# sys.exit()

k,z = 120, 70
n = 3

M = init_samples(k, z, n)
X,Y = np.where(M)

print(f"\nORIGINAL MEAN distance between samples: {mean_dist(M):.2f}")
print(f"ORIGINAL MIN distance between samples: {min_dist(M)}\n")

# get random permutation of column indices
p1 = np.random.permutation(z)
for i,t1 in enumerate(p1):
    p2 = np.random.permutation(p1[i+1:])
    for t2 in p2:
        t1list = np.where(M[:,t1])[0]
        t2list = np.where(M[:,t2])[0]
        # print(f"t1: {t1}, t2: {t2}")
        # print(f"t1list: {t1list}")
        # print(f"t2list: {t2list}")
        d1a = pair_dists(t1list)
        d2a = pair_dists(t2list)
        da = min(list(d1a) + list(d2a))
        # can t1list and t2list swap one element without either having 2 of the same?
        random.shuffle(t1list)
        random.shuffle(t2list)
        # for all pairs of elements in t1list and t2list, check if they can swap
        for k1, k2 in itertools.product(t1list, t2list):
            if k1 == k2:
                continue
            # swap k1 and k2
            t1list2 = np.copy(t1list)
            t1list2[np.where(t1list==k1)[0][0]] = k2
            t2list2 = np.copy(t2list)
            t2list2[np.where(t2list==k2)[0][0]] = k1
            d1b = pair_dists(t1list2)
            d2b = pair_dists(t2list2)
            db = min(list(d1b) + list(d2b))
            if db>da:
                M[k1,t1] = M[k2,t2] = False
                M[k1,t2] = M[k2,t1] = True
                # print(f"\nSwapped {k1} and {k2} in tasks {t1} and {t2}\t{db} > {da}")
                # print(f"Mean distance: {mean_dist(M):.2f}")
                # print(f"Min distance: {min_dist(M)}")
                break


# for each task, compute the distance between checkpoint samples
print(f"\nFINAL MEAN distance between samples: {mean_dist(M):.2f}")
print(f"FINAL MIN distance between samples: {min_dist(M)}\n")
    


sys.exit()
#---------------------------------------------------------------
# Example usage
# if __name__ == "__main__":
# Set the number of samples
n_samples = 25

# Generate samples using different methods
oa_samples = orthogonal_sampling_2d(n_samples)
lhs_samples = latin_hypercube_sampling_2d(n_samples)
# random_samples = random_sampling_2d(n_samples)

# Plot the samples
n_grid = int(np.ceil(np.sqrt(n_samples)))
plot_samples(oa_samples, title=f"Orthogonal Sampling (n={n_samples})", n=n_grid)
plot_samples(lhs_samples, title=f"Latin Hypercube Sampling (n={n_samples})", n=n_grid)
# plot_samples(random_samples, title=f"Random Sampling (n={n_samples})", n=n_grid)

# Calculate and print discrepancy (lower is better)
oa_disc = calculate_discrepancy(oa_samples)
lhs_disc = calculate_discrepancy(lhs_samples)
# random_disc = calculate_discrepancy(random_samples)

print(f"Discrepancy measures (lower is better):")
print(f"Orthogonal Sampling: {oa_disc:.6f}")
print(f"Latin Hypercube Sampling: {lhs_disc:.6f}")
# print(f"Random Sampling: {random_disc:.6f}")