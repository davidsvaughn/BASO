import numpy as np
import matplotlib.pyplot as plt
import os, sys
import itertools
from math import ceil
import random

#--------------------------------------------------------------

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

def maximize_min_dist(M, log=None):
    if log is None:
        log = lambda x: print(x)
    if log: log(f"ORIGINAL  starting samples - pairwise distance stats: min={min_dist(M)}\tmean={mean_dist(M):.4g}")

    # get random permutation of column indices
    p1 = np.random.permutation(M.shape[1])
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
                
    if log: log(f"OPTIMIZED starting samples - pairwise distance stats: min={min_dist(M)}\tmean={mean_dist(M):.4g}")
    return M

def init_samples(N, M, n_obs, log=None, optimize=False):
    """
    Initialize samples for N checkpoints and M tasks.
    
    Parameters:
    N (int): Number of checkpoints
    M (int): Number of tasks
    n_obs (float):  if >=1, Initial number of observations (i.e. checkpoints) per task
                if <1, Initial fraction of full sample space (KxZ)
    log (function, optional): Logging function to print messages
    """
    if n_obs >= 1:
        if n_obs < 2:
            n_obs = 2
            if log: log('FYI: increasing n_obs to 2 (minimum 2 obs/task allowed)')
        m = ceil(n_obs * M)
    else:
        min_frac = 2/N # == 2*M/(N*M)
        if n_obs < min_frac:
            m = 2*M
            if log: log(f'FYI: increasing n_obs to {min_frac:.4g} (minimum 2 obs/task allowed)')
        else:
            m = max(2*M, ceil(n_obs * N * M))
    if log: log(f'FYI: initializing sampler with {m} observations ( ~{m/(N*M):.4g} of all obs, ~{m/M:.4g} obs/task )\n')
    if log: log('-'*80)
    
    tasks = list(range(M))
    checkpoints = list(range(N))
    chk_tasks = [[] for _ in range(N)]
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
            tasks = list(range(M)) # reset task list
        if len(checkpoints) == 0:
            checkpoints = list(range(N))
    random.shuffle(chk_tasks)
    
    # convert to x,y indices, and create a boolean matrix of size (N, M)
    S = np.zeros((N, M), dtype=bool)
    for i, tasks in enumerate(chk_tasks):
        for j in tasks:
            S[i, j] = True
    
    # optimize the samples to increase the minimum distance between checkpoints within each task
    if optimize:
        S = maximize_min_dist(S, log=log)
    
    # use: X,Y = np.where(S)
    return S

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


#--------------------------------------------------------------
if __name__ == "__main__":

    N,M = 120, 70
    n_obs = 2

    S = init_samples(N, M, n_obs)
    # X,Y = np.where(S)
    
    print(S.sum(axis=0))
    print(S.sum(axis=1))
    print(S.sum())
