import sys, os
import numpy as np
import gc
import torch
import math
import gpytorch
from crossing import count_line_curve_intersections
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

#--------------------------------------------------------------------------

# function to convert tensor to numpy, first to cpu if needed
def to_numpy(x):
    x = x.cpu() if x.is_cuda else x
    return x.numpy()

#--------------------------------------------------------------------------

# degree metric
def degree_metric(model, X, z=None, verbose=False):
    if z is None:
        z = int(X[:,1].max().item() + 1)
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model.likelihood(model(X))
    mean = pred.mean.reshape(-1, z)
    model.train()
    model.likelihood.train()
    
    degrees = []
    for i in range(z):
        y = to_numpy(mean[:, i])
        x = to_numpy(X[X[:,1]==i][:,0])
        d = count_line_curve_intersections(x, y)
        # plt.plot(x, y)
        # plt.show()
        degrees.append(d)
    avg_degree = np.mean(degrees)
    # show histogram
    if verbose:
        print(f'Average degree: {avg_degree}')
        plt.hist(degrees, bins=np.ptp(degrees)+1)
        plt.show()
    return avg_degree

#--------------------------------------------------------------------------

def bayesian_std(y, Y, weight=None):
    """
    Calculate a stabilized standard deviation for small samples
    using a larger population as prior.
    
    Parameters:
    y (array-like): Small sample array
    Y (array-like): Larger population array
    weight (float): Weight for the prior (None = automatic weighting based on sample size)
    
    Returns:
    float: Stabilized standard deviation
    """
    n = len(y)
    N = len(Y)
    
    # Calculate standard deviations
    std_y = np.std(y, ddof=1) if n > 1 else 0
    std_Y = np.std(Y, ddof=1)
    
    # Automatic weighting based on sample size
    if weight is None:
        # As n increases, weight of prior decreases
        weight = 1 / (1 + n/5)  # Adjust the divisor to control how quickly prior influence fades
    
    # Weighted average of the two standard deviations
    return weight * std_Y + (1 - weight) * std_y

def empirical_bayes_std(y, Y):
    """
    Calculate standard deviation using an empirical Bayes approach.
    
    Parameters:
    y (array-like): Small sample array
    Y (array-like): Larger population array
    
    Returns:
    float: Stabilized standard deviation
    """
    n = len(y)
    
    if n <= 1:
        return np.std(Y, ddof=1)
    
    # Calculate variance components
    var_y = np.var(y, ddof=1)
    var_Y = np.var(Y, ddof=1)
    
    # Shrinkage factor (James-Stein type estimator)
    alpha = 1 - (n-3)/n * var_Y/var_y if var_y > 0 else 0
    alpha = max(0, min(1, alpha))  # Constrain to [0,1]
    
    # Shrink sample variance toward population variance
    stabilized_var = alpha * var_Y + (1 - alpha) * var_y
    
    return np.sqrt(stabilized_var)

#--------------------------------------------------------------------------

def task_standardize(Y, X):
    # deep copy Y
    Y = Y.clone()
    t = X[:,1].long()
    means, stds, ys = [], [], []
    Z = t.max() + 1
    for i in range(Z):
        y = Y[t==i].squeeze()
        mu = y.mean()
        means.append(mu)
        ys.append(y-mu)
    means = np.array(means)
    #-----------------------------------
    yy = torch.cat(ys).numpy()
    # sigma = np.std(yy)
    # stds = np.array([sigma for _ in range(Z)])
    stds = np.array([bayesian_std(y.numpy(), yy) for y in ys])
    # stds = np.array([empirical_bayes_std(y.numpy(), yy) for y in ys])
    #-----------------------------------
    # perform standardization
    for i in range(Z):
        Y[t==i] = (Y[t==i] - means[i]) / stds[i]
    return Y, (means, stds)

def inv_task_standardize(Y, X, means, stds):
    Y = Y.clone()
    t = X[:,1].long()
    for i in range(len(means)):
        Y[t==i] = (Y[t==i] * stds[i]) + means[i]
    return Y


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

#--------------------------------------------------------------------------

def clear_cuda_tensors(target_size=None): # (1, 8192, 32, 96)
    """Clear tensors of specific size from memory"""
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
    print(f"Cleared {count} tensors")
    
#--------------------------------------------------------------------------
# logEI
# https://arxiv.org/pdf/2310.20708v2
# https://chatgpt.com/c/67e31360-d748-8011-87e3-a24f38402e44

# get floating point precision
def get_float_precision():
    return sys.float_info.epsilon

EPS = 1e-16
# EPS = get_float_precision()

# Precompute constants used in the piecewise expansions
LOG_2PI_HALF = 0.5 * math.log(2 * math.pi)  # c1 = log(2π)/2
LOG_PI_OVER_2_HALF = 0.5 * math.log(math.pi / 2.0)  # c2 = log(π/2)/2

# We'll create a standard normal for PDF/CDF computations
_normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

def log_pdf(z):
    # log φ(z) = -z^2/2 - 0.5*log(2π)
    return -0.5 * z * z - LOG_2PI_HALF

def pdf(z):
    # φ(z)
    return torch.exp(log_pdf(z))

def cdf(z):
    # Φ(z)
    return _normal.cdf(z)

def erfcx(x):
    # erfcx(x) = e^{x^2} * erfc(x)
    # PyTorch >= 1.9 provides torch.special.erfcx. If not available, you can implement
    # an approximation directly or via e.g. SciPy. Here we assume torch.special.erfcx exists:
    return torch.special.erfcx(x)

def log1mexp(log_x):
    r"""
    A stable approximation of log(1 - exp(u)) given u in log form:
       log(1 - e^u) = log( -expm1(u) ), etc.
    See Mächler (2012) 'Accurately Computing log(1 − exp(−|a|))'.
    """
    # log_x = log(u). We want log(1 - u), but in log-domain.
    # We can do:  log(1 - exp(u)) = log(1 - e^u).
    # If e^u is close to 0, do a direct series. If e^u is close to 1, we must be careful.
    # For simplicity, we rely on a direct approach:
    x = torch.exp(log_x)  # u
    # if x ~ 1 or bigger, then 1-x <=0 -> no real log. Typically won't happen with this usage.
    return torch.log1p(-x + EPS)  # +EPS to keep from log(0)

def log_h(z):
    r"""
    A numerically stable approximation of log( φ(z) + z * Φ(z) ).
    
    Implements Eq. (9) in the "Unexpected Improvements ..." paper:
       Case 1:  z > -1
         log_h(z) = log( φ(z) + zΦ(z) )
       Case 2:  -1 >= z > -1/sqrt(EPS)
         expansion in terms of erfcx(...) 
       Case 3:  z <= -1/sqrt(EPS)
         asymptotic approximation: -z^2/2 - c1 - 2 * log|z|
    """
    # We'll build the output piecewise
    out = torch.empty_like(z)

    # CASE 1: z > -1
    mask1 = (z > -1)
    if mask1.any():
        z_sel = z[mask1]
        val = torch.log(pdf(z_sel) + z_sel * cdf(z_sel))  # direct log(φ(z)+zΦ(z)) is stable enough for z > -1
        out[mask1] = val

    # CASE 2: -1 >= z > -1/sqrt(EPS)
    mask2 = (z <= -1) & (z > -1.0 / math.sqrt(EPS))
    if mask2.any():
        z_sel = z[mask2]
        # rewrite:
        #   log( φ(z) + zΦ(z) ) = -z^2/2 - log(2π)/2 + log{ 1 - exp( log(1 - z*φ(z)/...) ) } ...
        # but more directly we do:
        #   φ(z) + zΦ(z) = φ(z) [1 + zΦ(z)/φ(z)] = φ(z) [1 + z * (Φ(z)/φ(z))]
        #   Φ(z)/φ(z) = ...
        # For large negative z, we use erfcx to get stable values.
        # The final expression from the paper is:
        #    - z^2/2 - c1 + log1mexp( log( erfcx(-z/sqrt(2)) ) + c2 )
        # We implement that carefully here:
        val = -0.5 * z_sel * z_sel - LOG_2PI_HALF
        #  inside the log(1 - exp(...)) expression
        #  note the difference from the paper: we have sign manipulations for z < 0
        #  the paper uses: log(erfcx(-z/sqrt(2))|z| ) + c2
        #  We'll do that in steps:
        arg = torch.log(erfcx(-z_sel / math.sqrt(2.0))) + torch.log(torch.abs(z_sel)) + LOG_PI_OVER_2_HALF
        # Then the final add to val is log1mexp(arg):
        val += log1mexp(arg)
        out[mask2] = val

    # CASE 3: z <= -1/sqrt(EPS)
    mask3 = (z <= -1.0 / math.sqrt(EPS))
    if mask3.any():
        z_sel = z[mask3]
        # Asymptotic form: -z^2/2 - c1 - 2*log|z|
        val = -0.5 * z_sel * z_sel - LOG_2PI_HALF - 2.0 * torch.log(torch.abs(z_sel))
        out[mask3] = val

    return out

def logEI(mu, sigma, best):
    r"""
    Computes log(EI) for each point’s GP posterior (mu, sigma).
    EI(x) = sigma(x) * [φ(z) + z Φ(z)], where z = (mu - best) / sigma.
    logEI = log_h(z) + log(sigma).

    Args:
        mu: Tensor of shape [N], posterior means
        sigma: Tensor of shape [N], posterior stddevs (>0)
        best: float or Tensor[()] scalar best (incumbent) so far

    Returns:
        Tensor of shape [N] with logEI values. (NaN if sigma=0.)
    """
    z = (mu - best) / sigma
    return log_h(z) + torch.log(sigma)

#--------------------------------------------------------------------------