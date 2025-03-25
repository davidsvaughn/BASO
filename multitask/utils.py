import sys, os
import numpy as np
import gc
import torch
import math

torch.set_default_dtype(torch.float64)

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

