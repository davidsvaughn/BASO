import sys, os
import numpy as np
import gc
import torch
import math
from functools import partial
import gpytorch
from crossing import count_line_curve_intersections_vectorized
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

#--------------------------------------------------------------------------

# function to convert tensor to numpy, first to cpu if needed
def to_numpy(x):
    if x.requires_grad:
        x = x.detach()
    if x.is_cuda:
        x = x.cpu()
    return x.numpy()

def display_fig(run_dir, fig=None, fn=None):
    if run_dir is not None:
        j = len([f for f in os.listdir(run_dir) if f.endswith('.png')])
        if fn is None:
            fn = os.path.join(run_dir, f'fig_{j+1}.png')
        if fig is None:
            plt.savefig(fn)
            plt.close()
        else:
            fig.savefig(fn)
            del fig
    else:
        plt.show()
        
#--------------------------------------------------------------------------

class StoppingTracker:
    """
    Track a moving average of values and detect when to trigger early stopping.
    
    This class encapsulates the logic for monitoring a metric/loss and 
    determining when to stop training based on different patterns:
    1. Consistent directional changes in moving average
    2. Improvement falling below threshold
    """
    
    def __init__(self,
                 interval=1,
                 window_size=3, 
                 patience=10,
                 min_iterations=100,
                 name="metric",
                 mode="direction",      # "direction" or "improvement"
                 direction="up",        # "up" or "down"
                 threshold=1.005,        # threshold factor for comparison
                 optimizer=None,
                 lr_gamma=0.5,
                 lr_steps=5,
                 logging=None,
                 prefix=None,
                 burnin=None,
                 verbosity=1,
                ):
        """
        Initialize the early stopping tracker.
        
        Args:
            window_size: Size of the moving average window
            patience: Number of consecutive events to trigger stopping
            min_iterations: Minimum iterations before allowing early stopping
            name: Name of the metric being tracked (for logging)
            mode: What to monitor:
                  - "direction": Stop when moving average consistently moves in specified direction
                  - "improvement": Stop when relative improvement is below threshold
            direction: Which direction to monitor:
                  - "up": Rising values (like accuracy increasing or loss getting worse)
                  - "down": Falling values (like loss decreasing or accuracy getting worse)
            threshold: 
                  - For direction mode: Factor by which current average must differ from previous
                  - For improvement mode: Relative improvement threshold to consider significant
        """
        self.interval = interval
        self.window_size = window_size
        self.patience = patience
        self.threshold = threshold
        self.min_iterations = min_iterations
        self.name = name
        self.optimizer = optimizer
        self.lr_gamma = lr_gamma
        self.lr_steps = lr_steps
        self.prefix = f'{prefix.strip()}\t' if prefix is not None else ''
        self.burnin = burnin if burnin is not None else window_size*2
        self.verbosity = verbosity
        self.pfunc = partial(self._print, printer=logging.info if logging is not None else print)
        
        # Validate and store parameters
        if mode not in ["direction", "improvement"]:
            raise ValueError("mode must be 'direction' or 'improvement'")
        if direction not in ["up", "down"]:
            raise ValueError("direction must be 'up' or 'down'")
            
        self.mode = mode
        self.direction = direction
        self.history = []
        self.consecutive_count = 0
        self.lr_step = 0
        self.message_log = []
        
    def _print(self, msg, verbosity_level=0, printer=print):
        """ Print message if verbosity level is sufficient. """
        if self.verbosity >= verbosity_level:
            printer(msg)
        else:
            self.message_log.append(msg)
    
    def step(self, value):
        """ Add a new value to the history.
            Checks if early stopping is triggered, 
            Otherwise triggers lr reduction if needed.
            Returns True if early stopping is triggered, False otherwise.
        """
        self.history.append(value)
        return self.check()
    
    def _get_average(self, offset=0):
        """Get moving average with specified offset from the end."""
        if len(self.history) >= self.window_size + offset:
            end_idx = -offset if offset else None
            return sum(self.history[-self.window_size-offset:end_idx]) / self.window_size
        return None
    
    def _is_condition_met(self, current_avg, prev_avg):
        """Check if the stopping condition is met based on monitoring type and direction."""
        # Direction multiplier: 1 for "up", -1 for "down"
        mult = 1 if self.direction == "up" else -1
        
        if self.mode == "direction":
            # For direction mode: check if moving in specified direction beyond threshold
            # For "up" direction: current > prev * threshold
            # For "down" direction: current < prev / threshold
            if self.direction == "up":
                return current_avg > prev_avg * self.threshold
            else:
                return current_avg < prev_avg / self.threshold
        else:  # "improvement" mode
            # Calculate relative improvement
            # For "up" direction (like accuracy): improvement = (current - prev) / |prev|
            # For "down" direction (like loss): improvement = (prev - current) / |prev|
            if prev_avg == 0:
                rel_improvement = 0
            else:
                rel_improvement = mult * (current_avg - prev_avg) / abs(prev_avg)
                
            # Return True when improvement is BELOW threshold (not significant)
            return rel_improvement < self.threshold
    
    def should_stop(self):
        """Check if early stopping criteria are met."""
        # Only check after collecting enough data points and exceeding minimum iterations
        if len(self.history) < self.window_size + 1: #  or self.iteration < self.min_iterations:
            return False
        
        # Get current and previous moving averages
        current_avg = self._get_average()
        prev_avg = self._get_average(offset=1)
        
        # Check if condition is met
        if self._is_condition_met(current_avg, prev_avg):
            self.consecutive_count += 1
            if self.consecutive_count >= self.patience:
                # return True
                if self.iteration >= self.min_iterations:
                    if self.mode == "improvement":
                        self.pfunc(f"{self.prefix}Check failed: {self.name} stagnating for {self.patience} iterations.", 2)
                    else: # mode == "direction"
                        self.pfunc(f"{self.prefix}Check succeeded: {self.name} going {self.direction} for {self.patience} iterations.", 2)
                    return True
                return False
        else:
            self.consecutive_count = 0
            
        # also check if trend is in right direction
        if self.mode == "improvement" and self.iteration >= self.min_iterations:
            if self.direction == "up":
                if current_avg < prev_avg:
                    self.pfunc(f"{self.prefix}Check failed: {self.name} is not going up consistently.", 2)
                    return True
            else:
                if current_avg > prev_avg:
                    self.pfunc(f"{self.prefix}Check failed: {self.name} is not going down consistently.", 2)
                    return True
        return False
    
    def check(self):
        """Check if early stopping criteria are met and adjust learning rate if needed."""
        if self.should_stop():
            if self.optimizer is None or self.lr_step==self.lr_steps:
                last_msg = self.message_log[-1] if self.message_log else ""
                self.pfunc(f"{self.prefix}STOPPING : {last_msg}", 1)
                return True
            self.reduce_lr()
        return False
    
    def reduce_lr(self):
        """Reduce learning rate if optimizer is set."""
        if self.optimizer is None or self.lr_step==self.lr_steps:
            return False
        self.lr_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_gamma
            lr = param_group['lr']
        self.pfunc(f"{self.prefix}Reduced learning rate to {lr:.4f}", 2)
        self.consecutive_count = 0
        if self.iteration >= self.min_iterations:
            self.min_iterations = self.iteration + self.burnin
        return True
    
    @property
    def latest_value(self):
        """Get the most recent value in history."""
        return self.history[-1] if self.history else None
    
    @property
    def latest_average(self):
        """Get the most recent moving average."""
        return self._get_average()
    
    @property
    def previous_average(self):
        """Get the previous moving average."""
        return self._get_average(offset=1)
    
    @property
    def iteration(self):
        """Get the current iteration count."""
        return len(self.history) * self.interval
    
    def reset(self):
        """Reset the tracker."""
        self.history = []
        self.consecutive_count = 0
        return self

#--------------------------------------------------------------------------

# degree metric
def degree_metric(model, X, z=None, num_trials=500, verbose=False):
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
        d = count_line_curve_intersections_vectorized(x, y, num_trials=num_trials)
        # plt.plot(x, y)
        # plt.show()
        degrees.append(d)
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    mean_degree = count_line_curve_intersections_vectorized(x, mean.mean(axis=1), num_trials=num_trials)
    # show histogram
    if verbose:
        print(f'Average degree: {avg_degree}')
        plt.hist(degrees, bins=np.ptp(degrees)+1)
        plt.show()
    return avg_degree, max_degree, mean_degree

# curvature metric
def curvature_metric(model, X, z=None, verbose=False):
    if z is None:
        z = int(X[:,1].max().item() + 1)
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model.likelihood(model(X))
    mean = pred.mean.reshape(-1, z)
    model.train()
    model.likelihood.train()
    
    deriv, maxes = [],[]
    for i in range(z):
        y = to_numpy(mean[:, i])
        x = to_numpy(X[X[:,1]==i][:,0])
        
        d = second_derivative(x, y)
        # d2 = second_derivative_old(x, y)
        # plt.plot(x, y)
        # plt.show()
        deriv.extend(np.abs(d))
        maxes.append(np.max(np.abs(d)))
    avg_deriv = np.mean(deriv)
    mean_max = np.mean(maxes)
    max_max = np.max(maxes)
    # show histogram
    if verbose:
        print(f'Mean: {avg_deriv}, Max: {max_max}, Mean Max: {mean_max}')
        plt.hist(deriv, bins=15)
        plt.show()
        plt.hist(maxes, bins=15)
        plt.show()
    return mean_max

def second_derivative(x, y):
    """
    Estimate the second derivative at each point along a curve using vectorized operations.
    
    Parameters:
    X : numpy.ndarray
        2D array where X[:, 0] contains x-coordinates and X[:, 1] contains y-coordinates.
        Points should be sorted by x-coordinate.
    
    Returns:
    numpy.ndarray
        Array of second derivative estimates at each point.
    """
    # Extract x and y coordinates
    # x = X[:, 0]
    # y = X[:, 1]
    
    n = len(x)
    second_derivatives = np.zeros_like(y)
    
    # Need at least 3 points for second derivative
    if n < 3:
        return second_derivatives
    
    # For interior points (1 to n-2), we can vectorize this
    # Creating slices for the interior calculations
    x_prev = x[:-2]  # x_{i-1} values
    x_curr = x[1:-1]  # x_i values
    x_next = x[2:]    # x_{i+1} values
    
    y_prev = y[:-2]  # y_{i-1} values
    y_curr = y[1:-1]  # y_i values
    y_next = y[2:]    # y_{i+1} values
    
    # Calculate steps
    h1 = x_curr - x_prev  # Backward steps
    h2 = x_next - x_curr  # Forward steps
    
    # Vectorized calculation for all interior points
    second_derivatives[1:-1] = (
        2 * y_prev / (h1 * (h1 + h2)) -
        2 * y_curr / (h1 * h2) +
        2 * y_next / (h2 * (h1 + h2))
    )
    
    # First point
    h1, h2 = x[1] - x[0], x[2] - x[1]
    second_derivatives[0] = (
        2 * y[0] / (h1 * (h1 + h2)) -
        2 * y[1] / (h1 * h2) +
        2 * y[2] / (h2 * (h1 + h2))
    )
    
    # Last point
    h1, h2 = x[n-2] - x[n-3], x[n-1] - x[n-2]
    second_derivatives[n-1] = (
        2 * y[n-3] / (h1 * (h1 + h2)) -
        2 * y[n-2] / (h1 * h2) +
        2 * y[n-1] / (h2 * (h1 + h2))
    )
    
    return second_derivatives

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
    Y = Y.clone()
    t = X[:,1].long()
    means, stds, ys = [], [], []
    Z = t.max() + 1
    for i in range(Z):
        y = Y[t==i][:,0]
        mu = y.mean()
        yc = y - mu
        ys.append(yc)
        means.append(mu)
        #-----------------------------------
    means = np.array(means)
    #-----------------------------------
    yy = torch.cat(ys).numpy()
    stds = np.array([bayesian_std(y.numpy(), yy) for y in ys])
    # stds = np.array([empirical_bayes_std(y.numpy(), yy) for y in ys])
    #-----------------------------------
    # actually perform the standardization
    for i in range(Z):
        Y[t==i] = (Y[t==i] - means[i]) / stds[i]
    # return the standardized Y, and the means and stds for later inverse transform
    return Y, (means, stds)

def inv_task_standardize(Y, X, means, stds):
    Y = Y.clone()
    t = X[:,1].long()
    for i in range(len(means)):
        Y[t==i] = (Y[t==i] * stds[i]) + means[i]
    return Y

#--------------------------------------------------------------------------
# debugging stuff...

def inspect_matrix(V):
    print(V.shape)
    # hist of stds
    stds = np.std(V, axis=0)
    plt.hist(stds, bins=10)
    plt.show()
    # for each column, plot histogram
    for i in range(V.shape[1]):
        plt.hist(V[:,i], bins=10)
        plt.title(f'Column {i}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()
        if i > 10:
            break
    sys.exit()
    
def task_standardize_old(Y, X):
    Y = Y.clone()
    t = X[:,1].long()
    means, stds, ys = [], [], []
    Z = t.max() + 1
    jj = 0
    for i in range(Z):
        y = Y[t==i][:,0]
        mu = y.mean()
        #-----------------------------------
        # center y around the mean
        yc = y - mu
        # if std of yc is too small, center with global mean
        if not yc.std() > 1e-6:
            jj+= 1
            print(f'HEY! {jj}')
            Yf = Y.flatten()
            # bootstrap sample from the full dataset
            ymu = Yf[np.random.randint(0, len(Yf), size=len(Yf))].mean()
            yc = y - ymu
            mu = ymu
        ys.append(yc)
        means.append(mu)
        #-----------------------------------
    means = np.array(means)
    #-----------------------------------
    yy = torch.cat(ys).numpy()
    stds = np.array([bayesian_std(y.numpy(), yy) for y in ys])
    # stds = np.array([empirical_bayes_std(y.numpy(), yy) for y in ys])
    # plt.hist(stds, bins=10)
    # plt.show()
    #-----------------------------------
    # actually perform the standardization
    for i in range(Z):
        Y[t==i] = (Y[t==i] - means[i]) / stds[i]
    # return the standardized Y, and the means and stds for later inverse transform
    return Y, (means, stds)

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

def clear_cuda_tensors(logging=None, target_size=None): # (1, 8192, 32, 96)
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
    msg = f"Cleared {count} tensors of size {target_size}" if target_size is not None else f"Cleared {count} tensors"
    if logging is not None:
        logging.info(msg)
    else:
        print(msg)
    
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