import sys, os
import numpy as np
from functools import partial
from utils import adict

class StoppingCondition:
    """
    Track a moving average of values during training and decide when to stop.
    
    This class encapsulates the logic for monitoring a metric/loss and 
    determining when to stop training based on different patterns:
    1. Consistent directional changes in moving average
    2. Improvement falling below threshold
    """
    
    def __init__(self,
                 value=None, # either keyword (string) or function to evaluate
                 condition=None, # condition to check for stopping : lambda function to call on self.average
                 alpha=1, # smoothing factor for EMA (<1) or window size for moving average (1==no lookback)
                 interval=1, # evaluate every n iterations
                 patience=1,
                 min_iterations=100,
                 t=None, # threshold factor for comparison
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

        """
        self.value = value
        self.condition = condition
        self.alpha = alpha
        self.interval = interval
        self.patience = patience
        self.min_iterations = min_iterations
        self.threshold = t
        self.optimizer = optimizer
        self.lr_gamma = lr_gamma
        self.lr_steps = lr_steps
        self.logging = logging
        self.prefix = f'{prefix.strip()}\t' if prefix is not None else ''
        self.burnin = burnin if burnin is not None else (alpha*2 if alpha>1 else min_iterations//10)
        self.verbosity = verbosity
        self.pfunc = partial(self._print, printer=logging.info if logging is not None else print)
        
        self.history = []
        self.average = None
        self.message_log = []
        self.iteration = 0
        self.consecutive_count = 0
        self.lr_step = 0
        
        # condition must be a string with 'x' as the variable, and optionally 't'
        # when called, the following bindings:  x <- self.average
        #                                       t <- self.threshold
        self.test = lambda x,t: eval(condition)

        
    def _print(self, msg, verbosity_level=0, printer=print):
        """ Print message if verbosity level is sufficient. """
        if self.verbosity >= verbosity_level:
            printer(msg)
        # else:
        self.message_log.append(msg)
            
    def _update(self, value):
        """ Add value to history and update the moving average with a new value. """
        self.history.append(value)
        if len(self.history) == 1:
            self.average = [value]
            return
        # update moving average
        if self.alpha < 1:
            # Exponential moving average
            self.average += [self.alpha * value + (1 - self.alpha) * self.average[-1]]
        else:
            # Simple sliding window average
            self.average += [np.mean(self.history[-self.alpha:])]

    def _run_condition(self, **kwargs):
        """ Run the condition function on the moving average. """
        try:
            success = self.test(self.average, self.threshold)#, **kwargs)
        except IndexError as e:
            # this happens when the moving average is empty
            success = False
            self.pfunc(f"{self.prefix}Error evaluating condition: {e}", 3)
            # print(f"{self.prefix}Error evaluating condition '{self.condition}': {e}")
            # print()
        except Exception as e:
            success = False
            self.pfunc(f"{self.prefix}Error evaluating condition: {e}", 3)
            print(f"{self.prefix}Error evaluating condition '{self.condition}': {e}")
            print()
        if success:
            self.consecutive_count += 1
            self.pfunc(f"{self.prefix}*{self.consecutive_count}/{self.patience}* Condition satisfied: '{self.condition}' ", 2)
            return True
        return False
    
    def _eval(self, **kwargs):
        """ Add a new value to the history.
            Evaluate if condition is met.
        """
        value = None
        
        # check if self.value is str
        if isinstance(self.value, str):
            # if self.value is str, assume it's a keyword in kwargs
            # value = kwargs.get(self.value, None)
            # pop it from kwargs if it exists
            value = kwargs.pop(self.value, None)
            
        elif callable(self.value):
            # if self.value is callable, call it with kwargs
            if self.iteration>self.min_iterations and self.iteration%self.interval==0:
                value = self.value(**kwargs)
        
        # if value is None, return False
        if value is None:
            return False
            
        # store and update the moving average
        self._update(value)
        
        # check if stop condition is met
        if self.iteration>self.min_iterations and self.iteration%self.interval==0:
            return self._run_condition(**kwargs)
        return False
    
    def _reset(self, lr_step=None):
        self.consecutive_count = 0
        if self.iteration >= self.min_iterations:
            self.min_iterations = self.iteration + self.burnin
        if lr_step is not None:
            self.lr_step = lr_step
    
    def _reduce_lr(self, payload=None, **kwargs):
        """Reduce learning rate if optimizer is set."""
        if self.optimizer is None or self.lr_step==self.lr_steps:
            return
        
        # check if lr was already reduced by another tracker
        if payload is not None and 'lr_reduced' in payload and payload.lr_reduced:
            return
        
        # reduce learning rate
        self.lr_step += 1
        if payload is not None:
            payload.lr_reduced = True
            payload.lr_step = self.lr_step
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.lr_gamma
            lr = param_group['lr']
        self.pfunc(f"{self.prefix}*{self.lr_step}/{self.lr_steps}* Reduced learning rate to {lr:.4f}", 2)
        self._reset()
    
    def step(self, i=1, **kwargs):
        """ 
        """
        self.iteration += i
        success = self._eval(**kwargs)
        
        if success:
            # self.consecutive_count += 1
            if self.consecutive_count >= self.patience:
                if self.optimizer is None or self.lr_step==self.lr_steps:
                    last_msg = self.message_log[-1] if self.message_log else ""
                    self.pfunc(f"{self.prefix}STOPPING : {last_msg}", 1)
                    return True
                else:
                    self._reduce_lr(**kwargs)
        else:
            self.consecutive_count = 0
        return False
    
# 
class StoppingConditions(StoppingCondition):
    
    def __init__(self, trackers=[], **kwargs):
        super().__init__(**kwargs)
        self.trackers = trackers

    def step(self, **kwargs):
        payload = adict({'lr_reduced': False})
        
        for tracker in self.trackers:
            if tracker.step(payload=payload, **kwargs):
                # a stopping condition was met, no need to check others
                return True
        
        # check if lr was reduced
        if payload.lr_reduced:
            # reset all trackers
            for tracker in self.trackers:
                tracker._reset(lr_step=payload.lr_step)
        
        # no stopping condition was met
        return False
