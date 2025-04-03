import sys, os
import numpy as np
from functools import partial

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
    
    def eval(self, value):
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