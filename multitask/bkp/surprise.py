import numpy as np

def initialize_surprise_tracker():
    """Initialize the tracking structure for the surprise-based stopping criterion"""
    return {
        'surprise_history': [],       # History of surprise values
        'prediction_history': [],     # History of model predictions
        'performance_history': [],    # History of estimated best performance
        'best_idx_history': []        # History of predicted best checkpoint indices
    }

def surprise_based_stopping(
    current_model,           # The current GP model 
    new_sample_idx,          # Indices (i,j) of the new sample
    observed_value,          # Actual observed value at the new sample
    prev_y_mean,             # Model's mean predictions before update
    curr_y_mean,             # Model's mean predictions after update
    curr_y_var,              # Model's variance predictions after update
    tracker,                 # Tracking structure from initialize_surprise_tracker
    checkpoint_nums,         # Array of checkpoint numbers
    step=0,                  # Current iteration number
    min_samples=10,          # Minimum samples before considering stopping
    surprise_threshold=1.0,  # Threshold for "not surprising"
    window_size=5,           # Number of consecutive points to check
    min_window_std=0.01,     # Minimum std in performance to ensure stability
    surprise_weights=(0.4, 0.3, 0.3)  # Weights for different surprise metrics
):
    """
    Determine if sampling should stop based on how surprising new samples are.
    
    Args:
        current_model: GP model after incorporating the new sample
        new_sample_idx: Tuple (i,j) of indices of the new sample
        observed_value: The observed value at the new sample
        prev_y_mean: Model's mean predictions before incorporating new sample
        curr_y_mean: Model's mean predictions after incorporating new sample
        curr_y_var: Model's variance predictions after incorporating new sample
        tracker: Dictionary tracking surprise metrics over time
        checkpoint_nums: Array of checkpoint numbers for reporting
        step: Current iteration number
        min_samples: Minimum samples before considering stopping
        surprise_threshold: Threshold below which a sample is considered "not surprising"
        window_size: Number of consecutive non-surprising samples needed to stop
        min_window_std: Minimum required standard deviation in performance to ensure stability
        surprise_weights: Weights for the different surprise metrics (prediction error, model update, best change)
        
    Returns:
        should_stop: Boolean indicating whether to stop sampling
        tracker: Updated tracking structure
        diagnostics: Dictionary with diagnostic information
    """
    i, j = new_sample_idx
    K, Z = curr_y_mean.shape
    
    # 1. Calculate prediction error surprise
    # How surprised we were by the actual value at the point we just sampled
    if prev_y_mean is not None:
        predicted_value = prev_y_mean[i, j]
        prediction_std = np.sqrt(curr_y_var[i, j])  # Using current model's estimate of uncertainty
        norm_prediction_error = abs(observed_value - predicted_value) / (prediction_std + 1e-10)
    else:
        # First iteration, can't calculate prediction error
        norm_prediction_error = 1.0  # Neutral value
    
    # 2. Calculate model update surprise
    # How much did our overall understanding of the space change?
    if prev_y_mean is not None:
        # Calculate relative change in predictions across all points
        abs_changes = np.abs(curr_y_mean - prev_y_mean)
        rel_changes = abs_changes / (np.abs(prev_y_mean) + 1e-10)
        model_update = np.mean(rel_changes)
    else:
        # First iteration, can't calculate model update
        model_update = 1.0  # Neutral value
    
    # 3. Calculate best checkpoint surprise
    # Has our understanding of which checkpoint is best changed?
    curr_mean_over_tasks = curr_y_mean.mean(axis=1)
    curr_best_idx = np.argmax(curr_mean_over_tasks)
    curr_best_performance = curr_mean_over_tasks[curr_best_idx]
    
    if tracker['best_idx_history']:
        prev_best_idx = tracker['best_idx_history'][-1]
        best_idx_changed = prev_best_idx != curr_best_idx
        best_surprise = 1.0 if best_idx_changed else 0.0
    else:
        # First iteration
        best_surprise = 1.0  # Neutral value
    
    # 4. Combined surprise metric (weighted average)
    w1, w2, w3 = surprise_weights
    overall_surprise = w1 * norm_prediction_error + w2 * model_update + w3 * best_surprise
    
    # 5. Update histories
    tracker['surprise_history'].append(overall_surprise)
    tracker['prediction_history'].append(curr_y_mean.copy())
    tracker['performance_history'].append(curr_best_performance)
    tracker['best_idx_history'].append(curr_best_idx)
    
    # 6. Determine if we should stop
    # Need enough samples
    if len(tracker['surprise_history']) < min_samples:
        should_stop = False
        reason = f"Not enough samples ({len(tracker['surprise_history'])} < {min_samples})"
    else:
        # Check the last 'window_size' samples
        recent_surprises = tracker['surprise_history'][-window_size:]
        recent_best_indices = tracker['best_idx_history'][-window_size:]
        recent_performances = tracker['performance_history'][-window_size:]
        
        # Check if recent samples are all non-surprising
        low_surprise = all(s < surprise_threshold for s in recent_surprises)
        
        # Check if best checkpoint has been stable
        stable_best = len(set(recent_best_indices)) <= 1 # TODO: change size??
        
        # Check if performance estimates are stable (not jumping around)
        performance_std = np.std(recent_performances)
        stable_performance = performance_std < min_window_std
        
        # All conditions must be met
        should_stop = low_surprise and stable_performance # and stable_best # TODO: change this??
        
        if not low_surprise:
            reason = f"Recent samples still surprising (avg={np.mean(recent_surprises):.4f})"
        elif not stable_performance:
            reason = f"Performance estimates not stable (std={performance_std:.4f})"
        # elif not stable_best:
        #     reason = f"Best checkpoint not stable: {[checkpoint_nums[idx] for idx in recent_best_indices]}"
        else:
            reason = "All criteria met - new samples no longer provide significant information"
    
    # 7. Prepare diagnostics
    diagnostics = {
        "step": step,
        "should_stop": should_stop,
        "reason": reason,
        "surprise": overall_surprise,
        "prediction_error": norm_prediction_error,
        "model_update": model_update,
        "best_changed": best_surprise > 0,
        "best_checkpoint": checkpoint_nums[curr_best_idx],
        "best_performance": curr_best_performance,
        "recent_surprises": tracker['surprise_history'][-min(window_size, len(tracker['surprise_history'])):],
        "recent_best_checkpoints": [checkpoint_nums[idx] for idx in tracker['best_idx_history'][-min(window_size, len(tracker['best_idx_history'])):]]
    }
    
    return should_stop, tracker, diagnostics