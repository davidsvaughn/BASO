import numpy as np
import math

# Simulated expensive test function.
# Replace this with your actual expensive evaluation.
def expensive_test(checkpoint, test):
    # For illustration, we simulate a function that peaks around checkpoint 26.
    # The true performance is modeled as a baseline plus a Gaussian bump.
    baseline = 0.84
    bump = 0.05 * np.exp(-((checkpoint - 26)**2) / 200)
    noise = np.random.normal(0, 0.01)
    return baseline + bump + noise

# Define problem dimensions.
N_checkpoints = 73  # rows
N_tests = 71        # columns

# For each checkpoint, we will store the test results we've observed and which test indices were sampled.
observations = {i: [] for i in range(1, N_checkpoints+1)}
sampled_tests = {i: [] for i in range(1, N_checkpoints+1)}

# Define a total budget for samples (each sample = running one test on one checkpoint).
budget = 200

# ---- Initial sampling ----
# Sample one random test for each checkpoint.
for checkpoint in range(1, N_checkpoints+1):
    test_index = np.random.randint(1, N_tests+1)
    y_val = expensive_test(checkpoint, test_index)
    observations[checkpoint].append(y_val)
    sampled_tests[checkpoint].append(test_index)

# ---- Adaptive sequential sampling using UCB ----
alpha = 1.96  # controls exploration (roughly a 95% confidence width)

for b in range(budget - N_checkpoints):
    # Compute current estimates for each checkpoint.
    means = {}
    uncertainties = {}
    for checkpoint in range(1, N_checkpoints+1):
        data = observations[checkpoint]
        n = len(data)
        if n > 0:
            means[checkpoint] = np.mean(data)
            # Use sample variance to estimate uncertainty of the mean.
            if n > 1:
                uncertainties[checkpoint] = np.sqrt(np.var(data, ddof=1)) / np.sqrt(n)
            else:
                uncertainties[checkpoint] = 1.0  # High uncertainty with only one sample.
        else:
            means[checkpoint] = 0.0
            uncertainties[checkpoint] = 1.0

    # Compute UCB score for each checkpoint.
    ucb_scores = {cp: means[cp] + alpha * uncertainties[cp] for cp in range(1, N_checkpoints+1)}
    
    # Select the checkpoint with the highest UCB score.
    chosen_checkpoint = max(ucb_scores, key=ucb_scores.get)
    
    # For the chosen checkpoint, select a test index that hasn't been run yet (if possible).
    available_tests = list(set(range(1, N_tests+1)) - set(sampled_tests[chosen_checkpoint]))
    if available_tests:
        chosen_test = np.random.choice(available_tests)
    else:
        # If all tests have been run on that checkpoint, pick one at random.
        chosen_test = np.random.randint(1, N_tests+1)
    
    # Run the expensive test.
    y_val = expensive_test(chosen_checkpoint, chosen_test)
    observations[chosen_checkpoint].append(y_val)
    sampled_tests[chosen_checkpoint].append(chosen_test)
    
    # Optionally, print progress every 10 iterations.
    if b % 10 == 0:
        best_checkpoint = max(means, key=means.get)
        print(f"Iteration {b}: Best checkpoint so far: {best_checkpoint} (mean performance {means[best_checkpoint]:.4f})")

# ---- Final estimation ----
final_means = {cp: np.mean(observations[cp]) for cp in range(1, N_checkpoints+1)}
best_checkpoint = max(final_means, key=final_means.get)
print("\nEstimated best checkpoint:", best_checkpoint)
print("Estimated performance:", final_means[best_checkpoint])
