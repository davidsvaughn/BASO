import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# -------------------------------
# Load your actual data table.
# Adjust the file name and delimiter as needed.
# For example, if the file is tab-delimited, use delimiter='\t'
df = pd.read_csv('data/phi4-math-4claude.txt', delimiter='\t')

# -------------------------------
# Determine the problem dimensions.
# The data table should have a column "CHECKPOINT" and test columns "TEST_1" ... "TEST_71".
N_checkpoints = df.shape[0]  # e.g. 73 checkpoints
# Identify the test columns (assuming they start with "TEST_")
test_columns = [col for col in df.columns if col.startswith("TEST_") and col != "TEST_AVERAGE"]
N_tests = len(test_columns)  # e.g. 71 tests

# -------------------------------
# Define a function that simulates an expensive test by looking up the value from the data table.
# We assume that checkpoints are ordered in the file, and we use 1-indexing for clarity.
def expensive_test(checkpoint, test):
    # checkpoint: an integer from 1 to N_checkpoints.
    # test: an integer from 1 to N_tests (corresponding to column "TEST_{test}").
    row_index = checkpoint - 1
    col_name = f"TEST_{test}"
    return df.loc[row_index, col_name]

# -------------------------------
# Prepare data structures for adaptive sampling.
# For each checkpoint, store the sampled test results and which test indices were sampled.
observations = {cp: [] for cp in range(1, N_checkpoints+1)}
sampled_tests = {cp: [] for cp in range(1, N_checkpoints+1)}

# Set a total budget for samples (each sample = acquiring one (checkpoint, test) entry).
budget = 200

# ---- Initial Sampling ----
# Sample one random test for each checkpoint.
for cp in range(1, N_checkpoints+1):
    test_index = random.choice(range(1, N_tests+1))
    value = expensive_test(cp, test_index)
    observations[cp].append(value)
    sampled_tests[cp].append(test_index)

# ---- Adaptive Sequential Sampling using UCB ----
# We use a simple UCB rule: score = sample mean + alpha × (standard error)
alpha = 1.96  # This value roughly corresponds to a 95% confidence interval.

for b in range(budget - N_checkpoints):
    # Compute the current sample mean and uncertainty for each checkpoint.
    means = {}
    uncertainties = {}
    for cp in range(1, N_checkpoints+1):
        data = observations[cp]
        n = len(data)
        if n > 0:
            means[cp] = np.mean(data)
            if n > 1:
                uncertainties[cp] = np.std(data, ddof=1) / np.sqrt(n)
            else:
                uncertainties[cp] = 1.0  # High uncertainty with only one sample.
        else:
            means[cp] = 0.0
            uncertainties[cp] = 1.0

    # Compute the UCB score for each checkpoint.
    ucb_scores = {cp: means[cp] + alpha * uncertainties[cp] for cp in range(1, N_checkpoints+1)}
    
    # Choose the checkpoint with the highest UCB score.
    chosen_checkpoint = max(ucb_scores, key=ucb_scores.get)
    
    # For the chosen checkpoint, choose a test that hasn't been run yet (if possible).
    available_tests = list(set(range(1, N_tests+1)) - set(sampled_tests[chosen_checkpoint]))
    if available_tests:
        chosen_test = random.choice(available_tests)
    else:
        # If all tests have already been sampled for this checkpoint, pick one at random.
        chosen_test = random.choice(range(1, N_tests+1))
    
    # "Run" the expensive test (i.e. look up the value) and update our records.
    value = expensive_test(chosen_checkpoint, chosen_test)
    observations[chosen_checkpoint].append(value)
    sampled_tests[chosen_checkpoint].append(chosen_test)
    
    # Optionally, print progress every 10 iterations.
    if b % 10 == 0:
        best_cp_so_far = max(means, key=means.get)
        cp_name = df.loc[best_cp_so_far - 1, "CHECKPOINT"]
        print(f"Iteration {b}: Best checkpoint so far: {cp_name} (mean performance {means[best_cp_so_far]:.4f})")

# ---- Final Estimation ----
# Use the collected samples to estimate the average performance for each checkpoint.
final_means = {cp: np.mean(observations[cp]) for cp in range(1, N_checkpoints+1)}
estimated_best_cp = max(final_means, key=final_means.get)
estimated_best_name = df.loc[estimated_best_cp - 1, "CHECKPOINT"]
print("\nEstimated best checkpoint:", estimated_best_name)
print("Estimated performance:", final_means[estimated_best_cp])

# Optionally, compare with the true best checkpoint from the full data.
true_means = df["TEST_AVERAGE"].values
true_best_index = np.argmax(true_means)
true_best_name = df.loc[true_best_index, "CHECKPOINT"]
print("\nTrue best checkpoint:", true_best_name)
print("True performance:", true_means[true_best_index])
