import numpy as np
import matplotlib.pyplot as plt
from boolean_kde import boolean_sample_kde, plot_kde_example, compare_bandwidths

# get current directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the path
import sys
sys.path.append(os.path.dirname(current_dir))


# Generate example data
np.random.seed(42)
X = np.sort(np.random.uniform(0, 10, 100))  # 100 points between 0 and 10

# Create a pattern for Y - more likely to be sampled in certain regions
underlying_density = 0.5 * np.exp(-(X - 3)**2 / 4) + 0.8 * np.exp(-(X - 7)**2 / 2)
Y = np.random.binomial(1, underlying_density).astype(bool)

# Plot with default bandwidth
fig1 = plot_kde_example(X, Y, bandwidth=0.3)
plt.savefig(os.path.join(current_dir, 'kde_example.png'))
plt.close(fig1)

# Compare different bandwidths
fig2 = compare_bandwidths(X, Y, bandwidths=[0.1, 0.3, 0.5, 1.0])
plt.savefig(os.path.join(current_dir, 'bandwidth_comparison.png'))
plt.close(fig2)

print(f"Total points: {len(X)}")
print(f"Sampled points: {np.sum(Y)}")
print(f"Proportion sampled: {np.mean(Y):.2f}")

# Demonstrate individual Gaussian components
fig3, ax = plt.subplots(figsize=(12, 6))

# Plot sampled points
sampled_X = X[Y]
ax.scatter(sampled_X, np.zeros_like(sampled_X), color='black', marker='|', s=30, label='Sampled Points')

# Plot individual Gaussians for first few points
bandwidth = 0.3
x_eval = np.linspace(X.min(), X.max(), 1000)
colors = ['red', 'blue', 'green', 'purple', 'orange']

for i, point in enumerate(sampled_X[:5]):
    gaussian = np.exp(-0.5 * ((x_eval - point) / bandwidth) ** 2)
    gaussian = gaussian / (bandwidth * np.sqrt(2 * np.pi))
    ax.plot(x_eval, gaussian, color=colors[i % len(colors)], 
            alpha=0.5, linestyle='--', label=f'Gaussian at x={point:.2f}')

# Plot the full KDE
x_eval, density = boolean_sample_kde(X, Y, bandwidth=bandwidth)
ax.plot(x_eval, density, 'k-', linewidth=2, label='Full KDE')

ax.set_title('Individual Gaussian Components of KDE')
ax.set_xlabel('X')
ax.set_ylabel('Density')
ax.legend()

plt.savefig(os.path.join(current_dir, 'individual_gaussians.png'))
plt.close(fig3)