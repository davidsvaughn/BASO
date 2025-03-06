see: [this](https://chat.deepseek.com/a/chat/s/6c840e55-3446-47bd-b6d6-273580e36e65)

The code implements an adaptive Bayesian model for school testing optimization, using a multivariate normal distribution with incrementally updated parameters. Here are the key equations and components:

### 1. **Grade-Level Distributions**
Each grade's scores are modeled as normally distributed with parameters updated online:

**Mean Update** (Sequential Bayesian Update):

$$
\mu_i^{(n+1)} = \mu_i^{(n)} + \frac{x_i^{(n+1)} - \mu_i^{(n)}}{n + 1}
$$

**Variance Update** (Welford's Algorithm):

$$
S^{(n+1)} = S^{(n)} + (x_i^{(n+1)} - \mu_i^{(n)}) \cdot (x_i^{(n+1)} - \mu_i^{(n+1)})
$$

$$
\sigma_i^{(n+1)} = \sqrt{\frac{S^{(n+1)}}{n + 1}}
$$

### 2. **Multivariate Correlation Structure**
Grades are modeled jointly with a correlation matrix **C** updated incrementally:

**Z-Scores**:

$$
z_i = \frac{x_i - \mu_i}{\sigma_i}, \quad z_j = \frac{x_j - \mu_j}{\sigma_j}
$$

**Correlation Update** (Exponential Moving Average):

$$
C_{i,j}^{(new)} = (1 - \alpha) \cdot C_{i,j}^{(old)} + \alpha \cdot (z_i \cdot z_j)
$$

where $\alpha = \min\left(0.1, \frac{1}{1 + N_{\text{schools}}}\right)$ is the adaptive learning rate.

### 3. **Conditional Prediction for Partial Data**
For a school with observed grades \(a\) and unobserved grades \(b\), the conditional distribution is:

**Conditional Mean**:

$$
\mu_{b|a} = \mu_b + \Sigma_{ba} \Sigma_{aa}^{-1} (a - \mu_a)
$$

**Conditional Covariance**:

$$
\Sigma_{b|a} = \Sigma_{bb} - \Sigma_{ba} \Sigma_{aa}^{-1} \Sigma_{ab}
$$

**In Z-Space** (using correlation matrix **C**):

$$
\mu_{b|a}^{(z)} = C_{ab}^\top C_{aa}^{-1} z_a
$$

$$
\Sigma_{b|a}^{(z)} = C_{bb} - C_{ab}^\top C_{aa}^{-1} C_{ab}
$$

### 4. **Decision Rule**
Monte Carlo samples from \(p(b|a)\) are used to estimate:

$$
P\left(\frac{1}{K}\sum_{k=1}^K \left(\sum_{a}x_a + \sum_{b}x_b^{(sim)}\right) > \text{current best}\right)
$$

Testing stops if this probability falls below a threshold.

### Full Model Specification
The complete data-generating process is:

$$
\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma), \quad \Sigma_{i,j} = C_{i,j} \sigma_i \sigma_j
$$

with parameters $(\boldsymbol{\mu}, \Sigma)$ updated adaptively as data arrives. The model uses Bayesian-inspired online updates for means/variances and a heuristic correlation update rule.