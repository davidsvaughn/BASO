# Adaptive Bayesian Testing Strategy

Below is a concise mathematical description of the statistical assumptions and updates used by this model. In broad terms, the code posits that:

1. **Each grade $g$ has a normal distribution of true test scores** (across all schools) with unknown mean $\mu_g$ and variance $\sigma_g^2$.  
2. **Across grades, scores are correlated** via a correlation matrix $\mathbf{R} = [\rho_{ij}]$.  
3. **When we observe partial grades for a school,** we treat its unobserved grades as drawn from the conditional distribution of a multivariate normal.

Below, let $G$ be the total number of grades.

---

## 1. Per-Grade Model

For each grade $g \in \{1,\dots,G\}$, we maintain running estimates of its (population) mean $\mu_g$ and standard deviation $\sigma_g$. Concretely, we assume:

$$
X_g \sim \mathcal{N}\bigl(\mu_g,\,\sigma_g^2\bigr),
$$

where $X_g$ is the random variable corresponding to a school's true score in grade $g$.

### 1.1 Online (Welford) Update for $\mu_g$ and $\sigma_g$

Whenever the model observes a new score $x$ for grade $g$, it updates $\mu_g$ and $\sigma_g$ incrementally. Let $n_g$ be the sample count for grade $g$ *before* the new observation; then:

1. **Update the mean**:

$$
\mu_g^{(\mathrm{new})} = \mu_g^{(\mathrm{old})} + \frac{x - \mu_g^{(\mathrm{old})}}{\,n_g + 1\,}.
$$

2. **Update the sum of squares** (an intermediate step in Welford's algorithm). Define

$$
S_g^{(\mathrm{old})} 
\;=\;
\bigl[\sigma_g^{(\mathrm{old})}\bigr]^2 \; \times \; n_g,
$$

then

$$
S_g^{(\mathrm{new})}
\;=\;
S_g^{(\mathrm{old})}
\;+\;
\bigl(x \;-\; \mu_g^{(\mathrm{old})}\bigr)\,\bigl(x \;-\; \mu_g^{(\mathrm{new})}\bigr).
$$

3. **Update the variance and standard deviation**:

$$
\sigma_g^{(\mathrm{new})}
\;=\;
\sqrt{\;\frac{S_g^{(\mathrm{new})}}{\,n_g + 1\,}\;}.
$$

Then $n_g$ is incremented by 1. This provides an online (incremental) estimate of the sample mean and variance for each grade $g$.

---

## 2. Correlation Structure Across Grades

We assume that after *standardizing* each grade's score, the vector of standardized scores

$$
Z_g \;=\; \frac{X_g - \mu_g}{\sigma_g}
\quad\text{for}\quad g=1,\dots,G
$$

has some correlation matrix $\mathbf{R} = [\rho_{ij}]$. Equivalently, in the original (unstandardized) space, the covariance between $X_i$ and $X_j$ is

$$
\mathrm{Cov}(X_i,\,X_j) \;=\; \rho_{ij}\,\sigma_i\,\sigma_j.
$$

Hence if we collect all grades into a vector $\mathbf{X} = (X_1,\dots,X_G)$, then

$$
\mathbf{X}
\;\sim\;
\mathcal{N}\!\Bigl(\,\boldsymbol{\mu},\;\boldsymbol{\Sigma}\Bigr),
\quad
\text{where}
\quad
\boldsymbol{\mu} \;=\; \bigl(\mu_1,\dots,\mu_G\bigr),
\quad
\boldsymbol{\Sigma}_{ij} \;=\; \rho_{ij}\,\sigma_i\,\sigma_j.
$$

### 2.1 Adaptive Correlation Updates

When the model observes two (or more) grades from the *same* school, it uses that joint observation to update $\rho_{ij}$. The core step is:

1. Compute the standardized scores $z_i$ and $z_j$ for the newly observed pair of grades $(i,j)$.
2. Update $\rho_{ij}$ by a small adaptive step. In the code, it appears roughly as:

$$
\rho_{ij}^{(\mathrm{new})}
\;=\;
(1 - w)\,\rho_{ij}^{(\mathrm{old})}
\;+\;
w\,(\,z_i\,z_j\,),
$$

where $z_i = \frac{x_i - \mu_i}{\sigma_i}$, $z_j = \frac{x_j - \mu_j}{\sigma_j}$, and $w$ is an adaptive weight (bounded by a maximum like 0.1). The code then clips the result to lie in $[-1,1]$ and symmetrically updates $\rho_{ji}$.

---

## 3. Predicting Unobserved Grades for a Partially Tested School

Suppose for a particular school we have observed *some subset* $O\subset \{1,\dots,G\}$ of grades, and we have not yet tested the remaining grades $U = \{1,\dots,G\}\setminus O$. Denote:

- $\mathbf{X}_O$ = the vector of observed scores (for that school) in grades $O$.
- $\mathbf{X}_U$ = the vector of unobserved scores in grades $U$.

Because we assume a multivariate normal model on $\mathbf{X}=(X_1,\dots,X_G)$, the conditional distribution of $\mathbf{X}_U$ given $\mathbf{X}_O = \mathbf{x}_O$ is also normal:

$$
\mathbf{X}_U \mid \mathbf{X}_O = \mathbf{x}_O
\;\sim\;
\mathcal{N}\Bigl(\,\boldsymbol{\mu}_{U|O},\;\boldsymbol{\Sigma}_{U|O}\Bigr),
$$

where (in the *unstandardized* space):

1. **Conditional mean**:

$$
\boldsymbol{\mu}_{U|O}
\;=\;
\boldsymbol{\mu}_U 
\;+\;
\boldsymbol{\Sigma}_{UO}\,\boldsymbol{\Sigma}_{OO}^{-1}
\bigl(\,\mathbf{x}_O \;-\; \boldsymbol{\mu}_O\bigr),
$$

2. **Conditional covariance**:

$$
\boldsymbol{\Sigma}_{U|O}
\;=\;
\boldsymbol{\Sigma}_{UU}
\;-\;
\boldsymbol{\Sigma}_{UO}\,\boldsymbol{\Sigma}_{OO}^{-1}\,\boldsymbol{\Sigma}_{OU}.
$$

Above, $\boldsymbol{\mu}_U$ and $\boldsymbol{\mu}_O$ are the portions of the global mean vector $\boldsymbol{\mu}$ corresponding to the unobserved and observed sets of grades, and the block matrices $\boldsymbol{\Sigma}_{UU}, \boldsymbol{\Sigma}_{OO}, \boldsymbol{\Sigma}_{UO}, \boldsymbol{\Sigma}_{OU}$ come from partitioning the covariance matrix $\boldsymbol{\Sigma}$ accordingly.

---

## 4. Monte Carlo Estimation of "Best School" Probability

Given a partially tested school's observed grades $\mathbf{x}_O$, the model does the following to decide whether that school can still exceed the current best known school:

1. **Sample** $\mathbf{X}_U^{(s)}$ from the conditional distribution above (for $s=1,\dots,N$, e.g.\ $N=10{,}000$).
2. **Form the complete set of scores** $\bigl\{\mathbf{x}_O,\;\mathbf{X}_U^{(s)}\bigr\}$ for each simulation $s$.
3. **Compute the school's overall average** in each simulation:

$$
\overline{X}^{(s)}
\;=\;
\frac{1}{G}
\sum_{g=1}^{G} X_g^{(s)}.
$$

4. Compare $\overline{X}^{(s)}$ to the best known average so far (from any fully tested school). The empirical frequency of $\overline{X}^{(s)}$ exceeding that best-known average is taken as

$$
\hat{p} 
\;=\;
\frac{1}{\,N\,}
\sum_{s=1}^N 
\mathbf{1}\!\Bigl\{\,\overline{X}^{(s)} > \text{(best known average)}\Bigr\}.
$$

This $\hat{p}$ is reported as the school's probability of ultimately being the best.  

### 4.1 Decision to Continue or Stop Testing

- If $\hat{p}$ falls **below** a stopping threshold (e.g.\ 0.05) **and** the model's "confidence" measure is sufficiently high, the algorithm decides to **stop** testing that school further.  
- Otherwise, it continues testing or "prunes" the school only temporarily (with the possibility of **revisiting** it if $\hat{p}$ later rises above some threshold).

---

## 5. Grade Selection Strategy

When deciding *which grade to test next* for a partially tested school, the code computes a custom "priority" score that takes into account:

1. **Global sample count** (fewer total samples in a grade $\rightarrow$ higher priority).  
2. **Uncertainty** (higher $\sigma_g$ $\rightarrow$ higher priority).  
3. **Correlation** with already tested grades (lower correlation $\rightarrow$ *higher* priority, for diversity).  
4. **School-specific pattern** (if the observed grades in this school deviate strongly from the global mean, the code may prioritize grades that confirm or challenge that pattern).  
5. **Random exploration** (a small random term).

Mathematically, each untested grade $g$ receives a score of the form

$$
\text{Priority}(g)
\;=\;
w_1 \times \text{(inverse of sample count)} 
\;+\;
w_2 \times \text{(relative std dev)}
\;+\;
w_3 \times \bigl[\,1 - (\text{average correlation})\bigr]
\;+\;
w_4 \times \text{(pattern-based term)}
\;+\;
\epsilon_{\mathrm{random}},
$$

where each $w_i$ is a weight, and $\epsilon_{\mathrm{random}}$ is a small random contribution. The model picks the grade $g$ with the largest overall priority.

---

## Putting It All Together

1. **We maintain** one global mean $\mu_g$ and standard deviation $\sigma_g$ per grade, and a global correlation matrix $\mathbf{R}$.  
2. **Each new observation** $(g, x)$ triggers an online update of $\mu_g$, $\sigma_g$, and (if multiple grades are observed for the same school) small updates to the correlation entries $\rho_{ij}$.  
3. **For a partially tested school**, given observed $\mathbf{X}_O$, we invoke the **multivariate normal conditional** on $\mathbf{X}_O$ to simulate the unobserved $\mathbf{X}_U$. This yields a distribution over the school's full average.  
4. **A Monte Carlo approach** estimates the probability that the school's final average beats the best known average so far. This probability, along with some confidence checks, determines whether to continue testing that school (and which grade to test next) or to stop early.

These steps define the *adaptive Bayesian* testing strategy the code implements.