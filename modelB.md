## Model

Bayesian Optimization using a Multi-task Gaussian Process model. This formulation uses an ICM (intrinsic co-regionalization model) kernel, which is a way to induce/learn inter-task correlations. The multi-task ICM kernel is represented as:

$$
K_{icm} ((x,t),(x^{1\prime},t^{1\prime})) = K_x(x,x^{1\prime}) \otimes K_t(t,t^{1\prime})
$$

