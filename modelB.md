## Model

Bayesian Optimization using a Multi-task Gaussian Process model. This formulation uses an ICM (intrinsic co-regionalization model) kernel, which is a way to induce/learn inter-task correlations. The multi-task ICM kernel is represented as:

$$
K_{icm} ((x,t),(x`,t`)) = K_x(x,x`) \otimes K_t(t,t`)
$$

