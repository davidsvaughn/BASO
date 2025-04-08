## Model

Bayesian Optimization using a Multi-task Gaussian Process model. This formulation uses an ICM (intrinsic co-regionalization model) kernel, which is a way to induce/learn inter-task correlations. The multi-task ICM kernel is represented as:

$$
K_{icm} \bigl( (x,t),(x{\prime},t{\prime}) \bigr) = K_x(x,x{\prime}) \otimes K_t(t,t{\prime})
$$

