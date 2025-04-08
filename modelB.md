## Model

Bayesian Optimization using a Multi-task Gaussian Process model. This formulation uses the ICM (intrinsic co-regionalization model) kernel, which is a way to induce/learn inter-task correlations.

The multi-task ICM kernel is defined on a pair of input-task tuples $ (x,t) $ as such:

$$
K_{icm} \left( (x,t),(x{\prime},t{\prime}) \right) = K_x(x,x{\prime}) \otimes K_t(t,t{\prime})
$$

Where x represents the input feature space, and t represents the task-space.  And so the kernel $K_x$ measures relationships between inputs, and the kernel $K_t$ measures relationships between tasks.  Since the input space $x$ is continuous/numerical, $K_x$ is an RBF kernel, which represents similarities between two inputs $x,x{\prime}$ as a function of the squared distance between them $|x-x{\prime}|^2$.