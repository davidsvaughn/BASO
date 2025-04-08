## Model

Bayesian Optimization using a Multi-task Gaussian Process model. This formulation uses the ICM (intrinsic co-regionalization model) kernel, which is a way to induce/learn inter-task correlations.

The multi-task ICM kernel is defined on pairs of input-task tuples $(x,t)$ as such:

$$
K_{ICM} \left( (x,t),(x{\prime},t{\prime}) \right) = K_x(x,x{\prime}) \otimes K_t(t,t{\prime})
$$

The kernel $K_x$ measures relationships between inputs, and the kernel $K_t$ measures relationships between tasks.  Since the input space $x$ is continuous/numerical, $K_x$ is an RBF kernel, which represents similarities between input pairs $x,x{\prime}$ as a function of the squared distance between them $|x-x{\prime}|^2$. However, since tasks are categorical in nature (no intrinsic ordering) the task kernel $K_t$ is just a matrix of inter-task correlations which is learned from the observed data.  Instead of using a full rank matrix, though, $K_t$ is represented using a lower rank Cholesky factor $L$ s.t. $K_t = LL^T$, which ensures that $K_t$ is P.S.D..  Additionally, using a lower rank matrix encourages the model to learn correlations between tasks.

