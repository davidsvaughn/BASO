## Model

Bayesian Optimization using Multi-task Gaussian Process Regression. 

We have a set of model checkpoints saved periodically during training (usually saved at equally spaced intervals but this is not a requirement). In addition, we have a set of benchmark tasks we can run on a model to get a score. We would like to find the model checkpoint that performs the best overall on the benchmarks (i.e. has the highest average score). However, running a model checkpoint on a benchmark is costly. Suppose we have 100 model checkpoints and 100 benchmark tasks, and running any model-benchmark pair takes around 1 minute. Running all models on all tasks would take a whole week. This case may be rather extreme, but even in less extreme cases we would like to avoid running all model-task pairs.

One possible solution might be to look at each benchmark separately, run it on some subset of all the model checkpoints, and then try to fit a simple regression model to these score outputs to estimate scores for the untested checkpoints. In my experience, these types of holdout-set performance curves are either roughly monotonic (rising or falling) or exhibit a roughly unimodal (rise-and-fall) shape, as the model first learns generalizable patterns and then eventually starts to overfit the training data. Gaussian Process Regression is one approach...

Although each benchmark task may have it's own unique curve, since we are running the same set of checkpoints through each task there is almost certainly some relationship between many of these curves.  Some pairs may be highly correlated, while others not so much.  We would like to be able to share information across tasks to reduce the number of observations required, but we don't know what these inter-task relationships are a-priori. A multi-task regression method that could jointly learn these inter-task correlations AND use these correlations to share information between task-specific regression curves would be useful in this scenario.


This formulation uses the ICM kernel (ICM = intrinsic co-regionalization model), which was proposed here as a way to induce/learn inter-task correlations.

The multi-task ICM kernel is defined on pairs of input-task tuples $(x,t)$ as such:

$$
K_{ICM} \left( (x,t),(x{\prime},t{\prime}) \right) = K_x(x,x{\prime}) \otimes K_t(t,t{\prime})
$$

In the current scenario, the "inputs" are our saved model checkpoints, and the "tasks" are the benchmarks we run each model on.  When we run a given checkpoint $x_i$ on a task $t_j$, we get a scalar valued output $y_{i,j}$ representing the performance of the model on the task. These are typically values in the $[0..1]$ range.

The kernel $K_x$ measures relationships between inputs, and the kernel $K_t$ measures relationships between tasks.  Since the input space $x$ is continuous/numerical, $K_x$ is an RBF kernel, which represents similarities between input pairs $x,x{\prime}$ as a function of the squared distance between them $|x-x{\prime}|^2$. However, since tasks are categorical in nature (no intrinsic ordering) the task kernel $K_t$ is just a matrix of inter-task correlations which is learned from the observed data.  Instead of using a full rank matrix, though, $K_t$ is represented using a lower rank Cholesky factor $L$ s.t. $K_t = LL^T$, which ensures that $K_t$ is P.S.D..  Additionally, using a lower rank matrix encourages the model to learn correlations between tasks. 