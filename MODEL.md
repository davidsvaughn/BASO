## Model

Bayesian Optimization using Multi-task Gaussian Process Regression. 

We have a set of model checkpoints saved periodically during training (usually saved at equally spaced intervals but this is not a requirement). In addition, we have a set of benchmark tasks we can run on a model to get a score. If we chose a single benchmark and ran it on all model checkpoints and then plotted these, with checkpoint number on the x-axis and benchmark score on the y-axis, we would get a *learning curve* showing how the model improves during training, and then starts to degrade. Typically we are interested in finding the model checkpoint at the peak of this curve. If we were to plot the learning curve for every benchmark and then average them, we would get an average learning curve. Ideally we would like to find the model checkpoint that maximizes this curve, because this model performs the best overall on the benchmarks (i.e. has the highest average score - averaged over all benchmarks). However, running a model checkpoint on a benchmark is costly. Suppose that running a single model on a single benchmark takes 1 minute, and we have 100 model checkpoints and 100 benchmark tasks. Running all models on all tasks would take 10000 minutes == 1 week. This case may be rather extreme, but even in less extreme cases we would like to find ways of reliably estimating the best overall model checkpoint without running all model-task pairs.


Bayesian Optimization is a common approach for such optimization scenarios, when the function we are trying to optimize is expensive to compute. Sometimes it's called "black-box" optimization since we treat the function $f$ as a "black-box" and only have access to it by querying it's value at specific points. The basic approach is to repeatedly query the function so as to acquire more sample points with which to estimate a regression model, and then use that regression model to optimize the function. Naturally this process is a loop, the main steps being:

1. fit a regression model using function evaluations queried so far
2. use the regression model to decide the next point to query


It's important to remember that the end goal is to optimize $f$ (i.e. find the max, or argmax...), not really to fit a full regression model to $f$.  Of course, if we had a really good regression model, we could exploit it to easily find the max. Since we don't, we are naturally more concerned with increasing the quality of the regression model in the region(s) where $f$ is high, since those are more likely to contain $f_{max}$. So, there is an inherent trade-off at play during Bayesian optimization between (a) optimizing the accuracy of the regression model and (b) optimizing the function we are modeling. Sometimes this is called the *exploration-vs-exploitation* trade-off. Since step 1 above is fairly straightforward, this trade-off comes into play only in step 2, where we must decide how to choose the next query point. At any point during the BO loop, we can either (a) poke around in (i.e. exploit) regions where our current regression model has greater certainty that $f$ is high, or (b) explore regions where the current regression model has less certainty about whether $f$ is high or low. Clearly it would be useful to have a regression model that can provide uncertainty estimates.


Gaussian Process regression models are a popular choice, since they provide explicit uncertainty estimates that can be used to guide step 2. Another reason they are convenient for Bayesian Optimization is that the priors, marginals, and posterior (conditional) means and variances are all expressible in closed form as multivariate Gaussian (normal) distributions, which makes it easy to repeatedly perform Bayesian updates to our model as we observe more data points.


Standard GP regression would be suitable only for modeling the learning curve for a single benchmark. The difference between a Gaussian Process and a standard multivariate Gaussian probability distribution is that, in a GP model, the input space (in this case the space of model benchmarks) is the source of the "multiple dimensions" even though it lies along 1 dimension.  Suppose we only save model benchmarks every 50 steps, so we can only make observations where x is a multiple of 50.  We can still use a GP model to define a continuous function on the x domain. To make things simpler, lets define a discrete input domain $X_I$ as the vector of positive integers 1 to 1000: $X_I = [1,2,3,...,1000]$.  We can imagine modeling the vector of function values at each of these points $f(X_I) = [f(x_1),…,f(x_n)]$ as a multivariate Gaussian. Before making any observations, our *GP Prior* over this domain is defined as a multivariate normal distribution:

```math
f(X_I) \sim \mathcal{N}(\mu_0 \, \Sigma_0) \\
{    s.t.   } \\
\begin{aligned} \\
\mu_0(X_I) &= 0 \\
\Sigma_0(X_I) &= K(X_I,X_I)
\end{aligned}
```

where $K$ is a pair-wise kernel function $k(x,x{\prime})$ used to express the correlation between function values $f(x)$ and $f(x{\prime})$. Since the input domain (of model benchmarks) is numeric (as opposed to categorical) we would use an RBF kernel, which represents similarities between input pairs $x,x{\prime}$ as a function of the squared distance between them $|x-x{\prime}|^2$, encoding the intuition that model checkpoints that are closer together are expected to have more similar function values (i.e. benchmark scores) than two checkpoints that are farther apart. The RBF (Radial Basis Function) kernel is expressed as: 

$$
K_{RBF}(x,x{\prime}) = σ^2 \exp{ \frac{ -|x-x{\prime}|^2}{2l^2} }
$$

with hyperparameters $l$ and $σ²$ representing input-scale and output-scale.


Now suppose we acquire a set of observations $O = {X_O,Y_O}$ by evaluating the function at a set of points $X_O = [x_1, x_2, ..., x_n]$ to obtain scores $Y_O = [y_1, y_2,...,y_n]$ where $y_i = f(X_i)$. We would update the model (conditional on the new observations) to obtain the posterior distribution 

```math
f(X_I)|O \sim \mathcal{N}(μ_1 \, \Sigma_1) \\
{    s.t.   } \\
\begin{aligned} \\
μ_1(X_I) &= K(X_I,X_O)^T   K(X_O,X_O)^{-1}Y_O \\
\Sigma_1(X_I) &= K(X_I,X_I) - K(X_I,X_O)^T K(X_O,X_O)^{-1}K(X_I,X_O)
\end{aligned}
```




One possible solution might be to take each benchmark separately, run it on a subset of all the model checkpoints, and then try to fit a simple regression model to these "observed" score outputs to estimate the unobserved scores (i.e. for the untested checkpoints). In most cases, these types of holdout-set (validation-set) performance curves follow a predictable pattern: they're either roughly monotonic (rising or falling) or exhibit a roughly unimodal (rise-and-fall) shape, as the model first learns generalizable patterns and then eventually starts to overfit the training data. Gaussian Process Regression is one approach...

Although each benchmark task may have it's own unique curve, since we are running the same set of checkpoints through each task there is almost certainly some relationship between many of these curves.  Some pairs may be highly correlated, while others not so much.  We would like to be able to share information across tasks to reduce the number of observations required, but we don't know what these inter-task relationships are a-priori. A multi-task regression method that could jointly learn these inter-task correlations AND use these correlations to share information between task-specific regression curves would be useful in this situation.

Gaussian process regression is a flexible, non-parametric regression model highly suitable for such cases, 

since there is an established extension to the single-output case called "multi-output" or "multi-task" Gaussian processes.


This formulation uses the ICM kernel (ICM = intrinsic co-regionalization model), which was proposed here as a way to induce/learn inter-task correlations.

The multi-task ICM kernel is defined on pairs of input-task tuples $(x,t)$ as such:

$$
K_{ICM} \left( (x,t),(x{\prime},t{\prime}) \right) = K_x(x,x{\prime}) \otimes K_t(t,t{\prime})
$$

In the current scenario, the "inputs" are our saved model checkpoints, and the "tasks" are the benchmarks we run each model on.  When we run a given checkpoint $x_i$ on a task $t_j$, we get a scalar valued output $y_{i,j}$ representing the performance of the model on the task. These are typically values in the $[0..1]$ range.

The kernel $K_x$ measures relationships between inputs, and the kernel $K_t$ measures relationships between tasks.  Since the input space $x$ is continuous/numerical, $K_x$ is an RBF kernel, which represents similarities between input pairs $x,x{\prime}$ as a function of the squared distance between them $|x-x{\prime}|^2$. However, since tasks are categorical in nature (no intrinsic ordering) the task kernel $K_t$ is just a matrix of inter-task correlations which is learned from the observed data.  Instead of using a full rank matrix, though, $K_t$ is represented using a lower rank Cholesky factor $L$ s.t. $K_t = LL^T$, which ensures that $K_t$ is P.S.D..  Additionally, using a lower rank matrix encourages the model to learn correlations between tasks. 