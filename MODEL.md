## Model

Bayesian Optimization using Multi-task Gaussian Process Regression. 

We have a set of model checkpoints saved periodically during training (usually saved at equally spaced intervals but this is not a requirement). In addition, we have a set of benchmark tasks we can run on a model to get a score. If we chose a single benchmark and ran it on all model checkpoints and then plotted these, with checkpoint number on the x-axis and benchmark score on the y-axis, we would get a *learning curve* showing how the model improves during training, and then starts to degrade. Typically we are interested in finding the model checkpoint at the peak of this curve. If we were to plot the learning curve for every benchmark and then average them, we would get an average learning curve. Ideally we would like to find the model checkpoint that maximizes this curve, because this model performs the best overall on the benchmarks (i.e. has the highest average score - averaged over all benchmarks). However, running a model checkpoint on a benchmark is costly. Suppose that running a single model on a single benchmark takes 1 minute, and we have 100 model checkpoints and 100 benchmark tasks. Running all models on all tasks would take 10000 minutes == 1 week. This case may be rather extreme, but even in less extreme cases we would like to find ways of reliably estimating the best overall model checkpoint without running all model-task pairs.

Noisy observations... in addition to interpolating between observations to impute missing data, regression helps absorb/smooth out noise to reveal underlying pattern.

Bayesian Optimization is a common approach for such optimization scenarios, when the function we are trying to optimize is expensive to compute. Sometimes it's called "black-box" optimization since we treat the function $f$ as a "black-box" and only have access to it by querying it's value at specific points. The basic approach is to repeatedly query (i.e. evaluate) the function so as to acquire more sample points with which to estimate a regression model, and then use that regression model to optimize the function. Naturally this process is a loop, the main steps being:

1. use the current regression model to decide which point $x_i$ to query next
2. update the regression model using newly observed function value $y_i=f(x_i)$

It's important to remember that the end goal is to optimize $f(x)$ (i.e. find $\argmax{x}{f(x)}$), not really to fit a full regression model to $f(x)$.  Of course, if we had a really good regression model, we could exploit it to easily find the max. Since we don't, we are naturally more concerned with increasing the quality of the regression model in the region(s) where $f$ is high, since those are more likely to contain $f_{max}$. So, there is an inherent trade-off at play during Bayesian optimization between (a) optimizing the accuracy of the regression model and (b) optimizing the function we are modeling. Sometimes this is called the *exploration-vs-exploitation* trade-off. Since step 2 above is fairly straightforward, this trade-off comes into play only in step 1, where we must decide how to choose the next query point. At any point during the BO loop, we can either (a) poke around in (i.e. exploit) regions where our current regression model has greater certainty that $f$ is high, or (b) explore regions where the current regression model has less certainty about whether $f$ is high or low. Clearly it would be useful to have a regression model that can provide uncertainty estimates.


Gaussian Process regression models are a popular choice, since they provide explicit uncertainty estimates that can be used to guide step 1. Another reason they are convenient for Bayesian Optimization is that the priors, marginals, and posterior (conditional) means and variances are all expressible in closed form as multivariate Gaussian (normal) distributions, which makes it easy to repeatedly perform Bayesian updates to our model as we observe more data points.


Standard GP regression is well suited for modeling the learning curve of a single benchmark. What makes a Gaussian Process different from a standard multivariate Gaussian probability distribution is that, in a GP model, the input space (in this case the space of model benchmarks) is the source of the "multiple dimensions" even though it lies along 1 dimension. Suppose we only save model benchmarks every 50 steps, up to 1000, so we can only make observations $f(x)$ when $x$ is drawn from $[0, 50, 100, 150,…, 1000]$.  We can still use a GP regression model to define a continuous function over the entire interval $[0…1000]$. To make things simpler, let's instead define our GP regression function over a discrete input domain $\mathcal{X}$ of the integers 1 to N: $X_I = [1,2,3,…,N]$.  We can imagine modeling the vector of function values at each of these input points $f(X_I) = [f(x_1),…,f(x_N)]$ as a multivariate Gaussian. Before making any observations, our *GP Prior* over this domain is defined as a multivariate normal distribution:

```math
f(X_I) \sim \mathcal{N}(\mu_0 , \Sigma_0) \\
{    s.t.   } \\
\begin{aligned} \\
\mu_0(X_I) &= 0 \\
\Sigma_0(X_I) &= K(X_I,X_I)
\end{aligned}
```

where $K$ is a pair-wise kernel function $K(x,x{\prime})$ used to model the correlation between output function values $f(x)$ and $f(x{\prime})$. Since the input domain (of model benchmarks) is numeric (as opposed to categorical) we would use an RBF kernel, which represents similarities between input pairs $x,x{\prime}$ as a function of the squared distance between them $\lVert x - x{\prime} \rVert^2$, encoding the intuition that model checkpoints that are closer together are expected to have more similar function values (i.e. benchmark scores) than two checkpoints that are farther apart. The RBF (Radial Basis Function) kernel is expressed as: 

$$
K_{RBF}(x, x{\prime}) = \gamma^2 \exp{ \left( -\frac{1}{2\sigma^2} \lVert x - x{\prime} \rVert^2 \right)}
$$

where learnable hyperparameters $\sigma$ and $\gamma$ represent the input-scale and output-scale, respectively.


Now suppose we acquire a set of observations $O = {X_O,Y_O}$ by evaluating the function at a set of points (i.e. model checkpoints) $X_O = [x_1,x_2,…,x_n]$ to obtain values (i.e. benchmark scores) $Y_O = [y_1,y_2,…,y_n]$ where $y_i = f(X_i)$. We could then update the model, conditional on these new observations, to obtain the posterior distribution:

```math
f(X_I)|O \sim \mathcal{N}(μ_1 , \Sigma_1) \\
{    s.t.   } \\
\begin{aligned} \\
μ_1(X_I) &= K(X_I,X_O)^T   K(X_O,X_O)^{-1}Y_O \\
\Sigma_1(X_I) &= K(X_I,X_I) - K(X_I,X_O)^T K(X_O,X_O)^{-1}K(X_I,X_O)
\end{aligned}
```

This is the standard Bayesian update process used when using a GP model for Bayesian optimization, and would work perfectly well for the case of optimizing the learning curve of a single benchmark. But you may recall the objective function we want to maximize is the *average learning curve* over a set of benchmark tasks. One possible solution might be to fit a separate GP regression model to each benchmark task, and then average these regression curves afterwards. However, there is almost certainly some relationship (correlation) between many of these learning curves. After all, the benchmark tasks tend to be very similar in nature, in addition to the fact that they are being run on the same set of model checkpoints. Some task pairs may be highly correlated, while others not so much.  We would like to be able to share information across similar (correlated) tasks to reduce the number of function evaluations required, but we don't know what these inter-task correlations are *a-priori*. A multi-task regression method that could jointly estimate these inter-task correlations AND use these correlations to share information between task-specific regression curves would be very useful.


Well, there exists a well known extension to the standard single-output GP model called a "multi-output" or "multi-task" Gaussian process. This formulation uses the *ICM Kernel* (ICM = intrinsic co-regionalization model), which was proposed [here] as a way to induce/learn inter-task correlations. While the previous RBF kernel was defined on input pairs $x,x{\prime}$ drawn from a 1D domain, the multi-task ICM kernel is defined on pairs of input-task tuples $(x,t),(x{\prime},t{\prime})$ drawn from a 2D grid. The ICM kernel $K_{ICM}$ factors into the Kronecker product of two kernels:

$$
K_{ICM} \left( (x,t),(x{\prime},t{\prime}) \right) = K_x(x,x{\prime}) \otimes K_t(t,t{\prime})
$$

In the current scenario, the "inputs" $x,x{\prime}$ are drawn from the set of model checkpoints, and the "tasks" $t,t{\prime}$ are drawn from the set of benchmarks.  When we run a given checkpoint model $x_i$ on a task $t_j$, the output is a scalar valued score $y_{i,j} = f(x_i,t_j)$ representing the performance of the model on the task. These scores are typically values in the $[0..1]$ range.

The $K_x$ kernel part accounts for correlations between outputs $f(x,t)$ and $f(x{\prime},t{\prime})$ that are due to checkpoint similarities (which we directly relate to checkpoint distance), and the $K_t$ kernel accounts for output correlations due to inter-task correlations. Since the input space $x$ is continuous/numerical, $K_x$ is an RBF kernel as before. However, since tasks are categorical in nature (i.e. no intrinsic ordering) the task kernel $K_t$ is just P.S.D. matrix of inter-task correlations which is learned from the observed data.  Instead of using a full rank matrix, though, $K_t$ is sometimes represented using a lower rank approximation of the Cholesky factor $L$ s.t. $K_t = LL^T$, which helps ensure that $K_t$ is P.S.D.  Additionally, using a lower rank approximation helps encourage the model to learn correlations between tasks.

Now we could picture our input domain as a 2D matrix (grid) with model checkpoints $X_I = [1,2,3,…,N]$ along the horizontal dimension, and the space of all benchmark tasks $T_I = [1,2,3,…,M]$ along the vertical dimension. As a convenience we *vectorize* (i.e. flatten) this $N x M$ matrix into a one dimensional *all-pairs* vector $P_{X \otimes T} = X_I \otimes T_I$ of length $MN$, but it represents the exact same quantity. 

In a standard Bayesian Optimization setup using a multi-task GP, the objective would be to find the optimal checkpoint-task pair. But we are interested in summing (averaging) over the task dimension and finding the optimal checkpoint. 


The one question remaining is how to perform step 1 -- how do we use the current regression model to decide which point to query next?  In the standard single-output setting, a common method is to maximize an *acquisition function* to decide. An acquisition function is typically defined in such a way to balance our desire to maximize the current regression model and to reduce uncertainty in the regression model. There are many choices, but one of the most popular is called *Expected Improvement*. The expected improvement is defined using 3 quantities: 

1. the posetrior mean of our current 


---------------------------------------------------------------------------------------------------------


 In most cases, these types of holdout-set (validation-set) performance curves follow a predictable pattern: they're either roughly monotonic (rising or falling) or exhibit a roughly unimodal (rise-and-fall) shape, as the model first learns generalizable patterns and then eventually starts to overfit the training data. Gaussian Process Regression is one approach...