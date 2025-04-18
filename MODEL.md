## Model

[https://www.xyzeditor.com/]

### Bayesian Optimization using Multi-task Gaussian Process Regression

We have a set of model checkpoints saved periodically during training (usually saved at equally spaced intervals but this is not a requirement). In addition, we have a set of benchmark tasks we can run on a model to get a score. If we chose a single benchmark and ran it on all model checkpoints and then plotted these, with checkpoint number on the x-axis and benchmark score on the y-axis, we would get a *learning curve* showing how the model improves during training, and then starts to degrade. 

[fig here]

Typically we are interested in finding the model checkpoint at the peak of this curve. If we were to plot the learning curve for every benchmark and then average them, we would get an average learning curve. Ideally we would like to find the model checkpoint that maximizes this curve, because this model performs the best overall on the benchmarks (i.e. has the highest average score - averaged over all benchmarks). 

However, running a model checkpoint on a benchmark is costly. Suppose that running a single model on a single benchmark takes 1 minute, and we have 100 model checkpoints and 100 benchmark tasks. Running all models on all tasks would take 10000 minutes == 1 week. This case may be rather extreme, but even in less extreme cases we would like to find ways of reliably estimating the best overall model checkpoint without running all model-task pairs.

Noisy observations... in addition to interpolating between observations to impute missing data, regression helps absorb/smooth out noise to reveal underlying pattern.

Bayesian Optimization is a common approach for such optimization scenarios, when the function we are trying to optimize is expensive to compute. Sometimes it's called "black-box" optimization since we treat the function $f$ as a "black-box" and only have access to it by querying it's value at specific points. The basic approach is to repeatedly query (i.e. evaluate) the function so as to acquire more sample points with which to estimate a regression model, and then use that regression model to optimize the function. Naturally this process is a loop, the main steps being:

1. use the current regression model to decide which point $x_i$ to query next
2. update the regression model using newly observed function value $y_i=f(x_i)$

It's important to remember that the end goal is to optimize $f(x)$ (i.e. find $\argmax{x}{f(x)}$), not really to fit a full regression model to $f(x)$. Of course, if we had a really good regression model, we could exploit it to easily find the max. Since we don't, we are naturally more concerned with increasing the quality of the regression model in the region(s) where $f$ is high, since those are more likely to contain $f_{max}$. So, there is an inherent trade-off at play during Bayesian optimization between (a) optimizing the accuracy of the regression model and (b) optimizing the function we are modeling.

Sometimes this is called the *exploration-vs-exploitation* trade-off. Since step 2 above is fairly straightforward, this trade-off comes into play only in step 1, where we must decide how to choose the next query point. At any point during the BO loop, we can either (a) poke around in (i.e. exploit) regions where our current regression model has greater certainty that $f$ is high, or (b) explore regions where the current regression model has less certainty about whether $f$ is high or low. Clearly it would be useful to have a regression model that can provide uncertainty estimates.

*Gaussian Process* regression models are a popular choice, since they provide explicit uncertainty estimates that can be used to guide step 1. Another reason they are convenient for Bayesian Optimization is that the priors, marginals, and posterior (conditional) means and variances are all expressible in closed form as multivariate Gaussian (normal) distributions, which makes it easy to repeatedly perform Bayesian updates to our model as we observe more data points.

Standard GP regression is well suited for modeling the learning curve of a single benchmark. What makes a Gaussian Process different from a standard multivariate Gaussian probability distribution is that, in a GP model, the input space (in this case the space of model benchmarks) is the source of the "multiple dimensions" even though it lies along 1 dimension. Suppose we only save model benchmarks every 50 steps, up to 1000, so we can only make observations $f(x)$ when $x$ is drawn from $[0, 50, 100, 150,…, 1000]$. We can still use a GP regression model to define a continuous function over the entire interval $[0…1000]$. To make things simpler, let's instead define our GP regression function over a discrete input domain $X$ of the integers 1 to N: $X = [1,2,3,…,N]$. We can imagine modeling the vector of function values at each of these input points $f_X = [f(x_1),…,f(x_N)]$ as a multivariate Gaussian. Before making any observations, our *GP Prior* over this domain is defined as a multivariate normal distribution:

$$
f_X \sim  \mathcal{N}(\mu_X , \Sigma_X) \\

{ s.t. } \\
\begin{aligned}
\mu_X &= 0 \\
\Sigma_X &= K(X,X)
\end{aligned}
$$

where $K$ is an $NxN$ matrix of pair-wise kernel function values $k(x,x')$ computed on all input pairs $x,x'$.  This kernel function models the correlation between output function values $f(x)$ and $f(x')$. Since the input domain (of model benchmarks) is numeric (as opposed to categorical) we would use an RBF kernel, which represents similarities between input pairs $x,x'$ as a function of the squared distance between them $\lVert x - x' \rVert^2$, encoding the intuition that model checkpoints that are closer together are expected to have more similar function values (i.e. benchmark scores) than two checkpoints that are farther apart. The RBF (Radial Basis Function) kernel is expressed as:

$$
K_{RBF}(x, x') = \gamma^2  \exp{ \left( -\frac{1}{2\sigma^2} \lVert x - x' \rVert^2  \right)}
$$

where learnable hyperparameters $\sigma$ and $\gamma$ represent the input-scale and output-scale, respectively.

Now suppose we acquire a set of observations $O = {X_O,Y_O}$ by evaluating the function at a set of points (i.e. model checkpoints) $X_O = [x_1,x_2,…,x_n]$ to obtain values (i.e. benchmark scores) $Y_O = [y_1,y_2,…,y_n]$ where $y_i = f(x_i)$. We could then update the model, conditional on these new observations, to obtain the *posterior* distribution over the input space $X$, which is also a multivariate Gaussian:

$$
f_X|O \sim  \mathcal{N}(μ_X|O , \Sigma_X|O) \\
{ s.t. } \\
\begin{aligned} 
μ_X|O &= K(X_O,X)^T K(X_O,X_O)^{-1}Y_O \\
\Sigma_X|O &= K(X,X) - K(X_O,X)^T K(X_O,X_O)^{-1}K(X_O,X)
\end{aligned}
$$

This is the standard Bayesian update procedure for a GP regression model. The *posterior mean* $μ_X|O$ is just an $N$-length vector of means  $[\mu_1, \mu_2,…,\mu_N]$ and would be considered the actual "value" of the GP regression function at each input point $x_i$, while the *posterior covariance* $\Sigma_X|O$ is an $N$x$N$ matrix of pairwise variances.  Taking the square root of the diagonal would yield a vector of standard deviations for drawing confidence bands around the regression function.

For clarity, we could just as well write an expression, at any particular input point $x_i$, for the *marginal* posterior distribution of $f(x_i)$:

$$
f_i|O \sim  \mathcal{N}(μ_i|O , \sigma^2_i|O) \\
{ s.t. } \\
\begin{aligned} 
μ_i|O &= K(X_O,x_i)^T K(X_O,X_O)^{-1}Y_O \\
\sigma^2_i|O &= K(x_i,x_i) - K(X_O,x_i)^T K(X_O,X_O)^{-1}K(X_O,x_i) \\
\end{aligned}
$$ 

where $μ_i$ and $\sigma_i$ are the scalar mean and standard deviation of the *marginal* distribution at $x_i$.  This illustrates another nice property of multivariate Gaussians, which is that marginals, which are simply a slice of the full distribution at a particular point $x_i$, remain Gaussian.

This GP model works perfectly well for the case of optimizing the learning curve of a single benchmark. In this  current formulation we would say we are looking for the $i^*$ (i.e. checkpoint model $x_i^*$) that maximizes $\mu_i|O$.

But you may recall the objective function we want to maximize is the *average learning curve* over a set of benchmark tasks. One possible solution might be to fit a separate GP regression model to each benchmark task, and then average these regression curves afterwards. However, there is almost certainly some relationship (correlation) between many of these learning curves. After all, the benchmark tasks tend to be very similar in nature, in addition to the fact that they are being run on the same set of model checkpoints. Some task pairs may be highly correlated, while others not so much. We would like to be able to share information across similar (correlated) tasks to reduce the number of function evaluations required, but we don't know what these inter-task correlations are *a-priori*. A multi-task regression method that could jointly estimate these inter-task correlations AND use these correlations to share information between task-specific regression curves would be very useful.

#### Multi-Task Gaussian Procceses

Well, there exists a well known extension to the standard single-output GP model called a "multi-output" or "multi-task" Gaussian process. This formulation uses the *ICM Kernel* (ICM = intrinsic co-regionalization model), which was proposed [here] as a way to induce/learn inter-task correlations. While the previous RBF kernel was defined on input pairs $x,x'$ drawn from a 1D domain, the multi-task ICM kernel is defined on pairs of input-task tuples $\langle x,t\rangle ,\langle x',t'\rangle$ drawn from a 2D grid. The ICM kernel $K_{ICM}$ factors into the Kronecker product of two kernels:

$$
K_{ICM} \left( \langle x,t\rangle,\langle x',t'\rangle \right) = K_x(x,x') \otimes K_t(t,t')
$$

In the current scenario, the "inputs" $x,x'$ are drawn from the set of model checkpoints, and the "tasks" $t,t'$ are drawn from the set of benchmarks. When we run a given checkpoint model $x_i$ on a task $t_j$, the output is a scalar valued score $y_{i,j} = f(x_i,t_j)$ representing the performance of the model on the task. These scores are typically values in the $[0..1]$ range.

The $K_x$ kernel accounts for correlations between two outputs $f(x,t)$ and $f(x',t')$ that are due to checkpoint similarities (which we directly relate to inter-checkpoint distance), and the $K_t$ kernel accounts for output correlations due to inter-task correlations. Since the input space $x$ is continuous/numerical, $K_x$ is an RBF kernel as before. However, since tasks are categorical in nature (i.e. no intrinsic ordering) the task kernel $K_t$ is just P.S.D. matrix of inter-task correlations which is learned from the observed data. Instead of using a full rank matrix, though, $K_t$ is sometimes represented using a lower rank approximation of the Cholesky factor $L$ s.t. $K_t = LL^T$, which helps ensure that $K_t$ is P.S.D. Additionally, using a lower rank approximation helps encourage the model to learn correlations between tasks.

Now we can picture our input domain as a 2D matrix (grid) with model checkpoints $X = [1,2,3,…,N]$ along the horizontal dimension, and the space of all benchmark tasks $T= [1,2,3,…,M]$ along the vertical dimension. As a notational convenience we sometimes *vectorize* (i.e. flatten) this $N$x$M$ matrix into a one dimensional *all-pairs* vector $V_{X,T} = X \otimes  T$ of length $MN$, but it represents the exact same quantity.

Now we have a posterior mean $\mu_{i,j}$ and standard deviation $\sigma_{i,j}$ defined for every checkpoint model $x_i$ and  benchmark task $t_j$ combination: $\langle x_i,t_j \rangle$. In a standard Bayesian Optimization setup using a multi-task GP, the objective would be to find the optimal checkpoint-task pair. In other words, we'd want to find the $\langle i^*,j^* \rangle$ (i.e. checkpoint model $x_i^*$ and benchmark task $t_j^*$)  where the posterior mean $\mu_{i,j}|O$ is maximized.

But we are interested in summing (averaging) over the task dimension and finding the optimal checkpoint. So we want to find the $i^*$ (model checkpoint $x_i^*$) where the average over tasks $ \frac{\sum_{j=1}^M \mu_{i,j}|O}{M}$ is maximized. We maximize the sum $\sum_{j=1}^M \mu_{i,j}|O$ instead since max is invariant to division by a constant.

The one question remaining is how to perform step 1 -- how do we use the current regression model to decide which point to query next? In the standard single-output setting, a common method is to maximize an *acquisition function* to decide. An acquisition function is typically defined in such a way to balance our desire to maximize the current regression model and to reduce uncertainty in the regression model. There are many choices, but one of the most popular is called *Expected Improvement*.  

### Expected Improvement

How EI  usually works is that, as we acquire new observations $y_o=f(x_o)$, we continually keep track of the maximum score $y^*$ observed so far.  After every model update step (step 2), we must consider all input points $x_i$ not yet observed as candidates to query for the next observation. At each of these points, the current GP posterior yields a marginal posterior mean $μ_i$ and standard deviation $\sigma_i$.

These parameters define a probability distribution over all possible values $f(x_i)$ the objective function might assume at $x_i$. Using these marginal distributions, we can compute, for each candidate input $x_i$, the *expected value* of the amount that $f(x_i)$ will improve over the current $y^*$:

$$
\mathbb{E}_{f \sim \mathcal{N}(\mu_i, \sigma^2_i)} [\max(0, f(x_i) - y^*)] 
=  \mathbb{E}_f \left[ \max \left(0, \frac{f(x_i) - \mu_i}{\sigma_i} - \frac{y^*-\mu_i}{\sigma_i}\right) \right] \sigma_i
$$

If $f(x_i) < y^*$ there is no improvement, so we use $0$ instead. Now, if we let $v = \frac{y^*-\mu_i}{\sigma_i}$, then the right side contains an expression of the form: $\mathbb{E}_{u \sim  \mathcal{N}(0,1)} [\max(0,u-v)]$, which can be solved analytically:
$$
\begin{aligned} 
\mathbb{E}_u [\max(0,u - v)]  = & \ \int_{v}^{\infty} (u - v) \cdot \phi(u) \, du \\
= & \ \ [\phi(u) - v \cdot (1-\Phi(u))]_{v}^{\infty} \\
= & \ \ \phi(v) - v (1 -\Phi(v))
\end{aligned} 
$$

where $\phi$ and $\Phi$ are the standard Gaussian PDF and CDF, respectively. We can further simplify using $v = \frac{y^*-\mu_i}{\sigma_i}$ plus the identities:
- $\phi(-v) = \phi(v)$
- $\Phi(-v) =1 -\Phi(v)$

Plugging these back in yields the Expected Improvement acquisition function:

$$
EI(x_i) = (\mu_i-y^*)\Phi \left( \frac{\mu_i - y^*}{\sigma_i} \right) + \sigma_i \phi \left( \frac{\mu_i - y^*}{\sigma_i} \right)
$$

The $x_i$ with the greatest $EI(x_i)$ value would be chosen for the next function evaluation.

#### Modification to EI
As mentioned before, we are interested in using BO (Bayesian Optimization) to find the model checkpoint that acheives the highest sum of scores over all benchmark tasks $\sum_{j=1}^M \mu_{i,j}|O$  , which is equivalent to maximizing the average score. We would like to used an acquisition function like EI, but in its standard form it does not quite work for this  modified objective function. At each iteration, we would like to evaluate each possible checkpoint-task pair $\langle x_i,t_j \rangle$ *not* based on the expected improvement over the best individual benchmark score observed so far, $y^*_{x,t}$, but instead by the expected improvement in the sum of scores over all tasks for the given checkpoint.

One problem with this is that we have no direct observations of this sum (of scores over all tasks for a checkpoint). All we have are observations of individual scores resulting from arbitrary checkpoint-task combinations. However, as a proxy we can simply, for each candidate checkpoint $x_i$, sum over all current posterior mean score estimates (i.e. over all tasks):

$$
S_i = \sum_{j=1}^M \mu_{i,j}|O
$$

as well as use the maximum value $S^*$ as a proxy for a hypothetical *highest sum so far observed*.  Now for the purposes of our acquisition function, we would like to be able to treat $S_i$ as a function of the single candidate pair $\langle x_i,t_j \rangle$ under evaluation, keeping all else constant.  We can acheive this by decomposing the sum as:
$$
S_i = f(x_i,t_j) + \sum_{k \neq j} \mu_{i,k}
$$
which allows us to write a modified Expected Improvement function:

$$
\begin{aligned} 
& \ \ \mathbb{E}_{f \sim \mathcal{N}(\mu_{i,j}, \sigma^2_{i,j})} [\max(0,S_i - S^*)]  \\
 = & \ \ \mathbb{E}_{f \sim \mathcal{N}(\mu_{i,j}, \sigma^2_{i,j})} [\max(0,f(x_i,t_j) + \sum_{k \neq j}^{M} \mu_{i,k} - S^*) ]  \\
 = & \ \ \mathbb{E}_{f \sim \mathcal{N}(\mu_{i,j}, \sigma^2_{i,j})} [\max(0,f(x_i,t_j)- (S^* -\sum_{k \neq j}^{M} \mu_{i,k} ))]  \\
& \\
 =  & \ \ \mathbb{E}_f  \left[ \max \left(0, \frac{f(x_i,t_j) - \mu_{i,j}}{\sigma_{i,j}} - \frac{(S^* -\sum_{k \neq j}^{M} \mu_{i,k})-\mu_{i,j}}{\sigma_{i,j}}\right) \right] \sigma_{i,j}
\end{aligned} 
$$

The purpose of the $S_i$ decomposition is that we once again have an expression of the form: $\mathbb{E}_{u \sim  \mathcal{N}(0,1)} [\max(0,u-v)]$, which means we can use the same closed form expression derived earlier for EI, except we replace $y^*$ with $S^* -\sum_{k \neq j}^{M} \mu_{i,k}$:

$$
\begin{aligned} 
EI(x_i,t_j) &= (\mu_{i,j}-S^* +\sum_{k \neq j}^{M} \mu_{i,k})\Phi \left( \frac{\mu_{i,j} - S^* +\sum_{k \neq j}^{M} \mu_{i,k}}{\sigma_{i,j}} \right) + \sigma_{i,j} \phi \left( \frac{\mu_{i,j} - S^* +\sum_{k \neq j}^{M} \mu_{i,k}}{\sigma_{i,j}} \right) \\
& \\
&= \boxed{ (S_i - S^*)\Phi \left( \frac{S_i - S^*}{\sigma_{i,j}} \right) + \sigma_{i,j} \phi \left( \frac{S_i - S^*}{\sigma_{i,j}} \right) }
\end{aligned} 
$$

which we can maximize in order to choose the next checkpoint-task pair $\langle x_i,t_j \rangle$ to evaluate.

