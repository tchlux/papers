Empirical Analysis, Evaluation, and Comparison of Convex Optimization Techniques
:: Tyler H. Chang :: thchang@vt.edu :: http://people.cs.vt.edu/thchang/
:: Thomas C. H. Lux :: tchlux@vt.edu :: http://people.cs.vt.edu/tchlux/


# Introduction

Convex optimization techniques such as stochastic gradient descent
(SGD), Newton's method, and their variants are widely used in machine
learning applications.  Perhaps most notable, is the wide use of
convex optimization techniques for minimizing neural network loss
functions.  Convex optimization is a well studied field, with many
theoretical and practical guarantees on algorithm convergence for
convex objective functions.  Unfortunately, given the non-convex
nature of most loss landscapes, none of the theoretical analyses apply
in the context of neural network training.

This means that in the context of machine learning, the efficiency of
convex optimization algorithms is measured completely empirically.
This experiment based analysis poses an issue for designing effective
optimization algorithms, since often the only way to measure
performance is by running the algorithms on complex real world
problems.  Even then, an algorithm that performs well on a handful of
problems is not guaranteed to perform well on other problems.  While
authors may argue that their algorithms should heuristically perform
better given some class of neural network loss functions, it is almost
impossible to make any theortical guarantees.

To address this issue, we present an empirical analysis for the
optimization of four analytic objective functions by four well-known
optimization algorithms.  Each of the four objective functions in this
paper has been carefully designed to exhibit some property that
contrasts with those of a "nice" convex function.  Based on literature
and industry usage, the four optimization algorithms considered here
are SGD [[nemirovski2009robust]], L-BFGS 
[[nocedal1980updating,liu1989limited]], AdaGrad [[duchi2013proximal]], 
and Adam [[kingma2014adam]].  It is our hope that by
empirically analyzing the convergence of the above four algorithms on
these functions, we can infer the effects of each designed property on
the convergence of each algorithm.  This offers deeper insight into
how the choice of algorithms can be effected by the properties of the
loss landscape.

In what follows, we will first introduce the four algorithms of
interest, along with a summary of their convergence properties for
convex functions.  Next, we will describe the construction and
rational behind the four analytic objective functions that will be
tested on, along with various parameters of those functions that will
be varied throughout the analysis.  Next, we will describe the
experiment and data collection process.  Finally, we will present
visualizations of the collected data and analyze the results.

# Algorithms

For this analysis, we consider four algorithms commonly used in
machine learning, specifically, for training neural networks.  They
are stochastic gradient descent (SGD), limited-memory BFGS (L-BFGS),
adaptive gradients (AdaGrad), and adaptive moment estimation (Adam).
For the sake of completeness, we will begin by summarizing these four
algorithms and presenting their theoretical convergence guarantees for
convex objective functions.

## SGD

SGD [[nemirovski2009robust]] is a slight modification to the classic 
first-order gradient descent algorithm:
$$
x^{(k+1)} = x^{(k)} - \alpha^{(k)} \nabla f\left(x^{(k)}\right)
$$
where $x^{(k)}$ denotes the $k$th iterate, $\alpha^{(k)}$ denotes the
$k$th step size, and $\nabla f(x)$ denotes the gradient of $f$ at the point
$x$.
For a strongly convex function, gradient descent can be thought of as a 
contraction over the gradient $\nabla f$, with the fixed-point $x^\star$ 
that satisfies $\nabla f(x^\star) = 0$.
Alternatively, gradient descent can be thought of as an iterative minimization 
of the original function $f$ based on its first-order Taylor expansion:
$$
f(x) \approx f(x^{(k)}) + \nabla f(x^{(k)})^T x
$$
subject to the constraint that each iteration can only "move" a distance of
$\alpha^{(k)}/\| f(x^{(k)})\|$.

The difference between SGD and the standard gradient descent algorithm, is that
SGD only assumes access to an approximation $g$ for the true gradient
$\nabla f$.
In theory, the only condition on the approximation $g$ is that the expected
value of $g$ satisfies:
$$
\mathbb{E}[g(x)] = \nabla f(x)
$$
for all $x$.
Because $g$ is only an approximation to $\nabla f$, it is possible that each
$g(x^{(k)})$ could actually be an ascent direction, making the convergence
of SGD non-monotone, even for strongly convex functions.
Because of these random "bad directions," for a fixed step size 
$\alpha^{(k)} = \alpha$, SGD can only converge to within some radius of the 
true optimum $x^\star$, at which point its convergence stalls.
To achieve absolute convergence, the step size must therefore be decayed.
In practice, this means that after holding $\alpha$ constant for some number
of iterations, $\alpha$ should be decayed by some factor $\tau$.

For a strongly convex function and a deterministic gradient, SGD reduces
to standard gradient descent and its convergence is linear.
I.e., given $t$ iterations,
$$
|f(x^{(k)}) - f(x^\star)| \approx \mathcal{O}\left( c^t ) \right)
$$
for some constant $0 < c < 1$.
If $g$ is indeed a stochastic estimate to $\nabla f$, the convergence rate is 
reduced to $\mathcal{O}\left(\frac{1}{t}\right)$.
If furthermore the objective function is only convex (as opposed to strongly
convex), this rate is futher reduced to 
$\mathcal{O}\left(\frac{1}{\sqrt{t}}\right)$.

## L-BFGS

In general, quasi-Newton methods use an approximation to the Hessian 
$\nabla^2 f$ to allow for bigger step sizes in directions of low variance.
For a perfect quadratic, this allows Newton methods to "jump" straight to
the minima; for strongly convex functions, this accomodates poorly
conditioned objective functions by normalizing the sub-level sets of $f$.
The classic Newton update can be derived from a second-order Taylor 
approximation to $f$, and is given by:
$$
x^{(k+1)} = x^{(k)} - \left(\nabla^2 f(x^{(k)})\right)^{-1}\nabla f(x^{(k)}).
$$

The original Broyden-Fletcher-Goldfarb-Shanno algorithm (BFGS) algorithm 
iteratively refines an approximation to the Hessian matrix $H^{(k)}$ by 
applying Rank-1 matrix updates to its current approximation, each of which 
satisfies the secant condition:
$$
H^{(k)}\left(x^{(k)}-x^{(k-1)}\right) = \nabla f(x^{(k)}) - \nabla f(x^{(k-1)}).
$$
Intuitively, this can be thought of as iteratively refining the Hessian based 
on a planar fit to each observed gradient.
The Newton update is defined in terms of the inverse Hessian, so BFGS avoids
performing a matrix inversion by leveraging the Rank-1 Sherman-Morrison-Woodbury
matrix identity:
$$
(H + xy^T)^{-1} = H^{-1} - H^{-1}x(I + v^T H^{-1}x)v^T H^{-1}
$$
where $xy^T$ is a Rank-1 matrix, and $I$ is the identity.
By leveraging this formula, BFGS is able to keep the iteration cost cheap,
since the cost of the Rank-1 update is sigificantly cheaper than the cost
of matrix inversion.

L-BFGS [[nocedal1980updating,liu1989limited]] is a slight modification to BFGS, 
which further reduces iteration and storage cost for high-dimensional problems.
Instead of keeping track of the entire Hessian matrix $H$, L-BFGS stores only
the previous $m$ update vectors ($x$ and $y$ in the Woodbury matrix formula), 
then reconstructs each $H^{(k)}$ on the fly.
If $m=1$, then L-BFGS is reduced to the secant method.
If $m=k_{max}$ (the max-iteration cost) then L-BFGS is equivalent to BFGS,
though the storage and computational cost may be greater or lesser depending
on whether $k_{max}$ is greater than or less than the dimension.
For a typical application, $m$ is strictly less than the dimension, making
this a computationally effecient algorithm.
As an added bonus, the memory limit ensures that L-BFGS can accomodate
non-constant Hessians.

For a strongly convex objective function, L-BFGS converges superlinearly to the 
optimum $x^\star$.
That is, L-BFGS is faster than linear but slower than the quadratic convergence
rate $\mathcal{O}\left( c^{b^t} \right)$ (where both $c$ and $b$ are positive
numbers less than one).
When convexity assumptions are dropped, L-BFGS has no convergence guarantees.
In fact, in the presence of local maxima and saddle-points, most quasi-Newton
methods will converge to both [[dauphin2014identifying]].

## AdaGrad

AdaGrad [[duchi2013proximal]] attempts to recreate the benefits of Newton's 
method without explicitly approximating the Hessian.
To achieve this, AdaGrad directly measures the "variance" in function value
with respect to each basis direction.
Specifically, AdaGrad derives a "variance" matrix $G$ that captures the same
approximate information as L-BFGS, but with much lower cost since $G$ is
always diagonal given an orthonormal basis.
It should be noted that for a strongly convex function, the Hessian $\nabla^2 f$
is always symmetric positive definite (SPD), which immediately implies that
it is columnwise diagonally dominant.
Therefore, for strongly convex functions, a diagonal matrix $G$ derived from 
variance information makes for an excellent approximation to the true Hessian 
$\nabla^2 f$.

The $k$th variance estimate for each dimension of $G$ is given by 
$$
G^{(k)} = diag\left(\sqrt{\sum_{n=1}^k (g^{(n)})^2} + \varepsilon\right)
$$
where each $g^{(n)}$ is a previous gradient (estimate) and $\varepsilon$
is a "fudge" factor, introduced to prevent $G$ from becoming singular
in degenerate cases.
Note that since $G^{(k)}$ is diagonal, it can be readily inverted.
Also, by storing $\left(G^{(k)}\right)^2 - \varepsilon I$ and performing the 
square root and "fudging" operations on demand, $G$ can be iteratively refined 
without tracking previous gradients.
To normalize the sublevel sets and improve the conditioning of $f$, a scaled
version of $G$ can be directly plugged in for $H$ in the quasi-Newton update:
$$
x^{(k+1)} = x^{(k)} - \alpha^{(k)} (G^{(k)})^{-1} g(x^{(k)})
$$
where $\alpha^{(k)}$ is a step size, and $g$ is an approximation to $\nabla f$ 
in the stochastic case and $g = \nabla f$ in the deterministic case.
More intuitively, AdaGrad can also be thought of as a trust region method,
where the variance estimate $G$ allows for larger steps in directions of
low variance.

AdaGrad is guaranteed the same convergence as SGD, but the constant terms
that are ignored by the Big-O notation are siginificantly better for
AdaGrad when the problem is poorly conditioned.

## Adam

Adam [[kingma2014adam]]is combines the idea of variance estimation from 
AdaGrad, with the idea of momentum.
Intuitively, momentum places some weight on previous iterates by replacing
the current gradient estimate $g$ with a weighted average of $g$ and the
previously seen gradients:
$$
\hat{g}^{(k+1)}(x^{(k)}) = \beta g(x^{(k)}) + (1-\beta)\hat{g}^{(k)}.
$$
For stochastic gradient estimates $g$, this has the effect of smoothing over
noise and avoiding wild oscillations;
for poorly conditioned problems, this prevents the iterates $x^{(k)}$ from
wildly oscillating about the optimum descent direction;
and for non-convex functions, this can allow Adam to blow through sharp
minima with poor generalization error.

Leveraging the variance estimate $G$ used by AdaGrad, Adam applies momentum
not only to the gradient (first moment) estimate, but also applies momentum
to the variance matrix $G$ (second moment).
Therefore, the Adam update can be summarized by:
$$
x^{(k+1)} = x^{(k)} - \alpha
\left(\sqrt{\beta_2 g^2(x^{(k)}) + (1-\beta_2)(G^{(k)})^2}\right)^{-1}
\left(\beta_1 g(x^{(k)}) + (1-\beta_1)\hat{g}\right)
$$
where $\beta_1$ and $\beta_2$ are the first and second moment coefficients
respectively, and $G^{(k)}$ is the $k$th "non-fudged" variance estimate
from AdaGrad.
Large momentum coefficients are most helpful for noisy and poorly conditioned
problems.
However, if the momentum coefficient is too large with respect to the step
size $\alpha$, Adam can fail to converge.
Most interesting problems are noisy and poorly conditioned, and
most algorithms tend to converge well for any well-conditioned problem.
So, it is common practice to set $\beta_1 \approx 1$ and $\beta_2 \approx 1$,
then choose the largest convergent step size $\alpha$.

Similarly to AdaGrad, Adam converges at the same rate as SGD but with more 
favorable hidden constants when the problem is poorly conditioned.


# Analytic Objective Functions

In order to empirically evaluate each optimization technique, we present four analytic functions for minimization that each have a single global minimum, but varying other properties. Roughly speaking, they are presented in order of increasing difficulty.

## Me Jr.

A convex sub-quadratic function that appears to come to a sudden point. Quadratic estimators will tend to overshoot the minimum of this function when stepping. The function is defined as

$$ \sum_{i=1}^{d} |x_i|^{\frac{2k}{2k-1}} $$

where $d$ is the dimension of the problem and $k > 1$. A one dimensional plot of the function and its derivative looks like:

{{Mejr-2D.html}}

## Basin

The basin is a convex super-quadratic function used to mimic a phenomenon observed in practice where the region surrounding an optimal point has a gradient whose magnitude goes to zero at a rapidly decreasing rate. Visually, this manifests as a 'flattening' surrounding the global minimum. The function is defined as

$$ \sum_{i=1}^{d} x_i^{2k} $$

where $d$ is the dimension of the problem and $k > 1$. A one dimensional plot of the function and its derivative looks like:

{{Basin-2D.html}}

## Saddle

In problems with more than tens of dimensions, the likelihood of non-uniform curvature between dimensions becomes increasingly likely. When dimensions have opposing curvature, *saddle points* are created. Recent work [[dauphin2014identifying]] has shown that saddle points are a very common occurrence when training neural networks. Analytically we define the following function with exponentially more saddle points with growing dimension.

$$ \sum_{i=1}^{d} \frac{s^4 x^2}{2} - \frac{s^2 x^4}{2} + \frac{x^6}{6}, $$

where $d$ is the dimension of the problem and $s$ is the constant defining the absolute value of the location of saddle points per-dimension. A one dimensional plot of the function and its derivative looks like:

{{Saddle-2D.html}}

## Multimin

Many problems that require optimization have local minima. Using Chebyshev polynomials, we construct a function that has a prescribed number of minima that grows exponentially with increasing dimension.

$$ \sum_{i=1}^{d} \big [  1 + a x_i^2 + f_{2m + 1}(x_i) \big ], $$
$$ \begin{align}
f_0(x_i) &= 1, \\
f_1(x_i) &= x_i, \\
f_{n+1}(x_i) &= 2 x_i f_{n}(x_i) - f_{n-1}(x_i), \\
\end{align} $$

where $d$ is the dimension of the data, $a$ is a multiplier for determining the relative effect size of the quadratic term, and $m$ is the number of local minima per dimension ((The number of local minima in the space will grow as $m^d$. When $m$ is odd there will be one global minimum, this is recommended. When $m$ is even there will be 2^d global minimizers.)). A one dimensional plot of the function and its derivative looks like:

{{Multimin-2D.html}}

# Function Transformations

We use the following transformations to analyze further properties of the functions.

## Noise

The analytic functions that we have defined thus far all have
well-defined, deterministic gradients almost everywhere.  In most
machine learning applications, a subset of the total training data
volume is used in each gradient evaluation, resulting in a stochastic
estimate of the true gradient.  To simulate this reality, we add
various amounts of uniform random noise to each gradient evaluation.
SGD and other first order methods (such as Adam and AdaGrad) are
expected to still converge under these conditions
[[nemirovski2009robust]].  Though the analysis mentions no constraint
on the variance of the noise, we choose to cap the maximum magnitude
of the uniform noise at 25\% of the maximum magnitude of the gradient
for our evaluations.  Let $\|g\|_{L^\infty}$ denote the maximum
magnitude of the gradient.  For each objective function, we run
optimizations with no noise, uniform noise with 12.5\% of
$\|g\|_{L^\infty}$, and uniform noise with 25\% of $\|g\|_{L^\infty}$.

## Skew

The condition number of the sub-level set $C$ of a function $f$ is defined as the ratio between the maximum diameter $W_{max}$ and the minimum diameter
$W_{min}$ across $C$:
$$
\kappa(C) = \frac{W_{max}}{W_{min}}.
$$
For $f$ convex, this is proportional to the conditioning of $f$ as an operator.
Notice that the sublevel sets for all the above functions are approximately square.
This means that without modification, the problems are all well-conditioned, i.e., $\kappa(f) \approx 1$.
To simulate poor problem conditioning, which is common in machine learning applications, we introduce various amounts of skew on $f$.
For each objective function, we run optimizations with no skew, an inverse conditioning ratio of $\frac{W_{min}}{W_{max}} = 0.5$, and
an inverse conditioning of $\frac{W_{min}}{W_{max}} = 0.01$.

## Rotation

Also note that each of the above functions is completely separable, in that it can be decomposed into the sum of its components in each dimension, which
can be optimized separately.
For Adam and AdaGrad, which use diagonal matrices to capture variance in each basis dimension, this means that all the necessarry information can be captured
without need for off-diagonal elements.
However, as the functions are rotated to a maximum angle of 45 degrees, the objective functions become non-separable, making Adam and AdaGrad's variance
approximations poor proxies for the true Hessian.
To simulate non-separability, we run the optimization algorithms on each objective function with no rotation, 22.5 degree rotation, and full 45 degree rotation.

# Implementation and Data Collection

All of the objective functions were implemented in Python, and their gradients were generated using the Python automatic differentiation tool AutoGrad.
The constants for the objective functions are presented in the table below:

| <160>            | <80> | <80> | <80> | <80> |
|                  | $m$  | $k$  | $a$  | $s$  |
|	           |      |      |      |      |
| **Me Jr.**       | N/A  | 2    | N/A  | N/A  |
| **Basin**        | N/A  | 2    | N/A  | N/A  |
| **Saddle Point** | N/A  | N/A  | N/A  | 0.75 |
| **Multi-Min**    | 3    | N/A  | 2    | N/A  |

The four algorithms discussed have been coded in Python, with the exception of L-BFGS, for which the SciPy implementation L-BFGS-B was used.
((As will be seen, due to incorrect usage of the SciPy implimentation of L-BFGS-B, the algorithm prematurely terminated before
converging and we did not have time to re-run those experiments.))
For L-BFGS-B, the default SciPy settings were used.
For all the other algorithms, the hyperparameter settings were based upon recommended settings for $\beta_1$, $\beta_2$, $\tau$, and $\varepsilon$,
while the step size $\alpha$ was tuned.
A table of the selected hyperparameter values is presented below.
For SGD, the decay factor $\tau$ was applied after every 5000 iterations.

| <160>       | <80>     | <80>      | <80>      | <80>   | <80>          |
|             | $\alpha$ | $\beta_1$ | $\beta_2$ | $\tau$ | $\varepsilon$ |
|	      |	     |		 |	     |	      |		      |
| **SGD**     | $0.1$    | N/A       | N/A       | $0.5$  | N/A           |
| **AdaGrad** | $0.01$   | N/A       | N/A       | N/A    | $10^{-6}$     |
| **Adam**    | $0.01$   | $0.9$     | $0.99$    | N/A    | $10^{-8}$     |

Each algorithm was run on each noise level, skew, and rotation of all four objective function in 10 dimensions.
((We originally intended to gather data for 100 and 1000 dimensions as well, but time constraints prevented us from doing so.))
For each noise level/skew/rotation of each objective function, 50,000 iterations were performed over 100 separate trials,
each initialized from a randomly generated starting position in the unit hypercube.
Note that each of the four objective functions has a single global minimum where $f(x) = 0$, and is upper bounded by $f(x) = 1$.

# Results

The two overall best performers for the analytic objective functions given varying noise, skew, and rotation were Stochastic Gradient Descent and ADAM. The following figure shows the median objective value obtained versus number of gradient evaluations for each optimization algorithm over all values for noise, skew, and rotation.

{{all.html}}

Multiple (expected) behaviors of each optimization algorithm can be observed in this plot. For the subquadratic *Me Jr.* function, SGD converges to the radius allowed by step size, hence the appearance of a step function. ADAM performs best on the *Basin* function because its momentum allows it to continue walking closer to the minimum. ADAM performs best on the *Saddle* function because of its strictly positive second-order estimate of function curvature that allows it to escape saddle points. None of the optimization algorithms are able to successfully minimize the *Multimin* problem.

Some unexpected and difficult-to-explain behaviors also occur. It is unclear why SGD suddenly obtains better performance with a specific step size on the *Multimin* problem. Perhaps the step size becomes just the right amount to step out of the local min.

### A Warning

<font color="#b44" face="arial"> The scaling on the following three plots does not stay perfect. By pressing 'Play', then 'Pause' on the desired frame, hover over the plot and press the 'autoscale' tool-tip button to the left of the 'home' button. This will correct the ranges. If you click on the slider itself, it will break the 'autoscale' button until the page is refreshed. Also the transitions between frames are clunky, we will attempt to pretty this up after submission, but working with these interactive plots is tedious and time consuming. </font>

## Convergence by Noise

First, we consider increasing the noise in the gradient of the objective function.

{{noise.html}}

As expected, the addition of noise significantly slows the convergence of all
the algorithms for all the functions.
However, the addition of noise seems to allow both SGD and Adam
to escape local minima in the Multimin function.
This manifests in the fact that both take longer to converge initially, but 
ultimately manage to converge much closer to the true minimum when noise is
added.

## Convergence by Skew

Next, we consider increasing the skew (i.e., deteriorating the conditioning)
of the objective functions.

{{skew.html}}

Interestingly, SGD seems to be less affected by the skew that Adam and AdaGrad.
This is unexpected behavior, since the theoretical analyses for both Adam and AdaGrad claim that they should exhibit better constants than SGD for poorly conditioned problems.

## Convergence by Rotation

Finally, we consider various rotations of the objective functions, which
correspond to non-separability.
We had considered that this could negatively impact Adam and AdaGrad, but as
seen below, none of the algorithms are significantly affected.

{{rotation.html}}

The only algorithm that exhibits any response to rotation is SGD, which fails
to drop exactly to zero for Me Jr, instead taking its signature stair step path
downward

# Conclusion

In this paper, we have presented a side-by-side convergence analysis for four
common convex optimization algorithms on four analytic objective functions
exhibiting specific properties, under various amounts of noise, skew, and
rotation.
Most of our empirical results back the theoretical results, but there are a 
few exceptions.
Most notably, Adam and AdaGrad do not display the same robustness to poor conditioning that was promised in their analyses.
Also interestingly, the presence of noise can be beneficial to both SGD and
Adam in a multimin space, allowing them to escape local minima.

============ Bibliography ============


%% Stochastic gradient descent
@article{nemirovski2009robust,
  title={Robust stochastic approximation approach to stochastic programming},
  author={Nemirovski, Arkadi and Juditsky, Anatoli and Lan, Guanghui and Shapiro, Alexander},
  journal={SIAM Journal on optimization},
  url={https://search.proquest.com/docview/920840954?pq-origsite=gscholar},
  volume={19},
  number={4},
  pages={1574--1609},
  year={2009},
  publisher={SIAM}
}

%% L-BFGS
@article{nocedal1980updating,
  title={Updating quasi-Newton matrices with limited storage},
  author={Nocedal, Jorge},
  journal={Mathematics of computation},
  url={http://www.ams.org/journals/mcom/1980-35-151/S0025-5718-1980-0572855-7/S0025-5718-1980-0572855-7.pdf},
  volume={35},
  number={151},
  pages={773--782},
  year={1980}
}

%% L-BFGS more recent paper.
@article{liu1989limited,
  title={On the limited memory BFGS method for large scale optimization},
  author={Liu, Dong C and Nocedal, Jorge},
  journal={Mathematical programming},
  url={http://people.sc.fsu.edu/~inavon/5420a/liu89limited.pdf},
  volume={45},
  number={1-3},
  pages={503--528},
  year={1989},
  publisher={Springer}
}

%% AdaGrad
@article{duchi2013proximal,
  title={Proximal and First-Order Methods for Convex Optimization},
  author={Duchi, John C and Singer, Yoram},
  url={https://pdfs.semanticscholar.org/2ddf/1465ec6026c16d47a5307ae94d834320cc5b.pdf},
  journal={Interpretation},
  volume={2},
  pages={2},
  year={2013}
}

%% ADAM
@article{kingma2014adam,
  title={Adam: A method for stochastic optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  url={https://arxiv.org/pdf/1412.6980.pdf},
  journal={arXiv preprint arXiv:1412.6980},
  year={2014}
}

%% Saddle Point problems
@inproceedings{dauphin2014identifying,
  title={Identifying and attacking the saddle point problem in high-dimensional non-convex optimization},
  author={Dauphin, Yann N and Pascanu, Razvan and Gulcehre, Caglar and Cho, Kyunghyun and Ganguli, Surya and Bengio, Yoshua},
  booktitle={Advances in neural information processing systems},
  pages={2933--2941},
  year={2014}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Convergence rate evaluation of DFO techniques
@inproceedings{lux2016convergence,
  title={Convergence Rate Evaluation of Derivative-Free Optimization Techniques},
  author={Lux, Thomas},
  booktitle={International Workshop on Machine Learning, Optimization and Big Data},
  url={https://link.springer.com/chapter/10.1007/978-3-319-51469-7_21?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3B5GElqwlOSk%2BfdW8%2Bl1aK3A%3D%3D},
  pages={246--256},
  year={2016},
  organization={Springer}
}

%% DFO comparison
@article{rios2013derivative,
  title={Derivative-free optimization: a review of algorithms and comparison of software implementations},
  author={Rios, Luis Miguel and Sahinidis, Nikolaos V},
  journal={Journal of Global Optimization},
  url={https://link.springer.com/article/10.1007/s10898-012-9951-y},
  volume={56},
  number={3},
  pages={1247--1293},
  year={2013},
  publisher={Springer}
}

%% DFO comparison
@article{civicioglu2013conceptual,
  title={A conceptual comparison of the Cuckoo-search, particle swarm optimization, differential evolution and artificial bee colony algorithms},
  author={Civicioglu, Pinar and Besdok, Erkan},
  journal={Artificial intelligence review},
  url={https://search.proquest.com/docview/1317666742},
  volume={39},
  number={4},
  pages={315--346},
  year={2013},
  publisher={Springer}
}

%% DFO comparison
@article{more2009benchmarking,
  title={Benchmarking derivative-free optimization algorithms},
  author={More, Jorge J and Wild, Stefan M},
  journal={SIAM Journal on Optimization},
  url={https://search.proquest.com/docview/880069333},
  volume={20},
  number={1},
  pages={172--191},
  year={2009},
  publisher={SIAM}
}

%% Optimization study
@book{floudas2008encyclopedia,
  title={Encyclopedia of optimization},
  author={Floudas, Christodoulos A and Pardalos, Panos M},
  url={https://scholar.google.com/scholar?q=Floudas%2C%20C.A.%2C%20Pardalos%2C%20P.M.%3A%20Encyclopedia%20of%20Optimization.%20Springer%20Science%20and%20Business%20Media%2C%20Heidelberg%20%282009%29},
  year={2008},
  publisher={Springer Science \& Business Media}
}

%%%%%%%%%%%%%%%%%%%%%% UNUSED REFERENCES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% MNIST image classification
@article{lecun1998mnist,
  title={The MNIST database of handwritten digits},
  author={LeCun, Yann},
  url={http://yann.lecun.com/exdb/mnist/},
  year={1998}
}

%% CIFAR image classificaiton
@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey},
  year={2009},
  url={https://www.cs.toronto.edu/~kriz/cifar.html},
  publisher={Citeseer}
}
