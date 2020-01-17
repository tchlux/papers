# This python module provides three different (popular) gradient based
# optimization techniques.


from numpy import finfo, float64, sqrt, ones, zeros, dot, roll

# Global parameters.
FLOAT64_EPSILON = finfo(float64).eps
STEPS = 100
# Stochastic gradient descent default parameters.
SGD_ALPHA = .1
SGD_TAU = 0.5
# AdaGrad default parameters.
ADG_ALPHA = .01
ADG_EPS = 10.0**(-6.0)
# ADAM default parameters.
ADAM_ALPHA = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99
ADAM_EPS = 10.0**(-8.0)


# L-BFGS minimization.
def L_BFGS(func, grad, start, budget=STEPS, m=0,
           alpha=.99, eps=2.**(-56)):
    # Automatically compile the "lbfgs.pyx" file in current directory.
    import pyximport; pyximport.install()
    from lbfgs import l_bfgs
    return l_bfgs(func, grad, start, budget, m, alpha, eps)


# Standard SGD optimization algorithm.
#    grad() - Function that returns the gradient at any point.
#    start  - Contains the starting iterate.
#    budget - Optional argument that sets the number of gradient evaluations.
#    alpha  - Optional argument containing the step size.
#    tau    - Optional argument containing the decay factor.
def SGD(func, grad, start, budget=STEPS, alpha=SGD_ALPHA, tau=SGD_TAU):
    points = [start]
    # initialize
    x = start
    # main loop
    for t in range(0,budget):
        # update step
        x = x - alpha * grad(x)
        # decay 10 times over the course of 
        if ((t > 0) and (not t%(budget//10))):
            alpha = max(alpha * tau, FLOAT64_EPSILON)
        points.append(x)
    return points


# ADAGRAD optimization algorithm as described by Duchi, Hazan, and Singer.
#   grad() - Function that returns the gradient at any point.
#   start  - Contains the starting iterate.
#   budget - Optional argument containing the gradient evaluation budget.
#   alpha  - Optional argument containing the step size in the space induced by G.
#   eps    - Optional argument containing the "Fudge factor" for adjusting G.
def ADAGRAD(func, grad, start, budget=STEPS, alpha=ADG_ALPHA, eps=ADG_EPS):
    points = [start]
    # initialize matrix norm
    x = start
    G = 0
    # main loop
    for t in range(0, budget):
        # get the gradient
        g = grad(x)
        # update the norm
        G = G + (g ** 2.0)
        # take the adjusted trust region step
        g = g / (eps + sqrt(G))
        x = x - (alpha * g)
        points.append(x)
    return points


# ADaM optimization algorithm as proposed by D. P. Kingma and J. L. Ba.
#   grad() - Function that returns the gradient at any point.
#   start  - Contains the starting iterate.
#   budget - Optional argument that sets the number of gradient evaluations.
#   alpha  - Optional argument containing the step size.
#   beta1  - Optional argument containing the decay rate for first moment.
#   beta2  - Optional argument containing the decay rate for second moment.
#   eps    - Optional argument containing the fudge factor.
def ADAM(func, grad, start, budget=STEPS, alpha=ADAM_ALPHA, 
         beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=ADAM_EPS):
    points = [start]
    # initialize
    x = start
    m = 0.0
    v = 0.0
    # main loop
    for t in range(0, budget):
        # get gradient
        g = grad(x)
        # compute biased first and second moments
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        # correct for bias
        m_hat = m / (1.0 - beta1 ** float(t+1))
        v_hat = v / (1.0 - beta2 ** float(t+1))
        # update step
        x = x - alpha * m_hat / (sqrt(v_hat) + eps)
        points.append(x)
    return points


algorithms = [L_BFGS, SGD, ADAGRAD, ADAM]
