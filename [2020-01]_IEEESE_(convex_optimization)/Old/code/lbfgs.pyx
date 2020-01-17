from numpy import zeros, dot, sqrt, roll, asarray, array

STEPS = 50000

# L-BFGS minimization (written in regular python).
def l_bfgs(func, grad, start, budget=STEPS, m=0, alpha=.99, eps=2.**(-56)):
    points = []
    dim = len(start)
    if (m == 0): m = max(10, int(sqrt(dim)))
    # Initialize storage for internal variables.
    x = start
    g = zeros(dim)
    s = zeros((m, dim))
    y = zeros((m, dim))
    rho = zeros(m)
    a = zeros(m)
    b = zeros(m)
    old_x = zeros(dim)
    old_g = zeros(dim)
    cdef int i,j
    # Loop until the budget is exhausted.
    for i in range(budget):
        points.append(x.copy())
        g = grad(x)
        # Very first step does different stuff for initialization.
        if (i == 0):
            old_x[:] = x[:]
            old_g[:] = g[:]
            x -= alpha * old_g
            continue
        # Normal case.
        # 
        # Roll the history arrays (free the first index) and also
        # check for termination (numerically unstable small step).
        # 
        s = roll(s, 1, axis=0)
        y = roll(y, 1, axis=0)
        rho = roll(rho, 1, axis=0)
        s[0,:] = x[:] - old_x[:]
        y[0,:] = g[:] - old_g[:]
        ys = (dot(y[0], s[0])) # <- Original L-BFGS doesn't "abs",
        #                              but for noisy functions this is
        #                              a necessary change to continue.
        if (sqrt(abs(ys)) < eps): break
        rho[0] = 1 / ys
        # Copy current iterates for next iteraction.
        old_x[:] = x[:]
        old_g[:] = g[:]
        # Reconstruct the BFGS update.
        for j in range(min(m, i)):
            a[j] = rho[j] * dot(s[j], g)
            g -= a[j] * y[j]
        g *= (ys / dot(y[0],y[0]))
        for j in range(min(m, i)-1, -1, -1):
            b[j] = rho[j] * dot(y[j], g)
            g += s[j] * (a[j] - b[j])
        # Compute the rescaled update.
        x -= alpha * g
    else:
        # If the for loop didn't break, then the last point needs to be recorded.
        points.append( x.copy() )
    # Return the set of points.
    return points


# # ====================================================================
# #            Failed attempt at optimizing the Cython code     
# # 
# # 
# # L-BFGS minimization (written in regular python).
# def l_bfgs(func, grad, double[:] start, int budget=STEPS, int m=0,
#            float alpha=.99, float eps=2.**(-56)):
#     points = []
#     cdef int dim, i, j, c
#     dim = len(start)
#     if (m == 0): m = max(10, int(sqrt(dim)))
#     # Initialize storage for internal variables.
#     cdef double[:] x = start
#     cdef double[:] g = zeros(dim)
#     cdef double[:,:] s = zeros((m, dim))
#     cdef double[:,:] y = zeros((m, dim))
#     cdef double[:] rho = zeros(m)
#     cdef double[:] a = zeros(m)
#     cdef double[:] b = zeros(m)
#     cdef double[:] old_x = zeros(dim)
#     cdef double[:] old_g = zeros(dim)
#     cdef double a_minus_b
#     # Loop until the budget is exhausted.
#     for i in range(budget):
#         points.append(x.copy())
#         # Very first step does different stuff for initialization.
#         if (i == 0):
#             # old_x = x
#             for c in range(dim): old_x[c] = x[c]
#             # old_g = grad(x)
#             g = grad(asarray(x))
#             for c in range(dim): old_g[c] = g[c]
#             # x -= alpha * old_g
#             for c in range(dim): x[c] -= alpha * old_g[c]
#             continue
#         # Normal case.
#         g = grad(asarray(x))
#         # Roll the history arrays (free the first index) and also
#         # check for termination (numerically unstable small step).
#         # 
#         s = roll(s, 1, axis=0)
#         # for j in range(m-1):
#         #     for c in range(dim): s[j+1,c] = s[j,c]
#         y = roll(y, 1, axis=0)
#         # for j in range(m-1):
#         #     for c in range(dim): y[j+1,c] = y[j,c]
#         rho = roll(rho, 1, axis=0)
#         # for j in range(m-1): rho[j+1] = rho[j]
#         # s[0,:] = x[:] - old_x[:]
#         for c in range(dim): s[0,c] = x[c] - old_x[c]
#         # y[0,:] = g[:] - old_g[:]
#         for c in range(dim): y[0,c] = g[c] - old_g[c]
#         ys = abs(dot(y[0], s[0])) # <- Original L-BFGS doesn't "abs",
#         #                              but for noisy functions this is
#         #                              a necessary change to continue.
#         if (sqrt(ys) < eps): break
#         rho[0] = 1 / ys
#         # Copy current iterates for next iteraction.
#         old_x[:] = x[:]
#         old_g[:] = g[:]
#         # Reconstruct the BFGS update.
#         for j in range(min(m, i)):
#             a[j] = rho[j] * dot(s[j], g)
#             # g -= a[j] * y[j]
#             for c in range(dim): g[c] -= a[j] * y[j,c]
#         g *= (ys / dot(y[0],y[0]))
#         for j in range(min(m, i)-1, -1, -1):
#             b[j] = rho[j] * dot(y[j], g)
#             # g += s[j] * (a[j] - b[j])
#             a_minus_b = a[j] - b[j]
#             for c in range(dim): g[c] = s[j,c] * a_minus_b
#         # Compute the rescaled update.
#         # x -= alpha * g
#         for c in range(dim): x[c] -= alpha * g[c]
#     else:
#         # If the for loop didn't break, then the last point needs to
#         # be recorded.
#         points.append( array(x) )
#     # Return the set of points.
#     return points
