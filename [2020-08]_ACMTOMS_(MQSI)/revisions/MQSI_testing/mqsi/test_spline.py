import fmodpy

SPLINE = fmodpy.fimport(
    "SPLINE.f90",
    dependencies=["REAL_PRECISION.f90", "EVAL_BSPLINE.f90"],
    blas=True,
    lapack=True,
    # verbose=True,
    rebuild=True,
)

# Fit a spline.
def fit(x, y):
    import numpy as np
    x = np.asarray(x, dtype="float64", order="F").flatten()
    y = np.asarray(y, dtype="float64", order="F").reshape(x.size, -1)
    assert x.shape[0] == y.shape[0], f"Number of points in x {x.shape} and y {y.shape} do not match."
    nb = x.shape[0]
    ncc = y.shape[1]
    t = np.zeros(nb*ncc+2*ncc, dtype="float64", order="F")
    bcoef = np.zeros(nb*ncc, dtype="float64", order="F")
    t, bcoef, info = SPLINE.fit_spline(xi=x, fx=y, t=t, bcoef=bcoef)
    assert (info == 0), f"FIT_SPLINE returned nonzero exit code {info}."
    # Given a spline "f", evaluate it.
    def evaluate(x, d=None, t=t, bcoef=bcoef):
        import numpy as np
        x = np.asarray(x, dtype="float64", order="F").flatten()
        y = x.copy()
        y, info = SPLINE.eval_spline(t=t, bcoef=bcoef, xy=y, d=d)
        assert (info == 0), f"EVAL_SPLINE returned nonzero exit code {info}."
        return y
    # Return the evaluator.
    return evaluate


import numpy as np
from tlux.plot import Plot, multiplot

# Seed for repeatability.
np.random.seed(0)

# Set the number of points.
# n = 4 
n = 10

# Set the number of continuity conditions.
cc = 2

# Set the number of evaluation points per interval.
ne = 100
randomized = True

# Random X (sorted)
x = np.random.random(size=(n,))
x *= 10
x.sort()

# Random Y.
y = np.random.random(size=(n,cc))*2-1

# Fit the data.
f = fit(x, y)

# Evaluate the function at data points.
fx = np.concatenate([np.linspace(x[i],x[i+1],ne) for i in range(n-1)])

# Randomize the order of the points.
if randomized:
    np.random.shuffle(fx)

fy = f(fx)

# Get the nonrandom order for plotting.
if randomized:
    i = np.argsort(fx)
else:
    i = np.arange(len(fx))

# Plot the data.
p = Plot("", "Random order data evaluation")
p.add("f", fx[i], fy[i], mode="lines", color=1)
p.add("data", x, y[:,0], color=0)
p.plot()
# p1 = p

# p = Plot("","Sorted order data evaluation")
# p.add("f", fx[i], f(fx[i]), mode="lines", show_in_legend=False, color=1)
# p.add("data", x, y[:,0], show_in_legend=False, color=0)
# p2 = p

# multiplot([p1, p2])
