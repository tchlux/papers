import os
# Turn off the tensorflow warning messages (about sub-optimality).
#   0 = all messages are logged (default behavior)
#   1 = INFO messages are not printed
#   2 = INFO and WARNING messages are not printed
#   3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------------------------------------------------------------------
# Watson test.
from util.math import Spline
y = [[0,1],[1,0],[1,0],[1,0],[0,0], # Plateu with curvature only on ends.
     [20,-1],[19,-1],[18,-1],[17,-1], # Sudden linear segment
     [0,0],[0,0],[3,0],[0,0], # Sudden flat with one peak
     [1,3],[6,9], # Exponential growth
     [16,.1],[16.1,.1], # small linear growth
     [1,-15]]
x = list(range(len(y)))
# Convert to cover the unit interval.
x = [v/(len(y)-1) for v in x]
for i in range(len(y)): y[i][1] *= (len(y)-1)
# Store the test.
f = Spline(x, y)
df = f.derivative()
ddf = df.derivative()
piecewise_polynomial = (f, df, ddf)
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# Atanh
import numpy as np
f = np.arctanh
df = lambda x: 1 / (1 - x**2)
ddf = lambda x: (2*x) / ((1-x**2)**2)
atanh = (f, df, ddf)
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Exponential.
import numpy as np
f = np.exp
df = np.exp
ddf = np.exp
exponential = (f, df, ddf)
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Trig.
import numpy as np
f = np.sin
df = np.cos
ddf = lambda x: -np.sin(x)
trig = (f, df, ddf)
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Signal function.
from jax import grad
from jax.numpy import sin, pi
f = lambda x: sin(4 * (2*pi) * x) / (x**2 + .1)
df = grad(f)
ddf = grad(df)
signal = (f, df, ddf)

# --------------------------------------------------------------------
# Large tangent test.
from jax import grad
f = lambda x: -(1.0 + ( 1.0 / (x-1.01) ))
df = grad(f)
ddf = grad(df)
large_tangent = (f, df, ddf)

# # --------------------------------------------------------------------
# # Random data test.
# from jax import grad
# from jax.numpy import random
# f = lambda x: random.random(size=len(x)) if hasattr(x,"__len__") else random.random()
# df = grad(f)
# ddf = grad(df)
# random = (f, df, ddf)

# # --------------------------------------------------------------------
# # Random monotone data test.
# from jax import grad
# from jax.numpy import random
# f = lambda x: sorted(random.random(size=len(x))) if hasattr(x,"__len__") else random.random()
# df = grad(f)
# ddf = grad(df)
# random_monotone = (f, df, ddf)


