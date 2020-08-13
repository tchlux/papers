import numpy as np

from util.plot import Plot

# --------------------------------------------------
#      Suggested functions (not useful here).
def oscillatory(x, c=1, w=0):
    return np.cos(2*np.pi*w + np.sum(c*x))
def product_peak(x, c=1, w=0):
    return np.prod((c**(-2) + (x-w)**2)**(-1))
def corner_peak(x, c=1):
    try:    d = len(x)
    except: d = 1
    return (1 + np.sum(c*x))**(-d-1)
def gaussian(x, c=1, w=0, t=1):
    return np.exp( - np.sum(c**2 * t * (x - w)**2) )
def continuous(x, c=1, w=0):
    return np.exp( - np.sum(c * np.abs(x - w)) )
# --------------------------------------------------

# A standard component-wise cosine function.
def cosine(x):
    return np.sum( np.cos(x*np.pi) )

# A quadratic 2-norm bowl about the origin.
def quadratic(x):
    return np.linalg.norm(x)**2

# Return the length of x.
def linear(x):
    return np.sum(x)

dim = 2
test_funcs = [linear, quadratic]

p = Plot()

name = lambda f: str(f).split()[1]

x_range = [0,1] #[-2,2]
y_range = None #[-3,3]

for f in test_funcs:
    p.add_func(name(f), f, *([x_range]*dim))
    
p.add_func("Error", lambda x: linear(x)-quadratic(x),
           *([x_range]*dim), opacity=.5)

p.show(y_range=y_range)
