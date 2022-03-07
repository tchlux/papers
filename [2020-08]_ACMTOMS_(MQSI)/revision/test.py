import numpy as np


# Given data points and values, construct and return a function that
# evaluates a shape preserving quadratic spline through the data.
def toms_574_spline(x, y, pts=1000):
    import toms574
    n = len(x)
    x = np.asarray(x, dtype=np.float32)
    x_min = x.min()
    x_max = x.max()
    y = np.asarray(y, dtype=np.float32)
    m = np.zeros(n, dtype=np.float32)
    # Get the slopes for the quadratic spline interpolant.
    toms574.slopes(x, y, m, n)
    # Define a function for evaluating the quadratic spline.
    def eval_quadratic(z, x=x, y=y, m=m):
        # Make sure "z" is an array.
        if (not issubclass(type(z), np.ndarray)): z = [z]
        # Clip input positions to be inside evaluated range.
        z = np.clip(z, min(x), max(x))
        # Initialize all arguments for calling TOMS 574 code.
        k = len(z)
        z = np.asarray(z, dtype=np.float32)
        z.sort()
        fz = np.zeros(k, dtype=np.float32)
        eps = 2**(-13)
        err = 0
        # Call TOMS 574 code and check for errors.
        err = toms574.meval(z, fz, x, y, m, n, k, eps, err)[-1]
        assert (err == 0), f"Nonzero error '{err}' when evaluating spline."
        # Return values.
        return fz
    def deriv_1(z, s=.001):
        try: return [deriv_1(v) for v in z]
        except:
            if min(abs(z - x_min), abs(z - x_max)) < 0.0001: return 0.0
            from util.math import fit_polynomial
            # Construct a local quadratic approximation.
            px = np.array([z-s, z, z+s])
            py = eval_quadratic(px)
            f = fit_polynomial(px, py)
            return f.derivative()(z)
    def deriv_2(z, s=.001):
        try: return [deriv_2(v) for v in z]
        except:
            if (min(abs(z - x_min), abs(z - x_max)) <= s): return 0.0
            from util.math import fit_polynomial
            # Construct a local quadratic approximation.
            # px = np.array([z-s, z-s/2, z, z+s/2, z+s])
            px = np.array([z-s, z, z+s])
            py = eval_quadratic(px)
            f = fit_polynomial(px, py)
            return f.derivative().derivative()(z)
    deriv_1.derivative = deriv_2
    eval_quadratic.derivative = deriv_1
    return eval_quadratic


# Construct and return a Schumaker spline fit to data.
def schumaker_spline(x, y):
    import numpy as np
    # Update the "EDITOR" environment variable, otherwise rpy2 fails to
    #  find standard libraries 'util' and 'stats'.
    import os
    os.environ.update({
        'EDITOR': '/Applications/Emacs.app/Contents/MacOS/Emacs',
    })
    # Import rpy2 to call R packages from Python.
    import rpy2
    # For managing and executing R codes, as well as importing packages
    from rpy2.robjects.packages import importr
    # Import functions for converting to and from R vector formats.
    from rpy2.robjects.numpy2ri import py2rpy, rpy2py
    # Import the "schumaker" package from R.
    schumaker = importr('schumaker')
    # Calculate the response for the given x with R
    edge_gradient = py2rpy(np.asarray([0.0, 0.0]))
    Spline, DerivativeSpline, SecondDerivativeSpline, IntervalTab = (
        schumaker.Schumaker(py2rpy(x), py2rpy(y), edgeGradients=edge_gradient, Vectorised=True)
    )
    # Construct python functions for evaluating the fit spline.
    def second_derivative(x, ddf=SecondDerivativeSpline):
        return rpy2py(ddf(py2rpy(x))).flatten()
    def first_derivative(x, df=DerivativeSpline):
        return rpy2py(df(py2rpy(x))).flatten()
    def function(x, f=Spline):
        return rpy2py(f(py2rpy(x))).flatten()
    function.derivative = first_derivative
    first_derivative.derivative = second_derivative
    # Return the function (with derivatives).
    return function


# PCHIP spline fit of data.
def pchip_spline(x, y):
    from scipy.interpolate import PchipInterpolator
    # Generate piecewise cubic monotone increasing spline
    spline_fit = PchipInterpolator(x, y)
    # Define a fit function that extrapolates properly.
    def function(x):
        return spline_fit(x)
    # Store a derivative (that is a PDF).
    def derivative(x=None, deriv=spline_fit.derivative(1)):
        return deriv(x)
    # Store a second derivative.
    def second_derivative(x=None, deriv=spline_fit.derivative(2)):
        return deriv(x)
    # Set details for the derivative function.
    function.derivative = derivative
    derivative.derivative = second_derivative
    # Return the fit functions.
    return function


# Spline algorithm from TOMS 770.
def bvspis_spline(x, y):
    import numpy as numpy
    import bvspis
    ftype = dict(dtype="float64", order='F')
    np = x.size - 1
    x = numpy.asarray(x, **ftype)
    y = numpy.asarray(y, **ftype)
    n = 6 # degree of the spline (minimum of 3*k)
    k = 2 # class of continuity
    r = 1 # monotonicity constraints
    q = 1 # no boundary condition is imposed
    p = 1 # constraint only derivative estimation
    opt = 100*p + 10*q + r
    eps = 2**(-26) # square root of machine precision for IEEE 64-bit float
    # The following parameters are unused (only used when q = 3).
    d0 = 0.0 
    dnp = 0.0
    d20 = 0.0
    d2np = 0.0
    # The following parameters are unused (only used when q = 2).
    beta = 0.0
    betai = 0.0
    rho = 0.0
    rhoi = 0.0
    kmax = 0
    maxstp = 0
    # Impose monotonicity constraints on all intervals, referenced for r=4.
    constr = numpy.ones(np+1, dtype="int32", order='F')
    # Derivative values (on output)
    d = numpy.zeros(np+1, **ftype)
    errc = 0 # error flag
    d2 = numpy.zeros(np+1, **ftype) # referenced when k=2
    diagn = numpy.zeros(np, dtype="int32", order='F') # diagnostic info
    nwork = 5 + (2+7)*np + (n*(n+11))//2 + 9
    work = numpy.zeros(nwork, **ftype)
    # Call the routine to fit the spline.
    output = bvspis.dbvssc(x, y, np, n, k, opt, d0, dnp, d20, d2np, constr,
                           eps, beta, betai, rho, rhoi, kmax, maxstp, errc,
                           d, d2, diagn, work, nwork)
    (x, y, np, n, k, opt, d0, dnp, d20, d2np, constr,
     eps, beta, betai, rho, rhoi, kmax, maxstp, errc,
     d, d2, diagn, work, nwork) = output
    assert (errc == 0), f"The DBVSSC routine reported a nonzero error {errc}. Check the documentation for more information about ERRC."
    # Use the evaluation routine separately.
    sbopt = 2 # use a binary search to find intervals for each evaluation point
    # Spline fit evaluation.
    def function(xtab, x=x, y=y, np=np, n=n, k=k, sbopt=sbopt,
                 errc=errc, d=d, d2=d2, work=work, nwork=nwork):
        ntab = xtab.size-1
        active_opt = numpy.ones(ntab+1)
        y0opt = 1
        y1opt = 0
        y2opt = 0
        y0tab = numpy.zeros(ntab+1, **ftype)
        y1tab = numpy.zeros(ntab+1, **ftype)
        y2tab = numpy.zeros(ntab+1, **ftype)
        erre = 0
        output = bvspis.dbvsse(x, y, np, n, k, xtab, ntab, sbopt, y0opt,
                               y1opt, y2opt, errc, d, d2, y0tab, y1tab,
                               y2tab, erre, work, nwork)
        (x, y, np, n, k, xtab, ntab, sbopt, y0opt, y1opt, y2opt, 
         errc, d, d2, y0tab, y1tab, y2tab, erre, work, nwork) = output
        assert (erre == 0), f"The DBVSSC routine reported a nonzero error ERRE = {erre}. Check the documentation for more information about ERRE."
        assert (errc == 0), f"The DBVSSC routine reported a nonzero error ERRC = {errc}. Check the documentation for more information about ERRC."
        return y0tab
    # First derivative evaluation.
    def df(xtab, x=x, y=y, np=np, n=n, k=k, sbopt=sbopt,
           errc=errc, d=d, d2=d2, work=work, nwork=nwork):
        ntab = xtab.size-1
        active_opt = numpy.ones(ntab+1)
        y0opt = 0
        y1opt = 1
        y2opt = 0
        y0tab = numpy.zeros(ntab+1, **ftype)
        y1tab = numpy.zeros(ntab+1, **ftype)
        y2tab = numpy.zeros(ntab+1, **ftype)
        erre = 0
        output = bvspis.dbvsse(x, y, np, n, k, xtab, ntab, sbopt, y0opt,
                               y1opt, y2opt, errc, d, d2, y0tab, y1tab, 
                               y2tab, erre, work, nwork)
        (x, y, np, n, k, xtab, ntab, sbopt, y0opt, y1opt, y2opt, 
         errc, d, d2, y0tab, y1tab, y2tab, erre, work, nwork) = output
        assert (erre == 0), f"The DBVSSC routine reported a nonzero error ERRE = {erre}. Check the documentation for more information about ERRE."
        assert (errc == 0), f"The DBVSSC routine reported a nonzero error ERRC = {errc}. Check the documentation for more information about ERRC."
        return y1tab
    # Second derivative evaluation.
    def ddf(xtab, x=x, y=y, np=np, n=n, k=k, sbopt=sbopt,
            errc=errc, d=d, d2=d2, work=work, nwork=nwork):
        ntab = xtab.size-1
        active_opt = numpy.ones(ntab+1)
        y0opt = 0
        y1opt = 0
        y2opt = 1
        y0tab = numpy.zeros(ntab+1, **ftype)
        y1tab = numpy.zeros(ntab+1, **ftype)
        y2tab = numpy.zeros(ntab+1, **ftype)
        erre = 0
        output = bvspis.dbvsse(x, y, np, n, k, xtab, ntab, sbopt, y0opt,
                               y1opt, y2opt, errc, d, d2, y0tab, y1tab, 
                               y2tab, erre, work, nwork)
        (x, y, np, n, k, xtab, ntab, sbopt, y0opt, y1opt, y2opt, 
         errc, d, d2, y0tab, y1tab, y2tab, erre, work, nwork) = output
        assert (erre == 0), f"The DBVSSC routine reported a nonzero error ERRE = {erre}. Check the documentation for more information about ERRE."
        assert (errc == 0), f"The DBVSSC routine reported a nonzero error ERRC = {errc}. Check the documentation for more information about ERRC."
        return y2tab
    # Stack and return the function.
    function.derivative = df
    function.derivative.derivative = ddf
    return function


# Produce a piecewise monotone spline fit of the specified continuity.
def monotone_fit(x, y, c=3, steps=10000, root=1, step_size=0.01, multiplier=0.0001):
    import monotone_fit as monofit
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Create a monotone fit of the data (using an optimization procedure).
    fx = np.zeros(shape=(x.size, c), dtype=float, order='F')
    t = np.zeros(shape=(x.size*c+2*c), dtype=float, order='F')
    bcoef = np.zeros(shape=(x.size*c), dtype=float, order='F')
    monofit.monotone_fit(x, y, c, steps//2, steps//2, root, step_size,
                         multiplier, fx, t, bcoef)
    # Evaluate the spline fit.
    def fit(x, t=t, bcoef=bcoef, d=0):
        xy = np.array(x, dtype=float)
        xy, info = monofit.eval_spline(t, bcoef, xy, d=d)
        assert info == 0, f"EVAL_SPLINE produced nonzero info {info}."
        return xy
    fit.derivative = lambda x: fit(x, d=1)
    fit.derivative.derivative = lambda x: fit(x, d=2)
    # Evaluate a b-spline with a given knot sequence.
    def bspline(x, t=np.asarray([0.,0,0,1,1,1]), d=0):
        t = t * 0.8 + 0.1
        xy = np.array(x, dtype=float)
        spline.eval_bspline(t, xy, d=d)
        return xy
    # Return the fit function.
    return fit


# Construct and returnn an MQSI spline fit to data.
def mqsi_spline(x, y):
    import numpy as np
    import mqsi
    # Initialize all arguments to MQSI.
    ftype = dict(dtype="float64", order='F')
    x = np.asarray(x, **ftype)
    y = np.asarray(y, **ftype)
    nd = x.size
    t = np.zeros(3*nd + 6, **ftype)
    bcoef = np.zeros(3*nd, **ftype)
    uv = np.zeros((nd, 2), **ftype)
    # Use MQSI to fit a spline to the data.
    t, bcoef, info, uv = mqsi.mqsi(x, y, t, bcoef, uv=uv)
    assert (info == 0), f"mqsi.mqsi subroutine returned nonzero info stats '{info}'. See help documentation."
    # Construct functions for evaluating the fit spline and derivatives.
    def function(x, t=t, bcoef=bcoef, d=None):
        x = np.array(x, dtype='float64', order='F')
        y, info = mqsi.eval_spline(t, bcoef, x, d=d)
        assert (info == 0), f"mqsi.eval_spline subroutine returned nonzero info stats '{info}'. See help documentation."
        return y
    function.derivative = lambda x: function(x, d=1)
    function.derivative.derivative = lambda x: function(x, d=2)
    # Return the fit function.
    return function


# Given a spline fit function and some data, plot the fit over the
#  interval covered by the data.
def plot_spline_fit(x, y, spline_fit, name=None, p=None, color=None,
                    dash=None, d1=False, d2=False, plot_points=5000):
    print("spline_fit: ",spline_fit)
    print()
    spline = spline_fit(x, y)
    # Evauate the spline.
    zx = np.linspace(1,6, round(5 / 0.01))
    zy = spline(zx)
    dzy = spline.derivative(zx)
    ddzy = spline.derivative.derivative(zx)
    # Create a Plot object if one was not given.
    if p is None:
        p = Plot()
    # Assign a color if none was given.
    if color is None:
        p.color_num += 1
        color = p.color_num
    # Assign a name if none was given.
    if name is None:
        name = str(spline_fit).split()[1].replace("_"," ").title()
    # Get the bounds of the data to evaluate.
    bounds = [x.min(), x.max()]
    # Evaluate the fit and two derivatives.
    p.add_func(f"{name}", spline, bounds, vectorized=True, color=color, group=color, dash=dash, plot_points=plot_points)
    if d1:
        p.add_func("First derivative", spline.derivative, bounds, vectorized=True,
                   color=color, group=color, dash="dash", plot_points=plot_points)
    if d2:
        p.add_func("Second derivative", spline.derivative.derivative, bounds, vectorized=True,
                   color=color, group=color, dash="dot", plot_points=plot_points)
    # Return the plot.
    return p



# from test_functions import signal as f_df_ddf
# from test_functions import trig as f_df_ddf
# from test_functions import large_tangent as f_df_ddf
from test_functions import piecewise_polynomial as f_df_ddf


# n = 8 
# x = np.linspace(0, 1, n)
# y = np.log(x)
# y = np.asarray(f_df_ddf[0](x), dtype=float)

x = np.asarray([
    0.025926231827891333, 0.13457994534493356, 0.18443986564691528, 0.2046486340378425, 0.26682727510286663, 0.29965467367452314, 0.3303348210038741, 0.42036780208748903, 0.4353223926182769, 0.43599490214200376, 0.5135781212657464, 0.5291420942770391, 0.5496624778787091, 0.6192709663506637, 0.6211338327692949, 0.7853351478166735
])
y = np.asarray([
    0.06528650438687811, 0.079645477009061, 0.09653091566061256, 0.10694568430998297, 0.12715997170127746, 0.20174322626496533, 0.2203062070705597, 0.22601200060423587, 0.34982628500329926, 0.42812232759738944, 0.46778748458230024, 0.4942368373819278, 0.505246090121704, 0.5967453089785958, 0.846561485357468, 0.8539752926394888
])

print()
print("x: ",x)
print("y: ",y)
print()

from util.plot import Plot

# Show the various fits of the data.
def show_fits(x, y, d1=False, d2=False, **plot_kwargs):
    p = Plot()
    color = (200,200,200)
    p.add("Data", x, y, marker_size=8, color=color+(0.0,),
          marker_line_width=2, marker_line_color="rgb{color}")
    c = 1
    plot_spline_fit(x, y, toms_574_spline, p=p, color=c, d1=d1, d2=d2)
    c += 1
    plot_spline_fit(x, y, schumaker_spline, p=p, color=c, d1=d1, d2=d2)
    c += 1
    plot_spline_fit(x, y, pchip_spline, p=p, color=c, d1=d1, d2=d2)
    c += 1
    plot_spline_fit(x, y, bvspis_spline, p=p, color=c, d1=d1, d2=d2)
    c += 2
    plot_spline_fit(x, y, mqsi_spline, p=p, color=c, d1=d1, d2=d2)
    c += 1
    plot_spline_fit(x, y, monotone_fit, p=p, color=c, d1=d1, d2=d2)
    p.show(**plot_kwargs)

show_fits(x, y,                       show=False)
show_fits(x, y, d1=True, append=True, show=False)
show_fits(x, y, d2=True, append=True, show=True)


# 2022-03-06 13:08:57
# 
########################################################################################
# # plot_spline_fit(x, y, mqsi_spline, p=p, color=0)#, d1=True, d2=True)               #
# # plot_spline_fit(x, y, bvspis_spline, p=p, color=1, dash="dot")#, d1=True, d2=True) #
# # p.show()                                                                           #
# # exit()                                                                             #
# color = (255,100,150)
# plot(xarray, Result, ylim=c(-0.5,2))
# lines(xarray, Result2, col = 2)
# lines(xarray, Result3, col = 3)
########################################################################################
