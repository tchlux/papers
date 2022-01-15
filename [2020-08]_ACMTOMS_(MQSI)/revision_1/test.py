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
    def deriv_2(z, s=.01):
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
    Spline, DerivativeSpline, SecondDerivativeSpline, IntervalTab = (
        schumaker.Schumaker(py2rpy(x), py2rpy(y), Vectorised=True)
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
        assert (erre == 0), f"The DBVSSC routine reported a nonzero error {erre}. Check the documentation for more information about ERRE."
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
        assert (erre == 0), f"The DBVSSC routine reported a nonzero error {erre}. Check the documentation for more information about ERRE."
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
        assert (erre == 0), f"The DBVSSC routine reported a nonzero error {erre}. Check the documentation for more information about ERRE."
        return y2tab
    # Stack and return the function.
    function.derivative = df
    function.derivative.derivative = ddf
    return function


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
                    dash=None, d1=False, d2=False):
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
    p.add_func(f"{name}", spline, bounds, vectorized=True, color=color, group=color, dash=dash)
    if d1:
        p.add_func("First derivative", spline.derivative, bounds, vectorized=True,
                   color=color, group=color, dash="dash")
    if d2:
        p.add_func("Second derivative", spline.derivative.derivative, bounds, vectorized=True,
                   color=color, group=color, dash="dot")
    # Return the plot.
    return p



from test_functions import signal as f_df_ddf

print()
x = np.linspace(0, 1, 20)
# y = np.log(x)
y = np.asarray(f_df_ddf[0](x), dtype=float)
print("type(x): ",type(x))
print("y.dtype: ",y.dtype)
print("type(y): ",type(y))
print("x: ",x)
print("y: ",y)
print()

from util.plot import Plot
p = Plot()
# color = (255,100,150)
color = (200,200,200)
p.add("Data", x, y, marker_size=8, color=color+(0.0,),
      marker_line_width=2, marker_line_color="rgb{color}")


# plot_spline_fit(x, y, mqsi_spline, p=p, color=0)#, d1=True, d2=True)
# plot_spline_fit(x, y, bvspis_spline, p=p, color=1, dash="dot")#, d1=True, d2=True)
# p.show()
# exit()

d1 = True
d2 = True

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


p.show()

# plot(xarray, Result, ylim=c(-0.5,2))
# lines(xarray, Result2, col = 2)
# lines(xarray, Result3, col = 3)
