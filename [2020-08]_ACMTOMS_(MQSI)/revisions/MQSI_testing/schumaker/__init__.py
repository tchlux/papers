
# Construct and return a Schumaker spline fit to data.
def spline_fit(x, y):
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
