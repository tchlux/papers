# PCHIP spline fit of data.
def spline_fit(x, y):
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
