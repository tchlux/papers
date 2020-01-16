# Given a list of numbers, generate two lists, one of the CDF x points
# and one of the CDF y points (for fitting).
def cdf_points(data):
    import numpy as np
    SMALL = 2**(-26)
    # Sort the data (without changing it) and get the min and max
    data = np.array(sorted(data))
    min_pt = data[0]
    max_pt = data[-1]
    # Initialize a holder for all CDF points
    data_vals = []
    # Add all of the CDF points for the data set
    for i,val in enumerate(data):
        if ((i > 0) and (val == data[i-1])): data_vals[-1] = (i+1)/len(data)
        else:                                data_vals.append( (i+1)/len(data) )
    # Add the 100 percentile point if it was not added already
    if (data_vals[-1] != 1.0): data_vals[-1] = 1.0
    # Convert data into its unique values.
    data = np.array(sorted(set(data)))
    # Convert it all to numpy format for ease of processing
    data_vals = np.array(data_vals)
    # Return the two lists that define the CDF points.
    from fraction import Fraction
    data = list(map(Fraction, data))
    data_vals = list(map(Fraction, data_vals))
    return data, data_vals


# Return a discontinuous fit.
def flat_fit(data):
    from numpy import searchsorted
    x, y = cdf_points(data)
    def edf(z):
        try:    return [value(_) for _ in z]
        except: return y[min(len(y)-1, max(0,searchsorted(x, z, side="left")-1)
                             + (1 if z == max(x) else 0))]
    return edf

# Return a linear fit function.
def linear_fit(data):
    from polynomial import fit
    x, y = cdf_points(data)    
    cdf = fit(x, y, continuity=0)
    def fit(z): return max(0,min(1,cdf(z)))
    fit.derivative = cdf.derivative
    return fit

# Return a cubic fit function.
def cubic_fit(data):
    from monotone import monotone_cubic_spline
    x, y = cdf_points(data)
    cdf = monotone_cubic_spline(x, y)
    def fit(z):
        val = max(0,min(1,cdf(z)))
        if (z < x[0]): val = min(val, y[0])
        if (z > x[-1]): val = max(val, y[-1])
        return val
    fit.derivative = cdf.derivative
    return fit

# Return a quintic fit function.
def quintic_fit(data):
    from monotone import monotone_quintic_spline
    x, y = cdf_points(data)
    cdf = monotone_quintic_spline(x, y)
    def fit(z):
        val = max(0,min(1,cdf(z)))
        if (z < x[0]): val = min(val, y[0])
        if (z > x[-1]): val = max(val, y[-1])
        return val
    fit.derivative = cdf.derivative
    return fit


