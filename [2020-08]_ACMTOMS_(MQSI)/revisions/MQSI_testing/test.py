import numpy as np

# Import all of the (python-wrapped) spline packages.
from toms574 import spline_fit as toms_574_spline
from schumaker import spline_fit as schumaker_spline
from pchip import spline_fit as pchip_spline
from bvspis import spline_fit as bvspis_spline
from l2_minimizer import spline_fit as l2min_spline
from mqsi import spline_fit as mqsi_spline


# Given a spline fit function and some data, plot the fit over the
#  interval covered by the data.
def plot_spline_fit(x, y, spline_fit, name=None, p=None, color=None,
                    dash=None, d1=False, d2=False, plot_points=5000):
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


# Show the various fits of the data.
def show_fits(x, y, methods={}, d1=False, d2=False, **plot_kwargs):
    # Ensure that the "x" and "y" are numpy arrays.
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    # Create a plot.
    from tlux.plot import Plot
    p = Plot()
    color = (200,200,200)
    p.add("Data", x, y, marker_size=8, color=color+(0.0,),
          marker_line_width=2, marker_line_color="rgb{color}")
    c = 0
    for name in sorted(methods):
        fit_func = methods[name]
        name = name.replace("_", " ")
        plot_spline_fit(x, y, fit_func,  name=name,  p=p, color=c, d1=d1, d2=d2)
        c += 1
    p.show(**plot_kwargs)


# Generate test data from a test function given a string name.
def generate_test_data(name):
    if name == "SIGNAL":
        # Signal function.
        from numpy import sin, pi
        f = lambda x: sin(4 * (2*pi) * x) / (x**2 + .1)
        x = np.linspace(0, 1, 20)
        y = np.asarray(f(x), dtype=float)
    elif name == "LARGE_TANGENT":
        # Large tangent test.
        f = lambda x: -(1.0 + ( 1.0 / (x-1.01) ))
        x = np.linspace(0, 1, 9)
        y = np.asarray(f(x), dtype=float)
    elif name == "PIECEWISE_POLYNOMIAL":
        # Piecewise polynomial test.
        from tlux.math import Spline
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
        x = np.linspace(0, 1, 16)
        y = np.asarray(f_df_ddf[0](x), dtype=float)
    elif name == "RANDOM_MONOTONE":
        # RANDOM MONOTONE.
        x = [ 0.025926231827891333, 0.13457994534493356, 0.18443986564691528,
              0.2046486340378425, 0.26682727510286663, 0.29965467367452314,
              0.3303348210038741, 0.42036780208748903, 0.4353223926182769,
              0.43599490214200376, 0.5135781212657464, 0.5291420942770391,
              0.5496624778787091, 0.6192709663506637, 0.6211338327692949,
              0.7853351478166735 ]
        y = [ 0.06528650438687811, 0.079645477009061, 0.09653091566061256,
              0.10694568430998297, 0.12715997170127746, 0.20174322626496533,
              0.2203062070705597, 0.22601200060423587, 0.34982628500329926,
              0.42812232759738944, 0.46778748458230024, 0.4942368373819278,
              0.505246090121704, 0.5967453089785958, 0.846561485357468,
              0.8539752926394888 ]
    else:
        raise(NotImplementedError(f"There is no test with name '{name}'."))
    # Return the points.
    return x, y


# --------------------------------------------------------------------
#                 AD HOC TESTING AND COMPARISON CODE
# 
if __name__ == "__main__":

    # x, y = generate_test_data("SIGNAL")
    # x, y = generate_test_data("LARGE_TANGENT")
    # x, y = generate_test_data("PIECEWISE_POLYNOMIAL")
    x, y = generate_test_data("RANDOM_MONOTONE")

    print()
    print("x: ",x)
    print("y: ",y)
    print()

    methods = dict(
        TOMS_574  = toms_574_spline, # C1 quadratic spline
        # Schumaker = schumaker_spline, # C1 quadratic spline
        PCHIP     = pchip_spline, # C1 cubic spline
        BVSPIS    = bvspis_spline, # C2 quintic spline
        MQSI      = mqsi_spline, # C2 quintic spline
        # L2_Min    = l2min_spline, # C2 quintic spline
    )

    # Generate 3 HTML plots and show them in the web browser.
    #   values of all splines
    show_fits(x, y, methods=methods,                       show=False)
    #   first derivatives of all splines
    show_fits(x, y, methods=methods, d1=True, append=True, show=False)
    #   second derivatives of all splines
    show_fits(x, y, methods=methods, d2=True, append=True, show=True)
