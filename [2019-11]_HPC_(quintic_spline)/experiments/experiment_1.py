import numpy as np

# Import:
#   - a Plotting and visualization library
#   - functions for fitting monotone splines
#   - a tabular Data object to record performances
import sys, os
sys.path += [os.path.abspath("code")]
from plot import Plot
from monotone import monotone_cubic_spline, monotone_quintic_spline
from data import Data
sys.path.pop(-1)


f = lambda x: np.sin(x) + x
df = lambda x: np.cos(x) + 1
ddf = lambda x: -np.sin(x)

num_points = 2
lower = 0
upper = 5/2 * np.pi

def fit_approximations(num_points=num_points, lower=lower, upper=upper):
    # Compute the node values, the function values, and two derivatives.
    x = np.linspace(lower, upper, num_points)
    y = f(x)
    dy = df(x)
    ddy = ddf(x)
    # Compute the fits for the nodes.
    return (monotone_cubic_spline(x,values=np.vstack((y,dy)).T),
            monotone_quintic_spline(x,values=np.vstack((y,dy,ddy)).T))

cubic_fit, quintic_fit = fit_approximations()

# -------------------------------------------------------------------
# Make a visual to show that the function looks like with initial
# approximations given 2 nodes.

legend = dict(
    xanchor = "center",
    yanchor = "top",
    x = .625,
    y = .205,
    orientation = "h",
    bgcolor="white",
    bordercolor="grey",
    borderwidth=.5
)

p = Plot("","","",font_family="times", font_size=20)
p.add("Points", [lower,upper], [f(lower), f(upper)])
p.add_func("sin(x) + x", f, [lower, upper], dash=None)
# p.add_func("df", df, [lower, upper])
# p.add_func("ddf", ddf, [lower, upper])
p.add_func("cubic", cubic_fit, [lower,upper], dash="dash")
p.add_func("quintic", quintic_fit, [lower,upper], dash="dot")
p.add("Points", np.linspace(lower,upper,num_points),
      f(np.linspace(lower,upper,num_points)),
      color=p.color(0), show_in_legend=False)
fig = p.plot(file_name="cubic-quintic-sin.html", legend=legend,
             width=700, height=400)
# -------------------------------------------------------------------

# Cycle an increasing number of nodes and compute the maximum error of
# the two different forms of interpolation.

d = Data(names=["Nodes", "Eval Points", "Cubic Errors", "Quintic Errors", "Max Cubic Error", "Max Quintic Error"],
         types=[int,     int,           list,           list,             float,             float])
data_file = "e1_data.pkl"
# Try loading previous experiment data if it exists.
try: d = Data.load(data_file)
except: pass

# Use 1000 evaluation points to estimate the error distribution 
eval_points = np.linspace(lower, upper, 10000)
# Get the truth values.
truth = f(eval_points)
# Define a function for collecting approximations and errors.
def get_approximations_and_error(nodes=2, parallel=True):
    if parallel: from code.parallel import map
    # Get the fits.
    cubic_fit, quintic_fit = fit_approximations(nodes)
    # Get the approximation values.
    cubic_vals = list(map(cubic_fit, eval_points))
    quintic_vals = list(map(quintic_fit, eval_points))
    # Compute the errors
    quintic_errors = quintic_vals - truth
    cubic_errors = cubic_vals - truth
    quintic_max_error = max(abs(quintic_errors))
    cubic_max_error = max(abs(cubic_errors))
    # Return the full row of information.
    return nodes, len(eval_points), cubic_errors, quintic_errors, cubic_max_error, quintic_max_error


print("Testing..")
# test_n = list(range(2, 101)) + list(range(110, 301, 10)) + list(range(400,1001,100)) + [999]
test_n = [2, 4, 10, 30] + list(range(100,1001,100))
for n in test_n:
    print(f" {n} nodes..", end="\r")
    # Skip tests that have already been run.
    if ((n,len(eval_points)) in d["Nodes","Eval Points"]): continue
    # Get the next row of approximation errors.
    d.append( get_approximations_and_error(n) )
    # Periodically save the test results to file.
    if (not (n % 10)): d.save(data_file)

print()
d = d[d["Eval Points"] == 10000]
d.sort()
print(d)

legend['x'] = .365
p = Plot("","Number of Points", "Max Absolute Error", font_family="times", font_size=18)
p.add("Cubic Max Errors", d["Nodes"], d["Max Cubic Error"], symbol="square")
p.add("Quintic Max Errors", d["Nodes"], d["Max Quintic Error"])
p.show(width=700, height=400, file_name="experiment_1_errors.html", legend=legend,
    y_axis_settings=dict(type="log"), x_axis_settings=dict(type="log"))
