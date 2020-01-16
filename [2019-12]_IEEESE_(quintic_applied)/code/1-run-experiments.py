MAKE_HISTOGRAM_PLOT = False
MAKE_SAMPLE_PREDICTION_FIGURE = True
COLLECT_ALL_DATA = not (MAKE_HISTOGRAM_PLOT or MAKE_SAMPLE_PREDICTION_FIGURE)

previous_step = __import__("0-clean-data")
test_name = previous_step.test
data_file_path = previous_step.prefix + test_name + ".pkl"

from util.data import Data
from util.system import AtomicOpen
from random import seed, sample


data = Data.load(data_file_path)
data.max_display = len(data)
print(data)
print()
print("Configuration values:")
print("  freq:   ", sorted(set(data["freq"])))
print("  fsize:  ", sorted(set(data["fsize"])))
print("  rsize:  ", sorted(set(data["rsize"])))
print("  threads:", sorted(set(data["threads"])))

interesting_config = [2800000, 16384, 16384, 64] # <- the
# configuration to use for distribution model testing.

config_cols = len(interesting_config)
interest_row = data[:,:config_cols].index( interesting_config )

print()
print("interest_row: ",interest_row)

# Get the configuration data and values for the example row.
config = data[interest_row, :config_cols]
values = sorted(data[interest_row, config_cols:])
print()
print("        ", data.names[:config_cols])
print("config: ",config)
print("values: ",len(values))
print()

# Normalize the range of the values.
min_value = values[0]
max_value = values[-1]
values = [(v - min_value) / (max_value - min_value) for v in values]

# Sample out the values (only keep 1000) of them.
from numpy import linspace
keep_count = 1000
values = [values[int(round(i))] for i in linspace(0,len(values)-1,keep_count)]

# Setup the experiment parameters.
trials = 100
sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
                150, 200, 250, 300, 400, 500]

# Get all of the different fits into one list.
from fits import cdf_points, flat_fit, linear_fit, cubic_fit, quintic_fit
functions = [flat_fit, linear_fit, cubic_fit, quintic_fit]

# Fit the "truth" function.
truth = linear_fit(values)
true_min_max = [0, 1]
true_x, true_y = cdf_points(values)

if MAKE_HISTOGRAM_PLOT:
    import numpy as np
    from util.plot import Plot
    from fits import cdf_points, quintic_fit

    values = np.array([v for row in data[:,config_cols:] for v in row])
    p = Plot("","Throughput (bytes/sec)","Count", font_family="times", font_size=17)
    p.add_histogram("throughput", values, num_bins=100)
    p.show(file_name="throughput_histogram.html", show_legend=False,
           width=400, height=200, x_range=[0,int(20e6)])

if MAKE_SAMPLE_PREDICTION_FIGURE:
    from random import seed, sample
    from util.plot import Plot
    # Create a visual of some sample distribution approximations.
    def make_plot(functions, prename):
        # Initialize some settings.
        seed(0); k = 10
        pop = sample(values, k)
        x, y = cdf_points(pop)
        styles = ["dashdot", "dot", "dash"]
        # Create the plot.
        p = Plot("", "x","CDF", font_family="times", font_size=18)
        p.add("Sample", x, y)
        p.add_func("Truth", truth, true_min_max, color=p.color((0,0,0,.3)))
        for f,s in zip(functions, styles):
            name = f.__name__.replace("_"," ").title().split()[0].replace("Flat","EDF")
            p.add_func(name, f(pop), true_min_max, dash=s, opacity=.8)
        # Set the legend properties.
        legend = dict(
            xanchor = "center",
            yanchor = "top",
            x = .25,
            y = .8,
            orientation = "v",
            bgcolor="white",
            bordercolor="grey",
            borderwidth=.5
        )
        # Create the plot.
        p.show(y_range=[-.1, 1.1], x_range=[0,.5], width=400,
               height=300, legend=legend, file_name=prename+"-sample-prediction.html")

    # Make the two different plots.
    functions = [flat_fit, linear_fit]
    make_plot(functions, "fl")
    functions = [cubic_fit, quintic_fit]
    make_plot(functions, "cq")

# Stop now if all data should not be collected, everything below this
# condition is related to the collection of the primary data for this
# paper. It takes about an hour to run on a 2016 MacBook Pro when
# using all cores (parallel version).
if not COLLECT_ALL_DATA: exit()

# --------------------------------------------------------------------
# Start collecting results.

# Try and load existing results, if that fails, oh well!
results_file = "[1]-temporary-results.csv"

# Move the old results files to a new spot.
import os, shutil
if os.path.exists(results_file):
    i = 0
    while os.path.exists(str(i) + '-'+results_file): i += 1
    shutil.move(results_file, str(i) + '-'+results_file)

# Construct a header row for the results file.
header = ["sample", "trial", "fit"] + [f"er {i+1}" for i in range(len(true_x))]
# Write the initial file (not in append mode, to clear it).
with open(results_file, "w") as out_file:
    print(",".join(header), file=out_file)

# Run experiments and collect data.
def run_predictions(k):
    seed(0)
    print("k: ",k)
    result_lines = []
    for i in range(trials):
        print("  i: ",i, end="\r")
        sub_population = sample(values, k)
        for f in functions:
            name = f.__name__
            cdf = f(sub_population)
            errors = [float(cdf(x) - y) for (x,y) in zip(true_x, true_y)]
            row = list(map(str, [k, i, name] + sorted(errors)))
            result_lines.append( ",".join(row) )
    # Save results as they are collected
    with AtomicOpen(results_file, "a") as out_file:
        for l in result_lines: print(l, file=out_file, flush=True)
    # Delete these intermediate results from memory.
    del( result_lines )
            
from util.parallel import map as pmap
for _ in pmap(run_predictions, sample_sizes, redirect=False): pass




# Sample sizes: 
# 
# Take many subsets of samples of one size, build approximations with
# 
#   flat, linear, cubic, and quintic splines.
# 
# Record full set of approximation error for all of them. (10K points).
# 

