import random, os
from numpy import array, linspace, zeros

from util.data import Data
from util.plot import Plot
from util.stats import cdf_fit

from fits import cdf_points, flat_fit, linear_fit, cubic_fit, quintic_fit


# --------------------------------------------------------------------
if not os.path.exists("[5]-small-readers.pkl"):
    data = Data.load("[0]-readers.pkl")
    data.max_display = 17
    measured = data.shape[1] - 4
    # Collapse the "throughput" columns into one list.
    data.pack("throughput")
    # Sample out the values (only keep 1000) of them.
    from numpy import linspace
    keep_count = 1000
    to_keep = [int(round(i)) for i in linspace(0,measured-1,keep_count)]
    for row in data:
        row["throughput"].sort()
        row["throughput"] = [row["throughput"][i] for i in to_keep]
    # Unpack the list-column again before saving.
    data.unpack("throughput")
    print(data)
    data.save("[5]-small-readers.pkl")
else:
    data = Data.load("[5]-small-readers.pkl")


# --------------------------------------------------------------------
#  Load the data.
data.max_display = 17
data.pack("throughput")

# --------------------------------------------------------------------
#  Compute the Delaunay weights for each row.
print("Computing Delaunay weights for each row..")
data["extrapolated"] = False
data["indices"] = None
data["weights"] = None
x = array( data[:,:4].to_matrix() )
shift = x.min(axis=0)
x -= shift
scale = x.max(axis=0)
x /= scale
indices = list(range(len(x)))
from util.approximate import Delaunay
to_remove = set()
for i in range(len(x)):
    print(i, x[i]*scale + shift)
    row = x[indices.pop(i)]
    # Get the indices and weights for this row.
    m = Delaunay()
    m.fit(x[indices])
    try:
        ids, wts = m(row, allow_extrapolation=False)
        data[i,"indices"] = [indices[_] for _ in ids]
        data[i,"weights"] = list(wts)
    except: data[i, "extrapolated"] = True
    # Re-insert the popped index.
    indices.insert(i,i)

print(f"  data shape: {data.shape}")


# --------------------------------------------------------------------
#   Generate samples to use for fits.
#   Compute the distribution fits.
#   Compute the weighted sum distribution fits.
k = 20
temp_file = f"[5]-temporary-sample-fit-prediction-{k}.dill"
functions = [flat_fit, linear_fit, cubic_fit, quintic_fit]
if not os.path.exists(temp_file):
    random.seed(0)
    print(f"Generating random samples of {k} points for each row..")
    data["sample"] = (sorted(random.sample(row["throughput"], k)) for row in data)
    # Create the initial functions.
    print("Fitting distribution interpolants over each sample..")
    for f in functions:
        print(f"  {f.__name__}")
        data[f.__name__] = (f(row["sample"]) for row in data)
    print(" "*70)
    print(data)
    data.save(temp_file)
else:
    print(f"Loading fit distributionsn from '{temp_file}'..")
    data = Data.load(temp_file)


# --------------------------------------------------------------------
# Evaluate weighted sum of all fits at each data point against truth.
# Record errors for each fit.

# Compute the "true" value of the function at these points.
true_cdf_values = linspace(0, 1, len(data[0,"throughput"])+1)[1:]

# Evaluate the error of all predictions.
p = Plot("","Absolute Approximation Error","CDF of Absolute Error", 
         font_family="times", font_size=16)
styles = [None, "dashdot", "dot", "dash"]
algs = [("EDF", "flat_fit"), ("Linear", "linear_fit"),
        ("Cubic", "cubic_fit"), ("Quintic", "quintic_fit")]
for (n,a),f,s in zip(algs,functions,styles):
    print("f: ",f)
    abs_errors = []
    for row in data:
        if row["extrapolated"]: continue
        print()
        print("",data.names[:4])
        print("  ",row[:4])
        # Initialize storage for the predicted values.
        sample = row['throughput']
        predicted_cdf_values = zeros(len(sample))

        # Make the prediction.
        for source, weight in zip(row["indices"], row["weights"]):
            source_func = data[source, f.__name__]
            for i in range(len(sample)):
                predicted_cdf_values[i] += weight * source_func(sample[i])

        # Add the absolute errors to the list of all errors achieved
        # by this function. These will be presented in aggregate.
        abs_errors += list(abs(true_cdf_values - predicted_cdf_values))

    cdf = cdf_fit(abs_errors)
    p.add_func(n, cdf, cdf(), dash=s)
    # p.add_histogram(f.__name__, abs_errors)

# TODO: verify that the predicted distributions look reasonable.

# Set the legend properties.
legend = dict(
    xanchor = "center",
    yanchor = "top",
    x = .7,
    y = .7,
    orientation = "v",
    bgcolor="white",
    bordercolor="grey",
    borderwidth=.5
)

p.show(file_name="delaunay-predictions.html",
       layout=dict(margin=dict(l=40, b=40)),
       width=400, height=290, legend=legend)
