import os, time, random
import numpy as np
from util.system import load, save, Timer
from util.math import fekete_points, pairwise_distance
from util.approximate import DelaunayP1, NeuralNetwork
from util.random import latin
from util.plot import Plot
from util.data import Data

# A modified version of the oscillatory test problem from:
#    'High Dimensional Polynomial Interpolation' by Barthelman et al. (2000)
# It is a cosine function of the norm of the input vectors. This
# function has no componentwise structure to take advantage of.
def oscillatory(x, c=1):
    if len(x.shape) <= 1:
        return np.cos(c*np.pi + np.linalg.norm(x))
    elif len(x.shape) == 2:
        return np.cos(c*np.pi + np.linalg.norm(x, axis=1))
    else:
        class UsageError(Exception): pass
        raise(UsageError(f"This objective function only supports arrays and matrices, received shape {x.shape}."))

# `True` if the plots and mesaurements should be of "error"
# `False` if the plots and measurements should be about simplices.
use_vals = False
# `True` if Fekete points should be used.
# `False` if a random Latin hypercube design should be used instead.
use_fekete = True
# Random seed.
seed = 0
dimension = 2
test_size = 1000
train_sizes = [2 * 4**(i) for i in range(1,6+1)]
train = train_sizes[-1]
snr = 0.0
# The models to test.
model = DelaunayP1

# Set the test function and declare some objects.
test_function = oscillatory
t = Timer()
m = DelaunayP1()
# Plot the distirbution of nearest points with increasing data
p = Plot(f"Delaunay in {dimension} Dimensions", "2 × 4^i data points", "Distance",
         font_family="times", font_size=25)

if use_vals:
    errors = []
else:
    vertex_distances = []
    edge_lengths = []
for train in train_sizes:
    if (train + 1 < dimension):
        if use_vals:
            errors.append( [] )
        else:
            vertex_distances.append( [] )
            edge_lengths.append( [] )
        continue

    # Seed each algorithm the same (in case it involved ranndomness).
    random.seed(seed)
    np.random.seed(seed)

    print()
    print('-'*70)
    print("train: ",train)
    print()
    if use_fekete:
        # Get the Fekete points.
        fekete_file = os.path.join(
            "fekete_points", f"fekete-{train}-{dimension}.pkl")
        print(f"  searching for '{fekete_file}'..")
        try:
            train_points = load(fekete_file)
        except:
            train_points = fekete_points(train, dimension)
            save(train_points, fekete_file)
    else:
        print("  generating train points..")
        train_points = latin(train, dimension)
    print("  generating test poitns..")
    test_points = np.random.random(size=(test_size, dimension))
    test_values = test_function(test_points)
    print("  using train points:", train_points.shape)
    print("  using test points: ", test_points.shape)
    # Cycle through different "signal-to-noise ratios"
    print()
    snr_string = f"{snr:.2f}"
    print("  SNR", snr_string)
    # Evaluate the test function at those points.
    train_values = test_function(train_points)
    # Generate noise (relative to values) and add it on to the values.
    noise_ratio = snr * (np.random.random(size=train_values.shape)*2 - 1)
    train_values += train_values * noise_ratio
    # Print out the time before starting the algorithm execution.
    print()
    print(" ",time.ctime())

    print()
    print(f"  fitting.. ")
    pts = train_points.copy()
    vals = train_values.copy()
    t.start()
    if use_vals:
        m.fit(pts, vals)
    else:
        m.fit(pts)
    t.stop()
    print(f"    {t()} seconds")
    # Evaluate the model.
    print("  predicting.. ")
    pts = test_points.copy()
    t.start()
    output = m(pts)
    if use_vals:
        errors.append( list(np.abs(output - test_values)) )
    else:
        # Measure the distance to the nearest vertex and the length of the
        # longest edge of each simplex being used to interpolate.
        nearest_vertex = []
        longest_edge = []
        for i, (pt, (ids, wts)) in enumerate(zip(pts, output)):
            # Compute the longest edge.
            longest_edge.append(
                np.max(pairwise_distance(train_points[ids])) )
            # Compute the nearest vertex.
            nearest_vertex.append(
                np.min(np.sqrt(np.sum((train_points[ids] - pt)**2, axis=1))) )
        vertex_distances.append( nearest_vertex )
        edge_lengths.append( longest_edge )
    t.stop()
    print(f"    {t()} seconds")


# Add box plot series for these two statistics.
if use_vals:
    p.add_box(f"Absolute Errors", errors, [i for i in range(1,len(train_sizes)+1)])
else:
    p.add_box(f"Nearest vertex", vertex_distances, [i for i in range(1,len(train_sizes)+1)])
    p.add_box(f"Longest edge", edge_lengths, [i for i in range(1,len(train_sizes)+1)])
legend = dict(
    xanchor = "center",
    yanchor = "top",
    x = .5,
    y = .13,
    orientation = "h",
    bgcolor="white",
    bordercolor="grey",
    borderwidth=.5
)
layout_settings = dict(
    margin = dict(l=100, t=60, r=20),
)
p.show(y_axis_settings=dict(type="log"), y_range=[-4,0], x_range=[.5,len(train_sizes)+.5],
       legend=legend, layout=layout_settings)
