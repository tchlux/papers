import os, time, random
import numpy as np
from util.system import load, save, Timer
from util.math import fekete_points
from util.approximate import DelaunayP1, NeuralNetwork
from util.plot import Plot

seed = 0
dimension = 2
test_size = 1000
train_sizes = [2 * 4**(i) for i in range(1,6+1)]
train = train_sizes[-1]
snr = 0.0
# The models to test.
models = [NeuralNetwork(epochs=epochs) for epochs in (100, 1000, 5000, 10000)]
names = [f"{epochs} epochs" for epochs in (100, 1000, 5000, 10000)]

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

test_function = oscillatory
# Get the Fekete points.
fekete_file = os.path.join(
    "fekete_points", f"fekete-{train}-{dimension}.pkl")
print(f"Searching for '{fekete_file}'..")
try:
    train_points = load(fekete_file)
except:
    train_points = fekete_points(train, dimension)
    save(train_points, fekete_file)

print("generating test poitns..")
test_points = np.random.random(size=(test_size, dimension))
test_values = test_function(test_points)
print("  using train points:", train_points.shape)
print("  using test points: ", test_points.shape)
# Cycle through different "signal-to-noise ratios"
print()
print("-"*35)
# Re-seed the random number generators for each test.
random.seed(seed)
np.random.seed(seed)
snr_string = f"{snr:.2f}"
print("SNR", snr_string)
# Evaluate the test function at those points.
train_values = test_function(train_points)
# Generate noise (relative to values) and add it on to the values.
noise_ratio = snr * (np.random.random(size=train_values.shape)*2 - 1)
train_values += train_values * noise_ratio
# Print out the time before starting the algorithm execution.
print()
print(time.ctime())
# Seed each algorithm the same (in case it involved ranndomness).
random.seed(seed)
np.random.seed(seed)
# Build the model and test it.
t = Timer()
# Fit the model.

p = Plot()
for n,m in zip(names,models):
    print()
    print(f"  fitting ({n}).. ")
    pts = train_points.copy()
    vals = train_values.copy()
    t.start()
    m.fit(pts, vals)
    t.stop()
    print(f"    {t()} seconds")
    # Evaluate the model.
    print("  predicting.. ")
    pts = test_points.copy()
    t.start()
    guesses = m(pts)
    t.stop()
    print(f"    {t()} seconds")
    total_time = t.total
    errors = list(guesses - test_values)
    p.add_histogram(f"Errors ({n})", errors)
p.show()
