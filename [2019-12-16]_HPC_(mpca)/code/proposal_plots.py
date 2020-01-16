from util.stats import mpca, pca
import numpy as np

GENERATE_APPROXIMATIONS = False
BIG_PLOTS = False
SHOW_2D_POINTS = False

# Generate some points for testing.
np.random.seed(4)
rgen = np.random.RandomState(17)
n = 50
points = (rgen.rand(n,2) - .5) * 2
# points *= np.array([.5, 1.])

# Create some testing functions (for learning different behaviors)
funcs = [
    lambda x: x[0]               , # Linear on x
    lambda x: abs(x[0] + x[1])   , # "V" function on 1:1 diagonal
    lambda x: abs(2*x[0] + x[1]) , # "V" function on 2:1 diagonal
    lambda x: x[0]**2            , # Quadratic on x
    lambda x: (x[0] + x[1])**2   , # Quadratic on 1:1 diagonal
    lambda x: (2*x[0] + x[1])**3 , # Cubic on 2:1 diagonal
    lambda x: (x[0]**3)          , # Cubic on x
    lambda x: rgen.rand()        , # Random function
]
# Calculate the response values associated with each function.
responses = np.vstack(tuple(tuple(map(f, points)) for f in funcs)).T

# Reduce to just the first function
choice = 3
func = funcs[choice]
response = responses[:,choice]

# Run the princinple response analysis function.
components, values = mpca(points, response)
conditioner = np.matmul(components, np.diag(values))

print()
print("Components")
print(components)
print()
print("Values")
print(values)
print()
print("Conditioner")
print(conditioner)
print()

# Generate a plot of the response surfaces.
from util.plot import Plot, multiplot
print("Generating plots of source function..")

# Add function 1
p1 = Plot(font_family="times")
p1.add("Points", *(points.T), response, opacity=.8)
p1.add_func("Surface", func, [-1,1], [-1,1], plot_points=1000, color=p1.color(1))

if GENERATE_APPROXIMATIONS:
    from util.algorithms import NearestNeighbor
    model = NearestNeighbor()
    model.fit(points, response)
    p1.add_func("Unconditioned Approximation", model, [-1,1], [-1,1],
                mode="markers", opacity=.8)
    # Generate a conditioned approximation
    model = NearestNeighbor()
    model.fit(np.matmul(points, conditioner), response)
    approx = lambda x: model(np.matmul(x, conditioner))
    p1.add_func("Best Approximation", approx, [-1,1], [-1,1],
                mode="markers", opacity=.8)


print("Generating metric principle components..")

# Return the between vectors and the differences between those points.
def between(x, y, unique=True):
    vecs = []
    diffs = []
    for i1 in range(x.shape[0]):
        start = i1+1 if unique else 0
        for i2 in range(start, x.shape[0]):
            if (i1 == i2): continue
            vecs.append(x[i2] - x[i1])
            diffs.append(y[i2] - y[i1])
    return np.array(vecs), np.array(diffs)

# Plot the between slopes to verify they are working.
# Calculate the between slopes
vecs, diffs = between(points, response)
vec_lengths = np.sqrt(np.sum(vecs**2, axis=1))
between_slopes = diffs / vec_lengths
bs = ((vecs.T / vec_lengths) * between_slopes).T
# Extrac a random subset for display
size = 100
random_subset = np.arange(len(bs))
rgen.shuffle(random_subset)
bs = bs[random_subset[:size],:]
# Normalize the between slopes so they fit on the plot
max_bs_len = np.max(np.sqrt(np.sum(bs**2, axis=1)))
bs /= max_bs_len
# Get a random subset of the between slopes and plot them.
p2 = Plot("","Metric PCA on Z","", font_family="times")
p2.add("Between Slopes", *(bs.T), color=p2.color(4, alpha=.4))

if SHOW_2D_POINTS:
    # Add the points and transformed points for demonstration.
    new_pts = np.matmul(np.matmul(points, conditioner), np.linalg.inv(components))
    p2.add("Original Points", *(points.T))
    p2.add("Transformed Points", *(new_pts.T), color=p2.color(6, alpha=.7))

# Normalize the values to sum to 1.
values /= np.sum(values)

# Add the principle response components 
for i,(vec,m) in enumerate(zip(components, values)):
    vec = vec * m
    p2.add(f"PC {i+1}", [0,vec[0]], [0,vec[1]], mode="lines")
    ax, ay = (vec / sum(vec**2)**.5) * 3
    p2.add_annotation(f"{m:.2f}", vec[0], vec[1], font_family="times")


p3 = Plot("", "PCA on X", "", font_family="times")
p3.add("Points", *(points.T), color=p3.color(4, alpha=.4))

# Add the normal principle components
components, values = pca(points)
for i,(vec,m) in enumerate(zip(components, values)):
    vec = vec * m
    p3.add(f"PC {i+1}", [0,vec[0]], [0,vec[1]], mode="lines")
    ax, ay = (vec / sum(vec**2)**.5) * 3
    p3.add_annotation(f"{m:.2f}", vec[0], vec[1], font_family="times")


if BIG_PLOTS:
    p1.plot(file_name="source_func.html", show=False)
    p2.plot(append=True, x_range=[-8,8], y_range=[-5,5])
else:
    # # Adjust the settings for display appearance
    p1.font_family = "times"
    p2.font_family = "times"
    p3.font_family = "times"
    # Make the plots (with manual ranges)
    p1 = p1.plot(html=False, show_legend=False)
    p2 = p2.plot(html=False, x_range=[-1,1], y_range=[-1,1], show_legend=False)
    p3 = p3.plot(html=False, x_range=[-1,1], y_range=[-1,1], show_legend=False)
    # Generate the multiplot of the two side-by-side figures
    # multiplot([p1,p2,p3], file_name="mpca_demo.html", height=126, width=600)
    multiplot([p1,p2,p3], file_name="mpca_demo.html", height=126, width=600)
