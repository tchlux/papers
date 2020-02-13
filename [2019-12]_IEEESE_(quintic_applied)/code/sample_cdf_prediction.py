import numpy as np
from fits import quintic_fit
from util.plot import Plot

# Distirbution Prediction Application
p = Plot("", "I/O Throughput", "CDF Value",
         font_family="times", font_size=18)

xs = [
    [0.0, 0.18, 0.53, 0.74, 1.0],
    [0.0, 0.17, 0.52, 0.75, 1.02],
    [0.0, 0.46, 0.62, 0.89, .98],
    [0.0, 0.15, .4, .85, 1.],
]    

# Modify the x values to look like throughput values.
scale = 4e6
shift = 1e6
rescale = lambda z: z*scale + shift

# Rescale the values.
for x in xs: x[:] = list(map(rescale, x))

# Generate some reasonable y values.
ys = [
    [0.0, 0.49, 0.88, 0.89, 1.0],
    [0.0, 0.33, 0.66, 0.85, 1.0],
    [0.0, 0.11, 0.25, 0.34, 1.0],
    [0.0, 0.16, .4, .7, 1. ],
]
# Make fits of the data.
fs = [quintic_fit(x=x, y=y) for (x,y) in zip(xs, ys)]
# Pick color numbers for the data.
cs = [0, 3, 1]
# Generate a guess function as a convex combination of other functions.
guess = lambda z, fs=fs: sum(f(z) * w for (f,w) in zip(fs, [.1, .3, .6]))

# # Compute the KS statistic difference between the guess and the truth.
# vs = np.linspace(0, 1, 1000)
# diffs = [abs(guess(v) - fs[-1](v)) for v in vs]
# print("np.max(diffs): ",np.max(diffs))
# print("np.argmax(diffs): ",np.argmax(diffs))
# print("vs[np.argmax(diffs)]: ",vs[np.argmax(diffs)])
# x = vs[np.argmax(diffs)]
# print("guess(x): ",guess(x))
# print("fs[-1](x): ",fs[-1](x))

for i,(x,y,f,c) in enumerate(zip(xs,ys,fs,cs)):
    # Skip the "truth" last one.
    if i >= 3: continue
    # p.add("Points", x, y, group=i, color=c)
    p.add_func("f", f, (min(x), max(x)), color=c, dash="dot", opacity=.5)


p.add_func("guess", guess, (min(map(min,xs)), max(map(max,xs))), color=6, dash="dash")
p.add_func("truth", fs[-1], (min(map(min,xs)), max(map(max,xs))), color=(100,150,200))

# add nodes and edges to draw a little line for the KS statistic.
p.add_node("1", rescale(.9), .6131, size=0)
p.add_node("2", rescale(.9), .8194, size=0)
p.add_edge(["1","2"])


p.show(width=600, height=500, show_legend=False, file_name="example_distribution_prediction.html")


# # Generate some data randomly to use as the three source distributions.
# 
# N = 5
# np.random.seed(0)
# noise_amount = .2
# 
# for i in range(3):
#     np.random.seed(i+2)
#     # x = np.random.random(size=(N,)); x.sort()
#     x = np.linspace(0, 1, N) + (np.random.random(size=(N,))*noise_amount - noise_amount/2)
#     y = np.linspace(0, 1, N) + (np.random.random(size=(N,))*noise_amount - noise_amount/2)
#     # Make these values match some CDF expectations.
#     x.sort(); y.sort()
#     x -= np.min(x); y -= np.min(y)
#     x /= np.max(x); y /= np.max(y)
#     print()
#     print("i: ",i)
#     print("x: ",[round(v,2) for v in x])
#     print("y: ",[round(v,2) for v in y])
 
