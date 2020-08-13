from util.plot import Plot, multiplot

# Return a linearly separated set of floats between "start" and "end"
def linspace(start, end, steps):
    return [start + (end-start)*i/(steps-1) for i in range(steps)]

# Transpose a python list of lists.
def transpose(l): return [_ for _ in zip(*l)]

# Given a center and a radius, produce a set of points equally spaced
# along the radius of the circle.
def circle_points(center, radius, steps=200):
    from math import sin, cos, pi
    points = []
    for i, step in enumerate(linspace(0, 2*pi, steps)):
        points.append( [center[0] + cos(step)*radius, center[1] + sin(step)*radius] )
    return points


# Generate an interesting random set of points
points = [
    [0.48, 0.52],
    [0.40, 0.24],
    [0.65, 0.93],
    [0.16, 0.56],
    [0.93, 0.68],
    [0.70, 0.16],
]

# Get the radius about each point.
from util.approximate import ShepMod
from numpy import array
m = ShepMod()
m.fit(array(points, dtype=float))
print(f"m.rw: {{{','.join(list(map(str,list(m.rw))))}}}")
# Now "m.rw" holds the radius about each point.


p = Plot()
# Add points.
for i,(pt,r) in enumerate(zip(points, m.rw)):
    circle_pts = circle_points(pt, r)
    if i == 0:
        p.add("circle", *transpose(circle_pts+[pt]),
              mode="lines", dash=None, color=(0,0,0,1.0))
    elif i == 1:
        p.add("circle", *transpose([pt]+circle_pts[125:]+circle_pts[:125]),
              mode="lines", dash="dot", color=p.color(i,alpha=.3))
    elif i == 2:
        p.add("circle", *transpose([pt]+circle_pts[25:]+circle_pts[:25]),
              mode="lines", dash="dash", color=p.color(i,alpha=.3))
    # Add the node second so that it is visually on top.
    p.add_node(f"point {i+1}", pt[0], pt[1], size=10, color=i)

height = 400
width = 500
y_bounds = [-.165, 1.55]
x_center = 0.5
x_bounds = [x_center - (y_bounds[1]-y_bounds[0]) * (width/height) / 2,
            x_center + (y_bounds[1]-y_bounds[0]) * (width/height) / 2]
print("x_bounds: ",x_bounds)
print("y_bounds: ",y_bounds)

p1 = p.graph(x_range=x_bounds, y_range=y_bounds,
             width=width, height=height,
             file_name="example_shepard_radius.html")


exit()

# The rest of the code was replaced by using Mathematica instead.

p1 = p.graph(x_range=x_bounds, y_range=y_bounds,
             width=width, height=height,
             file_name="example_shepard.html", show=False)




p = Plot("","","", "",
         font_family="times", font_size=18)

# Recall, this has already happened above.
# 
# m = ShepMod()
# m.fit(array(points, dtype=float))
# 
points = array(points)
def weight(x, i=0):
    distance_to_x = sum((points[i] - x)**2)**(1/2)
    if (distance_to_x < m.rw[i]):
        ids, wts = m(x)
        if (i in ids): return wts[list(ids).index(i)]
        else:          return 0
    else: return 0
# ^ Get the "weight of i" from the (ids, wts) return tuple.



# Add points to the 3D visual.
for i in range(len(points)):
    p.add(f"{i}", [points[i][0]], [points[i][1]],
          [1] if (i == 0) else [0], color="rgb(0,0,0)", marker_size=10)
    p.add(f"{i}", [points[i][0]], [points[i][1]],
          [1] if (i == 0) else [0], color=i, marker_line_color="#000")

p.add_func("Weight 0", weight, x_bounds, y_bounds, plot_points=4000)
p2 = p.show(x_range=x_bounds, y_range=y_bounds, z_range=[-.1, 2.1],
            show_legend=False, width=500, height=500)

# OLD 1D PLOT
# 
# weight = lambda x: (max(0,(1 - abs(x))) / (abs(x)))**2
# p.add("left rk", [-1, -1], [-.8, .8], mode="lines", color=(0,0,0,.7))
# p.add("center rk", [0, 0], [-.8, .8], mode="lines", color=(0,0,0,.7))
# p.add("right rk", [1, 1], [-.8, .8], mode="lines", color=(0,0,0,.7))
# p.add_annotation("r<sub>k</sub>", 1, 2, show_arrow=False,
#                  font_family="times", font_size=16)
# p.add_annotation("x<sup>(k)</sup>", 0, 2, show_arrow=False,
#                  font_family="times", font_size=16)
# p.add_annotation("-r<sub>k</sub>", -1, 2, show_arrow=False,
#                  font_family="times", font_size=16)
# p.add_func("Wk(x)", weight, [-1.5, 1.5], color=(0,0,0))
# p2 = p.show(x_range=[-1.5,1.5], y_range=[-1, 50], show_legend=False,
#             file_name="example_shepard_weight.html", show=False)

multiplot([p1, p2], width=600, height=200, gap=.18,
          file_name="example_shepard_with_weight.html")
