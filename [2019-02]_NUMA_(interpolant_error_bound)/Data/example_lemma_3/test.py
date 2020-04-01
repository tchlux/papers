from util.plot import Plot
from util.math.points import fekete_points, grid
from util.random import latin



lin = lambda x: x + .1
lin2 = lambda x: 10 * x
quad = lambda x: x**2
quad2 = lambda x: 10 * x**2
bounds = [.000000001, 1]

p = Plot()
# n = 512
# p.add("Grid", *(grid(n, 2).T), color=p.color(0,alpha=.5))
# p.add("Latin", *(latin(n, 2).T), color=p.color(1,alpha=.5))
# p.add("Fekete", *(fekete_points(n, 2).T), color=p.color(2,alpha=.5))

p.add_func("y = x", lin, bounds)
p.add_func("y = 10 x", lin2, bounds)
p.add_func("y = x^2", quad, bounds)
p.add_func("y = 10 x^2", quad2, bounds)
p.show(show=False)
p.show(x_range=[0,-10], y_range=[-10,0],
       y_axis_settings=dict(type="log"),
       x_axis_settings=dict(type="log"),
       append=True)


# 1 + 2D
# rotate and do 2D more
# repeat
# 
# ^ This process could be aimed at the megadiagnoals, involving the
#   combination of many components, sampling the point with all
#   ones.
