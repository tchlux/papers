from util.plot import Plot
from math import log

p = Plot()
f = lambda n, d: n * log(n)**(d-1)
f100 = lambda d: f(100, d)
p.add_func("O( N log(N)<sup>d-1</sup> )", f100, [1,100])
p.show()
