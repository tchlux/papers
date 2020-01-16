# Include the "code" directory into the path and import code custom to this project.
import sys, os
sys.path += [os.path.abspath("code")]
from fraction import Fraction as F
from polynomial import polynomial_piece, Polynomial, NewtonPolynomial
from plot import Plot, multiplot
sys.path.pop(-1)


x = [F(0),F(1)]
# A linear function.
y1 = [
    [F(1)],
    [F(0)]
]
# A cubic function.
y2 = [
    [F(1,2),F(4)],
    [F(1,2),F(4)]
]
# A quintic function.
y3 = [
    [F(1), F(-7), F( 100,1)],
    [F(0), F(-7), F(-100,1)]
]

f1 = Polynomial(polynomial_piece( *y1, x ))
f2 = Polynomial(polynomial_piece( *y2, x ))
f3 = Polynomial(polynomial_piece( *y3, x ))
print("f1: ",f1)
print("f2: ",f2)
print("f3: ",f3)

p1 = Plot("","","", font_family="times", font_size=20)
p1.add_func("f1", f1, x, color=p1.color(1))
p1.add_annotation("(1)&nbsp;&nbsp;to&nbsp;&nbsp;(0)", .5, .5, ax=.75,
                  ay=.8, font_family="times", font_size=16)

p2 = Plot("","","", font_family="times", font_size=20)
p2.add_func("f2", f2, x, color=p1.color(2))
p2.add_annotation("(½, 4)&nbsp;&nbsp;to&nbsp;&nbsp;(½, 4)", .5, .5, ax=40,
                  ay=.8, font_family="times", font_size=16)


p3 = Plot("","","", font_family="times", font_size=18)
p3.add_func("f3", f3, x, color=p1.color(3))
p3.add_annotation("(1,-7,100)&nbsp;&nbsp;to&nbsp;&nbsp;(0,-7,-100)",
                  .5, .5, ax=15, ay=.8, font_family="times",
                  font_size=16)


multiplot([p1,p2,p3], gap=.06, shared_y=True, show_legend=False,
          width=680, height=100, file_name="spline_demonstration.html")
