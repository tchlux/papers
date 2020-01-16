import numpy as np
# Include the "code" directory into the path and import code custom to this project.
import sys, os
sys.path += [os.path.abspath("code")]
from plot import Plot
from polynomial import polynomial_piece
sys.path.pop(-1)

c = "rgba(50,50,50,.5)"
a = .8
p = Plot("","α","β", font_family="times", font_size=18)
p.add("Region 1", [0, 3, 3], [3, 3, 0], mode="lines", fill="tonext",
      color=p.color(0), group=0)
p.add_func("Region 2", lambda x: (9 - x**2)**(1/2), [0,3], group=1,
           color=p.color(2), fill="tonext")
p.add("Region 3", [0, 3], [3, 0], mode="lines", fill="tonext", 
      color=p.color(3), group=2)
p.add("Region 4", [0, 1, 3], [3, 1, 0], mode="lines", fill="tozeroy", 
      color=p.color(1), group=3)

p.add("Point 1 to Region 1", [3,2], [4.5,3], color=c, group=0,
      line_color=p.color(0,alpha=a), mode="markers+lines", dash=None)
p.add("Point 2 to Region 2", [4,3*4/(4**2+3**2)**(1/2)], [3,3*3/(4**2+3**2)**(1/2)], 
      group=1, color=c, line_color=p.color(2, alpha=a),
      mode="markers+lines", dash="dot")
p.add("Point 3 to Region 3", [4,12/5], [1,3/5], color=c, group=2,
      line_color=p.color(3,alpha=a), mode="markers+lines", dash="dash")
p.add("Point 4 to Region 4", [1.5,9/13], [3.5,21/13], color=c, group=3,
      line_color=p.color(1,alpha=a), mode="markers+lines", dash="dashdot")

p.add_annotation(" 4 ", .3, .3, font_family="times", font_size=17,
                 show_arrow=False, border_width=1, x_anchor="left")
p.add_annotation(" 3 ", 1.02, 1.02, font_family="times", font_size=17,
                 show_arrow=False, border_width=1, x_anchor="left")
p.add_annotation(" 2 ", 1.55, 1.55, font_family="times", font_size=17,
                 show_arrow=False, border_width=1, x_anchor="left")
p.add_annotation(" 1 ", 2.3, 2.3, font_family="times", font_size=17,
                 show_arrow=False, border_width=1, x_anchor="left")

p.show(file_name="cubic_projection.html", width=400, height=400, show_legend=False)



VERIFY_CUBIC = False
# --------------------------------------------------------------------
#             VERIFY CORRECTNESS OF CUBIC INTERPOLANT
if VERIFY_CUBIC:
    # Knots
    k0 = 0
    k1 = 2
    h0 = k1 - k0
    # Truth function
    f = lambda x: 0. if x < (k0+k1)/2 else 1.
    df = lambda x: -1.
    # Basis functions.
    u = lambda t: 3*t**2 - 2*t**3
    p = lambda t: t**3 - t**2
    # Approximation function
    fh = lambda x: f(k0) * u((k1-x)/h0) + f(k1) * u((x - k0)/h0) - \
         h0 * df(k0) * p((k1-x)/h0) + h0 * df(k1) * p((x - k0)/h0)
    pp = polynomial_piece([0,-1], [1,-1], (k0,k1))

    # Plot this.
    plot = Plot()
    plot.add_func("F hat", fh, [k0, k1])
    plot.add_func("Pp", pp, [k0, k1])
    plot.show()
# --------------------------------------------------------------------

