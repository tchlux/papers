
import numpy as np
from util.math import Fraction
def exact(l):
    try:    return [exact(v) for v in l]
    except: return Fraction(l)

# Given data points and values, construct and return a function that
# evaluates a shape preserving quadratic spline through the data.
def quadratic(x, y, pts=1000):
    import fmodpy
    toms574 = fmodpy.fimport("toms574.f", implicit_typing=True, 
                             end_is_named=False, verbose=False)
    n = len(x)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    m = np.zeros(n, dtype=np.float32)
    # Get the slopes for the quadratic spline interpolant.
    toms574.slopes(x, y, m, n)
    # Define a function for evaluating the quadratic spline.
    def eval_quadratic(z, x=x, y=y, m=m):
        # Make sure "z" is an array.
        if (not issubclass(type(z), np.ndarray)): z = [z]
        # Clip input positions to be inside evaluated range.
        z = np.clip(z, min(x), max(x))
        # Initialize all arguments for calling TOMS 574 code.
        k = len(z)
        z = np.asarray(z, dtype=np.float32)
        z.sort()
        fz = np.zeros(k, dtype=np.float32)
        eps = 2**(-13)
        err = 0
        # Call TOMS 574 code and check for errors.
        err = toms574.meval(z, fz, x, y, m, n, k, eps, err)[-1]
        assert (err == 0), f"Nonzero error '{err}' when evaluating spline."
        # Return values.
        return fz

    # x = np.linspace(min(x), max(x), pts)
    # y = eval_quadratic(x)
    # return quintic(x,y)

    def deriv_1(z, s=.001):
        try: return [deriv_1(v) for v in z]
        except:
            from util.math import fit_polynomial
            # Construct a local quadratic approximation.
            px = np.array([z-s, z, z+s])
            py = eval_quadratic(px)
            f = fit_polynomial(px, py)
            return f.derivative()(z)
    def deriv_2(z, s=.01):
        try: return [deriv_2(v) for v in z]
        except:
            from util.math import fit_polynomial, Polynomial
            # Construct a local quadratic approximation.
            # px = np.array([z-s, z-s/2, z, z+s/2, z+s])
            px = np.array([z-s, z, z+s])
            py = eval_quadratic(px)
            f = fit_polynomial(px, py)
            return f.derivative().derivative()(z)
    deriv_1.derivative = lambda : deriv_2
    eval_quadratic.derivative = lambda : deriv_1
    return eval_quadratic

    # new_x = np.linspace(min(x), max(x), (len(x)-1)*10 + 1)
    # new_y = eval_quadratic(new_x)
    # # Return a function that evaluates the quadratic spline.
    # from util.math import fit_spline
    # return fit_spline(exact(new_x), exact(new_y), continuity=1)

# Use PCHIP and return.
def cubic(x, y):
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(x, y)

# Fit and evaluate a monotone quintic spline interpolant.
def quintic(x, y, spline=False):
    from mqsi import mqsi, eval_spline
    nd = len(x)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.zeros(3*nd+6, dtype=float)
    bcoef = np.zeros(3*nd, dtype=float)
    uv = np.zeros((nd,2), dtype=float, order='F')
    info = mqsi(x, y, t, bcoef, uv=uv)[-2]
    assert (info == 0), f"MQSI produced error code '{info}'."
    if spline:
        values = np.concatenate((y[:,None], uv), axis=1)
        from util.math import Spline
        return Spline(exact(x), exact(values))
    else:
        # Evaluate the quintic with the Fortran code:
        def eval_quintic(z, t=t, bcoef=bcoef, d=0):
            if (not issubclass(type(z), np.ndarray)): z = [z]
            xy = np.array(z, dtype=float)
            _, info = eval_spline(t, bcoef, xy, d=d)
            assert (info == 0), f"EVAL_SPLINE produced error code '{info}'."
            return xy
        def deriv_2(z): return eval_quintic(z, d=2)
        def deriv_1(z): return eval_quintic(z, d=1)
        deriv_1.derivative = lambda: deriv_2
        eval_quintic.derivative = lambda: deriv_1
        # Return the evaluator.
        return eval_quintic

# Import some testing functions.
from test_functions import piecewise_polynomial, signal, \
    large_tangent, atanh, exponential, trig

CONVERGENCE_TEST = False

if CONVERGENCE_TEST:
    # Test the decrease in maximum accuracy with more points.
    lower = 0
    upper = 1
    num_points = 8
    check_points = 10
    f, df, ddf = exponential
    # Evaluate the test points.
    tx = np.linspace(lower, upper, 1000000)
    ty = f(tx) # np.array([f(v) for v in tx])
    tdy = df(tx) # np.array([df(v) for v in tx])
    tddy = ddf(tx) # np.array([ddf(v) for v in tx])
    # Cycle up adding more data points.
    for n in range(16):
        # if (num_points <= 2**15): # 2**15 = ~32k
        #     num_points *= 2
        #     continue
        print()
        # print(f"{n} {num_points}  [{lower:.2e}, {upper:.2e}]")
        print(f"{num_points}")
        print("","constructing points..", end="\r")
        # Construct the set of points.
        x = np.linspace(lower, upper, num_points)
        y = f(x)
        print("","constructing approximations..", end="\r")
        # Construct the approximations.
        #  quintic
        m = quintic(x,y)
        dm = m.derivative()
        ddm = dm.derivative()
        #  cubic
        p = cubic(x,y)
        dp = p.derivative()
        ddp = dp.derivative()
        # #  quadratic
        # q = quadratic(x,y)
        # dq = q.derivative()
        # ddq = dq.derivative()

        print("","evaluating quintic..", end="\r")    
        tm = abs(m(tx) - ty)
        tdm = abs(dm(tx) - tdy)
        tddm = abs(ddm(tx) - tddy)
        print("", "%.3e  %.3e  %.3e"%(max(tm), max(tdm), max(tddm)))

        print("","evaluating cubic..", end="\r")    
        tp = abs(p(tx) - ty)
        tdp = abs(dp(tx) - tdy)
        tddp = abs(ddp(tx) - tddy)
        print("", "%.3e  %.3e  %.3e"%(max(tp), max(tdp), max(tddp)))

        # print("","evaluating quadratic..", end="\r")    
        # tq = max(abs(q(tx) - ty))
        # tdq = max(abs(dq(tx) - tdy))
        # tddq = max(abs(ddq(tx) - tddy))
        # print("", "%.3e  %.3e  %.3e"%(tq, tdq, tddq))

        # if (num_points == 131072):
        #     print("showing worst..")
        #     sorted_by_error = np.argsort(tm)
        #     to_keep = sorted_by_error[-1000:]
        #     worst_idx = np.argmax(abs(tm))
        #     to_keep = sorted(set(to_keep).union( set(
        #         range(max(0,worst_idx-50), min(len(tm),worst_idx+50))) ))
        #     my = m(tx)
        #     py = p(tx)
        #     from util.plot import Plot
        #     _ = Plot(f"Approximating e^x on [{lower},{upper}] given {num_points} points", "x", "Error")
        #     _.add("MQSI error", tx[to_keep], (my - ty)[to_keep], mode="lines")
        #     _.add("PCHIP error", tx[to_keep], (py - ty)[to_keep], mode="plines")
        #     _.show(show=False)
        #     _ = Plot()
        #     _.add("MQSI", tx[to_keep], my[to_keep], mode="lines")
        #     _.add("PCHIP", tx[to_keep], py[to_keep], mode="plines")
        #     _.show(append=True)

        # Set the new "lower" for the next test.x
        num_points *= 2
        # lower = (lower + upper) / 2

    from util.plot import Plot
    p = Plot()

    exit()



from util.plot import Plot, multiplot
legend = dict(
    xanchor = "center",
    yanchor = "top",
    x = .5,
    y = -.15,
    orientation = "h",
)
layout = dict(
    margin = dict(l=60, t=30, b=30),
)

# Construct a list of tests to run.
TESTS = [
    # n, min val, max val, function
    (7, 0, 1, *large_tangent),
    (5, 0, np.pi, np.sin, np.cos, lambda x: -np.sin(x)),
    (7, 0, np.pi, np.sin, np.cos, lambda x: -np.sin(x)),
    (10, 0, np.pi, np.sin, np.cos, lambda x: -np.sin(x)),
    (18, 0, 1, *piecewise_polynomial),
    (5, 0, 1, *signal),
    (13, 0, 1, *signal),
    (20, 0, 1, *signal),
]     

# Test the approximations (and their derivatives) on various functions.
for (n, min_val, max_val, f, df, ddf)  in TESTS:
    x = np.linspace(min_val,max_val,n)
    y = f(x)

    print()
    print(f)
    print("  Value plot..")
    print("   quintic..", end="\r")
    m = quintic(x,y)
    print("   cubic..", end="\r")
    p = cubic(x,y)
    print("   quadratic..", end="\r")
    q = quadratic(x,y)

    p1 = Plot("","x","f(x)")
    print("   data..", end="\r")
    p1.add("Data", x, y, group='d', marker_size=7, marker_line_width=1)
    print("   quintic..", end="\r")
    p1.add_func("MQSI", m, [min(x), max(x)], group='m')
    print("   cubic..", end="\r")
    p1.add_func("PCHIP", p, [min(x), max(x)], dash="dash", group='p')
    print("   quadratic..", end="\r")
    p1.add_func("Quadratic", q, [min(x), max(x)], dash="dot", group='q')
    # p1.add_func("truth", f, [min(x), max(x)], color=(0,0,0,.2), group='t')


    print("  First derivative..")
    dm = m.derivative()
    dp = p.derivative()
    dq = q.derivative()

    p2 = Plot("","x","f'(x)")
    print("   data..", end="\r")
    p2.add("Data", x, [df(v) for v in x], show_in_legend=False, group='d', marker_size=7, marker_line_width=1)
    print("   quintic..", end="\r")
    p2.add_func("MQSI", dm, [min(x), max(x)], show_in_legend=False, group='m', plot_points=5000)
    print("   cubic..", end="\r")
    p2.add_func("PCHIP", dp, [min(x), max(x)], dash="dash", show_in_legend=False, group='p', plot_points=5000)
    print("   quadratic..", end="\r")
    p2.add_func("Quadratic", dq, [min(x), max(x)], dash="dot", show_in_legend=False, group='q', plot_points=5000)
    # p2.add_func("truth", df, [min(x), max(x)], color=(0,0,0,.2), show_in_legend=False, group='t')

    multiplot([[p1],[p2]], legend=legend, layout=layout,
              append=True, show=(n == TESTS[-1][0]))

    # print("  Second derivative..")
    # ddm = dm.derivative()
    # ddp = dp.derivative()
    # # ddq = dq.derivative()

    # p3 = Plot("","x","f''(x)")
    # print("   data..", end="\r")
    # p3.add("Data", x, [ddf(v) for v in x], show_in_legend=False, group='d', marker_size=7, marker_line_width=1)
    # print("   quintic..", end="\r")
    # p3.add_func("MQSI", ddm, [min(x), max(x)], show_in_legend=False, group='m', plot_points=5000)
    # print("   cubic..", end="\r")
    # p3.add_func("PCHIP", ddp, [min(x), max(x)], dash="dash", show_in_legend=False, group='p', plot_points=5000)
    # print("   quadratic..", end="\r")
    # p3.add_func("Quadratic", ddq, [min(x), max(x)], dash="dot", show_in_legend=False, group='q', plot_points=5000)
    # p3.add_func("truth", ddf, [min(x), max(x)], color=(0,0,0,.2), show_in_legend=False, group='t')

    # multiplot([[p1],[p2],[p3]], legend=legend, layout=layout,
    #           append=True, show=(n == TESTS[-1][0]))



# 2021-01-23 20:05:53
# 
###########################################################################
# # Use the derivative estimations from TOMS 574, but a cubic spline fit. #
# def quad_spline(x, y):                                                  #
#     n = len(x)                                                          #
#     x = np.asarray(x, dtype=np.float32)                                 #
#     y = np.asarray(y, dtype=np.float32)                                 #
#     m = np.zeros(n, dtype=np.float32)                                   #
#     # Get the slopes for the quadratic spline interpolant.              #
#     toms574.slopes(x, y, m, n)                                          #
#     # Return a spline fit of the data.                                  #
#     from util.math import Spline                                        #
#     return Spline(x, np.concatenate((y[:,None], m[:,None]), axis=1))    #
###########################################################################

# 2021-01-23 20:06:02
# 
###################################################
# qs = quad_spline(x, y)                          #
# p.add_func("Quad spline", qs, [min(x), max(x)]) #
###################################################


# 2021-01-24 12:07:50
# 
###############################
#     # multiplot([p1,p2,p3]) #
# # p.show()                  #
###############################
