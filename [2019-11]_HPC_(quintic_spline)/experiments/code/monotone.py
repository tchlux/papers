# This file provides the function `monotone_quintic_spline` which
# constructs spline interpolant over provided data. It starts by using
# divided differences to construct approximations for the first and
# second derivative at each of the data points. Then it iteratively
# adjusts the derivatives and second derivatives until all quintic
# polynomial pieces are monotone. A `Spline` object is returned.

from polynomial import fit
IS_MONOTONE_CALLS = 0

# Given x and y values, construct a monotonic quintic spline fit.
#  shrink_rate = .9    -> 120 steps max
#              = .99   -> 386 steps max
#              = .999  -> 1221 steps max
#              = .9999 -> 3861 steps max
def monotone_quintic_spline(x, y=None, values=None, ends=2, mids=2, exact=True, 
                            max_steps=100, verbose=False, **kwargs):
    if verbose: print()
    from polynomial import Spline
    # Convert 'x' to exact values.
    if exact:
        from fraction import Fraction
        x = list(map(Fraction, x))
    # Build a set of 'values' if they were not provided.
    if (y is not None):
        if exact: y = list(map(Fraction, y))
        # Check for which type of monotonicity this function maintains.
        if   all(y[i+1] >= y[i] for i in range(len(y)-1)): kwargs.update(dict(min_d1=0))
        elif all(y[i] >= y[i+1] for i in range(len(y)-1)): kwargs.update(dict(max_d1=0))
        else: raise(Exception("Provided 'y' data is not monotone."))
        # Pass in the method of fitting end and midpoints as keyword arguments.
        kwargs.update(dict(mids=mids, ends=ends))
        # Construct an initial fit that is twice continuous.
        f = fit(x, y, continuity=2, **kwargs)
        values = f.values
    elif (values is not None):
        if exact: values = [list(map(Fraction, row)) for row in values]
        # Verify monotonicity.
        if   all(values[i+1][0] >= values[i][0] for i in range(len(values)-1)): pass
        elif all(values[i][0] >= values[i+1][0] for i in range(len(values)-1)): pass
        else: raise(Exception("Provided 'y' data is not monotone."))
        # Make a spline over the points and derivative values.
        f = Spline(x, values)
        values = f.values
    else:
        class UsageError(Exception): pass
        raise(UsageError("Must provided either a flat 'y' array or a (N,3) values array."))
    # Make all pieces monotone.
    check_counts = {}
    change_counts = {}
    change_values = {i:values[i][1] / max_steps for i in range(len(values))}
    to_check = list(range(len(values)-1))
    if verbose:
        print("Starting quintic monotonicity fix..\n")
        print("change_values: ",change_values)
    # Cycle over all intervals that need to be checked for monotonicity.
    while (len(to_check) > 0):
        i = to_check.pop(0)
        if verbose: print(f"\n  checking ({i+1}) ..")
        # Update this interval to make it monotone.
        changed = monotone_quintic(
            [x[i], x[i+1]], [values[i], values[i+1]])
        check_counts[i] = check_counts.get(i,0) + 1
        # If in fact this interval wasn't changed, remove the increment.
        if (not changed) and verbose: print(f"   not changed..")
        # Mark the adjacent intervals as needing to be checked,
        # since this adjustment may have broken their monotonicity.
        if changed:
            # Track the number of times this particular interval has been changed.
            change_counts[i] = change_counts.get(i,0) + 1
            # Shrink neighboring derivative values if this interval is looping.
            if change_counts[i] > 1:
                values[i][1]   = max(0, values[i][1] - change_values[i])
                values[i+1][1] = max(0, values[i+1][1] - change_values[i+1])
            # Queue up this interval and its neighbors to be checked again.
            next_value  = to_check[0]  if (len(to_check) > 0) else None
            second_last = to_check[-2] if (len(to_check) > 1) else None
            last_value  = to_check[-1] if (len(to_check) > 0) else None
            if (i > 0) and (i-1 not in {second_last, last_value}):
                to_check.append( i-1 )
            if (i not in {last_value}):
                to_check.append( i )
            if (i+1 < len(to_check)) and (i+1 != next_value):
                to_check.insert(0, i+1)
            # Maintain exactness if necessary!
            if exact:
                values[i][:] = map(Fraction, values[i])
                values[i+1][:] = map(Fraction, values[i+1])
            # Show some printouts to the user for verbosity.
            if verbose:
                print(f"   [update {change_counts[i]}] interval {i+1} changed ({changed})..")
                print(f"     {float(x[i]):.3f} {float(x[i+1]):.3f} ({float(values[i][0]):.2e} {float(values[i+1][0]):.2e})")
                print( "     ", ' '.join([f"{float(v): .5e}" for v in values[i]]))
                print( "     ", ' '.join([f"{float(v): .5e}" for v in values[i+1]]))
    if verbose:
        print("check_counts: ",check_counts)
        print("change_counts: ",change_counts)
        print("done.\n")
        return Spline(f.knots, f.values), check_counts, change_counts
        
    # Construct a new spline over the (updated) values for derivatives.
    return Spline(f.knots, f.values)

# Given a (x1, x2) and ([y1, d1y1, d2y1], [y2, d1y2, d2y2]), compute
# the rescaled derivative values for a monotone quintic piece.
def monotone_quintic(x, y, accuracy=2**(-26)):
    changed = False
    # Extract local variable names from the provided points and
    # derivative information (to match the source paper).
    U0, U1 = x
    X0, DX0, DDX0 = y[0]
    X1, DX1, DDX1 = y[1]
    function_change = X1 - X0
    interval_width = U1 - U0
    interval_slope = function_change / interval_width
    # Handle an unchanging interval.
    if (interval_slope == 0):
        changed = any(v != 0 for v in (DX0, DX1, DDX0, DDX1)) * 1
        y[0][:] = X0, 0, 0
        y[1][:] = X1, 0, 0
        return changed
    # Convert all the tests to the monotone increasing case (will undo).
    sign = (-1) ** int(function_change < 0)
    X0 *= sign
    X1 *= sign
    DX0 *= sign
    DX1 *= sign
    DDX0 *= sign
    DDX1 *= sign
    interval_slope *= sign
    # Set DX0 and DX1 to be the median of these three choices.
    if not (0 <= DX0 <= 14*interval_slope):
        new_val = (sorted([0, DX0, 14*interval_slope])[1])
        changed = (new_val != DX0)
        if changed: DX0 = new_val
    if not (0 <= DX1 <= 14*interval_slope):
        new_val = (sorted([0, DX1, 14*interval_slope])[1])
        changed += (new_val != DX1) + 10**1
        if changed: DX1 = new_val
    # Compute repeatedly used values "A" (left ratio) and "B" (right ratio).
    A = DX0 / interval_slope
    B = DX1 / interval_slope
    assert(A >= 0)
    assert(B >= 0)
    # Use a (simplified) monotone cubic over this region if AB = 0.
    # Only consider the coefficients less than the x^4, because that
    # term is strictly positive.
    if (A*B < accuracy):
        # Compute a temporary variable that is useful in the algebra.
        w = U0 - U1
        # First, find a DX0 and DX1 that makes DDX0 have nonempty feasible region.
        DX_multiplier = max(0, (- 20*(X1-X0) / w) / (5*DX0 + 4*DX1))
        if (DX_multiplier < 1):
            DX0 *= DX_multiplier
            DX1 *= DX_multiplier
            y[0][1] = DX0*sign
            y[1][1] = DX1*sign
            changed = True
        # Second, cap DDX0 so that the DDX1 feasible region is nonempty.
        max_DDX0 = (4*(2*DX0 + DX1) - 20*(X0-X1)/w) / w
        if (DDX0 > max_DDX0):
            DDX0 = max_DDX0
            y[0][2] = DDX0*sign
            changed = True
        # Enforce \gamma >= \delta
        min_DDX0 = 3 * DX0 / w
        if (min_DDX0 > DDX0):
            DDX0 = min_DDX0
            y[0][2] = DDX0*sign
            changed = True
        # Enforce \alpha >= 0 
        max_DDX1 = -4*DX1 / w
        if (DDX1 > max_DDX1):
            DDX1 = max_DDX1
            y[1][2] = DDX1*sign
            changed = True
        # Enforce \beta >= \alpha
        min_DDX1 = (3*DDX0*w - (24*DX0 + 32*DX1) + 60*(X0-X1)/w) / (5*w)
        if (DDX1 < min_DDX1):
            DDX1 = min_DDX1
            y[1][2] = DDX1*sign
            changed = True
        # Check for contradictions, which should never happen!
        assert(min_DDX0 <= max_DDX0)
        assert(min_DDX1 <= max_DDX1)
        # Return whether or not the values were changed.
        return changed

    # Clip derivative values that are too large (to ensure that
    # shrinking the derivative vectors on either end will not break
    # monotonicity). (clipping at 6 box is enough, with 8 needs more treatment)
    mult = (6 / max(A, B))
    if (mult < 1):
        DX0 *= mult
        DX1 *= mult
        A = DX0 / interval_slope
        B = DX1 / interval_slope
        changed = True
    # Make sure that the first monotonicity condition is met.
    tau_1 = 24 + 2*(A*B)**(1/2) - 3*(A+B)
    try: assert(tau_1 >= 0)
    except:
        class NonMonotoneTau(Exception): pass
        raise(NonMonotoneTau(f"Bad Tau 1 value: {tau_1}\n A: {A}\n B: {B}"))

    # Compute DDX0 and DDX1 that satisfy monotonicity by scaling (C,D)
    # down until montonicity is achieved (using binary search).
    alpha_constant   = 4 * (B**(1/4) / A**(1/4))
    alpha_multiplier = ((U0-U1) / DX1) * B**(1/4) / A**(1/4)
    gamma_constant   = 4 * (DX0 / DX1) * (B**(3/4) / A**(3/4))
    gamma_multiplier = ((U1-U0) / DX1) * (B**(3/4) / A**(3/4))
    beta_constant    = (12 * (DX0+DX1) * (U1-U0) + 30 * (X0-X1)) / ((X0-X1) * A**(1/2) * B**(1/2))
    beta_multiplier  = (3 * (U0-U1)**2) / (2 * (X0-X1) * A**(1/2) * B**(1/2))
    def is_monotone():
        global IS_MONOTONE_CALLS
        IS_MONOTONE_CALLS += 1
        a = alpha_constant + alpha_multiplier * DDX1
        g = gamma_constant + gamma_multiplier * DDX0
        b = beta_constant  + beta_multiplier  * (DDX0 - DDX1)
        if b <= 6: bound = - (b + 2) / 2
        else:      bound = -2 * (b - 2)**(1/2)
        return (a-accuracy > bound) and (g-accuracy > bound)

    # Move the second derivative towards a working value until
    # monotonicity is achieved.
    target_DDX0 = - A**(1/2) * (7*A**(1/2) + 3*B**(1/2)) * interval_slope / interval_width
    target_DDX1 =   B**(1/2) * (3*A**(1/2) + 7*B**(1/2)) * interval_slope / interval_width

    # If this function is not monotone, perform a binary
    # search for the nearest-to-original monotone DDX values.
    if not is_monotone():
        original_DDX0, original_DDX1 = DDX0, DDX1
        low_bound,     upp_bound     = 0,    1
        # Continue dividing the interval in 2 to find the smallest
        # "upper bound" (amount target) that satisfies monotonicity.
        while ((upp_bound - low_bound) > accuracy):
            to_target = upp_bound / 2  +  low_bound / 2
            # If we found the limit of floating point numbers, break.
            if ((to_target == upp_bound) or (to_target == low_bound)): break
            # Recompute DDX0 and DDX1 based on the to_target.
            DDX0 = (1-to_target) * original_DDX0 + to_target * target_DDX0
            DDX1 = (1-to_target) * original_DDX1 + to_target * target_DDX1
            # Otherwise, proceed with a binary seaarch.
            if is_monotone(): upp_bound = to_target
            else:             low_bound = to_target
        # Store the smallest amount "target" for DDX0 and DDX1 that is monotone.
        DDX0 = (1-upp_bound) * original_DDX0 + upp_bound * target_DDX0
        DDX1 = (1-upp_bound) * original_DDX1 + upp_bound * target_DDX1
        changed = True

    # Update "y" and return the updated version.
    y[0][:] = sign*X0, sign*DX0, sign*DDX0
    y[1][:] = sign*X1, sign*DX1, sign*DDX1
    return changed


# Given x and y values, construct a monotonic quintic spline fit.
def monotone_cubic_spline(x, y=None, values=None, exact=False):
    import numpy as np
    from polynomial import Spline
    # Convert 'x' to exact values.
    if exact:
        from fraction import Fraction
        x = list(map(Fraction, x))
    # Process the provided data to prepare for monotonicity checks.
    if (y is not None):
        if exact: y = list(map(Fraction, y))
        kwarg = {}
        if   all(y[i+1] >= y[i] for i in range(len(y)-1)): kwarg = dict(min_d1=0)
        elif all(y[i] >= y[i+1] for i in range(len(y)-1)): kwarg = dict(max_d1=0)
        else: raise(Exception("Provided 'y' data is not monotone."))
        f = fit(x, y, continuity=1, **kwarg)
        values = f.values
    elif (values is not None):
        if exact: values = [list(map(Fraction, row)) for row in values]
        # Verify monotonicity.
        if   all(values[i+1][0] >= values[i][0] for i in range(len(values)-1)): pass
        elif all(values[i][0] >= values[i+1][0] for i in range(len(values)-1)): pass
        else: raise(Exception("Provided 'y' data is not monotone."))
        # Make a spline over the points and derivative values.
        f = Spline(x, values)
        values = f.values
    else:
        class UsageError(Exception): pass
        raise(UsageError("Must provided either a flat 'y' array or a (N,2) values array."))
    # Make all pieces monotone.
    for i in range(len(values)-1):
        monotone_cubic([x[i], x[i+1]], [values[i], values[i+1]])
    # Construct a new spline over the (updated) values for derivatives.
    return Spline(f.knots, f.values)

# Given a (x1, x2) and ([y1, d1y1], [y2, d1y2]), compute
# the rescaled y values to make this piece monotone. 
def monotone_cubic(x, y):
    # Compute the secant slope, the left slope ratio and the
    # right slope ratio for this interval of the function.
    secant_slope = (y[1][0] - y[0][0]) / (x[1] - x[0])
    A = y[0][1] / secant_slope # (left slope ratio)
    B = y[1][1] / secant_slope # (right slope ratio)
    # ----------------------------------------------------------------
    #    USE PROJECTION ONTO CUBE FOR CUBIC SPLINE CONSTRUCTION
    # Determine which line segment it will project onto.
    mult = 3 / max(A, B)
    if (mult < 1):
        # Perform the projection onto the line segment by
        # shortening the vector to make the max coordinate 3.
        y[0][1] *= mult
        y[1][1] *= mult
        return True
    # ----------------------------------------------------------------
    return False



if __name__ == "__main__":
    # --------------------------------------------------------------------
    #               TEST CASES
    # 
    # 0, 4, (None)      -> Good
    # 1, 4, (None)      -> Bad (now good) (not monotone) [bad first derivative conditions]
    # 0, 13, (None)     -> Bad (now good) (not monotone after first pass) [no previous-interval fix]
    # 0, 30, (-3,None)  -> Bad (now good) (far right still not monotone after passes) [bad simplied conditions]
    # 0, 100, (None)    -> Ehh (now good) (barely non-monotone after 100 steps) [made lower ceiling for derivatives]
    # 0, 1000, (None)   -> Good
    # 
    # --------------------------------------------------------------------

    # SEED = 0
    # NODES = 100
    # SUBSET = slice(None) 

    SEED = 9
    INTERNAL_NODES = 53
    SUBSET = slice(None)

    # Generate random data to test the monotonic fit function.
    import numpy as np
    np.random.seed(SEED)
    nodes = INTERNAL_NODES + 2
    x = np.linspace(0, 1, nodes)
    y = sorted(np.random.normal(size=(nodes,)))
    x -= min(x); x /= max(x)
    y -= min(y); y /= max(y); y -= max(y)
    # Convert these arrays to lists.
    x, y = list(x)[SUBSET], list(y)[SUBSET]

    # Convert these arrays to exact arithmetic.
    from fraction import Fraction
    x = list(map(Fraction, x))
    y = list(map(Fraction, y))
    interval = [float(min(x)), float(max(x))]

    # Generate a plot to see what it all looks like.
    from plot import Plot

    p = Plot()
    p.add("Points", list(map(float, x)), list(map(float, y)))

    kwargs = dict(plot_points=1000)
    # Continuity 0
    p.add_func("continuity 0", fit(x,y,continuity=0),
               interval, group='0', **kwargs)
    p.add_func("c0 d1", fit(x,y,continuity=0, min_d1=0).derivative(),
               interval, dash="dash", color=p.color(p.color_num,alpha=.5), 
               group='0', **kwargs)
    # Continuity 1
    # p.add_func("continuity 1", fit(x,y,continuity=1, ends=1, mids=0), 
    #            interval, group='1', **kwargs)
    # p.add_func("c1 d1", fit(x,y,continuity=1, min_d1=0).derivative(), 
    #            interval, dash="dash", color=p.color(p.color_num,alpha=.5), 
    #            group='1', **kwargs)
    f = monotone_cubic_spline(x,y)
    p.add_func("monotone c1", f, interval, group='1m', **kwargs)
    p.add_func("monotone c1 d1", f.derivative(), interval, 
               dash="dash", color=p.color(p.color_num,alpha=.5), group='1m', **kwargs)
    # Continuity 2
    # p.add_func("continuity 2", fit(x,y,continuity=2, min_d1=0), 
    #            interval, group='2', **kwargs)
    # p.add_func("c2 d1", fit(x,y,continuity=2, min_d1=0).derivative(), 
    #            interval, dash="dash", color=p.color(p.color_num,alpha=.5), group='2', **kwargs)

    f,_,_ = monotone_quintic_spline(x,y, verbose=True)
    kwargs = dict(plot_points=10000)
    p.add_func("monotone c2", f, interval, group='2mf', color=p.color(7), **kwargs)
    p.add_func("monotone c2d1", f.derivative(), interval, 
               dash="dash", color=p.color(7,alpha=.5), group='2mf', **kwargs)

    p.show(file_name="monotone_quintic_interpolating_spline.html")
