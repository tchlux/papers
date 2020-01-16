# This file provides utilities for constructing polynomial interpolants.
# 
# The following objects are provided:
# 
#   Spline           -- A piecewise polynomial interpolant with
#                       evaluation, derivative, and a string method.
#   Polynomial       -- A monomial with stored coefficients,
#                       evaluation, derivative, and a string method.
#   NewtonPolynomial -- A Newton polynomial with stored coefficients,
#                       points, evaluation, derivative, and a string method.
# 
# The following functions are provided:
# 
#   polynomial       -- Given x and y values, this produces a minimum
#                       degree Newton polynomial that interpolates the
#                       provided points (this is a *single* polynomial).
#   polynomial_piece -- Given a function value and derivatives,
#                       produce a polynomial that interpolates these
#                       values at the endpoints of an interval.
#   fit              -- Use 'fill_derivative' to construct an interpolating
#                       Spline with a specified level of continuity.
#   fill_derivative  -- Compute all derivatives at points to be
#                       reasonable values using either a linear or a
#                       quadratic fit over neighboring points. 
#   solve_quadratic  -- Given three points, this will solve the equation
#                       for the quadratic function which interpolates
#                       all 3 values. Returns coefficient 3-tuple. 
# 

from fraction import Fraction

# This class-method wrapper function ensures that the class method
# recieves a fraction and returns a "float" if a fraction was not
# provided as input (ensuring internal methods only recieve fractions).
def float_fallback(class_method):
    def wrapped_method(obj, x):
        # Check for vector usage.
        try: return [wrapped_method(obj, v) for v in x]
        except:
            # Return perfect precision if that was provided.
            if (type(x) == Fraction): return class_method(obj, x)
            # Otherwise, use exact arithmetic internally and return as float.
            else: return float(class_method(obj, Fraction(x)))
    return wrapped_method

# A piecewise polynomial function that supports evaluation,
# differentiation, and stitches together an arbitrary sequence of
# function values (and any number of derivatives) at data points
# (knots). Provides evaluation, derivation, and string methods.
# It is recommended to use EXACT ARITHMETIC internally (which will
# automatically return floats unless Fraction objects are provided as
# evaluation points). Exact arithmetic is achieved by providing knots
# and values composed of Fraction objects.
# 
# Spline(knots, values):
#   Given a sequence of "knots" [float, ...] and an equal-length
#   sequence of "values" [[float, ...], ...] that define the
#   function value and any number of derivatives at every knot,
#   construct the piecewise polynomial that is this spline.
class Spline:
    # Define private internal variables for holding the knots, values,
    # and the functions. Provide access to "knots" and "values" properties.
    _knots = None
    _values = None
    _functions = None
    @property
    def knots(self): return self._knots
    @knots.setter
    def knots(self, knots): self._knots = [k for k in knots]
    @property
    def values(self): return self._values
    @values.setter
    def values(self, values): self._values = [[v for v in vals] for vals in values]
    
    def __init__(self, knots, values):
        assert(len(knots) == len(values))
        self.knots = knots
        self.values = values
        # Create the polynomial functions for all the pieces.
        self._functions = []
        for i in range(len(knots)-1):
            v0, v1 = self.values[i], self.values[i+1]
            k0, k1 = self.knots[i], self.knots[i+1]
            # Make the polynomial over a range starting a 0 to
            # increase stability of the resulting piece.
            f = polynomial_piece(v0, v1, (k0, k1))
            # Store the function (assuming it's correct).
            self._functions.append( f )

    # Evaluate this Spline at a given x coordinate.
    @float_fallback
    def __call__(self, x):
        # If "x" was given as a vector, then iterate over that vector.
        try:    return [self(v) for v in x]
        except: pass
        # Deterimine which interval this "x" values lives in.
        if (x <= self.knots[0]):    return self._functions[0](x)
        elif (x >= self.knots[-1]): return self._functions[-1](x)
        # Find the interval in which "x" exists.
        for i in range(len(self.knots)-1):
            if (self.knots[i] <= x <= self.knots[i+1]): break
        # If no interval was found, then something unexpected must have happened.
        else:
            class UnexpectedError(Exception): pass
            raise(UnexpectedError("This problem exhibited unexpected behavior."))
        # Now "self.knots[i] <= x <= self.knots[i+1]" must be true.
        return self._functions[i](x)
        
    # Compute the first derivative of this Spline.
    # WARNING: The returned spline does *not* have "values"!
    # TODO: Compute same number of values as parent, store them.
    def derivative(self, d=1):
        # Create a spline (do not bother filling values) with the same
        # knot sequence and all the derivative functions.
        s = Spline([], [])
        s.knots = self.knots
        s._functions = [f.derivative(d) for f in self._functions]
        return s

    # Produce a string description of this spline.
    def __str__(self):
        s = "Spline:\n"
        s += f" [-inf, {self.knots[1]}]  =  "
        s += str(self._functions[0]) + "\n"
        for i in range(1,len(self.knots)-2):
            s += f" ({self.knots[i]}, {self.knots[i+1]}]  =  "
            s += str(self._functions[i]) + "\n"
        s += f" ({self.knots[-2]}, inf)  =  "
        s += str(self._functions[-1])
        return s

# A generic Polynomial class that stores coefficients in monomial
# form. Provides numerically stable evaluation, derivative, and string
# operations for convenience. Coefficients should go highest to lowest order.
# 
# Polynomial(coefficients):
#    Given coefficients (or optionally a "NewtonPolynomial")
#    initialize this Monomial representation of a polynomial function.
class Polynomial:
    # Initialize internal storage for this Polynomial.
    _coefficients = None
    # Protect the "coefficients" of this class with a getter and
    # setter to ensure a user does not break them on accident.
    @property
    def coefficients(self): return self._coefficients
    @coefficients.setter
    def coefficients(self, coefs): self._coefficients = list(coefs)
    # Define an alternative alias shortform "coefs".
    @property
    def coefs(self): return self._coefficients
    @coefs.setter
    def coefs(self, coefs): self._coefficients = list(coefs)

    def __init__(self, coefficients):
        # If the user initialized this Polynomial with a Newton
        # Polynomial, then extract the points and coefficients.
        if (type(coefficients) == NewtonPolynomial):
            coefficients = to_monomial(coefficients.coefficients,
                                       coefficients.points)
        self.coefficients = coefficients

    # Evaluate this Polynomial at a point "x" in a numerically stable way.
    def __call__(self, x):
        if (len(self.coefficients) == 0): return 0
        total = self.coefficients[0]
        for d in range(1,len(self.coefficients)):
            total = self.coefficients[d] + x * total
        return total

    # Construct the polynomial that is the derivative of this polynomial.
    def derivative(self, d=1):
        if (d == 0):  return self
        elif (d > 1): return self.derivative().derivative(d-1)
        else:         return Polynomial([c*i for (c,i) in zip(
                self.coefficients, range(len(self.coefficients)-1,0,-1))])

    # Construct a string representation of this Polynomial.
    def __str__(self):
        s = ""
        for i in range(len(self.coefficients)):
            if (self.coefficients[i] == 0): continue
            if   (i == len(self.coefficients)-1): x = ""
            elif (i == len(self.coefficients)-2): x = "x"
            else:   x = f"x^{len(self.coefficients)-1-i}"
            s += f"{self.coefficients[i]} {x}  +  "
        # Remove the trailing 
        s = s.rstrip(" +")
        # Return the final string.
        return s

# Extend the standard Polymomial class to hold Newton polynomials with
# points in addition to the coefficients. This is more numerically
# stable when the points and coefficients are stored as "float" type.
# 
# NewtonPolynomial(coefficients, points):
#    Given a set of coefficients and a set of points (offsets), of
#    the same length, construct a standard Newton Polynomial.
class NewtonPolynomial(Polynomial):
    _points = None
    @property
    def points(self): return self._points
    @points.setter
    def points(self, points): self._points = list(points)

    # Store the coefficients and points for this Newton Polynomial.
    def __init__(self, coefficients, points):
        if (len(points) != len(coefficients)): raise(IndexError)
        self.coefficients = coefficients
        self.points = points

    # Construct the polynomial that is the derivative of this
    # polynomial by converting to monomial form and differntiating.
    def derivative(self, d=1): return Polynomial(self).derivative(d)

    # Evaluate this Newton Polynomial (in a numerically stable way).
    def __call__(self, x):
        total = self.coefficients[0]
        for d in range(1,len(self.coefficients)):
            total = self.coefficients[d] + (x - self.points[d]) * total
        return total

    # Construct a string representation of this Newton Polynomial.
    def __str__(self):
        s = f"{self.coefficients[0]}"
        for i in range(1,len(self.coefficients)):
            sign = "-" if (self.points[i] >= 0) else "+"
            s = f"{self.coefficients[i]} + (x {sign} {abs(self.points[i])})({s})"
        return s

# Given Newton form coefficients and points, convert them to monomial
# form (where all points are 0) having only coefficients.
def to_monomial(coefficients, points):
    coefs = [coefficients[0]]
    for i in range(1,len(coefficients)):
        # Compute the old coefficients multiplied by a constant and
        # add the lower power coefficients that are shifted up.
        coefs.append(coefficients[i])
        coefs = [coefs[0]] + [coefs[j+1]-points[i]*coefs[j] for j in range(len(coefs)-1)]
    return coefs

# Given unique "x" values and associated "y" values (of same length),
# construct an interpolating polynomial with the Newton divided
# difference method. Return that polynomial as a function.
def polynomial(x, y):
    # Sort the data by "x" value.
    indices = sorted(range(len(x)), key=lambda i: x[i])
    x = [x[i] for i in indices]
    y = [y[i] for i in indices]
    # Compute the divided difference table.
    dd_values = [y]
    for d in range(1, len(x)):
        slopes = []
        for i in range(len(dd_values[-1])-1):
            try:    dd = (dd_values[-1][i+1] - dd_values[-1][i]) / (x[i+d] - x[i])
            except: dd = 0
            slopes.append( dd )
        dd_values.append( slopes )
    # Get the divided difference (polynomial coefficients) in reverse
    # order so that the most nested value (highest order) is first.
    coefs = [row[0] for row in reversed(dd_values)]
    points = list(reversed(x))
    # Return the interpolating polynomial.
    return NewtonPolynomial(coefs, points)

# Given a (left value, left d1, ...), (right value, right d1, ...)
# pair of tuples, return the lowest order polynomial necessary to
# exactly match those values and derivatives at interval[0] on the left
# and interval[1] on the right ("interval" is optional, default [0,1]).
def polynomial_piece(left, right, interval=(0,1)):
    # Store the unscaled version for stability checks afterwards.
    v0, v1 = left, right
    # Make sure both are lists.
    left  = list(left)
    right = list(right)

    # Fill values by matching them on both sides of interval (reducing order).
    for i in range(len(left) - len(right)):
        right.append( left[len(right)] )
    for i in range(len(right) - len(left)):
        left.append( right[len(left)] )
    # Rescale left and right to make their usage in the divided
    # difference table correct (by dividing by the factorial).
    mult = 1
    for i in range(2,len(left)):
        mult *= i
        left[i] /= mult
        right[i] /= mult
    # First match the function value, then compute the coefficients
    # for all of the higher order terms in the polynomial.
    coefs = list(left)
    # Compute the divided difference up until we flatten out the
    # highest provided derivative between the two sides.
    interval_width = interval[1] - interval[0]
    if (len(left) == 1): dds = []
    else:                dds = [(right[0] - left[0]) / interval_width]
    for i in range(len(left)-2):
        new_vals = [ (dds[0] - left[i+1]) / interval_width ]
        for j in range(len(dds)-1):
            new_vals.append( (dds[j+1] - dds[j]) / interval_width )
        new_vals.append( (right[i+1] - dds[-1]) / interval_width )
        dds = new_vals
    # Now the last row of the dd table should be level with "left" and
    # "right", we can complete the rest of the table in the normal way.
    row = [left[-1]] + dds + [right[-1]]
    # Build out the divided difference table.
    while (len(row) > 1):
        row = [ (row[i+1]-row[i])/interval_width for i in range(len(row)-1) ]
        coefs.append(row[0])
    # Reverse the coefficients to go from highest order (most nested) to
    # lowest order, set the points to be the left and right ends.
    points = [interval[1]]*len(left) + [interval[0]]*len(left)
    coefs = list(reversed(coefs))
    # Finally, construct a Newton polynomial.
    f = NewtonPolynomial(coefs, points)

    # Check for errors in this polynomial, see if it its values are correct.
    error_tolerance = 2**(-26)
    k0, k1 = interval
    # Make sure all the function values match.
    for i in range(len(v0)):
        df = f.derivative(i)
        bad_left  = abs(df(k0) - (v0[i] if i < len(v0) else v1[i])) >= error_tolerance
        bad_right = abs(df(k1) - (v1[i] if i < len(v1) else v0[i])) >= error_tolerance
        if (bad_left or bad_right):
            # Convert knots and values to floats for printing.
            k0, k1 = map(float, interval)
            v0, v1 = list(map(float,left)), list(map(float,right))
            print()
            print("-"*70)
            print("error_tolerance: ",error_tolerance)
            print(f"Interval:              [{k0: .3f}, {k1: .3f}]")
            print("Assigned left values: ", v0)
            print("Assigned right values:", v1)
            print()
            lf = f"{'d'*i}f({k0: .3f})"
            print(f"Expected {lf} == {v0[i]: .3f}")
            print(f"     got {' '*len(lf)} == {df(k0): .3f}")
            print(f"     error {' '*(len(lf) - 2)} == {v0[i] - df(k0): .3e}")
            rf = f"{'d'*i}f({k1: .3f})"
            print(f"Expected {rf} == {v1[i]: .3f}")
            print(f"     got {' '*len(rf)} == {df(k1): .3f}")
            print(f"     error {' '*(len(rf) - 2)} == {v1[i] - df(k1): .3e}")
            print()
            print(f"{' '*i}coefs:",coefs)
            print(f"{' '*i}f:    ",f)
            print(f"{'d'*i}f:    ",df)
            print("-"*70)
            print()
            raise(Exception("The generated polynomial piece is numerically unstable."))

    # Return the polynomial function.
    return f

# Given data points "x" and data values "y", construct a monotone
# interpolating spline over the given points with specified level of
# continuity using the Newton divided difference method.
#  
# x: A strictly increasing sequences of numbers.
# y: The function values associated with each point.
# 
# continuity:
#   The level of continuity desired in the interpolating function.
# 
# kwargs:
#   "max_d{i}" -- A maximum value for a specific derivative.
#   "min_d{i}" -- A minimum value for a specific derivative.
# 
#   Otherwise, may include any keyword arguments for the `fill` function.
def fit(x, y, continuity=0, **kwargs):
    knots = [v for v in x]
    values = [[v] for v in y]
    fill_kwargs = {k:kwargs[k] for k in kwargs
                   if k[:5] not in {"max_d","min_d"}}
    # Construct further derivatives and refine the approximation
    # ensuring monotonicity in the process.
    for i in range(1,continuity+1):
        deriv = fill_derivative(knots, [v[-1] for v in values], **fill_kwargs)
        # Adjust for monotonicity conditions if appropriate.
        max_name = f"max_d{i}"
        if (max_name in kwargs): deriv = [min(kwargs[max_name], d) for d in deriv]
        min_name = f"min_d{i}"
        if (min_name in kwargs): deriv = [max(kwargs[min_name], d) for d in deriv]
        # Append all derivative values.
        for v,d in zip(values,deriv): v.append(d)
    # Return the interpolating spline.
    return Spline(knots, values)

# Compute all derivatives between points to be reasonable values
# using either a linear or a quadratic fit over adjacent points.
#  
# ends:
#   (zero)   endpoints to zero.
#   (lin)    endpoints to secant slope through endpoint neighbor.
#   (quad)   endpoints to capped quadratic interpolant slope.
#   (manual) a 2-tuple provides locked-in values for derivatives.
# 
# mids:
#   (zero)   all slopes are locked into the value 0.
#   (lin)    secant slope between left and right neighbor.
#   (quad)   slope of quadratic interpolant over three point window.
#   (manual) an (n-2)-tuple provides locked-in values for derivatives.
# 
# exact:
#   `True` if exact arithmetic should be used for derivative
#   computations, `False` otherwise.
def fill_derivative(x, y, ends=1, mids=1):
    # Initialize the derivatives at all points to be 0.
    deriv = [0] * len(x)
    # Set the endpoints according to desired method.
    if (ends == 0) or (len(x) < 2): pass
    # If the end slopes should be determined by a secant line..
    elif (ends == 1) or (len(x) < 3):
        deriv[0] = (y[1] - y[0]) / (x[1] - x[0])
        deriv[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    # If the end slopes should be determined by a quadratic..
    elif (ends == 2):
        # Compute the quadratic fit through the first three points and
        # use the slope of the quadratic as the estimate for the slope.
        a,b,c = solve_quadratic(x[:3], y[:3])
        deriv[0] = 2*a*x[0] + b
        # Do the same for the right endpoint.
        a,b,c = solve_quadratic(x[-3:], y[-3:])
        deriv[-1] = 2*a*x[-1] + b
    # If the ends were manually specified..
    elif (len(ends) == 2):
        deriv[0], deriv[-1] = ends
    else:
        raise(BadUsage("Manually defined endpoints must provide exactly two numbers."))
    # Initialize all the midpoints according to desired metohd.
    if (mids == 0) or (len(x) < 2): pass
    elif (mids == 1) or (len(x) < 3):
        for i in range(1, len(x)-1):
            deriv[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    elif (mids == 2):
        for i in range(1, len(x)-1):
            # Compute the quadratic fit of the three points and use
            # its slope at x[i] to estimate the derivative at x[i].
            a, b, c = solve_quadratic(x[i-1:i+1+1], y[i-1:i+1+1])
            deriv[i] = 2*a*x[i] + b
    elif (len(mids) == len(deriv)-2):
        deriv[1:-1] = mids
    else:
        raise(BadUsage("Manually defined endpoints must provide exactly two numbers."))
    # Return the computed derivatives.
    return deriv

# Given three points, this will solve the equation for the quadratic
# function which interpolates all 3 values. Returns coefficient 3-tuple.
class BadUsage(Exception): pass
def solve_quadratic(x, y):
    if len(x) != len(y): raise(BadUsage("X and Y must be the same length."))
    if len(x) != 3:      raise(BadUsage(f"Exactly 3 (x,y) coordinates must be given, received '{x}'."))
    x1, x2, x3 = x
    y1, y2, y3 = y
    a = -((-x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3)/((-x1 + x2) * (x2 - x3) * (-x1 + x3)))
    b = -(( x2**2 * y1 - x3**2 * y1 - x1**2 * y2 + x3**2 * y2 + x1**2 * y3 - x2**2 * y3)/((x1 - x2) * (x1 - x3) * (x2 - x3)))
    c = -((-x2**2 * x3 * y1 + x2 * x3**2 * y1 + x1**2 * x3 * y2 - x1 * x3**2 * y2 - x1**2 * x2 * y3 + x1 * x2**2 * y3)/((x1 - x2) * (x1 - x3) * (x2 - x3)))
    return (a,b,c)

# Given a function, use the Newton method to find the nearest inverse
# to a guess. If this method gets trapped, it will exit without having
# computed an inverse.
def inverse(f, y, x=0, accuracy=2**(-52), max_steps=100):
    # Get the derivative of the function.
    df = f.derivative()
    # Convert 'x' and 'y' to Fractions.
    x,y = Fraction(x), Fraction(y)
    # Search for a solution iteratively.
    for i in range(max_steps):
        diff = f(x) - y
        x -= diff / f.derivative()(x)
        # Stop the loop if we have gotten close to the correct answer.
        if (abs(diff) <= accuracy): break
    # Check for correctness.
    if (abs(f(x) - y) > accuracy):
        import warnings
        warnings.warn("\n\n  The calculated inverse has high error.\n"+
                      "  Consider providing better initial position.\n"+
                      "  This problem may also not be solvable.\n")
    # Return the final value.
    return Fraction(x)



# --------------------------------------------------------------------
#                            TESTING CODE

# Test the Polynomial class for basic operation.
def _test_Polynomial():
    f = Polynomial([3,0,1])
    assert(str(f) == "3 x^2  +  1")
    f = Polynomial([3,2,1])
    assert(str(f) == "3 x^2  +  2 x  +  1")
    assert(str(f.derivative()) == "6 x  +  2")
    assert(str(f.derivative(2)) == "6")
    assert(str(f.derivative(3)) == "")
    assert(str(f.derivative(4)) == "")
    assert(f.derivative(3)(10) == 0)
    f = Polynomial(NewtonPolynomial([3,2,1],[0,0,0]))
    assert(str(f) == "3 x^2  +  2 x  +  1")
    assert(str(f.derivative()) == "6 x  +  2")
    assert(str(f.derivative(2)) == "6")
    assert(str(f.derivative(3)) == "")
    assert(str(f.derivative(4)) == "")
    assert(f.derivative(3)(5) == 0)
    f = Polynomial(to_monomial([-1,10,-16,24,32,-32], [1,1,1,-1,-1,-1]))
    assert(str(f) == "-1 x^5  +  9 x^4  +  6 x^3  +  -22 x^2  +  11 x  +  -3")
    assert(str(f.derivative()) == "-5 x^4  +  36 x^3  +  18 x^2  +  -44 x  +  11")

# Test the Polynomial class for basic operation.
def _test_NewtonPolynomial():
    f = NewtonPolynomial([-1,2], [1,-1])
    assert(str(f) == "2 + (x + 1)(-1)")
    assert(str(Polynomial(f)) == "-1 x  +  1")
    f = NewtonPolynomial([-1,10,-16,24,32,-32], [1,1,1,-1,-1,-1])
    assert(str(f) == "-32 + (x + 1)(32 + (x + 1)(24 + (x + 1)(-16 + (x - 1)(10 + (x - 1)(-1)))))")

# Test the "polynomial" interpolation routine (uses Newton form).
def _test_polynomial():
    SMALL = 1.4901161193847656*10**(-8) 
    # ^^ SQRT(EPSILON(REAL(1.0)))
    x_vals = [0,1,2,3,4,5]
    y_vals = [1,2,1,2,1,10]
    f = polynomial(x_vals, y_vals)
    for (x,y) in zip(x_vals,y_vals):
        try:    assert( abs(y - f(x)) < SMALL )
        except:
            string =  "\n\nFailed test.\n"
            string += f" x:    {x}\n"
            string += f" y:    {y}\n"
            string += f" f({x}): {f(x)}"
            class FailedTest(Exception): pass
            raise(FailedTest(string))

# Test the "polynomial_piece" interpolation routine.
def _test_polynomial_piece(plot=False):
    if plot:
        from plot import Plot
        p = Plot("Polynomial Pieces")
    # Pick the (value, d1, ...) pairs for tests.
    left_rights = [
        ([0], [0]),
        ([0], [1]),
        ([0,1], [0,-1]),
        ([0,2], [0,-2]),
        ([0,1], [1,0]),
        ([1,0], [0,-1]),
        ([0,1], [0,1]),
        ([0,1,0], [0,-1,1]),
        ([0,1,10], [0,-1,-10]),
        ([-2,2,10,6], [0,0,20,-6]),
    ]
    interval = (1,3)
    # Plot a bunch of sample functions.
    for (left, right) in left_rights:
        name = f"{left}  {right}"
        # Convert to exact for testing correctness.
        left = list(map(Fraction, left))
        right = list(map(Fraction, right))
        interval = (Fraction(interval[0]), Fraction(interval[1]))
        # Construct the polynomial piece.
        f = polynomial_piece( left, right, interval=interval )
        exact_coefs = list(map(round,f.coefs))[::-1]
        left_evals = [f(interval[0])]
        right_evals = [f(interval[1])]
        for i in range(1,len(left)):
            df = f.derivative(i)
            left_evals.append( df(interval[0]) )
            right_evals.append( df(interval[1]) )
        # TODO: Print out an error if the assert statement fails.
        assert(0 == sum(abs(true - app) for (true, app) in zip(left,left_evals)))
        assert(0 == sum(abs(true - app) for (true, app) in zip(right,right_evals)))
        # Create a plot of the functions if a demo is desired.
        if plot: p.add_func(name, f, [interval[0]-.1, interval[1]+.1],
                            )#mode="markers", marker_size=2)
    if plot: p.show(file_name="piecewise_polynomial.html")


# Test the Spline class for basic operation.
def _test_Spline():
    knots = [0,1,2,3,4]
    values = [[0],[1,-1,0],[0,-1],[1,0,0],[0]]
    f = Spline(knots, values)
    for (k,v) in zip(knots,values):
        for d in range(len(v)):
            try: assert(f.derivative(d)(k) == v[d])
            except:
                print()
                print('-'*70)
                print("      TEST CASE")
                print("Knot:           ", k)
                print("Derivative:     ", d)
                print("Expected value: ", v[d])
                print("Received value: ", f.derivative(d)(k))
                print()
                print(f)
                print('-'*70)
                raise(Exception("Failed test case."))


# Test the "fit" function. (there is testing code built in, so this
# test is strictly for generating a visual to verify).
def _test_fit(plot=False):
    x_vals = list(map(Fraction, [0,.5,2,3.5,4,5.3,6]))
    # y_vals = [1,2,-1,3,1,4,3]
    y_vals = list(map(Fraction, [1,2,2.2,3,3.5,4,4]))
    # Execute with different operational modes, (tests happen internally).
    kwargs = dict(min_d1=0)
    f = fit(x_vals, y_vals, continuity=2, mids=0, ends=0, **kwargs)
    f = fit(x_vals, y_vals, continuity=2, mids=1, ends=1, **kwargs)
    f = fit(x_vals, y_vals, continuity=2, mids=2, ends=2, **kwargs)
    if plot:
        from plot import Plot
        plot_range = [min(x_vals)-.1, max(x_vals)+.1]
        p = Plot()
        p.add("Points", list(map(float,x_vals)), list(map(float,y_vals)))
        f = fit(x_vals, y_vals, continuity=2, mids=0, ends=0, **kwargs)
        p.add_func("f (mids=0)", f, plot_range)
        p.add_func("f deriv (m0)", f.derivative(1), plot_range, dash="dash")
        p.add_func("f dd (m0)", f.derivative(2), plot_range, dash="dot")
        f = fit(x_vals, y_vals, continuity=2, mids=1, ends=1, **kwargs)
        p.add_func("f (mids=1)", f, plot_range)
        p.add_func("f deriv (m1)", f.derivative(1), plot_range, dash="dash")
        p.add_func("f dd (m1)", f.derivative(2), plot_range, dash="dot")
        f = fit(x_vals, y_vals, continuity=2, mids=2, ends=2, **kwargs)
        p.add_func("f (mids=2)", f, plot_range)
        p.add_func("f deriv (m2)", f.derivative(1), plot_range, dash="dash")
        p.add_func("f dd (m2)", f.derivative(2), plot_range, dash="dot")
        p.show()

# Test "fill_derivative" function.
def _test_fill_derivative():
    x = list(map(Fraction, [0,1,2,4,5,7]))
    y = list(map(Fraction, [0,1,2,3,4,5]))
    # Test "0" values.
    d00 = [0, 0, 0, 0, 0, 0]
    assert( d00 == fill_derivative(x, y, ends=0, mids=0) )
    # Test "1" values (linear interpolation).
    d11 = [1, 1, Fraction(2, 3), Fraction(2, 3), Fraction(2, 3), Fraction(1, 2)]
    assert( d11 == fill_derivative(x, y, ends=1, mids=1) )
    # Test "2" values (quadratic interpolation).
    d22 = [1, 1, Fraction(5, 6), Fraction(5, 6), Fraction(5, 6), Fraction(1, 6)]
    assert( d22 == fill_derivative(x, y, ends=2, mids=2) )

# Test "solve_quadratic" function.
def _test_solve_quadratic():
    # Case 1
    x = [-1, 0, 1]
    y = [1, 0 , 1]
    a,b,c = solve_quadratic(x,y)
    assert(a == 1)
    assert(b == 0)
    assert(c == 0)
    # Case 2
    x = [-1, 0, 1]
    y = [-1, 0 , -1]
    a,b,c = solve_quadratic(x,y)
    assert(a == -1)
    assert(b == 0)
    assert(c == 0)
    # Case 3
    x = [-1, 0, 1]
    y = [0, 0 , 2]
    a,b,c = solve_quadratic(x,y)
    assert(a == 1)
    assert(b == 1)
    assert(c == 0)
    # Case 4
    x = [-1, 0, 1]
    y = [1, 1 , 3]
    a,b,c = solve_quadratic(x,y)
    assert(a == 1)
    assert(b == 1)
    assert(c == 1)

def _test_inverse(plot=False):
    # Construct a polynomial piece to test the inverse operation on.
    f = polynomial_piece([0,1,10], [0,-1], (1,3))
    i0 = inverse(f, 0)
    assert(abs(f(i0)) < 2**(-26))
    i12 = inverse(f, 1/2)
    assert(abs(f(i12) - 1/2) < 2**(-26))
    if plot:
        im1 = inverse(f, -1)
        from plot import Plot
        p = Plot()
        p.add_func("f", f, [0, 5])
        p.show()

if __name__ == "__main__":
    # Run the tests on this file.
    print()
    print("Running tests..")
    print(" Polynomial")
    _test_Polynomial()
    print(" NewtonPolynomial")
    _test_NewtonPolynomial()
    print(" polynomial")
    _test_polynomial()
    print(" polynomial_piece")
    _test_polynomial_piece(plot=False)
    print(" Spline")
    _test_Spline()
    print(" fit")
    _test_fit(plot=False)
    print(" fill_derivative")
    _test_fill_derivative()
    print(" solve_quadratic")
    _test_solve_quadratic()
    print(" inverse")
    _test_inverse(plot=False)
    print("tests complete.")


