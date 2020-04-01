# This file provides utilities for constructing polynomial interpolants.
# 
# The following objects are provided:
# 
#   Spline           -- A piecewise polynomial interpolant with
#                       evaluation, derivative, integration, negation,
#                       multiplication, addition, and a string method.
#   Polynomial       -- A polynomial with nonrepeating coefficients,
#                       evaluation, derivative, and a string method.
#   NewtonPolynomial -- A Newton polynomial with stored coefficients,
#                       points, evaluation, derivative, and a string method.
#   BSpline          -- 
# 
# The following functions are provided:
# 
#   polynomial       -- Given x and y values, this produces a minimum
#                       degree Newton polynomial that interpolates the
#                       provided points (this is a *single* polynomial).
#   polynomial_piece -- Given a function value and derivatives,
#                       produce a polynomial that interpolates these
#                       values at the endpoints of an interval.
#   fit              -- Use a local polynomial interpolant to estimate
#                       derivatives and construct an interpolating
#                       Spline with a specified level of continuity.
#   evaluate_bspline -- 
#   linspace         -- 
#   fill_derivative  -- Compute all derivatives at points to be
#                       reasonable values using either a linear or a
#                       quadratic fit over neighboring points. 
#   solve_quadratic  -- Given three points, this will solve the equation
#                       for the quadratic function which interpolates
#                       all 3 values. Returns coefficient 3-tuple. 
#   inverse          -- Given a function, use the Newton method to
#                       find the nearest inverse to a guess point.
# 

from fraction import Fraction

# This general purpose exception will be raised during user errors.
class UsageError(Exception): pass

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
    _derivative = 0
    @property
    def knots(self): return self._knots
    @knots.setter
    def knots(self, knots): self._knots = list(knots)
    @property
    def values(self): return self._values
    @values.setter
    def values(self, values): self._values = [[v for v in vals] for vals in values]
    @property
    def functions(self): return self._functions
    @functions.setter
    def functions(self, functions): self._functions = list(functions)
    
    def __init__(self, knots, values=None, functions=None):
        assert(len(knots) >= 1)
        self.knots = knots
        # Store the 'values' at each knot if they were provided.
        if (values is not None):
            assert(len(knots) == len(values))
            self.values = values
        # Store the 'functions' over each interval if they were provided.
        if (functions is not None):
            # TODO: Verify that the provided functions match the values.
            assert(len(functions) == len(knots)-1)
            self.functions = functions
        # Use the 'values' to generate 'functions' if no functions were provided.
        elif (values is not None):
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
        # No 'values' nor 'functions' were provided, usage error.
        else: raise(UsageError("Either 'values' or 'functions' must be provided with knots to construct a spline."))

    # Evaluate this Spline at a given x coordinate.
    def function_at(self, x):
        # If "x" was given as a vector, then iterate over that vector.
        try:    return [self.function_at(v) for v in x]
        except: pass
        # Deterimine which interval this "x" values lives in.
        if (x <= self.knots[0]):    return self._functions[0]
        elif (x >= self.knots[-1]): return self._functions[-1]
        # Find the interval in which "x" exists.
        for i in range(len(self.knots)-1):
            if (self.knots[i] <= x < self.knots[i+1]): break
        # If no interval was found, then something unexpected must have happened.
        else:
            class UnexpectedError(Exception): pass
            raise(UnexpectedError("This problem exhibited unexpected behavior."))
        # Return the applicable function.
        return self._functions[i]

    # Compute the integral of this Spline.
    def integral(self, i=1): return self.derivative(-i)

    # Compute the first derivative of this Spline.
    def derivative(self, d=1):
        # For integration, adjust the additive term to reflect the
        # expected lefthand side value of each function.
        if (d < 0):
            deriv_funcs = self._functions
            for i in range(-d):
                deriv_funcs = [f.integral(1) for f in deriv_funcs]
                total = self(self.knots[0])
                for i in range(len(deriv_funcs)):
                    deriv_funcs[i].coefficients[-1] = (
                        total - deriv_funcs[i](self.knots[i]))
                    total = deriv_funcs[i](self.knots[i+1])
        else:
            # Create a spline with the same knot sequence and all the
            # derivative functions and associated values.
            deriv_funcs = self._functions
            for i in range(d):
                deriv_funcs = [f.derivative(1) for f in deriv_funcs]

        # Construct the new spline, pass the "values" even though
        # nothing will be done with them. Assign the new "functions".
        s = Spline(self.knots, self.values, functions=deriv_funcs)
        s._derivative = self._derivative + d
        # Return the new derivative Spline.
        return s

    # Evaluate this Spline at a given x coordinate.
    @float_fallback
    def __call__(self, x):
        # If "x" was given as a vector, then iterate over that vector.
        try:    return [self(v) for v in x]
        except: pass
        # Get the appropriate function and compute the output value.
        return self.function_at(x)(x)
        
    # Add this "Spline" object to another "Spline" object.
    def __add__(self, other):
        # Check for correct usage.
        if (type(other) != type(self)):
            raise(UsageError(f"Only '{type(self)} objects can be added to '{type(self)}' objects, but '{type(other)}' was given."))
        # Generate the new set of knots.
        knots = sorted(set(self._knots + other._knots))
        # Compute the functions over each interval.
        functions = []
        for i in range(len(knots)-1):
            # Get the knot, nearby knots, and order of resulting
            # polynomial at this particular knot.
            left, right = knots[i], knots[i+1]
            k = knots[i]
            my_poly = self.function_at(k)
            other_poly = other.function_at(k)
            order = max(len(my_poly.coefficients), len(other_poly.coefficients))
            # Evaluate the function at equally spaced "x" values, TODO:
            # this should be Chebyshev nodes for numerical stability.
            x = [(step / (order-1)) * (right - left) + left for step in range(order)]
            y = [self(node) + other(node) for node in x]
            # Construct the interpolating polynomial.
            functions.append( polynomial(x, y) )
            f = functions[-1]
        # Return the new added function.
        return Spline(knots, functions=functions)

    # Multiply this "Spline" obect by another "Spline".
    def __mul__(self, other):
        # Check for correct usage.
        if (type(other) != type(self)):
            raise(UsageError(f"Only '{type(self)} objects can be multiplied by '{type(self)}' objects, but '{type(other)}' was given."))
        # Generate the new set of knots.
        knots = sorted(set(self._knots + other._knots))
        # Compute the functions over each interval.
        functions = []
        for i in range(len(knots)-1):
            # Get the knot, nearby knots, and order of resulting
            # polynomial at this particular knot.
            left, right = knots[i], knots[i+1]
            k = knots[i]
            my_poly = self.function_at(k)
            other_poly = other.function_at(k)
            order = len(my_poly.coefficients) + len(other_poly.coefficients) - 1
            # Evaluate the function at equally spaced "x" values, TODO:
            # this should be Chebyshev nodes for numerical stability.
            x = [(step / (order-1)) * (right - left) + left for step in range(order)]
            y = [self(node) * other(node) for node in x]
            # Construct the interpolating polynomial.
            functions.append( polynomial(x, y) )
            f = functions[-1]
        # Return the new added function.
        return Spline(knots, functions=functions)

    # Raise an existing spline to a power.
    def __pow__(self, number):
        if (type(number) != int) or (number <= 1):
            raise(TypeError(f"Only possible to raise '{type(self)}' to integer powers greater than 1."))
        # Start with a copy of "self", multiply in until complete.
        outcome = Spline(self.knots, values=self.values,
                         functions=[Polynomial(f) for f in self.functions])
        for i in range(number-1): outcome = outcome * self
        return outcome

    # Subtract another spline from this spline (add after negation).
    def __sub__(self, other): return self + (-other)

    # Negate this spline, create a new one that is it's negative.
    def __neg__(self):
        # Create a new spline, but negate all internal function coefficients.
        return Spline(self.knots, functions=[-f for f in self.functions])

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


# A generic Polynomial class that stores coefficients in a fully
# expanded form. Provides numerically stable evaluation, derivative,
# integration, and string operations for convenience. Coefficients
# should go highest to lowest order.
# 
# Polynomial(coefficients):
#    Given coefficients (or optionally a "NewtonPolynomial")
#    initialize this Polynomial representation of a polynomial function.
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
            coefficients = to_polynomial(coefficients.coefficients,
                                         coefficients.points)
        # If the user initialized this Polynomial, do a copy.
        elif (type(coefficients) == type(self)):
            coefficients = coefficients.coefficients
        # Remove all leading 0 coefficients.
        for i in range(len(coefficients)):
            if (coefficients[i] != 0): break
        else: i = len(coefficients)-1
        # Store the coeficients.
        self.coefficients = coefficients[i:]

    # Evaluate this Polynomial at a point "x" in a numerically stable way.
    def __call__(self, x):
        if (len(self.coefficients) == 0): return 0
        total = self.coefficients[0]
        for d in range(1,len(self.coefficients)):
            total = self.coefficients[d] + x * total
        return total

    # Construct the polynomial that is the integral of this polynomial.
    def integral(self, i=1): return self.derivative(-i)

    # Construct the polynomial that is the derivative of this polynomial.
    def derivative(self, d=1):
        if (d == 0): return self
        elif (d > 1):  return self.derivative(1).derivative(d-1)
        elif (d == 1): return Polynomial([c*i for (c,i) in zip(
                self.coefficients, range(len(self.coefficients)-1,0,-1))])
        elif (d < -1):  return self.derivative(-1).derivative(d+1)
        elif (d == -1): return Polynomial([c/(i+1) for (c,i) in zip(
                self.coefficients, range(len(self.coefficients)-1,-1,-1))]+[0])

    # Determines if two polynomials are equivalent.
    def __eq__(self, other):
        if type(other) == NewtonPolynomial: other = Polynomial(other)
        return all(c1 == c2 for (c1, c2) in zip(self.coefficients, other.coefficients))

    # Negate this polynomial.
    def __neg__(self): return Polynomial([-c for c in self.coefficients])

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
# points in addition to the coefficients. This is more convenient when
# constructing interpolating polynomials from divided difference tables.
# 
# NewtonPolynomial(coefficients, points):
#    Given a set of coefficients and a set of points (offsets), of
#    the same length, construct a standard Newton
#    Polynomial. Coefficients are stored from highest order term to
#    lowest order. Earlier points are evaluated earlier.
class NewtonPolynomial(Polynomial):
    _points = None
    @property
    def points(self): return self._points
    @points.setter
    def points(self, points): self._points = list(points)

    # Store the coefficients and points for this Newton Polynomial.
    def __init__(self, coefficients, points):
        if (len(points) != len(coefficients)): raise(IndexError)
        for i in range(len(coefficients)):
            if (coefficients[i] != 0): break
        else: i = len(coefficients)-1
        self.coefficients = coefficients[i:]
        self.points = points[i:]

    # Construct the polynomial that is the derivative of this
    # polynomial by converting to polynomial form and differntiating.
    def derivative(self, d=1): return Polynomial(self).derivative(d)

    # Evaluate this Newton Polynomial (in a numerically stable way).
    def __call__(self, x):
        total = self.coefficients[0]
        for d in range(1,len(self.coefficients)):
            total = self.coefficients[d] + (x - self.points[d]) * total
        return total

    # Negate this polynomial.
    def __neg__(self): return -Polynomial(self)

    # Construct a string representation of this Newton Polynomial.
    def __str__(self):
        s = f"{self.coefficients[0]}"
        for i in range(1,len(self.coefficients)):
            sign = "-" if (self.points[i] >= 0) else "+"
            s = f"{self.coefficients[i]} + (x {sign} {abs(self.points[i])})({s})"
        return s

# Given Newton form coefficients and points, convert them to polynomial
# form (where all points are 0) having only coefficients.
def to_polynomial(coefficients, points):
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
            if (x[i+d] != x[i]): 
                dd = (dd_values[-1][i+1] - dd_values[-1][i]) / (x[i+d] - x[i])
            else:
                dd = 0
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
def polynomial_piece(left, right, interval=(0,1), stable=True):
    # Make sure the code is used correctly.
    assert( len(interval) == 2 )
    assert( interval[0] != interval[1] )
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
    # If a stability check is not necessaary, return the function immediately.
    if not stable: return f

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
            import sys
            print(file=sys.stderr)
            print("-"*70, file=sys.stderr)
            print("bad_left:  ",bad_left, "  ", abs(df(k0) - (v0[i] if i < len(v0) else v1[i])))
            print("bad_right: ",bad_right, "  ", abs(df(k1) - (v1[i] if i < len(v1) else v0[i])))
            print("error_tolerance: ",error_tolerance, file=sys.stderr)
            print(f"Interval:              [{k0: .3f}, {k1: .3f}]", file=sys.stderr)
            print("Assigned left values: ", v0, file=sys.stderr)
            print("Assigned right values:", v1, file=sys.stderr)
            print(file=sys.stderr)
            lf = f"{'d'*i}f({k0: .3f})"
            print(f"Expected {lf} == {v0[i]}", file=sys.stderr)
            print(f"     got {' '*len(lf)} == {df(k0)}", file=sys.stderr)
            print(f"     error {' '*(len(lf) - 2)} == {v0[i] - df(k0): .3e}", file=sys.stderr)
            rf = f"{'d'*i}f({k1: .3f})"
            print(f"Expected {rf} == {v1[i]}", file=sys.stderr)
            print(f"     got {' '*len(rf)} == {df(k1)}", file=sys.stderr)
            print(f"     error {' '*(len(rf) - 2)} == {v1[i] - df(k1): .3e}", file=sys.stderr)
            print(file=sys.stderr)
            print(f"{' '*i}coefs:",coefs, file=sys.stderr)
            print(f"{' '*i}f:    ",f, file=sys.stderr)
            print(f"{'d'*i}f:    ",df, file=sys.stderr)
            print("-"*70, file=sys.stderr)
            print(file=sys.stderr)
            raise(Exception("The generated polynomial piece is numerically unstable."))

    # Return the polynomial function.
    return f

# Given data points "x" and data values "y", construct an
# interpolating spline over the given points with specified level of
# continuity using a sufficiently continuous polynomial fit over
# neighboring points.
#  
# x: A strictly increasing sequences of numbers.
# y: Function values associated with each point.
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
    assert( all(x[i] < x[i+1] for i in range(len(x)-1)) )
    knots = [v for v in x]
    values = [[v] for v in y]
    # Construct further derivatives and refine the approximation
    # ensuring monotonicity in the process.
    max_on_one_side = continuity // 2 + 1
    for i in range(0,len(x)):
        # Get initial candidates from the left and right.
        left_candidates = list(range(max(0,i-max_on_one_side),i))
        right_candidates = list(range(i+1,min(len(x), i+1+max_on_one_side)))
        # How many more points are needed:
        needed = (continuity+2) - (len(left_candidates) + len(right_candidates))
        # Add more candidates from the right side.
        if (len(left_candidates) < max_on_one_side):
            # Build out the right candidates to fill in.
            right_candidates += list(range(right_candidates[-1] + 1, min(
                len(x), right_candidates[-1] + 1 + needed)))
        # Add more candidates from the left side.
        elif (len(right_candidates) < max_on_one_side):
            # Build out the left candidates to fill in.
            left_candidates = list(range(max(0, left_candidates[0]-needed), 
                                         left_candidates[0])) + left_candidates
        # If the leftmost candidate is further away, remove it.
        elif (x[i] - x[left_candidates[0]]) > (x[right_candidates[-1]] - x[i]):
            left_candidates.pop(0)
        # If the rightmost candidate is further away, remove it.
        else: right_candidates.pop(-1)
        # Construct the polynomial.
        candidates = left_candidates + [i] + right_candidates
        p = polynomial([x[idx] for idx in candidates],
                       [y[idx] for idx in candidates])
        # Evaluate the derivatives of the polynomial.
        for d in range(1, continuity+1):
            # Compute the next derivative of this polynomial and evaluate.
            p = p.derivative()
            deriv = p(x[i])
            # Bound the derivatives if that was requested.
            max_name = f"max_d{d}"
            if (max_name in kwargs): deriv = min(kwargs[max_name], deriv)
            min_name = f"min_d{d}"
            if (min_name in kwargs): deriv = max(kwargs[min_name], deriv)
            # Store the derivative.
            values[i].append(deriv)
    # Return the interpolating spline.
    return Spline(knots, values)

# Class for constructing an exact equivalent to a B-spline out of
# exact Polynomial pieces.
class BSpline:
    def __init__(self, knots):
        # Generate "x" and "y" of sufficient size to match the
        # piecewise order of the resulting B-spline.
        x = []
        for i in range(len(knots)-1):
            if (knots[i] == knots[i+1]): continue
            x += list(linspace(knots[i], knots[i+1], len(knots)+1))[:-1]
        x += [knots[-1]]
        y = evaluate_bspline(knots, x)
        # Store the breakpoints as the unique knots.
        self.breakpoints = sorted(set(knots))
        self.funcs = {0:[]}
        for i in range(len(self.breakpoints)-1):
            local_x = [v for v in x if self.breakpoints[i] <= v < self.breakpoints[i+1]]
            local_y = [f for v,f in zip(x,y) if self.breakpoints[i] <= v < self.breakpoints[i+1]]
            self.funcs[0].append( polynomial(local_x, local_y) )
        # Special case for 1-breakpoint.
        if (len(self.breakpoints) == 1):
            self.funcs[0] = [polynomial([self.breakpoints[0]],[0])]

    # Retrieve a specific derivative (or integral) of this function.
    def __getitem__(self, d): 
        # Create the derivatives if they have not already been created.
        if (d not in self.funcs):
            # Derivatives are easy to compute.
            if   (d > 0): self.funcs[d] = [f.derivative(d) for f in self.funcs[0]]
            # Integrals need to match the values of their left neighbors.
            elif (d < 0):
                funcs = self[d+1]
                new_funcs = []
                start = 0
                for i in range(len(self.breakpoints)-1):
                    # Get the existing function, integrate it and evaluate
                    # the value of the function across the interval.
                    f = funcs[i].integral(1)
                    x = linspace(self.breakpoints[i],
                                 self.breakpoints[i+1],
                                 len(f.coefficients))
                    y = [f(v) for v in x]
                    # Shift the values of the function to match on the left.
                    change = start - y[0]
                    y = [v+change for v in y]
                    # Store the new integral piece.
                    new_funcs.append( polynomial(x, y) )
                    # Retreive the rightmost value to be used by next function.
                    start = y[-1]
                # Special case for 1-breakpoint.
                if (len(self.breakpoints) == 1):
                    new_funcs = funcs
                # Store the integral functions.
                self.funcs[d] = new_funcs
                
        # Return the set of functions.
        return self.funcs[d]

    # Evaluate this B-spline.
    def __call__(self, x, d=0):
        # Check to see if an iterable was given, unwrap it if so.
        try: return [self(v) for v in x]
        except: pass
        # Retrieve the appropriate functions.
        funcs = self[d]
        # Handle extrapolation the way the B-spline extrapolation will
        # be handled. 0 outside interval, except integrals on right.
        if   (x < self.breakpoints[0]): return 0
        elif (x >= self.breakpoints[-1]):
            if (d >= 0): return 0
            else:        return funcs[-1](self.breakpoints[-1])
        else:
            # Identify which interval this "x" belongs to.
            for i in range(len(self.breakpoints)):
                if (self.breakpoints[i] <= x < self.breakpoints[i+1]): break
            else: raise(Exception("ERROR: Logic error in code, this should not happen."))
            # Compute the value and return it.
            return funcs[i](x)

# Given x locations, evaluate a B-spline and return the value.
def evaluate_bspline(knots, x):
    # Try unpacking iterables if provided.
    try: return [evaluate_bspline(knots, v) for v in x]
    except: pass
    # Initialize by creating the constant valued basis functions.
    values = [0] * max(1,len(knots))
    for k in range(len(knots)-1):
        if   (knots[k] == knots[k+1]):     continue
        elif (knots[k] <= x < knots[k+1]): values[k] = 1
    # Compute the remainder of the B-spline by building up from the step function.
    for step in range(1, len(knots)-1):
        for k in range(0, len(knots) - step - 1):
            left_divisor  = (knots[k+step]   - knots[k])
            right_divisor = (knots[k+step+1] - knots[k+1])
            if   (left_divisor != 0 != right_divisor):
                values[k] = (
                    (x - knots[k])        / left_divisor  * values[k] +
                    (knots[k+step+1] - x) / right_divisor * values[k+1])
            elif (left_divisor != 0):
                values[k] = (
                    (x - knots[k])        / left_divisor  * values[k])
            elif (right_divisor != 0):
                values[k] = (
                    (knots[k+step+1] - x) / right_divisor * values[k+1])
    # Return the computed B-spline value.
    return values[0]

# Function for constructing a linearly spaced set of points over an interval.
def linspace(start, end, count):
    width = end - start
    return [start + width * i / (count-1) for i in range(count)]

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
        raise(UsageError("Manually defined endpoints must provide exactly two numbers."))
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
        raise(UsageError("Manually defined endpoints must provide exactly two numbers."))
    # Return the computed derivatives.
    return deriv

# Given three points, this will solve the equation for the quadratic
# function which interpolates all 3 values. Returns coefficient 3-tuple.
# 
# This could be done by constructing a Polynomial interpolant, however
# that is slightly less computationally efficient and less stable.
def solve_quadratic(x, y):
    if len(x) != len(y): raise(UsageError("X and Y must be the same length."))
    if len(x) != 3:      raise(UsageError(f"Exactly 3 (x,y) coordinates must be given, received '{x}'."))
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
    x, y = Fraction(x), Fraction(y)
    # Search for a solution iteratively.
    for i in range(max_steps):
        diff = f(x) - y
        x -= diff / df(x)
        x = Fraction(float(x)) # <- cap the accuracy to that of floats
        # Stop the loop if we have gotten close to the correct answer.
        if (abs(diff) <= accuracy): break
    # Check for correctness, warn the user if result is bad.
    if (abs(f(x) - y) > accuracy):
        import warnings
        warnings.warn(f"\n\n  The calculated inverse has high error ({float(abs(f(x)-y)):.2e}).\n"+
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
    f = Polynomial(to_polynomial([-1,10,-16,24,32,-32], [1,1,1,-1,-1,-1]))
    assert(str(f) == "-1 x^5  +  9 x^4  +  6 x^3  +  -22 x^2  +  11 x  +  -3")
    assert(str(f.derivative()) == "-5 x^4  +  36 x^3  +  18 x^2  +  -44 x  +  11")
    # Check that integrals work too.
    assert(str(f.derivative().derivative(-1)) == "-1.0 x^5  +  9.0 x^4  +  6.0 x^3  +  -22.0 x^2  +  11.0 x")
    assert(str(f.derivative().derivative(-1).derivative()) == "-5.0 x^4  +  36.0 x^3  +  18.0 x^2  +  -44.0 x  +  11.0")


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
    # Create the knots and values.
    knots = [Fraction(k) for k in knots]
    values = [[Fraction(v) for v in vals] for vals in values]
    # Create the spline.
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
    # TODO: Test cases for the addition, multiplication, negation
    #       and integration of a Spline object.
    # 
    # from util.plot import Plot
    # g = Spline(knots, values)
    # s_add = f + g
    # add = lambda x: f(x) + g(x)
    # s_mult = f * g
    # mult = lambda x: f(x) * g(x)
    # p = Plot()
    # p.add_func("f", f, [min(knots), max(knots)])
    # p.add_func("f'", f.derivative(1), [min(knots), max(knots)])
    # p.add_func("f''", f.derivative(2), [min(knots), max(knots)])
    # p.add_func("g", g, [min(knots), max(knots)])
    # p.add_func("s add", s_add, [min(knots), max(knots)])
    # p.add_func("add", add, [min(knots), max(knots)])
    # p.add_func("s mult", s_mult, [min(knots), max(knots)])
    # p.add_func("mult", mult, [min(knots), max(knots)])
    # # Negated
    # p.add_func("-f", -f, [min(knots), max(knots)])
    # p.add_func("-f true", lambda x: -(f(x)), [min(knots), max(knots)])    
    # p.add_func("-mult", -s_mult, [min(knots), max(knots)])
    # p.add_func("-mult true", lambda x: -s_mult(x), [min(knots), max(knots)])
    # p.add_func("int(f)", f.derivative(-1), [min(knots), max(knots)])
    # p.add_func("int(-mult)", (-s_mult).derivative(-1), [min(knots), max(knots)])
    # p.show()
    # exit()

# Test the "fit" function. (there is testing code built in, so this
# test is strictly for generating a visual to verify).
def _test_fit(plot=False):
    x_vals = list(map(Fraction, [0,.5,2,3.5,4,5.3,6]))
    y_vals = list(map(Fraction, [1,2,2.2,3,3.5,4,4]))
    # Execute with different operational modes, (tests happen internally).
    kwargs = dict(min_d1=0)
    f = fit(x_vals, y_vals, continuity=2, **kwargs)
    for i in range(len(f._functions)):
        f._functions[i] = Polynomial(f._functions[i])
    if plot:
        from util.plot import Plot
        plot_range = [min(x_vals)-.1, max(x_vals)+.1]
        p = Plot()
        p.add("Points", list(map(float,x_vals)), list(map(float,y_vals)))
        p.add_func("f (mids=0)", f, plot_range)
        p.add_func("f deriv (m0)", f.derivative(1), plot_range, dash="dash")
        p.add_func("f dd (m0)", f.derivative(2), plot_range, dash="dot")
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
    i0 = inverse(f, 0, accuracy=2**(-26))
    assert(abs(f(i0)) < 2**(-2))
    i12 = inverse(f, 1/2, accuracy=2**(-2))
    # print(float(abs(f(i12) - 1/2)), abs(f(i12) - 1/2))
    assert(abs(f(i12) - 1/2) < 2**(-2))
    if plot:
        print(f"inverse(f, 20): {float(inverse(f, 20, 4.2)):.4f}")
        im1 = inverse(f, -1)
        from util.plot import Plot
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


