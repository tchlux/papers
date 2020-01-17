# Get numpy data structures (in differentiable form)
from autograd.numpy import zeros, ones, arange, linspace, \
    identity, roll, radians, random, linalg
# Get numpy mathematical expressions (in differentiable form)
from autograd.numpy import pi, sum, exp, sin, cos, log, outer, dot, matmul
# Use an automatic elementwise-gradient function
from autograd import elementwise_grad as grad

# Set up a repeatable random number generator
SEED = 0
RANDOM = random.RandomState(SEED)
# Store all the objective functions in a global variable
FUNCTIONS = []


# Decorator for converting a function into an objective function
def objectify(lower=None, upper=None, max_val=None, solution=None, gradient=None):
    # Define the "normalize" decorator that handles given arguments
    def decorator(func):
        def obj_func(x, *args, **kwargs):
            # Rescale the x out of [-1,1] and into the domain of the
            # function if a normalization was necessary
            if (lower is not None) and (upper is not None):
                x = ((x + 1) * (upper - lower) / 2) + lower
            # Return the objective value
            value = func(x, *args, **kwargs)
            # Rescale the objective value if necessary
            if hasattr(x, "__len__"):
                if (max_val is not None):
                    value /= max_val*len(x)
            else: value /= max_val
            # Return the value.
            return value
        # Set a random point generator (in the -1,1 range)
        obj_func.rand = lambda d, gen=RANDOM: gen.rand(d) * 2 - 1
        # Set a solution
        if solution: obj_func.sol  = solution
        else:        obj_func.sol  = lambda d: zeros(d)
        # Set a gradient
        if gradient: obj_func.grad = gradient
        else:        obj_func.grad = grad(obj_func)
        obj_func.lower, obj_func.upper = -1., 1.
        # Store the maximum gradient magnitude
        obj_func.max_grad = max(abs(obj_func.grad(ones(1)*obj_func.lower)),
                                abs(obj_func.grad(ones(1)*obj_func.upper)))
        # Copy the name over to the new function
        obj_func.__name__ = func.__name__ + "_(objectified)"
        obj_func.func = func
        # Add the objective function to the list of functions
        FUNCTIONS.append(obj_func)
        # Return the new objective function
        return obj_func
    # Return the decoroator, which will be passed a single function as
    # an argument by python implicitly.
    return decorator

# ============================================
#      Objective Function Transformations     
# ============================================

# Errors that may be rasied
class IllegalNoise(Exception): pass
class IllegalSkew(Exception): pass
class IllegalRotation(Exception): pass
class IllegalRecenter(Exception): pass

# Given a noise ratio in [0,1], add noise to the gradient that is
# proportional to "ratio" times the max magnitude of the gradient.
def noise(ratio, dimension):
    # Check for proper usage
    if not (0 <= ratio <= 1): 
        raise(IllegalNoise("Only noise ratios in the range [0,1] are accepted."))
    # Create the decorator for an objective function
    def decorator(func):
        # Calculate the magnitude of the noise that will be added
        noise_magnitude = ratio * func.max_grad
        # Generate a new copy of the objective function (different gradient)
        def obj_func(x, *args, **kwargs):
            return func(x , *args, **kwargs)
        # Store the solution, gradient, lower, and upper bounds for this new function
        obj_func.rand = func.rand
        obj_func.sol = func.sol
        obj_func.grad = lambda x: func.grad(x) + (RANDOM.rand(dimension)-.5)*noise_magnitude/2
        obj_func.max_grad = func.max_grad
        obj_func.lower = ones(dimension) * func.lower
        obj_func.upper = ones(dimension) * func.upper
        obj_func.__name__ = func.__name__ + f"_(noised-{ratio:.2f})"
        obj_func.func = func
        # Return new function
        return obj_func
    # Return decorator function
    return decorator    

# Given a skew in [0,1) go from the original space at 0 to 1
# where the difference in stretched-ness between the least and most
# stretched dimension approaches infinity.
def skew(skew, dimension):
    # Check for proper usage
    if not (0 <= skew < 1): 
        raise(IllegalSkew("Only skews in the range [0,1) are accepted."))
    # Generate a skew vector to apply 
    skew_vec = linspace(1-skew, 1, dimension)
    # Create the decorator for an objective function
    def decorator(func):
        def obj_func(x, *args, **kwargs):
            return func(x * skew_vec, *args, **kwargs)
        # Store the new random point generation function
        obj_func.rand = lambda d, gen=RANDOM: func.rand(d,gen) / skew_vec
        # Store the new solution
        obj_func.sol = lambda d: func.sol(d) / skew_vec
        # Store the new gradient
        obj_func.grad = lambda x: func.grad(x*skew_vec) / skew_vec
        obj_func.max_grad = func.max_grad
        # Store the new bounds
        obj_func.lower = (ones(dimension)*func.lower) / skew_vec
        obj_func.upper = (ones(dimension)*func.upper) / skew_vec
        # Copy over the name
        obj_func.__name__ = func.__name__ + f"_(skewed-{skew:.2f})"
        obj_func.func = func
        # Return new function
        return obj_func
    # Return decorator function
    return decorator

# Given a ratio and a dimension, create a pair of matrices:
#    (rotation matrix, inverse rotation matrix)
def rotation_matrices(rotation_ratio, dimension):
    # Convert the rotation from degrees to radians
    rotation = radians(rotation_ratio * 45)
    # Determine the order of the dimension rotations (done in pairs)
    rotation_order = arange(dimension)
    # Generate the rotation matrix by rotating two dimensions at a time.
    rotation_matrix = identity(dimension)
    # Make sure "i" is a c integer (fast loops).
    # cdef int i
    for i in range(len(rotation_order)-1):
        d1, d2 = rotation_order[i], rotation_order[i+1]
        next_rotation = identity(dimension)
        # Establish the rotation
        next_rotation[d1,d1] =  cos(rotation)
        next_rotation[d2,d2] =  cos(rotation)
        next_rotation[d1,d2] =  sin(rotation)
        next_rotation[d2,d1] = -sin(rotation)
        # Compound the paired rotations
        rotation_matrix = matmul(next_rotation, rotation_matrix)
        # When there are two dimenions or fewer, do not keep iterating.
        if (dimension <= 2): break
    return rotation_matrix, rotation_matrix.T

# Given a rotation in [0,1] go from the original space as 0 to 1 when
# the rotation corresponds to all axes being rotated to be exactly
# between the original dimensions.
def rotate(rotation_ratio, dimension, rotation_matrix=None):
    # Check for proper usage
    if not (0 <= rotation_ratio <= 1): 
        raise(IllegalRotation("Only rotations in the range [0,1] are accepted."))
    # If not given a rotation matrix, make one.
    if (rotation_matrix is None):
        print("creating rotation..", end="\r")
        rotation_matrix, inverse_rotation = rotation_matrices(
            rotation_ratio, dimension)
    # Otherwise, compute the inverse (for orthonormal matrices, the transpose).
    else: inverse_rotation = rotation_matrix.T
    # Create the decorator for an objective function
    def decorator(func):
        def obj_func(x, *args, **kwargs):
            return func(matmul(rotation_matrix, x), *args, **kwargs)
        # Store the rotation matrix.
        obj_func.rotation_matrix = rotation_matrix
        # Store the new random point generation function
        obj_func.rand = lambda d, gen=RANDOM: matmul(inverse_rotation,func.rand(d,gen))
        # Store the new solution
        obj_func.sol = lambda d: matmul(inverse_rotation,func.sol(d))
        # Store the new gradient function
        obj_func.grad = lambda x: matmul(inverse_rotation, func.grad(matmul(rotation_matrix,x)))
        obj_func.max_grad = func.max_grad
        # Store the new bounds (must be vector values now).
        obj_func.lower = matmul(rotation_matrix, ones(dimension) * func.lower)
        obj_func.upper = matmul(rotation_matrix, ones(dimension) * func.upper)
        obj_func.lower[:] = min(obj_func.lower)
        obj_func.upper[:] = max(obj_func.upper)
        # Copy over the name
        obj_func.__name__ = func.__name__ + f"_(rotated-{rotation_ratio:.2f})"
        obj_func.func = func
        # Return new function
        return obj_func
    # Return decorator function
    return decorator

# Given a new center, shift all inputs to be centered about that point.
def recenter(new_center, dimension):
    # Create the decorator for an objective function
    def decorator(func):
        # Calculate the shift based on the new center
        shift = func.sol(dimension) - new_center
        # Check for proper usage
        if not (all(func.lower <= new_center) and 
                all(new_center <= func.upper)):
            raise(IllegalRecenter("New center must be within existing bounds."))
        def obj_func(x, *args, **kwargs):
            return func(x + shift, *args, **kwargs)
        # Store the solution, gradient, lower, and upper bounds for this new function
        obj_func.rand = lambda d, gen=RANDOM: func.rand(d,gen) - shift
        obj_func.sol = lambda d: func.sol(d) - shift
        obj_func.grad = lambda x: func.grad(x + shift)
        obj_func.max_grad = func.max_grad
        obj_func.lower = ones(dimension) * func.lower - shift
        obj_func.upper = ones(dimension) * func.upper - shift
        obj_func.__name__ = func.__name__ + f"_(recentered-{new_center})"
        obj_func.func = func
        # Return new function
        return obj_func
    # Return decorator function
    return decorator



# A standard quadratic function.
@objectify(max_val=1)
def quadratic(x):
    return sum(x**2)

# A sub-quadratic with sudden increase in steepness near solution at 0.
@objectify(max_val=1)
def subquadratic(x, k=2):
    return sum(abs(x)**((2*k)/(2*k-1)))

# A super-quadratic with apparent 'flatness' around solution at 0.
@objectify(max_val=1)
def superquadratic(x, k=2):
    return sum((x)**(2*k))

# A polynomial with saddle points at +/- "s", solution at 0.
@objectify(max_val=(1/6 - .75**2 / 2 + .75**4 / 2))
def saddle(x, s=.75):
    return sum((s**4)*(x**2)/2 - (s**2)*(x**4)/2 + (x**6)/6)

# A chebyshev polynomial with "m" minima, solution at 0.
@objectify(max_val=4)
def multimin(x, m=3, a=2):
    # Compute the chebyshev polynomial with "m" minima
    previous = 1
    current = x
    for n in range(2, m*2 + 1):
        previous, current = (current, 2*x*current - previous)
    # Return the chebyshev polynomial with a quadratic overture
    return len(x) + a*sum(x**2) + sum(current)

# An approximation of a nowhere differentiable function.
@objectify(max_val=3.998046875)
def weirerstrass(x, k=10, a=.5, b=3):
    # Calculate some constants
    ak = a**(arange(k+1))
    bk = b**(arange(k+1))
    # Calculate the function value
    ak_xi = cos( pi * outer(bk,(x+1)) )
    return sum(dot(ak, ak_xi)) - len(x) * sum(ak * cos(pi*bk))


functions = FUNCTIONS
