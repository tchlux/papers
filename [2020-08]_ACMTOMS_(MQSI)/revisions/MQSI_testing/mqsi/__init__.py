'''This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code is attached to each wrapped function.
'''

import os
import ctypes
import numpy

# --------------------------------------------------------------------
#               CONFIGURATION
# 
_verbose = True
_link_lapack = False
_fort_compiler = "gfortran"
_shared_object_name = "MQSI.so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
if _link_lapack:
    _compile_options = ['-fPIC', '-shared', '-O3', '-lblas', '-llapack']
    _ordered_dependencies = ['REAL_PRECISION.f90', 'EVAL_BSPLINE.f90', 'SPLINE.f90', 'MQSI.f90', 'MQSI_c_wrapper.f90']
else:
    _compile_options = ['-fPIC', '-shared', '-O3']
    _ordered_dependencies = ['blas.f', 'lapack.f', 'REAL_PRECISION.f90', 'EVAL_BSPLINE.f90', 'SPLINE.f90', 'MQSI.f90', 'MQSI_c_wrapper.f90']
# 
# --------------------------------------------------------------------
#               AUTO-COMPILING
#
# Try to import the existing object. If that fails, recompile and then try.
try:
    clib = ctypes.CDLL(_path_to_lib)
except:
    # Remove the shared object if it exists, because it is faulty.
    if os.path.exists(_shared_object_name):
        os.remove(_shared_object_name)
    # Compile a new shared object.
    _command = " ".join([_fort_compiler] + _compile_options + ["-o", _shared_object_name] + _ordered_dependencies)
    if _verbose:
        print("Running system command with arguments")
        print("  ", _command)
    # Run the compilation command.
    import subprocess
    subprocess.run(_command, shell=True, cwd=_this_directory)
    # Import the shared object file as a C library with ctypes.
    clib = ctypes.CDLL(_path_to_lib)
# --------------------------------------------------------------------


# ----------------------------------------------
# Wrapper for the Fortran subroutine MQSI

def mqsi(x, y, t, bcoef, uv=None):
    '''! Computes a monotone quintic spline interpolant (MQSI), Q(X), to data
! in terms of spline coefficients BCOEF for a B-spline basis defined by
! knots T. Q(X) is theoretically guaranteed to be monotone increasing
! (decreasing) over exactly the same intervals [X(I), X(J)] that the
! data Y(.) is monotone increasing (decreasing).
!
! INPUT:
!   X(1:ND) -- A real array of ND increasing values.
!   Y(1:ND) -- A real array of ND function values Y(I) = F(X(I)).
!
! OUTPUT:
!   T(1:3*ND+6) -- The knots for the MQSI B-spline basis.
!   BCOEF(1:3*ND) -- The coefficients for the MQSI B-spline basis.
!   INFO -- Integer representing the subroutine return status.
!     0  Normal return.
!     1  There are fewer than three data points, so there is nothing to do.
!     2  X(:) and Y(:) have different sizes.
!     3  The size of T(:) must be at least the number of knots 3*ND+6.
!     4  The size of BCOEF(:) must be at least the spline space dimension 3*ND.
!     5  The values in X(:) are not increasing or not separated by at
!        least the square root of machine precision.
!     6  The magnitude or spacing of the data (X(:), Y(:)) could lead to
!        overflow. Some differences |Y(I+1) - Y(I)| or |X(I+1) - X(I)| exceed
!        10**38, which could lead to local Lipschitz constants exceeding 10**54.
!     7  The computed spline does not match the provided data in Y(:) and
!        this result should not be used. This arises when the scaling of
!        function values and derivative values causes the linear system used
!        to compute the final spline interpolant to have a prohibitively
!        large condition number.
!     8  The optional array UV must have size at least ND x 2.
!   >10  20 plus the info flag as returned by DGBSV from LAPACK when
!        computing the final spline interpolant.
!   <-10 (negated) 20 plus the info flag as returned by DGESV from
!        LAPACK when computing the quadratic interpolants.
!   UV(1:ND,1:2) -- First and second derivatives of Q(X) at the breakpoints
!     (optional argument).
!'''
    
    # Setting up "x"
    if ((not issubclass(type(x), numpy.ndarray)) or
        (not numpy.asarray(x).flags.f_contiguous) or
        (not (x.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'x' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        x = numpy.asarray(x, dtype=ctypes.c_double, order='F')
    x_dim_1 = ctypes.c_int(x.shape[0])
    
    # Setting up "y"
    if ((not issubclass(type(y), numpy.ndarray)) or
        (not numpy.asarray(y).flags.f_contiguous) or
        (not (y.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'y' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        y = numpy.asarray(y, dtype=ctypes.c_double, order='F')
    y_dim_1 = ctypes.c_int(y.shape[0])
    
    # Setting up "t"
    if ((not issubclass(type(t), numpy.ndarray)) or
        (not numpy.asarray(t).flags.f_contiguous) or
        (not (t.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 't' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        t = numpy.asarray(t, dtype=ctypes.c_double, order='F')
    t_dim_1 = ctypes.c_int(t.shape[0])
    
    # Setting up "bcoef"
    if ((not issubclass(type(bcoef), numpy.ndarray)) or
        (not numpy.asarray(bcoef).flags.f_contiguous) or
        (not (bcoef.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'bcoef' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        bcoef = numpy.asarray(bcoef, dtype=ctypes.c_double, order='F')
    bcoef_dim_1 = ctypes.c_int(bcoef.shape[0])
    
    # Setting up "info"
    info = ctypes.c_int()
    
    # Setting up "uv"
    uv_present = ctypes.c_bool(True)
    if (uv is None):
        uv_present = ctypes.c_bool(False)
        uv = numpy.zeros(shape=(1,1), dtype=ctypes.c_double, order='F')
    elif (type(uv) == bool) and (uv):
        uv = numpy.zeros(shape=(1,1), dtype=ctypes.c_double, order='F')
    elif ((not issubclass(type(uv), numpy.ndarray)) or
          (not numpy.asarray(uv).flags.f_contiguous) or
          (not (uv.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'uv' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        uv = numpy.asarray(uv, dtype=ctypes.c_double, order='F')
    if (uv_present):
        uv_dim_1 = ctypes.c_int(uv.shape[0])
        uv_dim_2 = ctypes.c_int(uv.shape[1])
    else:
        uv_dim_1 = ctypes.c_int()
        uv_dim_2 = ctypes.c_int()

    # Call C-accessible Fortran wrapper.
    clib.c_mqsi(ctypes.byref(x_dim_1), ctypes.c_void_p(x.ctypes.data), ctypes.byref(y_dim_1), ctypes.c_void_p(y.ctypes.data), ctypes.byref(t_dim_1), ctypes.c_void_p(t.ctypes.data), ctypes.byref(bcoef_dim_1), ctypes.c_void_p(bcoef.ctypes.data), ctypes.byref(info), ctypes.byref(uv_present), ctypes.byref(uv_dim_1), ctypes.byref(uv_dim_2), ctypes.c_void_p(uv.ctypes.data))

    # Return final results, 'INTENT(OUT)' arguments only.
    return t, bcoef, info.value, (uv if uv_present else None)


# ----------------------------------------------
# Wrapper for the Fortran subroutine FIT_SPLINE

def fit_spline(xi, fx, t, bcoef):
    '''! Subroutine for computing a linear combination of B-splines that
! interpolates the given function value (and function derivatives)
! at the given breakpoints.
!
! INPUT:
!   XI(1:NB) -- The increasing real-valued locations of the NB
!               breakpoints for the interpolating spline.
!   FX(1:NB,1:NCC) -- FX(I,J) contains the (J-1)st derivative at
!                     XI(I) to be interpolated, providing NCC
!                     continuity conditions at all NB breakpoints.
!
! OUTPUT:
!   T(1:NB*NCC+2*NCC) -- The nondecreasing real-valued locations
!                        of the knots for the B-spline basis.
!   BCOEF(1:NB*NCC) -- The coefficients for the B-splines that define
!                      the interpolating spline.
!   INFO -- Integer representing the subroutine execution status:
!     0    Successful execution.
!     1    SIZE(XI) is less than 3.
!     3    SIZE(FX,1) does not equal SIZE(XI).
!     4    SIZE(T) too small, should be at least NB*NCC + 2*NCC.
!     5    SIZE(BCOEF) too small, should be at least NB*NCC.
!     6    Elements of XI are not strictly increasing.
!     7    The computed spline does not match the provided FX
!          and this fit should be disregarded. This arises when
!          the scaling of function values and derivative values
!          causes the resulting linear system to have a
!          prohibitively large condition number.
!   >10    20 plus the info flag as returned by DGBSV from LAPACK.
!
!
! DESCRIPTION:
!   This subroutine computes the B-spline basis representation of the
!   spline interpolant to given function values (and derivative values)
!   at unique breakpoints. The osculatory interpolating spline of order
!   2*NCC is returned in terms of knots T and coefficients BCOEF, defining
!   the underlying B-splines and their linear combination that interpolates
!   the given data. This function uses the subroutine EVAL_BSPLINE to
!   evaluate the B-splines at all knots and the LAPACK routine DGBSV to
!   compute the B-spline coefficients. The difference between the provided
!   function (and derivative) values and the actual values produced by the
!   computed interpolant can vary depending on the spacing of the knots
!   and the magnitudes of the values provided. When the condition number
!   of the linear system defining BCOEF is large, the computed interpolant
!   may fail to accurately reproduce the data, indicated by INFO = 7.
!'''
    
    # Setting up "xi"
    if ((not issubclass(type(xi), numpy.ndarray)) or
        (not numpy.asarray(xi).flags.f_contiguous) or
        (not (xi.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'xi' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        xi = numpy.asarray(xi, dtype=ctypes.c_double, order='F')
    xi_dim_1 = ctypes.c_int(xi.shape[0])
    
    # Setting up "fx"
    if ((not issubclass(type(fx), numpy.ndarray)) or
        (not numpy.asarray(fx).flags.f_contiguous) or
        (not (fx.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'fx' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        fx = numpy.asarray(fx, dtype=ctypes.c_double, order='F')
    fx_dim_1 = ctypes.c_int(fx.shape[0])
    fx_dim_2 = ctypes.c_int(fx.shape[1])
    
    # Setting up "t"
    if ((not issubclass(type(t), numpy.ndarray)) or
        (not numpy.asarray(t).flags.f_contiguous) or
        (not (t.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 't' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        t = numpy.asarray(t, dtype=ctypes.c_double, order='F')
    t_dim_1 = ctypes.c_int(t.shape[0])
    
    # Setting up "bcoef"
    if ((not issubclass(type(bcoef), numpy.ndarray)) or
        (not numpy.asarray(bcoef).flags.f_contiguous) or
        (not (bcoef.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'bcoef' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        bcoef = numpy.asarray(bcoef, dtype=ctypes.c_double, order='F')
    bcoef_dim_1 = ctypes.c_int(bcoef.shape[0])
    
    # Setting up "info"
    info = ctypes.c_int()

    # Call C-accessible Fortran wrapper.
    clib.c_fit_spline(ctypes.byref(xi_dim_1), ctypes.c_void_p(xi.ctypes.data), ctypes.byref(fx_dim_1), ctypes.byref(fx_dim_2), ctypes.c_void_p(fx.ctypes.data), ctypes.byref(t_dim_1), ctypes.c_void_p(t.ctypes.data), ctypes.byref(bcoef_dim_1), ctypes.c_void_p(bcoef.ctypes.data), ctypes.byref(info))

    # Return final results, 'INTENT(OUT)' arguments only.
    return t, bcoef, info.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine EVAL_SPLINE

def eval_spline(t, bcoef, xy, d=None):
    '''! Evaluate a spline constructed with FIT_SPLINE.
!
! INPUT:
!   T(1:NSPL+2*NCC) -- The nondecreasing real-valued locations of the
!      knots for the underlying B-splines, where the spline has NCC-1
!      continuous derivatives and is composed of NSPL B-splines.
!   BCOEF(1:NSPL) -- The coefficients of the B-spline basis functions
!      defining this interpolating spline.
!
! INPUT/OUTPUT:
!   XY(1:M) -- On input, the locations at which the spline is to be
!      evaluated; on output, holds the value (or Dth derivative) of
!      the spline with knots T and coefficients BCOEF evaluated at the
!      given locations.
!
! OUTPUT:
!   INFO -- Integer representing subroutine execution status.
!     0  Successful execution.
!     1  The sizes of T and BCOEF are incompatible.
!     2  Given the sizes of T and BCOEF, T is an invalid knot sequence.
!
! OPTIONAL INPUT:
!   D --  The order of the derivative of the spline to take at the
!     points in XY. If omitted, D = 0. When D < 0, the spline integral
!     over [T(1), XY(.)] is returned in XY(.).
!
! DESCRIPTION:
!
!   This subroutine serves as a convenient wrapper to the underlying
!   calls to EVAL_BSPLINE to evaluate the full spline. Internally this
!   evaluates the spline at each provided point XY(.) by first using a
!   bisection search to identify which B-splines could be nonzero at
!   that point, then computing the B-splines and taking a linear
!   combination of them to produce the spline value. Optimizations are
!   incorporated that make the evaluation of successive points in increasing
!   order most efficient, hence it is recommended that the provided points
!   XY(:) be increasing.
!'''
    
    # Setting up "t"
    if ((not issubclass(type(t), numpy.ndarray)) or
        (not numpy.asarray(t).flags.f_contiguous) or
        (not (t.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 't' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        t = numpy.asarray(t, dtype=ctypes.c_double, order='F')
    t_dim_1 = ctypes.c_int(t.shape[0])
    
    # Setting up "bcoef"
    if ((not issubclass(type(bcoef), numpy.ndarray)) or
        (not numpy.asarray(bcoef).flags.f_contiguous) or
        (not (bcoef.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'bcoef' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        bcoef = numpy.asarray(bcoef, dtype=ctypes.c_double, order='F')
    bcoef_dim_1 = ctypes.c_int(bcoef.shape[0])
    
    # Setting up "xy"
    if ((not issubclass(type(xy), numpy.ndarray)) or
        (not numpy.asarray(xy).flags.f_contiguous) or
        (not (xy.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'xy' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        xy = numpy.asarray(xy, dtype=ctypes.c_double, order='F')
    xy_dim_1 = ctypes.c_int(xy.shape[0])
    
    # Setting up "info"
    info = ctypes.c_int()
    
    # Setting up "d"
    d_present = ctypes.c_bool(True)
    if (d is None):
        d_present = ctypes.c_bool(False)
        d = ctypes.c_int()
    if (type(d) is not ctypes.c_int): d = ctypes.c_int(d)

    # Call C-accessible Fortran wrapper.
    clib.c_eval_spline(ctypes.byref(t_dim_1), ctypes.c_void_p(t.ctypes.data), ctypes.byref(bcoef_dim_1), ctypes.c_void_p(bcoef.ctypes.data), ctypes.byref(xy_dim_1), ctypes.c_void_p(xy.ctypes.data), ctypes.byref(info), ctypes.byref(d_present), ctypes.byref(d))

    # Return final results, 'INTENT(OUT)' arguments only.
    return xy, info.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine EVAL_BSPLINE

def eval_bspline(t, xy, d=None):
    '''! Subroutine for evaluating a B-spline with provided knot sequence, T.
! W A R N I N G : This routine has NO ERROR CHECKING and assumes
! informed usage for speed. It has undefined behavior for input that
! doesn't match specifications.
!
! INPUT:
!   T(1:N) -- The nondecreasing sequence of N knots for the B-spline.
!
! INPUT / OUTPUT:
!   XY(1:M) -- On input, the locations at which the B-spline is evaluated;
!     on output, holds the value of the Dth derivative of the B-spline
!     with prescribed knots evaluated at the given locations, or the
!     integral of the B-spline over [T(1), XY(.)] in XY(.).
!
! OPTIONAL INPUT:
!   D --  The order of the derivative to take of the B-spline. If omitted,
!     D = 0. When D < 0, this subroutine integrates the B-spline over each
!     interval [T(1), XY(.)].
!
! DESCRIPTION:
!
!   This function uses the recurrence relation defining a B-spline:
!
!   B_{I,1}(X) = {1, if T(I) <= X < T(I+1),
!                {0, otherwise,
!
!   where I is the spline index, J = 2, ..., N-MAX{D,0}-1 is the order, and
!
!                   X-T(I)                      T(I+J)-X
!   B_{I,J}(X) = ------------- B_{I,J-1}(X) + ------------- B_{I+1,J-1}(X).
!                T(I+J-1)-T(I)                T(I+J)-T(I+1)
!
!   All the intermediate steps (J) are stored in a single block of
!   memory that is reused for each step.
!
!   The computation of the integral of the B-spline proceeds from
!   the above formula one integration step at a time by adding a
!   duplicate of the last knot, raising the order of all
!   intermediate B-splines, summing their values, and rescaling
!   each sum by the width of the supported interval divided by the
!   degree plus the integration coefficient.
!
!   For the computation of the derivative of the B-spline, the divided
!   difference definition of B_{I,J}(X) is used, building from J = N-D,
!   ..., N-1 via
!
!                       (J-1) B_{I,J-1}(X)     (J-1) B_{I+1,J-1}(X)
!   D/DX[B_{I,J}(X)] =  ------------------  -  --------------------.
!                         T(I+J-1) - T(I)         T(I+J) - T(I+1)
!
!   The final B-spline is right continuous and has support over the
!   interval [T(1), T(N)).
!'''
    
    # Setting up "t"
    if ((not issubclass(type(t), numpy.ndarray)) or
        (not numpy.asarray(t).flags.f_contiguous) or
        (not (t.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 't' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        t = numpy.asarray(t, dtype=ctypes.c_double, order='F')
    t_dim_1 = ctypes.c_int(t.shape[0])
    
    # Setting up "xy"
    if ((not issubclass(type(xy), numpy.ndarray)) or
        (not numpy.asarray(xy).flags.f_contiguous) or
        (not (xy.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'xy' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        xy = numpy.asarray(xy, dtype=ctypes.c_double, order='F')
    xy_dim_1 = ctypes.c_int(xy.shape[0])
    
    # Setting up "d"
    d_present = ctypes.c_bool(True)
    if (d is None):
        d_present = ctypes.c_bool(False)
        d = ctypes.c_int()
    if (type(d) is not ctypes.c_int): d = ctypes.c_int(d)

    # Call C-accessible Fortran wrapper.
    clib.c_eval_bspline(ctypes.byref(t_dim_1), ctypes.c_void_p(t.ctypes.data), ctypes.byref(xy_dim_1), ctypes.c_void_p(xy.ctypes.data), ctypes.byref(d_present), ctypes.byref(d))

    # Return final results, 'INTENT(OUT)' arguments only.
    return xy



# Construct and returnn an MQSI spline fit to data.
def spline_fit(x, y):
    import numpy as np
    import mqsi
    # Initialize all arguments to MQSI.
    ftype = dict(dtype="float64", order='F')
    x = np.asarray(x, **ftype)
    y = np.asarray(y, **ftype)
    nd = x.size
    t = np.zeros(3*nd + 6, **ftype)
    bcoef = np.zeros(3*nd, **ftype)
    uv = np.zeros((nd, 2), **ftype)
    # Use MQSI to fit a spline to the data.
    t, bcoef, info, uv = mqsi.mqsi(x, y, t, bcoef, uv=uv)
    assert (info == 0), f"mqsi.mqsi subroutine returned nonzero info stats '{info}'. See help documentation."
    # Construct functions for evaluating the fit spline and derivatives.
    def function(x, t=t, bcoef=bcoef, d=None):
        x = np.array(x, dtype='float64', order='F')
        y, info = mqsi.eval_spline(t, bcoef, x, d=d)
        assert (info == 0), f"mqsi.eval_spline subroutine returned nonzero info stats '{info}'. See help documentation."
        return y
    function.derivative = lambda x: function(x, d=1)
    function.derivative.derivative = lambda x: function(x, d=2)
    # Return the fit function.
    return function


