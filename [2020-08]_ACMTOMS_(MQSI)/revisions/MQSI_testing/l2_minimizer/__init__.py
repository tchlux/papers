'''This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code follows.


'''

import os
import ctypes
import numpy

# --------------------------------------------------------------------
#               CONFIGURATION
# 
_verbose = True
_fort_compiler = "gfortran"
_shared_object_name = "monotone_fit.so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3'] #, '-lblas', '-llapack',]
_ordered_dependencies = ['blas.f90', 'lapack.f', 'monotone_fit.f90', 'monotone_fit_c_wrapper.f90']
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
# Wrapper for the Fortran subroutine EVAL_BSPLINE

def eval_bspline(t, xy, d=None):
    '''! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
!                        EVAL_BSPLINE.f90
!
! DESCRIPTION:
!   This file defines a subroutine EVAL_BSPLINE for computing the
!   value, integral, or derivative(s) of a B-spline given its knot
!   sequence.
!
! CONTAINS:
!   SUBROUTINE EVAL_BSPLINE(T, XY, D)
!     USE REAL_PRECISION, ONLY: R8
!     REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: T
!     REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
!     INTEGER, INTENT(IN), OPTIONAL :: D
!   END SUBROUTINE EVAL_BSPLINE
!
! EXTERNAL DEPENDENCIES:
!   MODULE REAL_PRECISION
!     INTEGER, PARAMETER :: R8
!   END MODULE REAL_PRECISION
!
! CONTRIBUTORS:
!   Thomas C.H. Lux (tchlux@vt.edu)
!   Layne T. Watson (ltwatson@computer.org)
!   William I. Thacker (thackerw@winthrop.edu)
!
! VERSION HISTORY:
!   June 2020 -- (tchl) Created file, (ltw / wit) reviewed and revised.
!
! Subroutine for evaluating a B-spline with provided knot sequence, T.
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
!   interval [T(1), T(N)).'''
    
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


# ----------------------------------------------
# Wrapper for the Fortran subroutine FIT_SPLINE

def fit_spline(xi, fx, t, bcoef):
    '''! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
!                           SPLINE.f90
!
! DESCRIPTION:
!   This file defines the subroutines FIT_SPLINE for computing the
!   coefficients of a B-spline basis necessary to reproduce given
!   function and derivative values, and EVAL_SPLINE for evaluating the
!   value, integral, and derivatives of a spline defined in terms of
!   its B-spline basis.
!
! CONTAINS:
!   SUBROUTINE FIT_SPLINE(XI, FX, T, BCOEF, INFO)
!     USE REAL_PRECISION, ONLY: R8
!     REAL(KIND=R8), INTENT(IN),  DIMENSION(:)   :: XI
!     REAL(KIND=R8), INTENT(IN),  DIMENSION(:,:) :: FX
!     REAL(KIND=R8), INTENT(OUT), DIMENSION(:)   :: T, BCOEF
!     INTEGER, INTENT(OUT) :: INFO
!   END SUBROUTINE FIT_SPLINE
!
!   SUBROUTINE EVAL_SPLINE(T, BCOEF, XY, INFO, D)
!     USE REAL_PRECISION, ONLY: R8
!     REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: T, BCOEF
!     REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
!     INTEGER, INTENT(OUT) :: INFO
!     INTEGER, INTENT(IN), OPTIONAL :: D
!   END SUBROUTINE EVAL_SPLINE
!
! EXTERNAL DEPENDENCIES:
!   MODULE REAL_PRECISION
!     INTEGER, PARAMETER :: R8
!   END MODULE REAL_PRECISION
!
!   SUBROUTINE EVAL_BSPLINE(T, XY, D)
!     USE REAL_PRECISION, ONLY: R8
!     REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: T
!     REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
!     INTEGER, INTENT(IN), OPTIONAL :: D
!   END SUBROUTINE EVAL_BSPLINE
!
! CONTRIBUTORS:
!   Thomas C.H. Lux (tchlux@vt.edu)
!   Layne T. Watson (ltwatson@computer.org)
!   William I. Thacker (thackerw@winthrop.edu)
!
! VERSION HISTORY:
!   June 2020 -- (tchl) Created file, (ltw / wit) reviewed and revised.
!
! Subroutine for computing a linear combination of B-splines that
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
!   may fail to accurately reproduce the data, indicated by INFO = 7.'''
    
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
!   XY(:) be increasing.'''
    
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
# Wrapper for the Fortran subroutine MONOTONE_FIT

def monotone_fit(x, y, c, steps, msteps, step_size, monotonicity_multiplier, fx=None, t=None, bcoef=None):
    '''! Prouce a monotone fit to data.'''
    
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
    
    # Setting up "c"
    if (type(c) is not ctypes.c_int): c = ctypes.c_int(c)
    
    # Setting up "steps"
    if (type(steps) is not ctypes.c_int): steps = ctypes.c_int(steps)
    
    # Setting up "msteps"
    if (type(msteps) is not ctypes.c_int): msteps = ctypes.c_int(msteps)
    
    # Setting up "step_size"
    if (type(step_size) is not ctypes.c_double): step_size = ctypes.c_double(step_size)
    
    # Setting up "monotonicity_multiplier"
    if (type(monotonicity_multiplier) is not ctypes.c_double): monotonicity_multiplier = ctypes.c_double(monotonicity_multiplier)
    
    # Setting up "fx"
    if (fx is None):
        fx = numpy.zeros(shape=(x.size, c), dtype=ctypes.c_double, order='F')
    elif ((not issubclass(type(fx), numpy.ndarray)) or
          (not numpy.asarray(fx).flags.f_contiguous) or
          (not (fx.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'fx' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        fx = numpy.asarray(fx, dtype=ctypes.c_double, order='F')
    fx_dim_1 = ctypes.c_int(fx.shape[0])
    fx_dim_2 = ctypes.c_int(fx.shape[1])
    
    # Setting up "t"
    if (t is None):
        t = numpy.zeros(shape=(x.size*c+2*c), dtype=ctypes.c_double, order='F')
    elif ((not issubclass(type(t), numpy.ndarray)) or
          (not numpy.asarray(t).flags.f_contiguous) or
          (not (t.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 't' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        t = numpy.asarray(t, dtype=ctypes.c_double, order='F')
    t_dim_1 = ctypes.c_int(t.shape[0])
    
    # Setting up "bcoef"
    if (bcoef is None):
        bcoef = numpy.zeros(shape=(x.size*c), dtype=ctypes.c_double, order='F')
    elif ((not issubclass(type(bcoef), numpy.ndarray)) or
          (not numpy.asarray(bcoef).flags.f_contiguous) or
          (not (bcoef.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'bcoef' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        bcoef = numpy.asarray(bcoef, dtype=ctypes.c_double, order='F')
    bcoef_dim_1 = ctypes.c_int(bcoef.shape[0])
    
    # Setting up "info"
    info = ctypes.c_int()

    # Call C-accessible Fortran wrapper.
    clib.c_monotone_fit(ctypes.byref(x_dim_1),
                        ctypes.c_void_p(x.ctypes.data), ctypes.byref(y_dim_1),
                        ctypes.c_void_p(y.ctypes.data), ctypes.byref(c),
                        ctypes.byref(steps), ctypes.byref(msteps),
                        ctypes.byref(step_size), ctypes.byref(monotonicity_multiplier),
                        ctypes.byref(fx_dim_1), ctypes.byref(fx_dim_2),
                        ctypes.c_void_p(fx.ctypes.data), ctypes.byref(t_dim_1),
                        ctypes.c_void_p(t.ctypes.data), ctypes.byref(bcoef_dim_1),
                        ctypes.c_void_p(bcoef.ctypes.data), ctypes.byref(info))

    # Return final results, 'INTENT(OUT)' arguments only.
    return step_size.value, monotonicity_multiplier.value, fx, t, bcoef, info.value


class real_precision:
    ''''''

    # Declare 'r8'
    def get_r8(self):
        r8 = ctypes.c_int()
        clib.real_precision_get_r8(ctypes.byref(r8))
        return r8.value
    def set_r8(self, r8):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    r8 = property(get_r8, set_r8)

real_precision = real_precision()


# Produce a piecewise monotone spline fit of the specified continuity.
def spline_fit(x, y, c=2, l2_steps=5000, monotone_steps=10000, step_size=0.001, multiplier=.01):
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Create a monotone fit of the data (using an optimization procedure).
    c += 1
    fx = np.zeros(shape=(x.size, c), dtype=float, order='F')
    t = np.zeros(shape=(x.size*c+2*c), dtype=float, order='F')
    bcoef = np.zeros(shape=(x.size*c), dtype=float, order='F')
    monotone_fit(x, y, c, l2_steps, monotone_steps, step_size,
                 multiplier, fx, t, bcoef)
    # Evaluate the spline fit.
    def fit(x, t=t, bcoef=bcoef, d=0):
        xy = np.array(x, dtype=float)
        xy, info = eval_spline(t, bcoef, xy, d=d)
        assert info == 0, f"EVAL_SPLINE produced nonzero info {info}."
        return xy
    fit.derivative = lambda x: fit(x, d=1)
    fit.derivative.derivative = lambda x: fit(x, d=2)
    # Evaluate a b-spline with a given knot sequence.
    def bspline(x, t=np.asarray([0.,0,0,1,1,1]), d=0):
        t = t * 0.8 + 0.1
        xy = np.array(x, dtype=float)
        spline.eval_bspline(t, xy, d=d)
        return xy
    # Return the fit function.
    return fit


