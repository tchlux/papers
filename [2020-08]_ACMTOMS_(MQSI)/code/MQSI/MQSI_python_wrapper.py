'''This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code follows.

! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
!                          MQSI.f90
!
! DESCRIPTION:
!   This file defines a subroutine MQSI for constructing a monotone
!   quintic spline interpolant of data in terms of its B-spline basis.
!
! CONTAINS:
!   SUBROUTINE MQSI(X, Y, T, BCOEF, INFO, UV)
!     USE REAL_PRECISION, ONLY: R8
!     REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: X
!     REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: Y
!     REAL(KIND=R8), INTENT(OUT),   DIMENSION(:) :: T, BCOEF
!     INTEGER, INTENT(OUT) :: INFO
!     REAL(KIND=R8), INTENT(OUT), DIMENSION(:,:), OPTIONAL :: UV
!   END SUBROUTINE MQSI
!
! DEPENDENCIES:
!   MODULE REAL_PRECISION
!     INTEGER, PARAMETER :: R8
!   END MODULE REAL_PRECISION
!
!   SUBROUTINE FIT_SPLINE(XI, FX, T, BCOEF, INFO)
!     USE REAL_PRECISION, ONLY: R8
!     REAL(KIND=R8), INTENT(IN),  DIMENSION(:)   :: XI
!     REAL(KIND=R8), INTENT(IN),  DIMENSION(:,:) :: FX
!     REAL(KIND=R8), INTENT(OUT), DIMENSION(:)   :: T, BCOEF
!     INTEGER, INTENT(OUT) :: INFO
!   END SUBROUTINE FIT_SPLINE
!
! CONTRIBUTORS:
!   Thomas C.H. Lux (tchlux@vt.edu)
!   Layne T. Watson (ltwatson@computer.org)
!   William I. Thacker (thackerw@winthrop.edu)
!
! VERSION HISTORY:
!   June 2020 -- (tchl) Created file, (ltw / wit) reviewed and revised.
!
'''

import os
import ctypes
import numpy

# --------------------------------------------------------------------
#               CONFIGURATION
# 
_verbose = True
_fort_compiler = "gfortran"
_shared_object_name = "MQSI.so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3', '-lblas', '-llapack']
_ordered_dependencies = ['REAL_PRECISION.f90', 'EVAL_BSPLINE.f90', 'SPLINE.f90', 'MQSI.f90', 'MQSI_c_wrapper.f90']
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
    t_dim_1 = ctypes.c_int(t.shape[0])
    
    # Setting up "bcoef"
    bcoef_dim_1 = ctypes.c_int(bcoef.shape[0])
    
    # Setting up "info"
    info = ctypes.c_int()
    
    # Setting up "uv"
    uv_present = ctypes.c_bool(True)
    if (uv is None):
        uv_present = ctypes.c_bool(False)
        uv = numpy.zeros(shape=(1,1), dtype=ctypes.c_double, order='F')
    elif (type(uv) == bool) and (uv):
        uv = numpy.zeros(shape=(x.size, 2), dtype=ctypes.c_double, order='F')
    uv_dim_1 = ctypes.c_int(uv.shape[0])
    uv_dim_2 = ctypes.c_int(uv.shape[1])

    # Call C-accessible Fortran wrapper.
    clib.c_mqsi(ctypes.byref(x_dim_1), ctypes.c_void_p(x.ctypes.data), ctypes.byref(y_dim_1), ctypes.c_void_p(y.ctypes.data), ctypes.byref(t_dim_1), ctypes.c_void_p(t.ctypes.data), ctypes.byref(bcoef_dim_1), ctypes.c_void_p(bcoef.ctypes.data), ctypes.byref(info), ctypes.byref(uv_present), ctypes.byref(uv_dim_1), ctypes.byref(uv_dim_2), ctypes.c_void_p(uv.ctypes.data))

    # Return final results, 'INTENT(OUT)' arguments only.
    return y, t, bcoef, info.value, (uv if uv_present else None)

