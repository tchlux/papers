! This automatically generated Fortran wrapper file allows codes
! written in Fortran to be called directly from C and translates all
! C-style arguments into expected Fortran-style arguments (with
! assumed size, local type declarations, etc.).


SUBROUTINE C_FIT_SPLINE(XI_DIM_1, XI, FX_DIM_1, FX_DIM_2, FX, T_DIM_1, T, BCOEF_DIM_1, BCOEF, INFO) BIND(C)
  USE REAL_PRECISION , ONLY : R8
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: XI_DIM_1
  REAL(KIND=R8), INTENT(IN), DIMENSION(XI_DIM_1) :: XI
  INTEGER, INTENT(IN) :: FX_DIM_1
  INTEGER, INTENT(IN) :: FX_DIM_2
  REAL(KIND=R8), INTENT(IN), DIMENSION(FX_DIM_1,FX_DIM_2) :: FX
  INTEGER, INTENT(IN) :: T_DIM_1
  REAL(KIND=R8), INTENT(OUT), DIMENSION(T_DIM_1) :: T
  INTEGER, INTENT(IN) :: BCOEF_DIM_1
  REAL(KIND=R8), INTENT(OUT), DIMENSION(BCOEF_DIM_1) :: BCOEF
  INTEGER, INTENT(OUT) :: INFO

  INTERFACE
    SUBROUTINE FIT_SPLINE(XI, FX, T, BCOEF, INFO)
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
      !   may fail to accurately reproduce the data, indicated by INFO = 7.
      !
      USE REAL_PRECISION , ONLY : R8
      IMPLICIT NONE
      REAL(KIND=R8), INTENT(IN), DIMENSION(:) :: XI
      REAL(KIND=R8), INTENT(IN), DIMENSION(:,:) :: FX
      REAL(KIND=R8), INTENT(OUT), DIMENSION(:) :: T
      REAL(KIND=R8), INTENT(OUT), DIMENSION(:) :: BCOEF
      INTEGER, INTENT(OUT) :: INFO
    END SUBROUTINE FIT_SPLINE
  END INTERFACE

  CALL FIT_SPLINE(XI, FX, T, BCOEF, INFO)
END SUBROUTINE C_FIT_SPLINE


SUBROUTINE C_EVAL_SPLINE(T_DIM_1, T, BCOEF_DIM_1, BCOEF, XY_DIM_1, XY, INFO, D_PRESENT, D) BIND(C)
  USE REAL_PRECISION , ONLY : R8
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: T_DIM_1
  REAL(KIND=R8), INTENT(IN), DIMENSION(T_DIM_1) :: T
  INTEGER, INTENT(IN) :: BCOEF_DIM_1
  REAL(KIND=R8), INTENT(IN), DIMENSION(BCOEF_DIM_1) :: BCOEF
  INTEGER, INTENT(IN) :: XY_DIM_1
  REAL(KIND=R8), INTENT(INOUT), DIMENSION(XY_DIM_1) :: XY
  INTEGER, INTENT(OUT) :: INFO
  LOGICAL, INTENT(IN) :: D_PRESENT
  INTEGER, INTENT(IN) :: D

  INTERFACE
    SUBROUTINE EVAL_SPLINE(T, BCOEF, XY, INFO, D)
      ! Evaluate a spline constructed with FIT_SPLINE.
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
      !
      USE REAL_PRECISION , ONLY : R8
      IMPLICIT NONE
      REAL(KIND=R8), INTENT(IN), DIMENSION(:) :: T
      REAL(KIND=R8), INTENT(IN), DIMENSION(:) :: BCOEF
      REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
      INTEGER, INTENT(OUT) :: INFO
      INTEGER, INTENT(IN), OPTIONAL :: D
    END SUBROUTINE EVAL_SPLINE
  END INTERFACE

  IF (D_PRESENT) THEN
    CALL EVAL_SPLINE(T=T, BCOEF=BCOEF, XY=XY, INFO=INFO, D=D)
  ELSE
    CALL EVAL_SPLINE(T=T, BCOEF=BCOEF, XY=XY, INFO=INFO)
  END IF
END SUBROUTINE C_EVAL_SPLINE

