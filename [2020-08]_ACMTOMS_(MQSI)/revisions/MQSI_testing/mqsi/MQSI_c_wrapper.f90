! This automatically generated Fortran wrapper file allows codes
! written in Fortran to be called directly from C and translates all
! C-style arguments into expected Fortran-style arguments (with
! assumed size, local type declarations, etc.).


SUBROUTINE C_MQSI(X_DIM_1, X, Y_DIM_1, Y, T_DIM_1, T, BCOEF_DIM_1, BCOEF, INFO, UV_PRESENT, UV_DIM_1, UV_DIM_2, UV) BIND(C)
  USE REAL_PRECISION , ONLY : R8
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: X_DIM_1
  REAL(KIND=R8), INTENT(IN), DIMENSION(X_DIM_1) :: X
  INTEGER, INTENT(IN) :: Y_DIM_1
  REAL(KIND=R8), INTENT(INOUT), DIMENSION(Y_DIM_1) :: Y
  INTEGER, INTENT(IN) :: T_DIM_1
  REAL(KIND=R8), INTENT(OUT), DIMENSION(T_DIM_1) :: T
  INTEGER, INTENT(IN) :: BCOEF_DIM_1
  REAL(KIND=R8), INTENT(OUT), DIMENSION(BCOEF_DIM_1) :: BCOEF
  INTEGER, INTENT(OUT) :: INFO
  LOGICAL, INTENT(IN) :: UV_PRESENT
  INTEGER, INTENT(IN) :: UV_DIM_1
  INTEGER, INTENT(IN) :: UV_DIM_2
  REAL(KIND=R8), INTENT(OUT), DIMENSION(UV_DIM_1,UV_DIM_2) :: UV

  INTERFACE
    SUBROUTINE MQSI(X, Y, T, BCOEF, INFO, UV)
      ! Computes a monotone quintic spline interpolant (MQSI), Q(X), to data
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
      !
      USE REAL_PRECISION , ONLY : R8
      IMPLICIT NONE
      REAL(KIND=R8), INTENT(IN), DIMENSION(:) :: X
      REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: Y
      REAL(KIND=R8), INTENT(OUT), DIMENSION(:) :: T
      REAL(KIND=R8), INTENT(OUT), DIMENSION(:) :: BCOEF
      INTEGER, INTENT(OUT) :: INFO
      REAL(KIND=R8), INTENT(OUT), OPTIONAL, DIMENSION(:,:) :: UV
    END SUBROUTINE MQSI
  END INTERFACE

  IF (UV_PRESENT) THEN
    CALL MQSI(X=X, Y=Y, T=T, BCOEF=BCOEF, INFO=INFO, UV=UV)
  ELSE
    CALL MQSI(X=X, Y=Y, T=T, BCOEF=BCOEF, INFO=INFO)
  END IF
END SUBROUTINE C_MQSI


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


SUBROUTINE C_EVAL_BSPLINE(T_DIM_1, T, XY_DIM_1, XY, D_PRESENT, D) BIND(C)
  USE REAL_PRECISION , ONLY : R8
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: T_DIM_1
  REAL(KIND=R8), INTENT(IN), DIMENSION(T_DIM_1) :: T
  INTEGER, INTENT(IN) :: XY_DIM_1
  REAL(KIND=R8), INTENT(INOUT), DIMENSION(XY_DIM_1) :: XY
  LOGICAL, INTENT(IN) :: D_PRESENT
  INTEGER, INTENT(IN) :: D

  INTERFACE
    SUBROUTINE EVAL_BSPLINE(T, XY, D)
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
      !   B_{{I,1}}(X) = {{1, if T(I) <= X < T(I+1),
      !                {{0, otherwise,
      !
      !   where I is the spline index, J = 2, ..., N-MAX{{D,0}}-1 is the order, and
      !
      !                   X-T(I)                      T(I+J)-X
      !   B_{{I,J}}(X) = ------------- B_{{I,J-1}}(X) + ------------- B_{{I+1,J-1}}(X).
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
      !   difference definition of B_{{I,J}}(X) is used, building from J = N-D,
      !   ..., N-1 via
      !
      !                       (J-1) B_{{I,J-1}}(X)     (J-1) B_{{I+1,J-1}}(X)
      !   D/DX[B_{{I,J}}(X)] =  ------------------  -  --------------------.
      !                         T(I+J-1) - T(I)         T(I+J) - T(I+1)
      !
      !   The final B-spline is right continuous and has support over the
      !   interval [T(1), T(N)).
      !
      USE REAL_PRECISION , ONLY : R8
      IMPLICIT NONE
      REAL(KIND=R8), INTENT(IN), DIMENSION(:) :: T
      REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
      INTEGER, INTENT(IN), OPTIONAL :: D
    END SUBROUTINE EVAL_BSPLINE
  END INTERFACE

  IF (D_PRESENT) THEN
    CALL EVAL_BSPLINE(T=T, XY=XY, D=D)
  ELSE
    CALL EVAL_BSPLINE(T=T, XY=XY)
  END IF
END SUBROUTINE C_EVAL_BSPLINE

