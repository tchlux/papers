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

