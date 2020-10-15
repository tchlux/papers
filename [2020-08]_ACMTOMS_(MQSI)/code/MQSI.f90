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
USE REAL_PRECISION, ONLY: R8
IMPLICIT NONE
! Arguments.
REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: X
REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: Y
REAL(KIND=R8), INTENT(OUT),   DIMENSION(:) :: T, BCOEF
INTEGER, INTENT(OUT) :: INFO
REAL(KIND=R8), INTENT(OUT), DIMENSION(:,:), OPTIONAL :: UV
! Local variables.
!  Estimated first and second derivatives (columns) by quadratic
!  facet model at all data points (rows).
REAL(KIND=R8), DIMENSION(SIZE(X),2) :: FHATX 
!  Spline values, first, and second derivatives (columns) at all points (rows).
REAL(KIND=R8), DIMENSION(SIZE(X),3) :: FX
!  Execution queues holding intervals to check for monotonicity
!  CHECKING, TO_CHECK, derivative values to grow (after shrinking)
!  GROWING, TO_GROW, and derivative values to shrink (because of
!  nonmonotonicity) SHRINKING, TO_SHRINK. The arrays GROWING and
!  SHRINKING are repurposed to identify locations of local maxima and
!  minima of provided Y values early in the routine.
LOGICAL, DIMENSION(SIZE(X)) :: CHECKING, GROWING, SHRINKING
INTEGER, DIMENSION(SIZE(X)) :: TO_CHECK, TO_GROW, TO_SHRINK
!  Coefficients for a quadratic interpolant A and B, the direction of
!  function change DIRECTION, a derivative value DX, the exponent
!  scale difference between X and Y values SCALE, and the current
!  bisection search (relative) step size STEP_SIZE.
REAL(KIND=R8) :: A, B, DIRECTION, DX, SCALE, STEP_SIZE
!  The smallest spacing of X and step size ACCURACY, the machine
!  precision at unit scale EPS, the largest allowed spacing of X or Y
!  values 10^38 TP38, and the placeholder large magnitude curvature
!  value 10^54 TP54.
REAL(KIND=R8), PARAMETER :: ACCURACY = SQRT(EPSILON(1.0_R8)), &
     EPS = EPSILON(1.0_R8), &
     TP38 = 10.0_R8**38, TP54 = 10.0_R8**54
!  Iteration indices I and J, number of data points ND, counters for
!  execution queues, checking, NC, growing, NG, and shrinking, NS.
INTEGER :: I, IM1, IP1, J, NC, ND, NG, NS
!  Boolean indicating whether the bisection search is in progress.
LOGICAL :: SEARCHING
INTERFACE
   SUBROUTINE FIT_SPLINE(XI, FX, T, BCOEF, INFO)
     USE REAL_PRECISION, ONLY: R8
     REAL(KIND=R8), INTENT(IN),  DIMENSION(:)   :: XI
     REAL(KIND=R8), INTENT(IN),  DIMENSION(:,:) :: FX
     REAL(KIND=R8), INTENT(OUT), DIMENSION(:)   :: T, BCOEF
     INTEGER, INTENT(OUT) :: INFO
   END SUBROUTINE FIT_SPLINE
END INTERFACE
ND = SIZE(X)
! Check the shape of incoming arrays.
IF      (ND .LT. 3)             THEN; INFO = 1; RETURN
ELSE IF (SIZE(Y) .NE. ND)       THEN; INFO = 2; RETURN
ELSE IF (SIZE(T) .LT. 3*ND + 6) THEN; INFO = 3; RETURN
ELSE IF (SIZE(BCOEF) .LT. 3*ND) THEN; INFO = 4; RETURN
END IF
! Verify that X values are increasing, and that (X,Y) data is not extreme.
DO I = 1, ND-1
   IP1 = I+1
   IF ((X(IP1) - X(I)) .LT. ACCURACY)    THEN; INFO = 5; RETURN
   ELSE IF (((X(IP1) - X(I)) .GT. TP38) .OR. &
         (ABS(Y(IP1) - Y(I)) .GT. TP38)) THEN; INFO = 6; RETURN
   END IF
END DO
IF (PRESENT(UV)) THEN
   IF ((SIZE(UV,DIM=1) .LT. ND) .OR. (SIZE(UV,DIM=2) .LT. 2)) THEN
      INFO = 8; RETURN; END IF
END IF
! Scale Y by an exact power of 2 to make Y and X commensurate, store
! scaled Y in FX and also back in Y.
J = INT(LOG((1.0_R8+MAXVAL(ABS(Y(:))))/MAXVAL(ABS(X(:)))) / LOG(2.0_R8))
SCALE = 2.0_R8**J
FX(:,1) = (1.0_R8/SCALE)*Y(:)
Y(:) = FX(:,1)

! ==================================================================
!          Algorithm 1: Estimate initial derivatives by 
!         using a minimum curvature quadratic facet model.
! 
! Identify local extreme points and flat points. Denote location
! of flats in GROWING and extrema in SHRINKING to save memory.
GROWING(1) = ABS(Y(1) - Y(2)) .LT. EPS*(1.0_R8+ABS(Y(1))+ABS(Y(2)))
GROWING(ND) = ABS(Y(ND-1) - Y(ND)) .LT. EPS*(1.0_R8+ABS(Y(ND-1))+ABS(Y(ND)))
SHRINKING(1) = .FALSE.
SHRINKING(ND) = .FALSE.
DO I = 2, ND-1
   IM1 = I-1
   IP1 = I+1
   IF ((ABS(Y(IM1)-Y(I)) .LT. EPS*(1.0_R8+ABS(Y(IM1))+ABS(Y(I)))) .OR. &
       (ABS(Y(I)-Y(IP1)) .LT. EPS*(1.0_R8+ABS(Y(I))+ABS(Y(IP1))))) THEN
      GROWING(I) = .TRUE.
      SHRINKING(I) = .FALSE.
   ELSE
      GROWING(I) = .FALSE.
      SHRINKING(I) = (Y(I) - Y(IM1)) * (Y(IP1) - Y(I)) .LT. 0.0_R8
   END IF
END DO
! Use local quadratic interpolants to estimate slopes and second
! derivatives at all points. Use zero-sloped quadratic interpolants
! at extrema with left/right neighbors to estimate curvature.
estimate_derivatives : DO I = 1, ND
   IM1 = I-1
   IP1 = I+1
   ! Initialize the curvature to be maximally large.
   FX(I,3) = TP54
   ! If this is locally flat, then first and second derivatives are zero valued.
   pick_quadratic : IF (GROWING(I)) THEN
      FX(I,2:3) = 0.0_R8
   ! If this is an extreme point (local minimum or maximum Y),
   ! construct quadratic interpolants that have zero slope here and
   ! hit left/right neighbors.
   ELSE IF (SHRINKING(I)) THEN
      ! Set the first derivative to zero.
      FX(I,2) = 0.0_R8
      ! Compute the coefficient A in  Ax^2+Bx+C  that interpolates at X(I-1).
      FX(I,3) = (Y(IM1) - Y(I)) / (X(IM1) - X(I))**2
      ! Compute the coefficient A in  Ax^2+Bx+C  that interpolates at X(I+1).
      A = (Y(IP1) - Y(I)) / (X(IP1) - X(I))**2
      IF (ABS(A) .LT. ABS(FX(I,3))) FX(I,3) = A
      ! Compute the actual second derivative (instead of coefficient A).
      FX(I,3) = 2.0_R8 * FX(I,3)
   ELSE
      ! Determine the direction of change at the point I.
      IF (I .EQ. 1) THEN
         IF      (Y(I) .LT. Y(IP1)) THEN; DIRECTION =  1.0_R8
         ELSE IF (Y(I) .GT. Y(IP1)) THEN; DIRECTION = -1.0_R8
         END IF
      ELSE
         IF      (Y(IM1) .LT. Y(I)) THEN; DIRECTION =  1.0_R8
         ELSE IF (Y(IM1) .GT. Y(I)) THEN; DIRECTION = -1.0_R8
         END IF
      END IF
      ! --------------------
      ! Quadratic left of I.
      IF (IM1 .GT. 1) THEN
         ! If a zero derivative at left point, use its right interpolant.
         IF (SHRINKING(IM1) .OR. GROWING(IM1)) THEN
            A = (Y(I) - Y(IM1)) / (X(I) - X(IM1))**2
            B = -2.0_R8 * X(IM1) * A
         ! Otherwise use the standard quadratic on the left.
         ELSE; CALL QUADRATIC(X, Y, IM1, A, B)
         END IF
         DX = 2.0_R8*A*X(I) + B
         IF (DX*DIRECTION .GE. 0.0_R8) THEN
            FX(I,2) = DX
            FX(I,3) = A
         END IF
      END IF
      ! ------------------------
      ! Quadratic centered at I (require that it has at least one
      ! neighbor that is not forced to zero slope).
      IF ((I .GT. 1) .AND. (I .LT. ND)) THEN
         IF (.NOT. ((SHRINKING(IM1) .OR. GROWING(IM1)) .AND. &
              (SHRINKING(IP1) .OR. GROWING(IP1)))) THEN
            ! Construct quadratic interpolant through this point and neighbors.
            CALL QUADRATIC(X, Y, I, A, B)
            DX = 2.0_R8*A*X(I) + B
            ! Keep this new quadratic if it has less curvature.
            IF ((DX*DIRECTION .GE. 0.0_R8) .AND. (ABS(A) .LT. ABS(FX(I,3)))) THEN
               FX(I,2) = DX
               FX(I,3) = A
            END IF
         END IF
      END IF
      ! ---------------------
      ! Quadratic right of I.
      IF (IP1 .LT. ND) THEN
         ! If a zero derivative at right point, use its left interpolant.
         IF (SHRINKING(IP1) .OR. GROWING(IP1)) THEN
            A = (Y(I) - Y(IP1)) / (X(I) - X(IP1))**2
            B = -2.0_R8 * X(IP1) * A
         ! Otherwise use the standard quadratic on the right.
         ELSE; CALL QUADRATIC(X, Y, IP1, A, B)
         END IF
         DX = 2.0_R8*A*X(I) + B
         ! Keep this new quadratic if it has less curvature.
         IF ((DX*DIRECTION .GE. 0.0_R8) .AND. (ABS(A) .LT. ABS(FX(I,3)))) THEN
            FX(I,2) = DX
            FX(I,3) = A
         END IF
      END IF
      ! Set the final quadratic.
      IF (FX(I,3) .EQ. TP54) THEN
         FX(I,2:3) = 0.0_R8
      ! Compute curvature of quadratic from coefficient of x^2.
      ELSE
         FX(I,3) = 2.0_R8 * FX(I,3)
      END IF
   END IF pick_quadratic
END DO estimate_derivatives

! ==================================================================
!          Algorithm 3: Identify viable piecewise monotone
!        derivative values by doing a quasi-bisection search.
! 
! Store the initially estimated first and second derivative values.
FHATX(:,1:2) = FX(:,2:3)
! Identify which spline segments are not monotone and need to be fixed.
CHECKING(:) = .FALSE.
SHRINKING(:) = .FALSE.
NC = 0; NG = 0; NS = 0
DO I = 1, ND-1
   IP1 = I+1
   ! Check for monotonicity on all segments that are not flat.
   IF ((.NOT. (GROWING(I) .AND. GROWING(IP1))) .AND. &
        (.NOT. IS_MONOTONE(X(I), X(IP1), FX(I,1), FX(IP1,1), &
        FX(I,2), FX(IP1,2), FX(I,3), FX(IP1,3)))) THEN
      ! Store points bounding nonmonotone segments in the TO_SHRINK queue.
      IF (.NOT. SHRINKING(I)) THEN
         SHRINKING(I) = .TRUE.
         NS = NS+1
         TO_SHRINK(NS) = I
      END IF
      IF (.NOT. SHRINKING(IP1)) THEN
         SHRINKING(IP1) = .TRUE.
         NS = NS+1
         TO_SHRINK(NS) = IP1
      END IF
   END IF
END DO
! Initialize step size to 1.0 (will be halved at beginning of loop).
STEP_SIZE = 1.0_R8
SEARCHING = .TRUE.
GROWING(:) = .FALSE.
! Loop until the accuracy is achieved and *all* intervals are monotone.
DO WHILE (SEARCHING .OR. (NS .GT. 0))
   ! Compute the step size for this iteration.
   IF (SEARCHING) THEN
      STEP_SIZE = STEP_SIZE / 2.0_R8
      IF (STEP_SIZE .LT. ACCURACY) THEN
         SEARCHING = .FALSE.
         STEP_SIZE = ACCURACY
         NG = 0
      END IF
   ! Grow the step size (at a slower rate than the step size reduction
   ! rate) if there are still intervals to fix.
   ELSE
      STEP_SIZE = STEP_SIZE * 1.5_R8
   END IF
   ! Grow all those first and second derivatives that were previously 
   ! shrunk, and correspond to currently monotone spline pieces.
   grow_values : DO J = 1, NG
      I = TO_GROW(J)
      ! Do not grow values that are actively related to a nonmonotone
      ! spline segment.
      IF (SHRINKING(I)) CYCLE grow_values
      ! Otherwise, grow those values that have been modified previously.
      FX(I,2) = FX(I,2) + STEP_SIZE * FHATX(I,1)
      FX(I,3) = FX(I,3) + STEP_SIZE * FHATX(I,2)
      ! Make sure the first derivative does not exceed its original value.
      IF ((FHATX(I,1) .LT. 0.0_R8) .AND. (FX(I,2) .LT. FHATX(I,1))) THEN
         FX(I,2) = FHATX(I,1)
      ELSE IF ((FHATX(I,1) .GT. 0.0_R8) .AND. (FX(I,2) .GT. FHATX(I,1))) THEN
         FX(I,2) = FHATX(I,1)
      END IF
      ! Make sure the second derivative does not exceed its original value.
      IF ((FHATX(I,2) .LT. 0.0_R8) .AND. (FX(I,3) .LT. FHATX(I,2))) THEN
         FX(I,3) = FHATX(I,2)
      ELSE IF ((FHATX(I,2) .GT. 0.0_R8) .AND. (FX(I,3) .GT. FHATX(I,2))) THEN
         FX(I,3) = FHATX(I,2)
      END IF
      ! Set this point and its neighboring intervals to be checked for
      ! monotonicity. Use sequential IF statements for short-circuiting.
      IF (I .GT. 1) THEN; IF (.NOT. CHECKING(I-1)) THEN
         CHECKING(I-1) = .TRUE.
         NC = NC+1
         TO_CHECK(NC) = I-1
      END IF; END IF
      IF (I .LT. ND) THEN; IF (.NOT. CHECKING(I)) THEN
         CHECKING(I) = .TRUE.
         NC = NC+1
         TO_CHECK(NC) = I
      END IF; END IF
   END DO grow_values
   ! Shrink the first and second derivatives that cause nonmonotonicity.
   shrink_values : DO J = 1, NS
      I = TO_SHRINK(J)
      SHRINKING(I) = .FALSE.
      IF (SEARCHING .AND. (.NOT. GROWING(I))) THEN
         GROWING(I) = .TRUE.
         NG = NG+1
         TO_GROW(NG) = I
      END IF
      ! Shrink the values that are causing nonmonotonicity.
      FX(I,2) = FX(I,2) - STEP_SIZE * FHATX(I,1)
      FX(I,3) = FX(I,3) - STEP_SIZE * FHATX(I,2)
      ! Make sure the first derivative does not pass zero.
      IF ((FHATX(I,1) .LT. 0.0_R8) .AND. (FX(I,2) .GT. 0.0_R8)) THEN
         FX(I,2) = 0.0_R8
      ELSE IF ((FHATX(I,1) .GT. 0.0_R8) .AND. (FX(I,2) .LT. 0.0_R8)) THEN
         FX(I,2) = 0.0_R8
      END IF
      ! Make sure the second derivative does not pass zero.
      IF ((FHATX(I,2) .LT. 0.0_R8) .AND. (FX(I,3) .GT. 0.0_R8)) THEN
         FX(I,3) = 0.0_R8
      ELSE IF ((FHATX(I,2) .GT. 0.0_R8) .AND. (FX(I,3) .LT. 0.0_R8)) THEN
         FX(I,3) = 0.0_R8
      END IF
      ! Set this point and its neighboring intervals to be checked for
      ! monotonicity.
      IF ((I .GT. 1) .AND. (.NOT. CHECKING(I-1))) THEN
         CHECKING(I-1) = .TRUE.
         NC = NC+1
         TO_CHECK(NC) = I-1
      END IF
      IF ((I .LT. ND) .AND. (.NOT. CHECKING(I))) THEN
         CHECKING(I) = .TRUE.
         NC = NC+1
         TO_CHECK(NC) = I
      END IF
   END DO shrink_values
   ! Identify which spline segments are nonmonotone after the updates.
   NS = 0
   check_monotonicity : DO J = 1, NC
      I = TO_CHECK(J)
      IP1 = I+1
      CHECKING(I) = .FALSE.
      IF (.NOT. IS_MONOTONE(X(I), X(IP1), FX(I,1), FX(IP1,1), &
           FX(I,2), FX(IP1,2), FX(I,3), FX(IP1,3))) THEN
         IF (.NOT. SHRINKING(I)) THEN
            SHRINKING(I) = .TRUE.
            NS = NS+1
            TO_SHRINK(NS) = I
         END IF
         IF (.NOT. SHRINKING(IP1)) THEN
            SHRINKING(IP1) = .TRUE.
            NS = NS+1
            TO_SHRINK(NS) = IP1
         END IF
      END IF
   END DO check_monotonicity
   NC = 0
END DO
! ------------------------------------------------------------------

! Use FIT_SPLINE to fit the final MQSI. For numerical stability and accuracy
! this routine requires the enforced separation of the values X(I).
CALL FIT_SPLINE(X, FX, T, BCOEF, INFO)

! Restore Y to its original value and unscale spline and derivative values.
BCOEF(1:3*ND) = SCALE * BCOEF(1:3*ND)
Y(:) = SCALE*Y(:) ! Restore original Y.
IF (PRESENT(UV)) THEN; UV(1:ND,1:2) = FX(1:ND,2:3); END IF ! Return first
! and second derivative values at breakpoints.

CONTAINS

! ==================================================================
!          Algorithm 2: Check for monotonicity using tight
!        theoretical constraints on a quintic polynomial piece.
! 
FUNCTION IS_MONOTONE(U0, U1, F0, F1, DF0, DF1, DDF0, DDF1)
! Given an interval [U0, U1] and function values F0, F1, first
! derivatives DF0, DF1, and second derivatives DDF0, DDF1 at U0, U1,
! respectively, IS_MONOTONE = TRUE if the quintic polynomial matching
! these values is monotone over [U0, U1], and IS_MONOTONE = FALSE otherwise.
!
REAL(KIND=R8), INTENT(IN) :: U0, U1, F0, F1, DF0, DF1, DDF0, DDF1
LOGICAL :: IS_MONOTONE
! Local variables.
REAL(KIND=R8), PARAMETER :: EPS = EPSILON(1.0_R8)
REAL(KIND=R8) :: ALPHA, BETA, GAMMA, SIGN, TEMP, W

! When the function values are flat, everything *must* be 0.
IF (ABS(F1 - F0) .LT. EPS*(1.0_R8 + ABS(F1) + ABS(F0))) THEN
   IS_MONOTONE = (DF0  .EQ. 0.0_R8) .AND. (DF1  .EQ. 0.0_R8) .AND. &
                 (DDF0 .EQ. 0.0_R8) .AND. (DDF1 .EQ. 0.0_R8)
   RETURN
END IF
! Identify the direction of change of the function (increasing or decreasing).
IF (F1 .GT. F0) THEN
   SIGN =  1.0_R8
ELSE
   SIGN = -1.0_R8
END IF
W = U1 - U0
! Determine which set of monotonicity conditions to use based on the
! assigned first derivative values at either end of the interval.
IF ((ABS(DF0) .LT. EPS) .OR. (ABS(DF1) .LT. EPS)) THEN
   ! Simplified monotone case, which reduces to a test of cubic
   ! positivity studied in
   ! 
   ! J. W. Schmidt and W. He{\ss}, ``Positivity of cubic polynomials on
   ! intervals and positive spline interpolation'', {\sl BIT Numerical
   ! Mathematics}, 28 (1988) 340--352.
   ! 
   ! Notably, monotonicity results when the following conditions hold:
   !    alpha >= 0,
   !    delta >= 0,
   !    beta  >= alpha - 2 * sqrt{ alpha delta },
   !    gamma >= delta - 2 * sqrt{ alpha delta },
   ! 
   ! where alpha, beta, delta, and gamma are defined in the paper. The
   ! equations that follow are the result of algebraic simplifications
   ! of the terms as they are defined by Schmidt and He{\ss}.
   ! 
   ! The condition delta >= 0 was enforced when first estimating
   ! derivative values (with correct sign). Next check for alpha >= 0.
   IF (SIGN*DDF1*W .GT. SIGN*4.0_R8*DF1) THEN
      IS_MONOTONE = .FALSE.
   ELSE
      ! Compute a simplification of 2 * sqrt{ alpha delta }.
      TEMP = DF0 * (4*DF1 - DDF1*W)
      IF (TEMP .GT. 0.0_R8) TEMP = 2.0_R8 * SQRT(TEMP)
      ! Check for gamma >= delta - 2 * sqrt{ alpha delta }
      IF (TEMP + SIGN*(3.0_R8*DF0 + DDF0*W) .LT. 0.0_R8) THEN
         IS_MONOTONE = .FALSE.
      ! Check for beta >= alpha - 2 * sqrt{ alpha delta }
      ELSE IF (60.0_R8*(F1-F0)*SIGN - W*(SIGN*(24.0_R8*DF0 + 32.0_R8*DF1) &
           - 2.0_R8*TEMP + W*SIGN*(3.0_R8*DDF0 - 5.0_R8*DDF1)) .LT. 0.0_R8) THEN
         IS_MONOTONE = .FALSE.
      ELSE
         IS_MONOTONE = .TRUE.
      END IF
   END IF
ELSE
   ! Full quintic monotonicity case related to the theory in
   !
   ! G. Ulrich and L. T. Watson, ``Positivity conditions for quartic
   ! polynomials'', {\sl SIAM J. Sci. Comput.}, 15 (1994) 528--544.
   ! 
   ! Monotonicity results when the following conditions hold:
   !    tau_1 > 0,
   !    if (beta <= 6) then
   !       alpha > -(beta + 2) / 2,
   !       gamma > -(beta + 2) / 2,
   !    else
   !       alpha > -2 sqrt(beta - 2),
   !       gamma > -2 sqrt(beta - 2),
   !    end if
   ! 
   ! where alpha, beta, gamma, and tau_1 are defined in the paper. The
   ! following conditions are the result of algebraic simplifications
   ! of the terms as defined by Ulrich and Watson.
   ! 
   ! Check for tau_1 > 0.
   IF (W*(2.0_R8*SQRT(DF0*DF1) - SIGN * 3.0_R8*(DF0+DF1)) - &
        SIGN*24.0_R8*(F0-F1) .LE. 0.0_R8) THEN
      IS_MONOTONE = .FALSE.
   ELSE
      ! Compute alpha, gamma, beta from theorems to determine monotonicity.
      TEMP = (DF0*DF1)**(0.75_R8)
      ALPHA = SIGN * (4.0_R8*DF1 - DDF1*W) * SQRT(SIGN*DF0) / TEMP
      GAMMA = SIGN * (4.0_R8*DF0 + DDF0*W) * SQRT(SIGN*DF1) / TEMP
      BETA = SIGN * (3.0_R8 * ((DDF1-DDF0)*W - 8.0_R8*(DF0+DF1)) + &
           60.0_R8 * (F1-F0) / W) / (2.0_R8 * SQRT(DF0*DF1))
      IF (BETA .LE. 6.0_R8) THEN
         TEMP = -(BETA + 2.0_R8) / 2.0_R8
      ELSE
         TEMP = -2.0_R8 * SQRT(BETA - 2.0_R8)
      END IF
      IS_MONOTONE = (ALPHA .GT. TEMP) .AND. (GAMMA .GT. TEMP)
   END IF
END IF
END FUNCTION IS_MONOTONE

SUBROUTINE QUADRATIC(X, Y, I2, A, B)
! Given data X, Y, an index I2, compute the coefficients A of x^2 and B
! of x for the quadratic interpolating Y(I2-1:I2+1) at X(I2-1:I2+1).
REAL(KIND=R8), INTENT(IN),  DIMENSION(:) :: X, Y
INTEGER, INTENT(IN) :: I2
REAL(KIND=R8), INTENT(OUT) :: A, B
! Local variables.
REAL(KIND=R8) :: C1, C2, C3, & ! Intermediate terms for computation.
  D ! Denominator for computing A and B via Cramer's rule.
INTEGER :: I1, I3
I1 = I2-1
I3 = I2+1
! The earlier tests for extreme data (X,Y) keep the quantities below 
! within floating point range. Compute the shared denominator.
D = (X(I1) - X(I2)) * (X(I1) - X(I3)) * (X(I2) - X(I3))
! Compute coefficients A and B in quadratic interpolant  Ax^2 + Bx + C.
C1 = X(I1) * (Y(I3) - Y(I2))
C2 = X(I2) * (Y(I1) - Y(I3))
C3 = X(I3) * (Y(I2) - Y(I1))
A = (C1 + C2 + C3) / D
B = - (X(I1)*C1 + X(I2)*C2 + X(I3)*C3) / D
END SUBROUTINE QUADRATIC

END SUBROUTINE MQSI
