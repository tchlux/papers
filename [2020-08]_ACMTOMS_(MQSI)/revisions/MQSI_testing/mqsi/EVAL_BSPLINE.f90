! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
! 
USE REAL_PRECISION, ONLY: R8
IMPLICIT NONE
! Arguments.
REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: T
REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
INTEGER, INTENT(IN), OPTIONAL :: D
! Local variables.
!  Iteration variables.
INTEGER :: I, J
!  Derivative being evaluated, order of B-spline K, one less than the 
!  order L, and number of knots defining the B-spline N.
INTEGER :: DERIV, K, L, N
!  Evaluations of T constituent B-splines (columns) at all points (rows).
REAL(KIND=R8), DIMENSION(SIZE(XY), SIZE(T)) :: BIATX
!  Temporary storage for compute denominators LEFT, RIGHT, and last knot TN.
REAL(KIND=R8) :: LEFT, RIGHT, TN
! Assign the local value of the optional derivative "D" argument.
IF (PRESENT(D)) THEN
   DERIV = D
ELSE
   DERIV = 0
END IF
! Set local variables that are used for notational convenience.
N = SIZE(T) ! Number of knots.
K = N - 1 ! Order of B-spline.
L = K - 1 ! One less than the order of the B-spline.
TN = T(N) ! Value of the last knot, T(N).

! If this is a large enough derivative, we know it is zero everywhere.
IF (DERIV+1 .GE. N) THEN
   XY(:) = 0.0_R8
   RETURN

! ---------------- Performing standard evaluation ------------------
! This is a standard B-spline with multiple unique knots, right continuous.
ELSE
   ! Initialize all values to 0.
   BIATX(:,:) = 0.0_R8
   ! Assign the first value for each knot index.
   first_b_spline : DO J = 1, K
      IF (T(J) .EQ. T(J+1)) CYCLE first_b_spline
      ! Compute all right continuous order 1 B-spline values.
      WHERE ( (T(J) .LE. XY(:)) .AND. (XY(:) .LT. T(J+1)) )
         BIATX(:,J) = 1.0_R8
      END WHERE
   END DO first_b_spline
END IF
! Compute the remainder of B-spline by building up from the order one B-splines.
! Omit the final steps of this computation for derivatives.
compute_spline : DO I = 2, K-MAX(DERIV,0)
   ! Cycle over each knot accumulating the values for the recurrence.
   DO J = 1, N - I
      ! Check divisors, intervals with 0 width add 0 value to the B-spline.
      LEFT = (T(J+I-1) - T(J))
      RIGHT = (T(J+I) - T(J+1))
      ! Compute the B-spline recurrence relation (cases based on divisor).
      IF (LEFT .GT. 0) THEN
         IF (RIGHT .GT. 0) THEN
            BIATX(:,J) = &
                 ((XY(:) - T(J)) / LEFT) * BIATX(:,J) + &
                 ((T(J+I) - XY(:)) / RIGHT) * BIATX(:,J+1)
         ELSE
            BIATX(:,J) = ((XY(:) - T(J)) / LEFT) * BIATX(:,J)
         END IF
      ELSE
         IF (RIGHT .GT. 0) THEN
            BIATX(:,J) = ((T(J+I) - XY(:)) / RIGHT) * BIATX(:,J+1)
         END IF
      END IF
   END DO
END DO compute_spline

! -------------------- Performing integration ----------------------
int_or_diff : IF (DERIV .LT. 0) THEN
! Integrals will be 1 for TN <= X(J) < infinity.
WHERE (TN .LE. XY(:))
   BIATX(:,N) = 1.0_R8
END WHERE
! Currently the first B-spline in BIATX(:,1) has full order and covers
! all knots, but each following B-spline spans fewer knots (having
! lower order). This loop starts at the back of BIATX at the far right
! (order 1) constituent B-spline, raising the order of all constituent
! B-splines to match the order of the first by using the standard
! forward evaluation.
raise_order : DO I = 1, L
   DO J = N-I, K
      LEFT = (TN - T(J))
      RIGHT = (TN - T(J+1))
      IF (LEFT .GT. 0) THEN
         IF (RIGHT .GT. 0) THEN
            BIATX(:,J) = &
                 ((XY(:) - T(J)) / LEFT) * BIATX(:,J) + &
                 ((TN - XY(:)) / RIGHT) * BIATX(:,J+1)
         ELSE
            BIATX(:,J) = ((XY(:) - T(J)) / LEFT) * BIATX(:,J)
         END IF
      ELSE
         IF (RIGHT .GT. 0) THEN
            BIATX(:,J) = ((TN - XY(:)) / RIGHT) * BIATX(:,J+1)
         END IF
      END IF
   END DO
END DO raise_order

! Compute the integral of the B-spline.
! Do a forward evaluation of all constituents.
DO J = 1, K
   LEFT = (TN - T(J))
   RIGHT = (TN - T(J+1))
   IF (LEFT .GT. 0) THEN
      IF (RIGHT .GT. 0) THEN
         BIATX(:,J) = &
              ((XY(:) - T(J)) / LEFT) * BIATX(:,J) + &
              ((TN - XY(:)) / RIGHT) * BIATX(:,J+1)
      ELSE
         BIATX(:,J) = ((XY(:) - T(J)) / LEFT) * BIATX(:,J)
      END IF
   ELSE
      IF (RIGHT .GT. 0) THEN
         BIATX(:,J) = ((TN - XY(:)) / RIGHT) * BIATX(:,J+1)
      END IF
   END IF
END DO
! Sum the constituent functions at each knot (from the back).
DO J = K, 1, -1
   BIATX(:,J) = (BIATX(:,J) + BIATX(:,J+1))
END DO
! Divide by the degree plus one.
BIATX(:,1) = BIATX(:,1) / REAL(L+1,R8)
! Rescale the integral by its width.
BIATX(:,1) = BIATX(:,1) * (TN - T(1))

! ------------------ Performing differentiation --------------------
ELSE IF (DERIV .GT. 0) THEN
! Compute a derivative of the B-spline (if D > 0).
compute_derivative : DO J = N-DERIV, K
   ! Cycle over each knot just as for standard evaluation, however
   ! instead of using the recurrence relation for evaluating the value
   ! of the B-spline, use the recurrence relation for computing the
   ! value of the derivative of the B-spline.
   DO I = 1, N-J
      ! Assure that the divisor will not cause invalid computations.
      LEFT = (T(I+J-1) - T(I))
      RIGHT = (T(I+J) - T(I+1))
      ! Compute the derivative recurrence relation.
      IF (LEFT .GT. 0) THEN
        IF (RIGHT .GT. 0) THEN
          BIATX(:,I) = REAL(J-1,R8)*(BIATX(:,I)/LEFT-BIATX(:,I+1)/RIGHT)
        ELSE
          BIATX(:,I) = REAL(J-1,R8)*(BIATX(:,I)/LEFT)
        END IF
      ELSE
         IF (RIGHT .GT. 0) THEN
            BIATX(:,I) = REAL(J-1,R8)*(-BIATX(:,I+1)/RIGHT)
         END IF
      END IF
   END DO
END DO compute_derivative
END IF int_or_diff

! Assign the values to the output XY.
XY(:) = BIATX(:,1)

END SUBROUTINE EVAL_BSPLINE
