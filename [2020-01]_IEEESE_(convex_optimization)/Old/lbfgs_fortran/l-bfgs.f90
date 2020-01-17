
! R8 data type from Hompack.
MODULE REAL_PRECISION
INTEGER, PARAMETER:: R8=SELECTED_REAL_KIND(13)
END MODULE REAL_PRECISION

! Module containing LBFGS and interfaces.
MODULE LBFGS_MOD
USE REAL_PRECISION

! Interface for gradient.
INTERFACE
   SUBROUTINE GRAD_INT(X, G)
      USE REAL_PRECISION, ONLY : R8
      REAL(KIND=R8), INTENT(IN) :: X(:)
      REAL(KIND=R8), INTENT(OUT) :: G(SIZE(X))
   END SUBROUTINE GRAD_INT
END INTERFACE

CONTAINS

SUBROUTINE LBFGS(D, GRAD, X, IERR, LIM, IBUDGET, ALPHA, EPS, HISTORY)
! The subroutine minimizes a function f(x) using the L-BFGS algorithm.
!
! On input:
!
! D is an integer specifying the dimension of the input space.
!
! GRAD is a subroutine such that E[ GRAD(x) ] = \nabla f(x).
!    The interface for GRAD must match GRAD_INT.
!
! X(1:D) is a real valued vector containing the starting point.
!
!
! On output:
!
! X(:) contains the solution point.
!
! IERR is an integer valued error flag. 0 indicates a successful run.
!
!
! Optional parameters:
!
! In input, LIM is an integer specifying the memory limit for L-BFGS.
!    By default, LIM = CEIL(SQRT(D)).
!
! On input, IBUDGET is a budget for evaluations of G. By default, IBUDGET
!    is 1000.
!
! On input, ALPHA is the learning rate or rescale factor. By default,
!    ALPHA = 1.0.
!
! On input, EPS is a nonzero stopping tolerance. EPS must be nonzero to
!    prevent numerical instability. By default, EPS is the unit roundoff.
!
! When present, HISTORY(1:D,1:IBUDGET) is a real valued matrix. On output,
!    the columns of HISTORY contain the iterates.
!
IMPLICIT NONE
! Input/output parameter list.
INTEGER, INTENT(IN) :: D
PROCEDURE(GRAD_INT) :: GRAD
REAL(KIND=R8), INTENT(INOUT) :: X(D)
INTEGER, INTENT(OUT) :: IERR
! Optional inputs.
INTEGER, OPTIONAL, INTENT(IN) :: LIM
INTEGER, OPTIONAL, INTENT(IN) :: IBUDGET
REAL(KIND=R8), OPTIONAL, INTENT(IN) :: ALPHA, EPS
REAL(KIND=R8), ALLOCATABLE, OPTIONAL, INTENT(OUT) :: HISTORY(:,:)
! Local variables.
INTEGER :: IBUDGETL, LIML
INTEGER :: I, J, K
REAL(KIND=R8) :: ALPHAL, EPSL, Y1_DOT_S1
REAL(KIND=R8) :: G(D)
REAL(KIND=R8) :: OLD_X(D), OLD_G(D)
REAL(KIND=R8), ALLOCATABLE :: A(:), B(:), RHO(:), S(:,:), Y(:,:)

! Assign optional inputs.
LIML = CEILING(SQRT(REAL(D)))
IF (PRESENT(LIM)) THEN
   IF (LIML .LE. 0) THEN
      IERR = -1; RETURN; END IF
   LIML = LIM
END IF
IBUDGETL = 1000
IF (PRESENT(IBUDGET)) THEN
   IF (IBUDGET .LE. 0) THEN
      IERR = -1; RETURN; END IF
   IBUDGETL = IBUDGET
END IF
ALPHAL = 1.0_R8
EPSL = EPSILON(0.0_R8)
IF (PRESENT(EPS)) THEN
   IF (EPS .LE. 0.0_R8) THEN
      IERR = -1; RETURN; END IF
   EPSL = EPS
END IF
IF (PRESENT(ALPHA)) THEN
   IF (ALPHA .LE. SQRT(EPSL)) THEN
      IERR = -1; RETURN; END IF
   ALPHAL = ALPHA
END IF
IF (PRESENT(HISTORY)) THEN
   ALLOCATE(HISTORY(D, IBUDGETL), STAT=IERR)
   IF (IERR .NE. 0) RETURN
   HISTORY(:,:) = 0.0_R8
END IF
! Allocate the limited memory.
ALLOCATE(RHO(LIML), S(D, LIML), Y(D, LIML), A(LIML), B(LIML), STAT=IERR)
IF (IERR .NE. 0) RETURN
S(:,:) = 0.0_R8
Y(:,:) = 0.0_R8

! Begin the iteration.
DO I = 1, IBUDGETL
   ! Store the current iterate.
   IF (PRESENT(HISTORY)) THEN
      HISTORY(:,I) = X(:)
   END IF
   ! In the first iteration, do a simple update.
   IF (I .EQ. 1) THEN
      CALL GRAD(X, G)
      OLD_X = X; OLD_G = G
      X(:) = X(:) - ALPHAL * G(:)
      CYCLE
   END IF
   ! Otherwise, compute the L-BFGS update.
   ! Evaluate the current iterate.
   CALL GRAD(X, G)
   ! Update the history arrays.
   S = CSHIFT(S, -1, DIM=2)
   Y = CSHIFT(Y, -1, DIM=2)
   RHO = CSHIFT(RHO, -2)
   S(:,1) = X(:) - OLD_X(:)
   Y(:,1) = G(:) - OLD_G(:)
   Y1_DOT_S1 = DOT_PRODUCT(Y(:,1), S(:,1))
   IF (SQRT(ABS(Y1_DOT_S1)) .LT. EPSL) THEN
      IERR = I; RETURN; END IF
   RHO(1) = 1.0_R8 / Y1_DOT_S1
   ! Copy current iterates for next iteration.
   OLD_X = X; OLD_G = G
   ! Reconstruct the BFGS update.
   DO J = 1, MIN(LIML,I-1)
      A(J) = RHO(J) * DOT_PRODUCT(S(:,J), G(:))
      G(:) = G(:) - A(J) * Y(:,J)
   END DO
   G(:) = (Y1_DOT_S1 / DOT_PRODUCT(Y(:,1), Y(:,1))) * G(:)
   DO J = MIN(LIML,I-1), 1, -1
      B(J) = RHO(J) * DOT_PRODUCT(Y(:,J), G(:))
      G(:) = G(:) + S(:,J) * (A(J) - B(J))
   END DO
   ! Now compute the rescaled update.
   X(:) = X(:) - ALPHAL * G(:)
END DO
RETURN
END SUBROUTINE LBFGS

END MODULE LBFGS_MOD
