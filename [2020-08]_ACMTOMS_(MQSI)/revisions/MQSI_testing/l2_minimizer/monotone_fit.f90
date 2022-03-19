MODULE REAL_PRECISION  ! module for 64-bit real arithmetic
  INTEGER, PARAMETER:: R8=SELECTED_REAL_KIND(13)
END MODULE REAL_PRECISION

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


! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
USE REAL_PRECISION, ONLY: R8
IMPLICIT NONE
! Arguments.
REAL(KIND=R8), INTENT(IN),  DIMENSION(:)   :: XI
REAL(KIND=R8), INTENT(IN),  DIMENSION(:,:) :: FX
REAL(KIND=R8), INTENT(OUT), DIMENSION(:)   :: T, BCOEF
INTEGER, INTENT(OUT) :: INFO
! Local variables.
!  LAPACK pivot indices.
INTEGER, DIMENSION(SIZE(BCOEF)) :: IPIV
!  Storage for linear system that is solved to get B-spline coefficients.
REAL(KIND=R8), DIMENSION(1 + 3*(2*SIZE(FX,2)-1), SIZE(FX)) :: AB
!  Maximum allowed (relative) error in spline function values.
REAL(KIND=R8), PARAMETER :: MAX_ERROR = SQRT(SQRT(EPSILON(1.0_R8)))
INTEGER :: DEGREE, & ! Degree of B-spline.
     DERIV, & ! Derivative loop index.
     I, I1, I2, J, J1, J2, & ! Miscellaneous loop indices.
     K, &    ! Order of B-splines = 2*NCC.
     NB, &   ! Number of breakpoints.
     NCC, &  ! Number of continuity conditions.
     NK, &   ! Number of knots = NSPL + 2*NCC.
     NSPL    ! Dimension of spline space = number of B-spline
             ! coefficients = NB * NCC.
! LAPACK subroutine for solving banded linear systems.
EXTERNAL :: DGBSV
INTERFACE
   SUBROUTINE EVAL_BSPLINE(T, XY, D)
     USE REAL_PRECISION, ONLY: R8
     REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: T
     REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
     INTEGER, INTENT(IN), OPTIONAL :: D
   END SUBROUTINE EVAL_BSPLINE
   SUBROUTINE EVAL_SPLINE(T, BCOEF, XY, INFO, D)
     USE REAL_PRECISION, ONLY: R8
     REAL(KIND=R8), INTENT(IN), DIMENSION(:) :: T, BCOEF
     REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
     INTEGER, INTENT(OUT) :: INFO
     INTEGER, INTENT(IN), OPTIONAL :: D
   END SUBROUTINE EVAL_SPLINE
END INTERFACE

! Define some local variables for notational convenience.
NB = SIZE(XI)
NCC = SIZE(FX,2)
NSPL = SIZE(FX)
NK = NSPL + 2*NCC
K = 2*NCC
DEGREE = K - 1
INFO = 0

! Check the shape of incoming arrays.
IF      (NB .LT. 3)             THEN; INFO = 1; RETURN
ELSE IF (SIZE(FX,1) .NE. NB)    THEN; INFO = 3; RETURN
ELSE IF (SIZE(T) .LT. NK)       THEN; INFO = 4; RETURN
ELSE IF (SIZE(BCOEF) .LT. NSPL) THEN; INFO = 5; RETURN
END IF
! Verify that breakpoints are increasing.
DO I = 1, NB - 1
   IF (XI(I) .GE. XI(I+1)) THEN
      INFO = 6
      RETURN
   END IF
END DO

! Copy the knots that will define the B-spline representation.
! Each internal knot will be repeated NCC times to maintain the
! necessary level of continuity for this spline.
T(1:K) = XI(1)
DO I = 2, NB-1
   T(I*NCC+1 : (I+1)*NCC) = XI(I)
END DO
! Assign the last knot to exist a small step outside the supported
! interval to ensure the B-spline basis functions are nonzero at the
! rightmost breakpoint.
T(NK-DEGREE:NK) = MAX( XI(NB) + ABS(XI(NB))*SQRT(SQRT(EPSILON(XI(NB)))),  &
                       XI(NB) + SQRT(SQRT(EPSILON(XI(NB)))) )

! The next block of code evaluates each B-spline and it's derivatives
! at all breakpoints. The first and last elements of XI will be
! repeated K times and each internal breakpoint will be repeated NCC
! times. As a result, the support of each B-spline spans at most three
! breakpoints. The coefficients for the B-spline basis are determined
! by solving a linear system with NSPL columns (one column for each
! B-spline) and NB*NCC rows (one row for each value of the
! interpolating spline).
! 
! For example, a C^1 interpolating spline over three breakpoints
! will match function value and first derivative at each breakpoint
! requiring six fourth order (third degree) B-splines each composed
! from five knots. Below, the six B-splines are numbered (first
! number, columns) and may be nonzero at the three breakpoints
! (middle letter, rows) for each function value (odd rows, terms end
! with 0) and first derivative (even rows, terms end with 1). The
! linear system will look like:
! 
!       B-SPLINE VALUES AT BREAKPOINTS      SPLINE          VALUES
!        1st  2nd  3rd  4th  5th  6th    COEFFICIENTS
!      _                              _     _   _           _    _ 
!     |                                |   |     |         |      |
!   B |  1a0  2a0  3a0  4a0            |   |  1  |         |  a0  |
!   R |  1a1  2a1  3a1  4a1            |   |  2  |         |  a1  |
!   E |  1b0  2b0  3b0  4b0  5b0  6b0  |   |  3  |   ===   |  b0  |
!   A |  1b1  2b1  3b1  4b1  5b1  6b1  |   |  4  |   ===   |  b1  |
!   K |            3c0  4c0  5c0  6c0  |   |  5  |         |  c0  |
!   S |            3c1  4c1  5c1  6c1  |   |  6  |         |  c1  |
!     |_                              _|   |_   _|         |_    _|
!   
! Notice this matrix is banded with lower/upper bandwidths KL/KU equal
! to (one less than the maximum number of breakpoints for which a
! spline takes on a nonzero value) times (the number of continuity
! conditions) minus (one). In general KL = KU = DEGREE = K - 1.

! Initialize all values in AB to zero.
AB(:,:) = 0.0_R8
! Evaluate all B-splines at all breakpoints (walking through columns).
DO I = 1, NSPL
   ! Compute index of the last knot for the current B-spline.
   J = I + K
   ! Compute the row indices in the coefficient matrix A.
   I1 = ((I-1)/NCC - 1) * NCC + 1 ! First row.
   I2 = I1 + 3*NCC - 1            ! Last row.
   ! Only two breakpoints will be covered for the first NCC
   ! B-splines and the last NCC B-splines.
   IF (I .LE. NCC)      I1 = I1 + NCC
   IF (I+NCC .GT. NSPL) I2 = I2 - NCC
   ! Compute the indices of the involved breakpoints.
   J1 = I1 / NCC + 1 ! First breakpoint.
   J2 = I2 / NCC     ! Last breakpoint.
   ! Convert the i,j indices in A to the banded storage scheme in AB.
   ! The mapping looks like   A[i,j] --> AB[KL+KU+1+i-j,j] .
   I1 = 2*DEGREE+1 + I1 - I
   I2 = 2*DEGREE+1 + I2 - I
   ! Evaluate this B-spline, computing function value and derivatives.
   DO DERIV = 0, NCC-1
      ! Place the evaluations into a subsection of a column in AB,
      ! shift according to which derivative is being evaluated and use
      ! a stride determined by the number of continuity conditions.
      AB(I1+DERIV:I2:NCC,I) = XI(J1:J2)
      CALL EVAL_BSPLINE(T(I:J), AB(I1+DERIV:I2:NCC,I), D=DERIV)
   END DO
END DO
! Copy the FX into the BCOEF (output) variable.
DO I = 1, NCC ; BCOEF(I:NSPL:NCC) = FX(:,I) ; END DO
! Call the LAPACK subroutine to solve the banded linear system.
CALL DGBSV(NSPL, DEGREE, DEGREE, 1, AB, SIZE(AB,1), IPIV, BCOEF, NSPL, INFO)
! Check for errors in the execution of DGBSV, (this should not happen).
IF (INFO .NE. 0) THEN; INFO = INFO + 20; RETURN; END IF
! Check to see if the linear system was correctly solved by looking at
! the difference between produced B-spline values and provided values.
DO DERIV = 0, NCC-1
   DO I = 1, NB
      ! Compute the indices of the first and last B-splines that have a knot
      ! at this breakpoint.
      I1 = MAX(1, (I-2)*NCC+1)
      I2 = MIN(NB*NCC, (I+1)*NCC)
      ! Evaluate this spline at this breakpoint. Correct usage is
      ! enforced here, so it is expected that INFO=0 always.
      AB(1,1) = XI(I)
      CALL EVAL_SPLINE(T(I1:I2+2*NCC), BCOEF(I1:I2), AB(1,1:1), INFO, D=DERIV)
      ! Check the precision of the reproduced value.
      ! Return an error if the precision is too low.
      IF (ABS( (AB(1,1) - FX(I,DERIV+1)) / &
           (1.0_R8 + ABS(FX(I,DERIV+1))) ) .GT. MAX_ERROR) THEN
         INFO = 7; RETURN
      END IF
   END DO
END DO

END SUBROUTINE FIT_SPLINE


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
USE REAL_PRECISION, ONLY: R8
IMPLICIT NONE
! Arguments.
REAL(KIND=R8), INTENT(IN), DIMENSION(:) :: T, BCOEF
REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
INTEGER, INTENT(OUT) :: INFO
INTEGER, INTENT(IN), OPTIONAL :: D
! Local variables.
INTEGER :: DERIV, I, I1, I2, ITEMP, J, K, NB, NCC, NK, NSPL
REAL(KIND=R8), DIMENSION(SIZE(T)-SIZE(BCOEF)) :: BIATX
INTERFACE
   SUBROUTINE EVAL_BSPLINE(T, XY, D)
     USE REAL_PRECISION, ONLY: R8
     REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: T
     REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
     INTEGER, INTENT(IN), OPTIONAL :: D
   END SUBROUTINE EVAL_BSPLINE
END INTERFACE
NK = SIZE(T) ! Number of knots.
NSPL = SIZE(BCOEF) ! Number of spline basis functions (B-splines).
! Check for size-related errors.
IF ( ((NK - NSPL)/2 .LT. 1) .OR. (MOD(NK - NSPL, 2) .NE. 0) )&
  THEN; INFO = 1; RETURN; ENDIF
! Compute the order for each B-spline (number of knots per B-spline minus one).
K = NK - NSPL ! = 2*NCC, where NCC = number of continuity conditions at each
   ! breakpoint.
NCC = K/2
NB = NSPL/NCC ! Number of breakpoints.
! Check for nondecreasing knot sequence.
DO I = 1, NK-1
   IF (T(I) .GT. T(I+1)) THEN; INFO = 2; RETURN; END IF
END DO
! Check for valid knot sequence for splines of order K.
DO I = 1, NK-K
   IF (T(I) .EQ. T(I+K)) THEN; INFO = 2; RETURN; END IF
END DO
! Assign the local value of the optional derivative argument D.
IF (PRESENT(D)) THEN; DERIV = D;  ELSE; DERIV = 0; END IF

! In the following code, I1 (I2, respectively) is the smallest (largest,
! respectively) index of a B-spline B_{I1}(.) (B_{I2}(.), respectively)
! whose support contains XY(I).  I1 = I2 + 1 - K .
!
! Initialize the indices I1 and I2 before looping.
I1 = 1; I2 = K ! For evaluation points in first breakpoint interval.
! Evaluate all the B-splines that have support at each point XY(I) in XY(:).
evaluate_at_x : DO I = 1, SIZE(XY)
   ! Return zero for points that are outside the spline's support.
   IF ((XY(I) .LT. T(1)) .OR. (XY(I) .GE. T(NK))) THEN
      XY(I) = 0.0_R8
      CYCLE evaluate_at_x
   ELSE IF ( (T(I2) .LE. XY(I)) .AND. (XY(I) .LT. T(I2+1)) ) THEN
      CONTINUE
   ELSE IF ( (I2 .NE. NB*NCC) .AND. (T(I2+NCC) .LE. XY(I)) .AND. &
      (XY(I) .LT. T(I2+NCC+1)) ) THEN
      I1 = I1 + NCC; I2 = I2 + NCC
   ELSE ! Find breakpoint interval containing XY(I) using a bisection method
        ! on the breakpoint indices.
      I1 = 1; I2 = NB
      DO WHILE (I2 - I1 .GT. 1)
         J = (I1+I2)/2 ! Breakpoint J = knot T((J+1)*NCC).
         IF ( T((J+1)*NCC) .LE. XY(I) ) THEN; I1 = J; ELSE; I2 = J; END IF
      END DO ! Now I2 = I1 + 1, and XY(I) lies in the breakpoint interval
             ! [ T((I1+1)*NCC), T((I1+2)*NCC) ).
      I2 = (I1 + 1)*NCC ! Spline index = knot index.
      I1 = I2 + 1 - K ! The index range of B-splines with support containing
                      ! XY(I) is I1 to I2.
   END IF
   ! Store only the single X value that is relevant to this iteration.
   BIATX(1:K) = XY(I) ! K = I2-I1+1.
   ! Compute the values of selected B-splines.
   ITEMP = 0
   DO J = I1, I2
      ITEMP = ITEMP + 1
      CALL EVAL_BSPLINE(T(J:J+K), BIATX(ITEMP:ITEMP), D=DERIV)
   END DO
   ! Evaluate spline interpolant at XY(I) as a linear combination of B-spline
   ! values, returning values in XY(:).
   XY(I) = DOT_PRODUCT(BIATX(1:K), BCOEF(I1:I2))
END DO evaluate_at_x
INFO = 0  ! Set INFO to indicate successful execution.
END SUBROUTINE EVAL_SPLINE


! Prouce a monotone fit to data.
SUBROUTINE MONOTONE_FIT(X, Y, C, STEPS, MSTEPS, STEP_SIZE, &
     MONOTONICITY_MULTIPLIER, FX, T, BCOEF, INFO)
  USE REAL_PRECISION, ONLY: R8
  IMPLICIT NONE
  ! Arguments.
  REAL(KIND=R8), INTENT(IN),  DIMENSION(:) :: X, Y
  INTEGER, INTENT(IN) :: C, STEPS, MSTEPS
  REAL(KIND=R8), INTENT(INOUT) :: STEP_SIZE, MONOTONICITY_MULTIPLIER
  REAL(KIND=R8), INTENT(OUT), DIMENSION(SIZE(X),C) :: FX
  REAL(KIND=R8), INTENT(OUT), DIMENSION(SIZE(X)*C+2*C) :: T
  REAL(KIND=R8), INTENT(OUT), DIMENSION(SIZE(X)*C) :: BCOEF
  INTEGER, INTENT(OUT) :: INFO
  ! Interfaces.
  INTERFACE
     SUBROUTINE EVAL_BSPLINE(T, XY, D)
       USE REAL_PRECISION, ONLY: R8
       REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: T
       REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
       INTEGER, INTENT(IN), OPTIONAL :: D
     END SUBROUTINE EVAL_BSPLINE
     SUBROUTINE FIT_SPLINE(XI, FX, T, BCOEF, INFO)
       USE REAL_PRECISION, ONLY: R8
       REAL(KIND=R8), INTENT(IN),  DIMENSION(:)   :: XI
       REAL(KIND=R8), INTENT(IN),  DIMENSION(:,:) :: FX
       REAL(KIND=R8), INTENT(OUT), DIMENSION(:)   :: T, BCOEF
       INTEGER, INTENT(OUT) :: INFO
     END SUBROUTINE FIT_SPLINE
     SUBROUTINE EVAL_SPLINE(T, BCOEF, XY, INFO, D)
       USE REAL_PRECISION, ONLY: R8
       REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: T, BCOEF
       REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
       INTEGER, INTENT(OUT) :: INFO
       INTEGER, INTENT(IN), OPTIONAL :: D
     END SUBROUTINE EVAL_SPLINE
  END INTERFACE
  ! Internal variables.
  REAL(KIND=R8), DIMENSION(1000) :: XY, DIRECTION, XY_TEMP
  REAL(KIND=R8), DIMENSION(SIZE(X),SIZE(BCOEF)) :: X_B_VALS
  REAL(KIND=R8), DIMENSION(SIZE(X)) :: X_F_VALS
  REAL(KIND=R8), DIMENSION(SIZE(XY),SIZE(BCOEF),3) :: XY_B_VALS
  REAL(KIND=R8), DIMENSION(SIZE(XY),3) :: XY_F_VALS
  REAL(KIND=R8), DIMENSION(SIZE(BCOEF)) :: C_STEP, C_TEMP, C_GRAD, C_CURV
  REAL(KIND=R8) :: D1_2NORM, SHIFT
  INTEGER :: I, J, NK, NSPL, K, STEP
  ! Monotonicity multiplier shift amount.
  SHIFT = MONOTONICITY_MULTIPLIER / 2.0_R8
  ! Number of knots.
  NK = SIZE(T) 
  ! Number of spline basis functions (B-splines).
  NSPL = SIZE(BCOEF)
  ! Compute the order for each B-spline (number of knots per B-spline minus one).
  K = NK - NSPL ! = 2*NCC, where NCC = number of continuity conditions at each breakpoint.
  ! Initialize the values to be zero slope and derivative at all points.
  FX(:,1) = Y(:)
  FX(:,2:) = 0.0_R8
  ! Fit the initial spline (we will minimize the 2nd derivative 2-norm).
  CALL FIT_SPLINE(X, FX, T, BCOEF, INFO)
  ! Assign all the evaluation points (equally spaced over X interval).
  DO I = 1, SIZE(XY)
     XY(I) = MINVAL(X(:)) + (MAXVAL(X(:)) - MINVAL(X(:))) * &
          (REAL(I-1, R8) / REAL(SIZE(XY)-1, R8))
  END DO
  ! Assign the direction of all evaluation points.
  DO I = 1, SIZE(Y)-1
     WHERE ((XY(:) .GE. X(I)) .AND. (XY(:) .LT. X(I+1)))
        DIRECTION(:) = Y(I+1) - Y(I)
     END WHERE
  END DO
  WHERE (ABS(DIRECTION(:)) .GT. 0.0_R8)
     DIRECTION(:) = DIRECTION(:) / ABS(DIRECTION(:))
  END WHERE
  ! Store the B-spline basis function values at all data points.
  DO I = 1, SIZE(BCOEF)
     X_B_VALS(:,I) = X(:)
     CALL EVAL_BSPLINE(T(I:I+K), X_B_VALS(:,I), D=0)
  END DO
  ! Store the B-spline basis function values at all test points
  !   and evaluate all the B-spline basis function values.
  DO I = 1, SIZE(BCOEF)
     XY_B_VALS(:,I,1) = XY(:)
     XY_B_VALS(:,I,2) = XY(:)
     XY_B_VALS(:,I,3) = XY(:)
     CALL EVAL_BSPLINE(T(I:I+K), XY_B_VALS(:,I,1), D=0)
     CALL EVAL_BSPLINE(T(I:I+K), XY_B_VALS(:,I,2), D=1)
     CALL EVAL_BSPLINE(T(I:I+K), XY_B_VALS(:,I,3), D=2)
  END DO

  ! Initial step size and multiplier for the monotonicity constraints.
  C_GRAD(:) = 0.0_R8
  C_CURV(:) = 1.0_R8
  D1_2NORM = HUGE(1.0_R8)
  BCOEF(:) = 0.0_R8

  ! Do the gradient descent.
  DO STEP = 1, STEPS + MSTEPS
     ! Get the current function values and initialize the step.
     XY_F_VALS(:,1) = MATMUL(XY_B_VALS(:,:,1), BCOEF(:))
     XY_F_VALS(:,2) = MATMUL(XY_B_VALS(:,:,2), BCOEF(:))
     XY_F_VALS(:,3) = MATMUL(XY_B_VALS(:,:,3), BCOEF(:))
     C_STEP(:) = 0.0_R8
     ! 
     ! The gradient points all these values towards zero, so all basis
     !   functions that are nonzero should be proportionally pushed towards zero.
     !     - basis value * 2nd deriv value = gradient
     C_TEMP(:) = MATMUL(XY_F_VALS(:,3), XY_B_VALS(:,:,3))
     C_STEP(:) = C_STEP(:) + C_TEMP(:) ! Step to minimize L2 of second derivative.
     ! 
     ! The gradient of monotonicity points the derivative towards zero
     !   in all places where the derivative sign is incorrect:
     !     where(wrong sign) - basis value * 1st deriv value
     ! Clip all the gradients that are in fine regions of monotinicity to zero.
     IF (STEP .GT. STEPS) THEN
        WHERE (XY_F_VALS(:,2) * DIRECTION(:) .GT. 0.0_R8)
           XY_F_VALS(:,2) = 0.0
        END WHERE
        C_TEMP(:) = MATMUL(XY_F_VALS(:,2), XY_B_VALS(:,:,2))
        C_TEMP(:) = C_TEMP(:) * MONOTONICITY_MULTIPLIER ! Lagrangian optimization
        C_STEP(:) = C_STEP(:) + C_TEMP(:) ! Step to minimize nonmonotonicity.
        ! 
        ! Update the Lagrangian multiplier for satisfying monotonicity.
        D1_2NORM = NORM2(XY_F_VALS(:,2))
        ! Increase the monotonicity term weight when convergence is not met.
        IF (D1_2NORM .GT. 0.0_R8) THEN
           MONOTONICITY_MULTIPLIER = MIN(1.0E50_R8, MONOTONICITY_MULTIPLIER + SHIFT)
           SHIFT = MIN(1.0E50_R8, SHIFT * 1.005_R8)
        ELSE
           SHIFT = MAX(MONOTONICITY_MULTIPLIER / 8.0_R8, SHIFT / 1.005_R8)
           MONOTONICITY_MULTIPLIER = MAX(0.0_R8, MONOTONICITY_MULTIPLIER - SHIFT)
        END IF
     END IF
     ! 
     ! Project the gradient step onto a direction where the function
     !   values are still correctly matched. I.e., find the minimum norm
     !   solution to the coeficient set that perfectly cancels
     !   the function values produced by the current computed step.
     ! 
     X_F_VALS(:) = MATMUL(X_B_VALS(:,:), C_STEP(:))
     CALL LEAST_SQUARES(X_B_VALS(:,:), X_F_VALS(:), C_TEMP(:))
     C_STEP(:) = C_STEP(:) - C_TEMP(:)
     ! 
     ! Update exponential moving average of gradient and curvature.
     C_GRAD(:) = 0.1 * C_STEP(:) + 0.9 * C_GRAD(:)
     C_CURV(:) = 0.01 * (C_STEP(:) - C_GRAD(:))**2 + 0.99 * C_CURV(:)
     C_CURV(:) = MAX(C_CURV(:), EPSILON(1.0_R8))
     ! 
     ! Take the step following the gradient (weighted by curvature).
     BCOEF(:) = BCOEF(:) - STEP_SIZE * (C_GRAD(:) / SQRT(C_CURV(:)))
     ! 
     ! Update the coefficients to interpolate the data.
     X_F_VALS(:) = Y(:) - MATMUL(X_B_VALS(:,:), BCOEF(:))
     CALL LEAST_SQUARES(X_B_VALS(:,:), X_F_VALS(:), C_STEP(:))
     BCOEF(:) = BCOEF(:) + C_STEP(:)
  END DO

  ! TODO: Find a way to structure the basis function values matrix
  !       and the points matrix so that all the multiplications with
  !       zeros are not done to evaluate the spline.
  !       Since the known the points aren't moving, then we only need
  !       to keep the basis functions that are not zero for each point.
  ! 

CONTAINS

  SUBROUTINE LEAST_SQUARES(A_IN, B, X)
    USE REAL_PRECISION, ONLY: R8
    IMPLICIT NONE
    REAL(KIND=R8), INTENT(IN), DIMENSION(:,:) :: A_IN
    REAL(KIND=R8), INTENT(IN), DIMENSION(:) :: B
    REAL(KIND=R8), INTENT(OUT), DIMENSION(:) :: X
    ! LAPACK routine for solving least squares.
    EXTERNAL :: DGELS
    ! Local variables.
    REAL(KIND=R8), DIMENSION(SIZE(A_IN,1),SIZE(A_IN,2)) :: A
    REAL(KIND=R8), DIMENSION(1) :: WORK_SIZE
    REAL(KIND=R8), DIMENSION(:), ALLOCATABLE :: WORK
    INTEGER :: LWORK, M, N, LDA, LDB, NRHS
    ! Store a local copy of A to prevent A_IN from being overwritten.
    A(:,:) = A_IN(:,:)
    ! Store all DGELS parameters literally for readability.
    M = SIZE(A,1)
    N = SIZE(A,2)
    LDA = SIZE(A,1)
    LDB = SIZE(X,1)
    NRHS = 1
    ! Allocate the correctly sized work array for DGELS.
    CALL DGELS('N', M, N, NRHS, A(:,:), LDA, X(:), LDB, WORK_SIZE(:), -1, INFO)
    IF (INFO .NE. 0) THEN
       PRINT *, 'WARNING (monotone_fit.f90:least_squares) DGELS produced nonzero info on work size query', INFO
       LWORK = MAX( 1, M*N + MAX( M*N, NRHS ) )
    ELSE
       LWORK = INT(WORK_SIZE(1))
    END IF
    ALLOCATE(WORK(1:LWORK))
    ! Copy the right hand side into the output variable for DGELS.
    X(1:SIZE(B)) = B(:)
    IF (SIZE(X) .GT. SIZE(B)) X(SIZE(B)+1:SIZE(X)) = 0.0_R8
    ! Solve the linear system.
    CALL DGELS('N', M, N, NRHS, A(:,:), LDA, X(:), LDB, WORK(:), LWORK, INFO)
    ! Deallocate and check for errors.
    DEALLOCATE(WORK)
    IF (INFO .NE. 0) THEN
       PRINT *, 'ERROR (monotone_fit.f90:least_squares) DGELS produced nonzero info on solve', INFO
    END IF
  END SUBROUTINE LEAST_SQUARES

  
END SUBROUTINE MONOTONE_FIT
