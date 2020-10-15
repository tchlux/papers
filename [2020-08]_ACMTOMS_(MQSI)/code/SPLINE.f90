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
T(NK-DEGREE:NK) = MAX( XI(NB) + ABS(XI(NB))*SQRT(EPSILON(XI(NB))),  &
                       XI(NB) + SQRT(EPSILON(XI(NB))) )

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
