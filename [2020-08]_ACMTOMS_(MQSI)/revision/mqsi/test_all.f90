! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
!                           test_all.f90
! 
! DESCRIPTION:
!   This file contains code for testing a local installation of the MQSI
!   package. Here is an example command to compile and run all tests:
! 
!    $F03 $OPTS REAL_PRECISION.f90 EVAL_BSPLINE.f90 SPLINE.f90 MQSI.f90 \
!      test_all.f90 -o test_all $LIB && ./test_all
!
!   where '$F03' is the name of the Fortran 2003 compiler, '$OPTS' are
!   compiler options such as '-O3', and '$LIB' provides a flag to link
!   BLAS and LAPACK. If the BLAS and LAPACK libraries are not
!   available on your system, then replace $LIB with the filenames
!   'blas.f lapack.f'; these files contain the routines from the BLAS
!   and LAPACK libraries that are necessary for this package.
! 
!   All tests will be run and an error message will be printed for any
!   failed test cases. Otherwise a message saying tests have passed
!   will be directed to standard output. A small timing test will also
!   be run to display the expected time required to fit and evaluate a
!   monotone quintic spline interpolant to data of a given size.
! 
! CONTAINS:
!   PROGRAM TEST_ALL
! 
! DEPENDENCIES:
!   MODULE REAL_PRECISION
!     INTEGER, PARAMETER :: R8
!   END MODULE REAL_PRECISION
! 
!   SUBROUTINE MQSI(X, Y, T, BCOEF, INFO, UV)
!     USE REAL_PRECISION, ONLY: R8
!     REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: X
!     REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: Y
!     REAL(KIND=R8), INTENT(OUT),   DIMENSION(:) :: T, BCOEF
!     INTEGER, INTENT(OUT) :: INFO
!     REAL(KIND=R8), INTENT(OUT), DIMENSION(:,:), OPTIONAL :: UV
!   END SUBROUTINE MQSI
! 
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
! CONTRIBUTORS:
!   Thomas C.H. Lux (tchlux@vt.edu)
!   Layne T. Watson (ltwatson@computer.org)
!   William I. Thacker (thackerw@winthrop.edu)
! 
! VERSION HISTORY:
!   July 2020 -- (tchl) Created file, (ltw / wit) reviewed and revised.
! 
PROGRAM TEST_ALL
USE REAL_PRECISION, ONLY: R8
IMPLICIT NONE
! ------------------------------------------------------------------
!                         Testing parameters
! 
! The number of equally spaced points used when checking monotonicity,
! and the number of repeated trials used when collecting timing data.
INTEGER, PARAMETER :: TRIALS = 100
! The maximum allowable error in spline approximations to data.
REAL(KIND=R8), PARAMETER :: ERROR_TOLERANCE = SQRT(EPSILON(1.0_R8))
! The different sizes of data provided for testing monotone interpolation.
INTEGER, PARAMETER :: NS(6) = (/ 2**3, 2**4, 2**5, 2**6, 2**7, 2**8 /)
! A switch determining whether data is randomly spaced or equally spaced.
LOGICAL, PARAMETER :: EQ_SPACED(2) = (/ .TRUE., .FALSE. /)
! The names of the test functions provided in this file.
CHARACTER(LEN=20), PARAMETER :: TEST_FUNC_NAMES(7) = (/ &
     "Large tangent       ", "Piecewise polynomial", "Random              ", &
     "Random monotone     ", "Signal decay        ", "Tiny test           ", &
     "Huge test           "/)
! The size of the data used in the demonstration timing experiment.
INTEGER, PARAMETER :: TIME_SIZE = 100
! The number of percentiles of timing data to write to standard output.
! It is best when this is an odd number (includes min, median, and max).
INTEGER, PARAMETER :: TIMING_PERCENTILES = 5
! ------------------------------------------------------------------
INTERFACE
  SUBROUTINE MQSI(X, Y, T, BCOEF, INFO, UV)
    USE REAL_PRECISION, ONLY: R8
    REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: X
    REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: Y
    REAL(KIND=R8), INTENT(OUT),   DIMENSION(:) :: T, BCOEF
    INTEGER, INTENT(OUT) :: INFO
    REAL(KIND=R8), INTENT(OUT), DIMENSION(:,:), OPTIONAL :: UV
  END SUBROUTINE MQSI
 SUBROUTINE EVAL_SPLINE(T, BCOEF, XY, INFO, D)
   USE REAL_PRECISION, ONLY: R8
   REAL(KIND=R8), INTENT(IN), DIMENSION(:) :: T, BCOEF
   REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: XY
   INTEGER, INTENT(OUT) :: INFO
   INTEGER, INTENT(IN), OPTIONAL :: D
 END SUBROUTINE EVAL_SPLINE
 SUBROUTINE FIT_SPLINE(XI, FX, T, BCOEF, INFO)
   USE REAL_PRECISION, ONLY: R8
   REAL(KIND=R8), INTENT(IN),  DIMENSION(:)   :: XI
   REAL(KIND=R8), INTENT(IN),  DIMENSION(:,:) :: FX
   REAL(KIND=R8), INTENT(OUT), DIMENSION(:)   :: T, BCOEF
   INTEGER, INTENT(OUT) :: INFO
 END SUBROUTINE FIT_SPLINE
END INTERFACE
! Iteration indices.
INTEGER :: I, J, K, L
! Boolean for whether or not all tests have successfully passed.
LOGICAL :: ALL_PASSED
! Time recording variables for spline evaluation EVAL_TIMES, and for
! construction of the spline interpolant FIT_TIMES. These are REAL
! valued because the system call for TIME requires a REAL value.
REAL :: EVAL_TIMES(TIMING_PERCENTILES,SIZE(TEST_FUNC_NAMES),SIZE(EQ_SPACED))
REAL :: FIT_TIMES( TIMING_PERCENTILES,SIZE(TEST_FUNC_NAMES),SIZE(EQ_SPACED))
! ------------------------------------------------------------------
ALL_PASSED = .TRUE.

! Run all tests.
WRITE (*,111) ('-', J=1,32), TRIALS
111 FORMAT(/,32A1,/,'Running tests,',I4,' trials each.')
main_loop : DO I = 1, SIZE(NS)
  WRITE (*,'(/,"N:",I6)') NS(I)
  ! Run a test on each test function (with this number of points).
  DO J = 1, SIZE(TEST_FUNC_NAMES)
    WRITE (*,"('  ',A)") TEST_FUNC_NAMES(J)
    DO K = 1, SIZE(EQ_SPACED)
      ! This next check executes a test.
      IF (.NOT. PASSES_TEST(J, NS(I), EQ_SPACED(K), TRIALS)) THEN
        WRITE (*,"(/,A)") 'FAILED TEST'
        ! Describe the test that failed and exit.
        WRITE (*,112,ADVANCE='NO') TEST_FUNC_NAMES(J)
112 FORMAT(/,'Test configuration:',/,9X,'function:',3X,A,/,10X,'spacing:',3X)
        IF (EQ_SPACED(K)) THEN
           WRITE (*,'(A)') 'equally spaced'
        ELSE
           WRITE (*,'(A)') 'randomly spaced'
        END IF
        WRITE (*,113) NS(I), TRIALS, ('_',L=1,34)
113 FORMAT(8X,'data size:',I5,/,' number of trials:',I5,/,34A1,/)
        ALL_PASSED = .FALSE.
        EXIT main_loop
      END IF
    END DO
  END DO
END DO main_loop

! End of testing code, beginning of timing code.
run_timing_test : IF (ALL_PASSED) THEN
WRITE (*,'(/,A)') 'All tests PASSED.'
WRITE (*,114) ('-', J=1,65), TIME_SIZE, TRIALS
114 FORMAT(/,65A1,/,'Computing timing data for fitting',I5,' points with',I5,' repeated trials.')
DO J = 1, SIZE(TEST_FUNC_NAMES)
  DO K = 1, SIZE(EQ_SPACED)
    ! When the '*_TIMES' variables are provided, the PASSES_TEST
    ! function runs a timing test instead of a monotonicity test. The
    ! logical result is ignored (since this is not a correctness test).
    ALL_PASSED = PASSES_TEST(J, TIME_SIZE, EQ_SPACED(K), TRIALS, &
         EVAL_TIMES(:,J,K), FIT_TIMES(:,J,K))
  END DO
END DO
! Average the timing percentiles over all executed tests.
EVAL_TIMES(:,1,1) = SUM(SUM(EVAL_TIMES(:,:,:), DIM=3), DIM=2) / &
     REAL(SIZE(EVAL_TIMES,2) * SIZE(EVAL_TIMES,3))
FIT_TIMES(:,1,1) = SUM(SUM(FIT_TIMES(:,:,:), DIM=3), DIM=2) / &
     REAL(SIZE(FIT_TIMES,2) * SIZE(FIT_TIMES,3))
! Send the results to standard output, showing some equally spaced
! percentiles of the timing data. All outputs are rounded to the
! microsecond (10^{-6} seconds) because that is the accuracy of common
! Fortran compilers CPU_TIME routine.
WRITE (*,115) TIME_SIZE
115 FORMAT(/,' Fit time for MQSI of',I5,' points:')
J = TIMING_PERCENTILES
DO I = 1, J
  IF (I .EQ. 1) THEN
    WRITE (*,116) 'min',FIT_TIMES(I,1,1)
116 FORMAT(A9,':',F10.6,' seconds')
  ELSE IF ((I-1 .EQ. (J-1)/2) .AND. (MOD(J,2) .EQ. 1)) THEN
    WRITE (*,116) 'median',FIT_TIMES(I,1,1)
  ELSE IF (I .EQ. J) THEN
    WRITE (*,116) 'max',FIT_TIMES(I,1,1)
  ELSE
    K = INT(1.0 + REAL((99)*(I-1))/REAL(J-1))
    WRITE (*,'(I9,":",F10.6," seconds")') K, FIT_TIMES(I,1,1)
  END IF
END DO
WRITE (*,117) TIME_SIZE
117 FORMAT(/,' Evaluation time per point for MQSI built from',I6,' points:')
DO I = 1, J
  IF (I .EQ. 1) THEN
    WRITE (*,116) 'min',EVAL_TIMES(I,1,1)
  ELSE IF ((I-1 .EQ. (J-1)/2) .AND. (MOD(J,2) .EQ. 1)) THEN
    WRITE (*,116) 'median',EVAL_TIMES(I,1,1)
  ELSE IF (I .EQ. J) THEN
    WRITE (*,116) 'max',EVAL_TIMES(I,1,1)
  ELSE
    K = INT(1.0 + REAL((99)*(I-1))/REAL(J-1))
    WRITE (*,'(I9,":",F10.6," seconds")') K, EVAL_TIMES(I,1,1)
  END IF
END DO
END IF run_timing_test

STOP ! <- End of TEST_ALL program execution.

CONTAINS

FUNCTION PASSES_TEST(F_NUM, N, EQUALLY_SPACED_X, TRIALS, EVAL_TIME, FIT_TIME)
  ! Wrapper function for running a test given the test configuration
  ! information. Offers a "correct output" test by default, when
  ! EVAL_TIME and FIT_TIME are provided this executes a "timed" test.
  ! 
  ! Arguments.
  !  Function number and data size.
  INTEGER, INTENT(IN) :: F_NUM, N 
  !  Equally or randomly spaced data.
  LOGICAL, INTENT(IN) :: EQUALLY_SPACED_X
  !  Number of test points for monotonicity, or repeated trials for timing.
  INTEGER, INTENT(IN) :: TRIALS 
  !  Optional storage for timing results.
  REAL, INTENT(OUT), OPTIONAL, DIMENSION(:) :: EVAL_TIME, FIT_TIME
  !  Boolean for if a test was passed.
  LOGICAL :: PASSES_TEST
  ! Local variables.
  !  Data location X, values Y, spline knots SK, and spline coefficients SC.
  REAL(KIND=R8) :: X(1:N), Y(1:N), SK(1:3*N+6), SC(1:3*N)
  !  Iteration index I, random seed size J, subroutine execution flag INFO.
  INTEGER :: I, INFO, J
  !  Storage for seeding the random number generator (for repeatability).
  INTEGER, DIMENSION(:), ALLOCATABLE :: SEED
  ! Initialize X values.
  IF (EQUALLY_SPACED_X) THEN
     DO I = 1, N
        X(I) = REAL(I-1, KIND=R8)
     END DO
     X(:) = X(:) / REAL(N-1, KIND=R8)
  ELSE
     ! Initialize random seed.
     CALL RANDOM_SEED(SIZE=J)
     ALLOCATE(SEED(J))
     SEED(:) = 7919
     CALL RANDOM_SEED(PUT=SEED)
     ! Generate random (increasing) data.
     CALL RANDOM_NUMBER(X)
     CALL SORT(X)
     ! Make sure the random points have ample spacing.
     DO I = 1, N-1
        X(I+1) = MAX(X(I+1), X(I)+ERROR_TOLERANCE*2.0_R8)
     END DO
  END IF
  ! Large tangent
  IF (F_NUM .EQ. 1) THEN
     CALL LARGE_TANGENT(X, Y)
  ! Piecewise polynomial
  ELSE IF (F_NUM .EQ. 2) THEN
     CALL PIECEWISE_POLYNOMIAL(X, Y)
  ! Random data
  ELSE IF (F_NUM .EQ. 3) THEN
     CALL RANDOM(X, Y)
  ! Random monotone data
  ELSE IF (F_NUM .EQ. 4) THEN
     CALL RANDOM_MONOTONE(X, Y)
  ! Decaying signal
  ELSE IF (F_NUM .EQ. 5) THEN
     CALL SIGNAL_DECAY(X, Y)
  ! Tiny test
  ELSE IF (F_NUM .EQ. 6) THEN
     CALL PIECEWISE_POLYNOMIAL(X, Y)
     ! Construct the smallest spacing of X and Y values
     ! that is allowed by the MQSI routine.
     X(:) = X(:) * SQRT(SQRT(TINY(1.0_R8)))
     Y(:) = Y(:) * SQRT(SQRT(TINY(1.0_R8)))
     INFO = 5
     grow_xy : DO WHILE ((INFO .EQ. 5) .OR. (INFO .EQ. 7))
        CALL MQSI(X, Y, SK, SC, INFO)
        IF ((INFO .NE. 5) .AND. (INFO .NE. 7)) EXIT grow_xy
        X(:) = X(:) * 10.0_R8
        Y(:) = Y(:) * 10.0_R8
     END DO grow_xy
  ! Huge test
  ELSE IF (F_NUM .EQ. 7) THEN
     CALL PIECEWISE_POLYNOMIAL(X, Y)
     ! Construct the largest spacing of X and Y values
     ! that is allowed by the MQSI routine.
     X(:) = X(:) * SQRT(SQRT(HUGE(1.0_R8)))
     Y(:) = Y(:) * SQRT(SQRT(HUGE(1.0_R8)))
     INFO = 6
     shrink_xy : DO WHILE ((INFO .EQ. 6) .OR. (INFO .EQ. 7))
        CALL MQSI(X, Y, SK, SC, INFO)
        IF ((INFO .NE. 6) .AND. (INFO .NE. 7)) EXIT shrink_xy
        X(:) = X(:) / 10.0_R8
        Y(:) = Y(:) / 10.0_R8
     END DO shrink_xy
  ! Unknown test number (incorrect usage of this testing routine).
  ELSE
     WRITE (*,100) F_NUM
100  FORMAT(/,'Unknown test number',I3,/)
     PASSES_TEST = .FALSE.
     IF (PRESENT(EVAL_TIME) .AND. PRESENT(FIT_TIME)) THEN     
        EVAL_TIME(:) = -1.0
        FIT_TIME(:) = -1.0
     END IF
     RETURN
  END IF
  ! If this test is supposed to record times, then run the TIME_TEST.
  IF (PRESENT(EVAL_TIME) .AND. PRESENT(FIT_TIME)) THEN
     CALL TIME_TEST(X, Y, FIT_TIME, EVAL_TIME, TRIALS)
     PASSES_TEST = .TRUE.
  ! Otherwise, run a correctness test (low error and monotone).
  ELSE
     CALL RUN_TEST(X, Y, PASSES_TEST, TRIALS)
  END IF
END FUNCTION PASSES_TEST

! ====================================================================
!                       Test execution routines.
! 
SUBROUTINE RUN_TEST(X, Y, PASSES, N_TEST)
! Runs a correctness test on MQSI, verifying reproduction
! of given function values and maintenance of monotonicity.
! 
! Arguments.
!  Data location.
REAL(KIND=R8), DIMENSION(:), INTENT(IN)    :: X
!  Data values.
REAL(KIND=R8), DIMENSION(:), INTENT(INOUT) :: Y
!  Result, TRUE if passed, FALSE otherwise.
LOGICAL, INTENT(OUT) :: PASSES
!  Number of test points between values of X.
INTEGER, INTENT(IN) :: N_TEST
! Local variables.
!  Maximum absolute observed error.
REAL(KIND=R8) :: MAX_ERROR
!  Spline coefficients SC, spline knots SK, temporary
!  input/output storage U, and test point storage Z.
REAL(KIND=R8) :: SC(1:3*SIZE(X)), SK(1:3*SIZE(X)+6), U(1:SIZE(X)), Z(1:N_TEST)
!  Iteration variables I, J, and subroutine status integer INFO.
INTEGER :: I, INFO, J
! Construct the monotone quintic spline interpolant.
CALL MQSI(X, Y, SK, SC, INFO)
IF (INFO .NE. 0) THEN
   WRITE (*,100)  INFO
100 FORMAT(/,'Failed to construct MQSI, error code',I4,'.')
   PASSES = .FALSE.
   RETURN
END IF
! Check that the spline reproduces the function values correctly.
U(:) = X(:)
CALL EVAL_SPLINE(SK, SC, U, INFO, D=0)
IF (INFO .NE. 0) THEN
   WRITE (*,101) INFO
101 FORMAT(/,'Failed to evaluate spline, error code',I4,'.')
   PASSES = .FALSE.
   RETURN
END IF
MAX_ERROR = MAXVAL( ABS((U(:) - Y(:))) / (1.0_R8 + ABS(Y(:))) )
IF (MAX_ERROR .GT. ERROR_TOLERANCE) THEN
   WRITE (*,102) MAX_ERROR, ERROR_TOLERANCE
102 FORMAT(/,'Value test: FAILED',/,'  relative error:', ES11.3,/, &
           ' error tolerance:',ES11.3)
   PASSES = .FALSE.
   RETURN
END IF
! Check for monotonicity over all intervals.
check_monotonicity :DO I = 1, SIZE(X)-1
   DO J = 1, N_TEST
      Z(J) = REAL(J-1, KIND=R8)
   END DO
   Z(:) = X(I) + (Z(:) / REAL(N_TEST-1, KIND=R8)) * (X(I+1) - X(I))
   CALL EVAL_SPLINE(SK, SC, Z, INFO, D=1)
   IF (.NOT. VALUES_ARE_MONOTONE(X(I), X(I+1), Y(I), Y(I+1), Z, &
        SQRT(ERROR_TOLERANCE))) THEN
      PASSES = .FALSE.
      RETURN
   END IF
END DO check_monotonicity
PASSES = .TRUE.
! End of test subroutine.
END SUBROUTINE RUN_TEST

! --------------------------------------------------------------------
SUBROUTINE TIME_TEST(X, Y, TFIT, TEVAL, TRIALS)
! Runs a batch of tests and records equally spaced percentiles of
! times into "TFIT" (fit time) and "TEVAL" (evaluation time).
! 
! Arguments.
!  Data location.
REAL(KIND=R8), INTENT(IN),    DIMENSION(:) :: X
!  Data values.
REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: Y
!  Evenly spaced percentiles of fit (TFIT) and evaluation (TEVAL) times.
REAL, INTENT(OUT), DIMENSION(:) :: TFIT, TEVAL
!  Number of repeated trials to run when collecting timing data.
INTEGER, INTENT(IN) :: TRIALS
! Local variables.
!  Spline coefficients SC, spline knots SK, temporary timing data
!  storage T, and temporary input/output storage U.
REAL(KIND=R8) :: SC(1:3*SIZE(X)), SK(1:3*SIZE(X)+6), T(TRIALS), U(TRIALS)
!  Temporary holders for start and finish times of tasks.
REAL :: FINISH_TIME_SEC, START_TIME_SEC
!  Iteration variables I,J, and subroutine status integer INFO.
INTEGER :: I, INFO, J
! Repeatedly measure spline construction time.
DO I = 1, TRIALS
   CALL CPU_TIME(START_TIME_SEC)
   CALL MQSI(X, Y, SK, SC, INFO)
   CALL CPU_TIME(FINISH_TIME_SEC)
   T(I) = REAL(FINISH_TIME_SEC - START_TIME_SEC, KIND=R8)
   IF (INFO .NE. 0) THEN
      WRITE (*,100)  INFO
100 FORMAT(/,'Failed to construct MQSI, error code',I4,'.')
      RETURN
   END IF
END DO
CALL SORT(T)
! Extract percentiles of construction times.
J = SIZE(TFIT)
DO I = 0, J-1
   TFIT(I+1) = REAL(T( INT(1.0 + REAL((TRIALS-1)*I)/REAL(J-1)) ))
END DO
! Repeatedly measure spline evaluation time.
DO I = 1, TRIALS
   CALL RANDOM_NUMBER(U)
   CALL CPU_TIME(START_TIME_SEC)
   CALL EVAL_SPLINE(SK, SC, U, INFO, D=0)
   IF (INFO .NE. 0) THEN
      WRITE (*,101) INFO
101 FORMAT(/,'Failed to evaluate spline, error code',I4,'.')
      RETURN
   END IF
   CALL CPU_TIME(FINISH_TIME_SEC)
   T(I) = REAL(FINISH_TIME_SEC - START_TIME_SEC, KIND=R8)
END DO
CALL SORT(T)
! Extract percentiles of evaluation times.
J = SIZE(TEVAL)
DO I = 0, J-1
   TEVAL(I+1) = REAL(T( INT(1.0 + REAL((TRIALS-1)*I)/REAL(J-1)) ))
END DO
! Rescale the timings to be relative to a single approximation point.
TEVAL(:) = TEVAL(:) / REAL(TRIALS)
END SUBROUTINE TIME_TEST


! ====================================================================
!                       Testing functions.
! 
SUBROUTINE LARGE_TANGENT(X, Y)
  ! "large tangent" is a function that rapidly grows from near 0 to 99 on [0,1].
  REAL(KIND=R8), INTENT(IN),  DIMENSION(:) :: X
  REAL(KIND=R8), INTENT(OUT), DIMENSION(:) :: Y
  Y(:) = -(1.0_R8 + ( 1.0_R8 / (X(:) - 1.01_R8) ))
END SUBROUTINE LARGE_TANGENT

SUBROUTINE PIECEWISE_POLYNOMIAL(X, Y)
  !  "piecewise polynomial" is a handcrafted function over [0,1] 
  !  represented as a C^1 spline.
  ! Arguments.
  !  Data location.
  REAL(KIND=R8), INTENT(IN),  DIMENSION(:) :: X
  !  Data values.
  REAL(KIND=R8), INTENT(OUT), DIMENSION(:) :: Y
  ! Local variables.
  !  Number of data points in the piecewise polynomial.
  INTEGER, PARAMETER :: N = 18
  !  Function values FX, spline coefficients SC, spline knots SK, 
  !  and data locations XI.
  REAL(KIND=R8) :: FX(1:N,1:2), SC(1:N*2), SK(1:N*2+2*2), XI(1:N)
  !  Iteration index I and subroutine status integer INFO.
  INTEGER :: I, INFO
  !   xi            f(x)      Df(x)  
  FX(1, 1:2)  = (/  0.0_R8,  1.0_R8 /)
  FX(2, 1:2)  = (/  1.0_R8,  0.0_R8 /)
  FX(3, 1:2)  = (/  1.0_R8,  0.0_R8 /)
  FX(4, 1:2)  = (/  1.0_R8,  0.0_R8 /)
  FX(5, 1:2)  = (/  0.0_R8,  0.0_R8 /)
  FX(6, 1:2)  = (/ 20.0_R8, -1.0_R8 /)
  FX(7, 1:2)  = (/ 19.0_R8, -1.0_R8 /)
  FX(8, 1:2)  = (/ 18.0_R8, -1.0_R8 /)
  FX(9, 1:2)  = (/ 17.0_R8, -1.0_R8 /)
  FX(10, 1:2) = (/  0.0_R8,  0.0_R8 /)  
  FX(11, 1:2) = (/  0.0_R8,  0.0_R8 /)  
  FX(12, 1:2) = (/  3.0_R8,  0.0_R8 /)
  FX(13, 1:2) = (/  0.0_R8,  0.0_R8 /)  
  FX(14, 1:2) = (/  1.0_R8,  3.0_R8 /)
  FX(15, 1:2) = (/  6.0_R8,  9.0_R8 /)
  FX(16, 1:2) = (/ 16.0_R8,  0.1_R8 /)
  FX(17, 1:2) = (/ 16.1_R8,  0.1_R8 /)
  FX(18, 1:2) = (/ 1.0_R8, -15.0_R8 /)
  DO I = 1, N
     XI(I) = REAL(I-1, KIND=R8)
  END DO
  ! Rescale XI and FX to be on the unit interval.
  XI(:) = XI(:) / REAL(N-1, KIND=R8)
  FX(:,2) = FX(:,2) * REAL(N-1, KIND=R8)
  CALL FIT_SPLINE(XI, FX, SK, SC, INFO)
  ! Make sure all provided locations are inside the support of the spline.
  Y(:) = MIN(MAX(X(:), 0.0_R8), 1.0_R8)
  CALL EVAL_SPLINE(SK, SC, Y, INFO, D=0)
END SUBROUTINE PIECEWISE_POLYNOMIAL

SUBROUTINE RANDOM(X, Y)
  !  "random" function that is just that, purely random data. It does
  !  not use X, but requires it as input to match the interface of
  !  other testing functions.
  ! Arguments.
  !  Data location.
  REAL(KIND=R8), INTENT(IN),  DIMENSION(:) :: X
  !  Data values.
  REAL(KIND=R8), INTENT(OUT), DIMENSION(:) :: Y
  ! Perform a null operation to prevent compiler warnings.
  Y(1) = X(1) 
  CALL RANDOM_NUMBER(Y)
END SUBROUTINE RANDOM

SUBROUTINE RANDOM_MONOTONE(X, Y)
  !  "random monotone" function that generates random monotone data.
  !  It does not use X, but requires it as input to match the interface
  !  of other testing functions.
  ! Arguments.
  !  Data location.
  REAL(KIND=R8), INTENT(IN),  DIMENSION(:) :: X
  !  Data values.
  REAL(KIND=R8), INTENT(OUT), DIMENSION(:) :: Y
  Y(1) = X(1) ! Null operation to prevent compiler warnings.
  CALL RANDOM_NUMBER(Y)
  CALL SORT(Y)
END SUBROUTINE RANDOM_MONOTONE

SUBROUTINE SIGNAL_DECAY(X, Y)
  !  "signal" function that is a decaying magnitude sine wave.
  ! Arguments.
  !  Data location.
  REAL(KIND=R8), INTENT(IN),  DIMENSION(:) :: X
  !  Data values.
  REAL(KIND=R8), INTENT(OUT), DIMENSION(:) :: Y
  ! Value of mathematical constant pi.
  REAL(KIND=R8) :: PI
  PI = ACOS(-1.0_R8)
  Y(:) = SIN(8.0_R8 * PI * X(:)) / (X(:)**2 + 0.1_R8)
END SUBROUTINE SIGNAL_DECAY

! ====================================================================
!                       Utility functions.
! 
FUNCTION VALUES_ARE_MONOTONE(U0, U1, F0, F1, DQ, ALLOWABLE_ERROR)
! Returns TRUE if derivative values are monotone (within tolerance)
! in the direction of function change, FALSE otherwise.
! 
! Arguments.
!  Ends of interval [U0, U1], function values at ends F0 and F1, equally
!  spaced spline derivative values on interval DQ, and the maximum
!  allowable magnitude of derivative deviation from monotonicity.
REAL(KIND=R8), INTENT(IN) :: U0, U1, F0, F1, ALLOWABLE_ERROR
REAL(KIND=R8), DIMENSION(:) :: DQ
!  Output Boolean, TRUE if values are monotone, FALSE otherwise.
LOGICAL :: VALUES_ARE_MONOTONE
! Local variables.
!  Machine precision.
REAL(KIND=R8), PARAMETER :: EPS = EPSILON(1.0_R8)
!  Minimum difference necessary to be considered increasing (decreasing).
REAL(KIND=R8) :: MIN_DIFF
! Rescale the DQ (derivative) values by the function change on the interval.
MIN_DIFF = EPS*(1.0_R8 + ABS(F1) + ABS(F0))
DQ(:) = DQ(:) / (1.0_R8 + ABS(F1 - F0))
! Test the extreme values of the derivative depending on direction 
! of function change (or lack of function change).
IF ( ((F1-F0 .GT. MIN_DIFF) .AND. (MINVAL(DQ(:)) .LT. -ALLOWABLE_ERROR)) .OR. &
     ((F1-F0 .LT. MIN_DIFF) .AND. (MAXVAL(DQ(:)) .GT.  ALLOWABLE_ERROR)) .OR. &
     ((ABS(F1-F0) .LT. MIN_DIFF) .AND. (MAXVAL(ABS(DQ(:))) .GT. ALLOWABLE_ERROR)) ) THEN
   WRITE (*,100) U0, U1, F1 - F0, MINVAL(DQ(:)), MAXVAL(DQ(:)), ALLOWABLE_ERROR
100 FORMAT(/,'Monotonicity test: FAILED',/,           &
           '  interval [',ES11.3,', ',ES11.3,']',/,   &
           '  interval function change:',ES11.3,/,    &
           '  minimum derivative value:',ES11.3,/,    &
           '  maximum derivative value:',ES11.3,/,    &
           '  error tolerance:         ',ES11.3,/)
   VALUES_ARE_MONOTONE = .FALSE.
ELSE
   VALUES_ARE_MONOTONE = .TRUE.
END IF
END FUNCTION VALUES_ARE_MONOTONE

SUBROUTINE SORT(VALUES)
  ! Insertion sort an array of values.
  ! 
  ! Arguments.
  !  Collection of unsorted numbers.
  REAL(KIND=R8), INTENT(INOUT), DIMENSION(:) :: VALUES
  ! Local variables.
  !  Temporary storage for swapping elements of VALUES.
  REAL(KIND=R8) :: TEMP_VAL
  !  Iteration indices I, J, and K.
  INTEGER :: I, J, K
  ! Return for the base case.
  IF (SIZE(VALUES) .LE. 1) RETURN
  ! Put the smallest value at the front of the list.
  I = MINLOC(VALUES,1)
  TEMP_VAL = VALUES(1)
  VALUES(1) = VALUES(I)
  VALUES(I) = TEMP_VAL
  ! Insertion sort the rest of the array.
  DO I = 3, SIZE(VALUES)
     TEMP_VAL = VALUES(I)
     ! Search backwards in the list until the insertion location is found. 
     J = I - 1
     K  = I
     DO WHILE (TEMP_VAL .LT. VALUES(J))
        VALUES(K) = VALUES(J)
        J = J - 1
        K  = K - 1
     END DO
     ! Put the value into its place (where it is greater than the
     ! element before it, but less than all values after it).
     VALUES(K) = TEMP_VAL
  END DO
END SUBROUTINE SORT

END PROGRAM TEST_ALL

