! Compile and execute the following program with:
!   $F03 bvspis.f test_bvspis.f03 -o test_bvspis
!   ./test_bvspis
! 
! Where F03 is the Fortran 2003 compatible compiler. For example:
!   gfortran bvspis.f test_bvspis.f03 -o test_bvspis
!   ./test_bvspis
! 
PROGRAM TEST_BVSPIS
  IMPLICIT NONE
  EXTERNAL :: DBVSSC, DBVSSE
  INTEGER :: NP, N, K, R, Q, P, OPT, KMAX, MAXSTP, NWORK, SBOPT, NTAB, &
       Y0OPT, Y1OPT, Y2OPT, ERRC, ERRE
  DOUBLE PRECISION :: EPS, D0, DNP, D20, D2NP, BETA, BETAI, RHO, RHOI
  INTEGER , ALLOCATABLE :: CONSTR(:), DIAGN(:)
  DOUBLE PRECISION , ALLOCATABLE :: X(:), Y(:), XTAB(:), D(:), D2(:), &
       WORK(:), Y0TAB(:), Y1TAB(:), Y2TAB(:)
  ! Set parameters.
  EPS = SQRT(EPSILON(0.0D0)) ! Default precision.
  NP = 3-1 ! Number of points minus one.
  N = 6 ! Degree of the spline (minimum of 3*k).
  K = 2 ! Class of continuity (C2).
  R = 1 ! Enforce monotonicity constraints.
  Q = 1 ! No boundary conditions are imposed.
  P = 1 ! Derivative estimation by constraints only.
  OPT = 100*P + 10*Q + R ! Contains PQR in decimal format.
  NWORK = 5+(2+7)*NP+(N*(N+11))/2+9 ! Explained in DBVSSC comments.
  SBOPT = 2 ! Do a binary search to find intervals for evaluation.
  NTAB = 3-1 ! Number of evaluation points minus one.
  Y0OPT = 1 ! Enable evaluation of the zeroth derivative.
  Y1OPT = 0 ! Disable evaluation of the first derivative.
  Y2OPT = 0 ! Disable evaluation of the second derivative.
  ERRC = 0 ! Error code, initialize to 0, should be overwritten.
  ERRE = 0 ! Error code, Initialize to 0, should be overwritten.
  ! Allocate and initialize arrays.
  ALLOCATE( &
       X(0:NP), & ! X values that cause failure.
       Y(0:NP), & ! Y values that cause failure.
       CONSTR(0:NP), & ! Used for setting shape preserving constraints.
       XTAB(0:NTAB), & ! Z evaluation point (known failure location).
       D(0:NP), & ! Derivative values (on output).
       D2(0:NP), & ! Referenced with K=2.
       DIAGN(0:NP-1), & ! Diagnostic info.
       WORK(1:NWORK), & ! Work space.
       Y0TAB(0:NTAB), & ! Spline values.
       Y1TAB(0:NTAB), & ! First derivative values.
       Y2TAB(0:NTAB) & ! Second derivative values.
       )
  X(0:NP) = (/ 0.0D0, 0.09D0, 0.091D0 /)
  Y(0:NP) = (/ 0.0D0, 0.1D0, 0.2D0 /)
  XTAB(0:NTAB) = (/ 0.0D0, 0.06D0, 0.09D0 /)
  CONSTR(0:NP) = 1
  D(0:NP) = 0.0D0
  D2(0:NP) = 0.0D0
  DIAGN(0:NP-1) = 0.0D0
  WORK(1:NWORK) = 0.0D0
  Y0TAB(0:NTAB) = 0.0D0
  Y1TAB(0:NTAB) = 0.0D0
  Y2TAB(0:NTAB) = 0.0D0
  ! Unused parameters, ignored unless Q=3.
  D0 = 0.0D0
  DNP = 0.0D0
  D20 = 0.0D0
  D2NP = 0.0D0
  ! Unused parameters, ignored unless Q=2.
  BETA = 0.0D0
  BETAI = 0.0D0
  RHO = 0.0D0
  RHOI = 0.0D0
  KMAX = 0
  MAXSTP = 0
  ! Construct the fit.
  PRINT *, 'Constructing fit.'
  PRINT *, 'X(:)', X(:)
  PRINT *, 'Y(:)', Y(:)
  CALL DBVSSC( &
       X(:), Y(:), NP, N, K, OPT, D0, DNP, D20, D2NP, CONSTR(:), &
       EPS, BETA, BETAI, RHO, RHOI, KMAX, MAXSTP, ERRC, &
       D(:), D2(:), DIAGN(:), WORK(:), NWORK &
       )
  PRINT *, 'ERRC ', ERRC ! = 0
  IF (ERRC .EQ. 0) THEN
     PRINT *, '^ No error code was produced by DBVSSC for the spline fit.'
  END IF
  PRINT *, ''
  ! Evaluate the fit.
  PRINT *, 'Evaluate the spline.'
  CALL DBVSSE( &
       X(:), Y(:), NP, N, K, XTAB(:), NTAB, SBOPT, Y0OPT, Y1OPT, &
       Y2OPT, ERRC, D(:), D2(:), Y0TAB(:), Y1TAB(:), Y2TAB(:), &
       ERRE, WORK(:), NWORK &
       )
  PRINT *, 'XTAB(:) ', XTAB(:) ! = 0.0, 0.06, 0.09 as expected.
  PRINT *, 'Y0TAB(:)', Y0TAB(:) ! = 0.0, 0.152, 0.1 not monotone, unexpected.
  IF (Y0TAB(2) .GT. Y0TAB(3)) THEN
     PRINT *, '^ Notice that these values produced by the spline are NOT monotone.'
  ENDIF
  PRINT *, ''
  PRINT *, 'ERRE ', ERRE ! = 0 no error code, unexpected.
  PRINT *, 'ERRC ', ERRC ! = 0 no error code, unexpected.
  IF ((ERRE .EQ. 0) .AND. (ERRC .EQ. 0)) THEN
     PRINT *, '^ No error codes were produced by DBVSSE on spline evaluation.'
     PRINT *, ''
  END IF

END PROGRAM TEST_BVSPIS
