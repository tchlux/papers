! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
!                          sample_main.f90
! 
! DESCRIPTION:
!   This file (sample_main.f90) contains a sample main program that
!   illustrates the usage of subroutines MQSI and EVAL_SPLINE to interpolate
!   data read from a file and compute derivatives of the monotone quintic
!   spline interpolant Q(x) at the data points (via the optional argument
!   UV to MQSI) and at other points (via EVAL_SPLINE).  Compile with:
! 
!    $F03 $OPTS REAL_PRECISION.f90 EVAL_BSPLINE.f90 SPLINE.f90 MQSI.f90 \
!       sample_main.f90 -o main $LIB
!
!   where '$F03' is the name of the Fortran 2003 compiler, '$OPTS' are
!   compiler options such as '-O3', and '$LIB' provides a flag to link
!   BLAS and LAPACK. If the BLAS and LAPACK libraries are not
!   available on your system, then replace $LIB with the filenames
!   'blas.f lapack.f'; these files contain the routines from the BLAS
!   and LAPACK libraries that are necessary for this package.
!
! CONTAINS:
!   PROGRAM SAMPLE_MAIN
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
!   August 2020 -- (tchl) Created file, (ltw / wit) reviewed and revised.
! 
PROGRAM SAMPLE_MAIN
USE REAL_PRECISION, ONLY: R8  
IMPLICIT NONE

! Define the interfaces for relevant MQSI package subroutines.
INTERFACE
 SUBROUTINE MQSI(X, Y, T, BCOEF, INFO, UV)
   USE REAL_PRECISION, ONLY: R8
   REAL(KIND=R8), INTENT(IN),  DIMENSION(:) :: X
   REAL(KIND=R8), INTENT(INOUT),  DIMENSION(:) :: Y
   REAL(KIND=R8), INTENT(OUT), DIMENSION(:) :: T, BCOEF
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
END INTERFACE

INTEGER :: I, INFO, J, ND
INTEGER, PARAMETER :: NEV=6 ! Number of spline Q(x) evaluation points.
REAL(KIND=R8) :: DIFF
REAL(KIND=R8), ALLOCATABLE :: BCOEF(:), QVAL(:,:), T(:), X(:), XY(:), &
   XYSAVE(:), Y(:), UV(:,:)

OPEN(UNIT=10, FILE="sample_main.dat", STATUS="OLD", ACTION="READ")
READ (10,*) ND ! Number of data points in file "sample_main.dat".
ALLOCATE(BCOEF(3*ND), QVAL(NEV,3), T(3*ND+6), X(ND), XY(NEV), XYSAVE(NEV), &
   Y(ND), UV(ND,2))
READ (10,*) (X(I), Y(I), I=1,ND) ! Data points to be interpolated.
CALL MQSI(X,Y,T,BCOEF,INFO,UV) ! Compute monotone quintic spline interpolant
   ! Q(x) to data (X,Y), returning knot sequence T, B-spline coefficients
   ! BCOEF, and derivative information for Q(x) in UV.
IF (INFO .NE. 0) THEN
   WRITE (*,"(/A/)") "This test data should not produce an error!"
   STOP
ENDIF
DIFF = (X(ND) - X(1))/REAL(NEV-1,R8)
XYSAVE(1:NEV) = [X(1), (X(1)+REAL(J,R8)*DIFF, J=1,NEV-2), X(ND)]
DO I=1,3
   XY(1:NEV) = XYSAVE(1:NEV)
   CALL EVAL_SPLINE(T,BCOEF,XY,INFO,I-1) ! Evaluate d^(I-1)Q(x)/dx at XY(.).
   IF (INFO .NE. 0) THEN
      WRITE (*,"(/A/)") "This test data should not produce an error!"
      STOP
   ENDIF
   QVAL(1:NEV,I) = XY(1:NEV)
END DO
WRITE (*,"(/A/)") "Spline values at evaluation points:"
WRITE (*,100) (XYSAVE(J),QVAL(J,1), QVAL(J,2),QVAL(J,3), J=1,NEV)
100 FORMAT(11X,"X",8X,"Q(X)",7X,"Q'(X)",6X,"Q''(X)",/ (4ES12.4))
WRITE (*,"(/A/)") "Spline values at interpolation points:"
WRITE (*,100) (X(J), Y(J), UV(J,1), UV(J,2), J=1,ND)
STOP
END PROGRAM SAMPLE_MAIN
