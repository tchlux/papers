! ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
!                             cli.f90
! 
! DESCRIPTION:
!   This file defines a command line interface program to the MQSI
!   package. The command line interface accepts data (to build a
!   monotone quintic spline interpolant) and points (to approximate
!   with MQSI) in files and outputs approximated values at points
!   to a file. Compile with:
! 
!    $F03 $OPTS REAL_PRECISION.f90 EVAL_BSPLINE.f90 SPLINE.f90 MQSI.f90 \
!       cli.f90 -o cli $LIB
!
!   where '$F03' is the name of the Fortran 2003 compiler, '$OPTS' are
!   compiler options such as '-O3', and '$LIB' provides a flag to link
!   BLAS and LAPACK. If the BLAS and LAPACK libraries are not
!   available on your system, then replace $LIB with the filenames
!   'blas.f lapack.f'; these files contain the routines from the BLAS
!   and LAPACK libraries that are necessary for this package. Running
!   the program with no arguments causes usage information to be
!   printed to standard output.
! 
! CONTAINS:
!   PROGRAM CLI
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
! 
! CONTRIBUTORS:
!   Thomas C.H. Lux (tchlux@vt.edu)
! 
! VERSION HISTORY:
!   June 2020 -- (tchl) Created file.
! 
PROGRAM CLI
USE REAL_PRECISION, ONLY: R8  
IMPLICIT NONE

! Define the interfaces for relevant MQSI package subroutines.
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
END INTERFACE

! MQSI status integer INFO, number of command line arguments C, number
! of points to approximate M, and number of data points N.
INTEGER :: INFO, C, M, N
! Holder for the input data point file name FILE_NAME, reused as the
! file name for the output file of approximated values. 
CHARACTER(LEN=218) :: FILE_NAME
! Spline coefficients SC, spline knots SK, approximation points U,
! data locations X, and data values Y.
REAL(KIND=R8), ALLOCATABLE :: SC(:), SK(:), U(:), X(:), Y(:)

! Get the number of command line arguments provided.
C = COMMAND_ARGUMENT_COUNT()

! Correct usage of this CLI provides either 2 or 3 command line arguments.
IF ((C .EQ. 2) .OR. (C .EQ. 3)) THEN
   ! Open the data file and read the number of points.
   CALL GET_COMMAND_ARGUMENT(1,FILE_NAME)
   OPEN(10,FILE=FILE_NAME,STATUS='OLD')
   READ(10,*) N
   ! Allocate storage based on N.
   ALLOCATE(X(N), Y(N), SC(1:3*N), SK(1:3*N+6))
   ! Read in the X and Y data.
   READ(10,*) X(:)
   READ(10,*) Y(:)
   CLOSE(10)

   ! Open the points file and read the number of points.
   CALL GET_COMMAND_ARGUMENT(2,FILE_NAME)
   OPEN(11,FILE=FILE_NAME,STATUS='OLD')
   READ(11,*) M
   ! Allocate storage based on M.
   ALLOCATE(U(M))
   ! Read in the points.
   READ(11,*) U(:)
   CLOSE(11)

   ! Construct a MQSI.
   CALL MQSI(X, Y, SK, SC, INFO)
   IF (INFO .NE. 0) WRITE(*,101) INFO
101 FORMAT('MQSI returned info =',I4)
   ! Evaluate the spline at all points (result is updated in-place in U).
   CALL EVAL_SPLINE(SK, SC, U, INFO)
   IF (INFO .NE. 0) WRITE(*,111) INFO
111 FORMAT(/,'EVAL_SPLINE returned info = ',I4)

   ! Set the output file name (if it was provided).
   IF (C .EQ. 3) THEN ; CALL GET_COMMAND_ARGUMENT(3,FILE_NAME)      
   ELSE;                FILE_NAME = "output.txt"
   END IF

   ! Write the output.
   OPEN(12,FILE=FILE_NAME,STATUS='NEW')
   WRITE (12,*) U(:)

   ! Explicityl deallocate all allocated memory.
   DEALLOCATE(X,Y,U)

ELSE
! This command line interface was not called correctly, so give the
! usage documentation and exit without doing anything.
   WRITE (*,121) 
121 FORMAT(/,&
         'MQSI command line interface. Usage:',/,/,&
         '   cli <data-file> <points-file> [<output-file>]',/,/,&
         'where "<data-file>" is a path to a file whose contents begin',/,&
         'with an integer N immediately followed by real valued locations',/,&
         'X(1:N) and real function values Y(1:N), and "<points-file>" is',/,&
         'a path to a file starting with an integer M immediately followed',/,&
         'by real valued locations U(1:M) of points to be approximated',/&
         'with MQSI over (X,Y).',/,/&
         'Values at all U(1:M) are produced by MQSI and written to "<output-file>"',/&
         'if it is provided, otherwise "output.txt".',/)
END IF

STOP

END PROGRAM CLI
  
