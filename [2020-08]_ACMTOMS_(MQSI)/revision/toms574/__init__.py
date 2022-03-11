'''This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code follows.

!$$$C     PROGRAM MAINDR                                                    MAIN  10
!$$$      REAL X(100),Y(100),XVAL(152),YVAL(152),M(100),XVAL1(200),         MAIN  20
!$$$     *     DELTAH,EPS                                                   MAIN  30
!$$$      INTEGER OUT,ERR                                                   MAIN  40
!$$$C                                                                       MAIN  50
!$$$C THIS PROGRAM IS A TEST DRIVER FOR THE SHAPE PRESERVING QUADRATIC      MAIN  60
!$$$C SPLINES BY D.F.MCALLISTER AND J.A.ROULIER.                            MAIN  70
!$$$C                                                                       MAIN  80
!$$$C ON INPUT--                                                            MAIN  90
!$$$C                                                                       MAIN 100
!$$$C   X CONTAINS THE ABSCISSAS OF THE POINTS OF INTERPOLATION.            MAIN 110
!$$$C                                                                       MAIN 120
!$$$C   Y CONTAINS THE ORDINATES OF THE POINTS OF INTERPOLATION.            MAIN 130
!$$$C                                                                       MAIN 140
!$$$C   N IS THE NUMBER OF DATA POINTS.                                     MAIN 150
!$$$C                                                                       MAIN 160
!$$$C   K IS THE NUMBER OF POINTS AT WHICH THE SPLINE IS TO BE EVALUATED.   MAIN 170
!$$$C                                                                       MAIN 180
!$$$C                                                                       MAIN 190
!$$$C UPON EXIT FROM SUBROUTINE 'SLOPES'--                                  MAIN 200
!$$$C                                                                       MAIN 210
!$$$C   M CONTAINS THE COMPUTED FIRST DERIVATIVES AT EACH DATA POINT.       MAIN 220
!$$$C                                                                       MAIN 230
!$$$C-----------------------------------------------------------------------MAIN 240
!$$$C                                                                       MAIN 250
!$$$C SUPPLY THE UNIT NUMBERS FOR I/O OPERATIONS.                           MAIN 260
!$$$      IN=1                                                              MAIN 270
!$$$      OUT=3                                                             MAIN 280
!$$$C                                                                       MAIN 290
!$$$      READ (IN,10)N                                                     MAIN 300
!$$$  10  FORMAT (I4)                                                       MAIN 310
!$$$      READ (IN,20) (X(I),Y(I),I=1,N)                                    MAIN 320
!$$$  20  FORMAT (2(E20.8,5X))                                              MAIN 330
!$$$C                                                                       MAIN 340
!$$$C CALCULATE THE SLOPES AT EACH DATA POINT.                              MAIN 350
!$$$      CALL SLOPES(X,Y,M,N)                                              MAIN 360
!$$$      WRITE (OUT,30)                                                    MAIN 370
!$$$  30  FORMAT('1','THESE ARE THE POINTS OF INTERPOLATION AND THE COMPUTEDMAIN 380
!$$$     * SLOPES.' //)                                                     MAIN 390
!$$$      WRITE (OUT,40) (I,X(I),Y(I),M(I),I=1,N)                           MAIN 400
!$$$  40  FORMAT('0','I=',I4,2X,'X=',E15.6,2X,'Y=',E15.6,2X,'M=',E15.6)     MAIN 410
!$$$C                                                                       MAIN 420
!$$$C SET THE ERROR TOLERANCE EPS, WHICH IS USED IN SUBROUTINE 'CHOOSE'.    MAIN 430
!$$$      EPS=1.E-04                                                        MAIN 440
!$$$C                                                                       MAIN 450
!$$$C TEST 1-- TEST FOR A SINGLE POINT WHICH IS LESS THAN THE ABSCISSA OF   MAIN 460
!$$$C          OF THE FIRST DATA POINT.                                     MAIN 470
!$$$      K=1                                                               MAIN 480
!$$$      XVAL(1)= -1.E0                                                    MAIN 490
!$$$      CALL MEVAL(XVAL,YVAL,X,Y,M,N,K,EPS,ERR)                           MAIN 500
!$$$      WRITE (OUT,50)                                                    MAIN 510
!$$$  50  FORMAT('1','TEST 1-- ERR SHOULD EQUAL 1.' //)                     MAIN 520
!$$$      WRITE (OUT,60)XVAL(1),YVAL(1),ERR                                 MAIN 530
!$$$  60  FORMAT('0','XVAL(1)=',E15.6,3X,'YVAL(1)=',E15.6,3X,'ERR=',I4 ////)MAIN 540
!$$$C                                                                       MAIN 550
!$$$C TEST 2-- TEST FOR A SINGLE POINT WHICH IS GREATER THAN THE ABSCISSA   MAIN 560
!$$$C          OF THE LAST DATA POINT.                                      MAIN 570
!$$$      XVAL(1)=12.E0                                                     MAIN 580
!$$$      CALL MEVAL(XVAL,YVAL,X,Y,M,N,K,EPS,ERR)                           MAIN 590
!$$$      WRITE(OUT,70)                                                     MAIN 600
!$$$  70  FORMAT('0','TEST 2-- ERR SHOULD EQUAL 1.' //)                     MAIN 610
!$$$      WRITE(OUT,80)XVAL(1),YVAL(1),ERR                                  MAIN 620
!$$$  80  FORMAT('0','XVAL(1)=',E15.6,3X,'YVAL(1)=',E15.6,3X,'ERR=',I4 ////)MAIN 630
!$$$C                                                                       MAIN 640
!$$$C TEST 3-- TEST FOR A SINGLE POINT WHICH IS IN RANGE BUT IS NOT A DATA  MAIN 650
!$$$C          POINT.                                                       MAIN 660
!$$$      XVAL(1)= 5.6E0                                                    MAIN 670
!$$$      CALL MEVAL(XVAL,YVAL,X,Y,M,N,K,EPS,ERR)                           MAIN 680
!$$$      WRITE(OUT,90)                                                     MAIN 690
!$$$  90  FORMAT('0','TEST 3-- ERR SHOULD EQUAL 0.' //)                     MAIN 700
!$$$      WRITE(OUT,100)XVAL(1),YVAL(1),ERR                                 MAIN 710
!$$$  100 FORMAT('0','XVAL(1)=',E15.6,3X,'YVAL(1)=',E15.6,3X,'ERR=',I4 ////)MAIN 720
!$$$C                                                                       MAIN 730
!$$$C TEST 4-- TEST FOR POINTS OF EVALUATION WHICH ARE DECREASING.          MAIN 740
!$$$      K=2                                                               MAIN 750
!$$$      XVAL(1)=5.E0                                                      MAIN 760
!$$$      XVAL(2)=4.E0                                                      MAIN 770
!$$$      CALL MEVAL(XVAL,YVAL,X,Y,M,N,K,EPS,ERR)                           MAIN 780
!$$$      WRITE (OUT,110)                                                   MAIN 790
!$$$  110 FORMAT('0','TEST 4-- ERR SHOULD EQUAL 2.' //)                     MAIN 800
!$$$      WRITE(OUT,120)ERR                                                 MAIN 810
!$$$  120 FORMAT('0','ERR=',I4 ////)                                        MAIN 820
!$$$C                                                                       MAIN 830
!$$$C TEST 5-- TEST FOR CORRECT EVALUATION OF SUCCESSIVE POINTS NOT IN      MAIN 840
!$$$C          ADJACENT INTERVALS.                                          MAIN 850
!$$$      XVAL(1)=1.2E0                                                     MAIN 860
!$$$      XVAL(2)=9.9E0                                                     MAIN 870
!$$$      CALL MEVAL(XVAL,YVAL,X,Y,M,N,K,EPS,ERR)                           MAIN 880
!$$$      WRITE (OUT,130)                                                   MAIN 890
!$$$  130 FORMAT('0','TEST 5-- ERR SHOULD EQUAL 0.' //)                     MAIN 900
!$$$      WRITE(OUT,140)XVAL(1),YVAL(1),XVAL(2),YVAL(2),ERR                 MAIN 910
!$$$  140 FORMAT('0','XVAL(1)=',E15.6,3X,'YVAL(1)=',E15.6,3X,'XVAL(2)=',    MAIN 920
!$$$     *       E15.6,3X,'YVAL(2)=',E15.6,3X,'ERR=',I4 ////)               MAIN 930
!$$$C                                                                       MAIN 940
!$$$C TEST 6-- EVALUATE EQUALLY SPACED POINTS IN THE ENTIRE INTERVAL        MAIN 950
!$$$C          DETERMINED BY THE DATA.                                      MAIN 960
!$$$C                                                                       MAIN 970
!$$$C COMPUTE THE POINTS OF EVALUATION.                                     MAIN 980
!$$$      K=150                                                             MAIN 990
!$$$      XVAL(1)=X(1)                                                      MAIN1000
!$$$      DELTAH=(X(N)-X(1))/FLOAT(K-1)                                     MAIN1010
!$$$      DO 150 I=2,K                                                      MAIN1020
!$$$          XVAL(I)= X(1) + FLOAT(I-1)*DELTAH                             MAIN1030
!$$$  150 CONTINUE                                                          MAIN1040
!$$$C                                                                       MAIN1050
!$$$      CALL MEVAL(XVAL,YVAL,X,Y,M,N,K,EPS,ERR)                           MAIN1060
!$$$      WRITE (OUT,160)                                                   MAIN1070
!$$$  160 FORMAT('1','TEST 6-- ERR SHOULD EQUAL 0.' //)                     MAIN1080
!$$$      WRITE(OUT,170)ERR                                                 MAIN1090
!$$$  170 FORMAT('0','ERR=',I4 ////)                                        MAIN1100
!$$$      WRITE(OUT,180) (I,XVAL(I),YVAL(I),I=1,K)                          MAIN1110
!$$$  180 FORMAT('0','I=',I4,3X,'XVAL=',E15.6,3X,'YVAL=',E15.6)             MAIN1120
!$$$C                                                                       MAIN1130
!$$$C TEST 7--EVALUATE EQUALLY SPACED POINTS IN THE INTERVAL                MAIN1140
!$$$C          X(1)-3.,X(N)+3. .  EXTRAPOLATION IS TESTED HERE.             MAIN1150
!$$$C                                                                       MAIN1160
!$$$C COMPUTE THE POINTS OF EVALUATION.                                     MAIN1170
!$$$      K=150                                                             MAIN1180
!$$$      XVAL1(1)=X(1)-3.E0                                                MAIN1190
!$$$      DELTAH=((X(N)+3.E0)-XVAL1(1))/FLOAT(K-1)                          MAIN1200
!$$$      DO 190 I=2,K                                                      MAIN1210
!$$$          XVAL1(I)= XVAL1(1) + FLOAT(I-1)*DELTAH                        MAIN1220
!$$$  190 CONTINUE                                                          MAIN1230
!$$$C                                                                       MAIN1240
!$$$      CALL MEVAL(XVAL1,YVAL,X,Y,M,N,K,EPS,ERR)                          MAIN1250
!$$$      WRITE(OUT,200)                                                    MAIN1260
!$$$  200 FORMAT('1','TEST 7-- ERR SHOULD EQUAL 1.' //)                     MAIN1270
!$$$      WRITE(OUT,210)ERR                                                 MAIN1280
!$$$  210 FORMAT('0','ERR=',I4 ////)                                        MAIN1290
!$$$      WRITE(OUT,220) (I,XVAL1(I),YVAL(I),I=1,K)                         MAIN1300
!$$$  220 FORMAT('0','I=',I4,3X,'XVAL=',E15.6,3X,'YVAL=',E15.6)             MAIN1310
!$$$C                                                                       MAIN1320
!$$$C TEST 8-- OVERIDE THE CALCULATED SLOPES AND USE THE SAME POINTS OF     MAIN1330
!$$$C          EVALUATION AS IN TEST 6.                                     MAIN1340
!$$$      READ(IN,230) (M(I),I=1,N)                                         MAIN1350
!$$$  230 FORMAT(E20.6)                                                     MAIN1360
!$$$      WRITE (OUT,240)                                                   MAIN1370
!$$$  240 FORMAT('1','THE SLOPES USED IN TEST 8 ARE GIVEN BELOW.' //)       MAIN1380
!$$$      WRITE(OUT,250) (M(I),I=1,N)                                       MAIN1390
!$$$  250 FORMAT('0','M=',E15.6)                                            MAIN1400
!$$$C                                                                       MAIN1410
!$$$      CALL MEVAL(XVAL,YVAL,X,Y,M,N,K,EPS,ERR)                           MAIN1420
!$$$      WRITE (OUT,260)                                                   MAIN1430
!$$$  260 FORMAT('1','TEST 8-- ERR SHOULD EQUAL 0.' //)                     MAIN1440
!$$$      WRITE(OUT,270)ERR                                                 MAIN1450
!$$$  270 FORMAT('0','ERR=',I4 ////)                                        MAIN1460
!$$$      WRITE(OUT,280) (I,XVAL(I),YVAL(I),I=1,K)                          MAIN1470
!$$$  280 FORMAT('0','I=',I4,3X,'XVAL=',E15.6,3X,'YVAL=',E15.6)             MAIN1480
!$$$C                                                                       MAIN1490
!$$$      STOP                                                              MAIN1500
!$$$      END                                                               MAIN1510
'''

import os
import ctypes
import numpy

# --------------------------------------------------------------------
#               CONFIGURATION
# 
_verbose = True
_fort_compiler = "gfortran"
_shared_object_name = "toms574.so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3']
_ordered_dependencies = ['toms574.f', 'toms574_c_wrapper.f90']
# 
# --------------------------------------------------------------------
#               AUTO-COMPILING
#
# Try to import the existing object. If that fails, recompile and then try.
try:
    clib = ctypes.CDLL(_path_to_lib)
except:
    # Remove the shared object if it exists, because it is faulty.
    if os.path.exists(_shared_object_name):
        os.remove(_shared_object_name)
    # Compile a new shared object.
    _command = " ".join([_fort_compiler] + _compile_options + ["-o", _shared_object_name] + _ordered_dependencies)
    if _verbose:
        print("Running system command with arguments")
        print("  ", _command)
    # Run the compilation command.
    import subprocess
    subprocess.run(_command, shell=True, cwd=_this_directory)
    # Import the shared object file as a C library with ctypes.
    clib = ctypes.CDLL(_path_to_lib)
# --------------------------------------------------------------------


# ----------------------------------------------
# Wrapper for the Fortran subroutine SLOPES

def slopes(xtab, ytab, mtab, num):
    '''!
!                                 SHAPE PRESERVING QUADRATIC SPLINES
!                                   BY D.F.MCALLISTER & J.A.ROULIER
!                                     CODED BY S.L.DODD & M.ROULIER
!                                       N.C.STATE UNIVERSITY
!
!
! SLOPES CALCULATES THE DERIVATIVE AT EACH OF THE DATA POINTS.  THE
! SLOPES PROVIDED WILL INSURE THAT AN OSCULATORY QUADRATIC SPLINE WILL
! HAVE ONE ADDITIONAL KNOT BETWEEN TWO ADJACENT POINTS OF INTERPOLATION.
! CONVEXITY AND MONOTONICITY ARE PRESERVED WHEREVER THESE CONDITIONS
! ARE COMPATIBLE WITH THE DATA.
!
! ON INPUT--
!
!   XTAB CONTAINS THE ABSCISSAS OF THE DATA POINTS.
!
!   YTAB CONTAINS THE ORDINATES OF THE DATA POINTS.
!
!   NUM IS THE NUMBER OF DATA POINTS (DIMENSION OF XTAB,YTAB).
!
!
! ON OUTPUT--
!
!   MTAB CONTAINS THE VALUE OF THE FIRST DERIVATIVE AT EACH DATA POINT.
!
! AND
!
!   SLOPES DOES NOT ALTER XTAB,YTAB,NUM.
!
!-----------------------------------------------------------------------'''
    
    # Setting up "xtab"
    if ((not issubclass(type(xtab), numpy.ndarray)) or
        (not numpy.asarray(xtab).flags.f_contiguous) or
        (not (xtab.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'xtab' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        xtab = numpy.asarray(xtab, dtype=ctypes.c_float, order='F')
    xtab_dim_1 = ctypes.c_int(xtab.shape[0])
    
    # Setting up "ytab"
    if ((not issubclass(type(ytab), numpy.ndarray)) or
        (not numpy.asarray(ytab).flags.f_contiguous) or
        (not (ytab.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'ytab' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        ytab = numpy.asarray(ytab, dtype=ctypes.c_float, order='F')
    ytab_dim_1 = ctypes.c_int(ytab.shape[0])
    
    # Setting up "mtab"
    if ((not issubclass(type(mtab), numpy.ndarray)) or
        (not numpy.asarray(mtab).flags.f_contiguous) or
        (not (mtab.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'mtab' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        mtab = numpy.asarray(mtab, dtype=ctypes.c_float, order='F')
    mtab_dim_1 = ctypes.c_int(mtab.shape[0])
    
    # Setting up "num"
    if (type(num) is not ctypes.c_int): num = ctypes.c_int(num)

    # Call C-accessible Fortran wrapper.
    clib.c_slopes(ctypes.byref(xtab_dim_1), ctypes.c_void_p(xtab.ctypes.data), ctypes.byref(ytab_dim_1), ctypes.c_void_p(ytab.ctypes.data), ctypes.byref(mtab_dim_1), ctypes.c_void_p(mtab.ctypes.data), ctypes.byref(num))

    # Return final results, 'INTENT(OUT)' arguments only.
    return xtab, ytab, mtab, num.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine MEVAL

def meval(xval, yval, xtab, ytab, mtab, num, nume, eps, err):
    '''!
!                                 SHAPE PRESERVING QUADRATIC SPLINES
!                                   BY D.F.MCALLISTER & J.A.ROULIER
!                                     CODED BY S.L.DODD & M.ROULIER
!                                       N.C. STATE UNIVERSITY
!
!
! MEVAL CONTROLS THE EVALUATION OF AN OSCULATORY QUADRATIC SPLINE.  THE
! USER MAY PROVIDE HIS OWN SLOPES AT THE POINTS OF INTERPOLATION OR USE
! THE SUBROUTINE 'SLOPES' TO CALCULATE SLOPES WHICH ARE CONSISTENT WITH
! THE SHAPE OF THE DATA.
!
!
!
! ON INPUT--
!
!   XVAL MUST BE A NONDECREASING VECTOR OF POINTS AT WHICH THE SPLINE
!   WILL BE EVALUATED.
!
!   XTAB CONTAINS THE ABSCISSAS OF THE DATA POINTS TO BE INTERPOLATED.
!   XTAB MUST BE INCREASING.
!
!   YTAB CONTAINS THE ORDINATES OF THE DATA POINTS TO BE INTERPOLATED.
!
!   MTAB CONTAINS THE SLOPE OF THE SPLINE AT EACH POINT OF INTERPOLA-
!   TION.
!
!   NUM IS THE NUMBER OF DATA POINTS (DIMENSION OF XTAB AND YTAB).
!
!   NUME IS THE NUMBER OF POINTS OF EVALUATION (DIMENSION OF XVAL AND
!   YVAL).
!
!   EPS IS A RELATIVE ERROR TOLERANCE USED IN SUBROUTINE 'CHOOSE'
!   TO DISTINGUISH THE SITUATION MTAB(I) OR MTAB(I+1) IS RELATIVELY
!   CLOSE TO THE SLOPE OR TWICE THE SLOPE OF THE LINEAR SEGMENT
!   BETWEEN XTAB(I) AND XTAB(I+1).  IF THIS SITUATION OCCURS,
!   ROUNDOFF MAY CAUSE A CHANGE IN CONVEXITY OR MONOTONICITY OF THE
!   RESULTING SPLINE AND A CHANGE IN THE CASE NUMBER PROVIDED BY
!   CHOOSE.  IF EPS IS NOT EQUAL TO ZERO, THEN EPS SHOULD BE GREATER
!   THAN OR EQUAL TO MACHINE EPSILON.
!
!
! ON OUTPUT--
!
! YVAL CONTAINS THE IMAGES OF THE POINTS IN XVAL.
!
!   ERR IS AN ERROR CODE--
!      ERR=0 - MEVAL RAN NORMALLY.
!      ERR=1 - XVAL(I) IS LESS THAN XTAB(1) FOR AT LEAST ONE I OR
!              XVAL(I) IS GREATER THAN XTAB(NUM) FOR AT LEAST ONE I.
!              MEVAL WILL EXTRAPOLATE TO PROVIDE FUNCTION VALUES FOR
!              THESE ABSCISSAS.
!      ERR=2 - XVAL(I+1) .LT. XVAL(I) FOR SOME I.
!
! AND
!
!   MEVAL DOES NOT ALTER XVAL,XTAB,YTAB,MTAB,NUM,NUME.
!
!
!   MEVAL CALLS THE FOLLOWING SUBROUTINES OR FUNCTIONS:
!      SEARCH
!      CASES
!      CHOOSE
!      SPLINE
!
!-----------------------------------------------------------------------'''
    
    # Setting up "xval"
    if ((not issubclass(type(xval), numpy.ndarray)) or
        (not numpy.asarray(xval).flags.f_contiguous) or
        (not (xval.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'xval' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        xval = numpy.asarray(xval, dtype=ctypes.c_float, order='F')
    xval_dim_1 = ctypes.c_int(xval.shape[0])
    
    # Setting up "yval"
    if ((not issubclass(type(yval), numpy.ndarray)) or
        (not numpy.asarray(yval).flags.f_contiguous) or
        (not (yval.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'yval' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        yval = numpy.asarray(yval, dtype=ctypes.c_float, order='F')
    yval_dim_1 = ctypes.c_int(yval.shape[0])
    
    # Setting up "xtab"
    if ((not issubclass(type(xtab), numpy.ndarray)) or
        (not numpy.asarray(xtab).flags.f_contiguous) or
        (not (xtab.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'xtab' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        xtab = numpy.asarray(xtab, dtype=ctypes.c_float, order='F')
    xtab_dim_1 = ctypes.c_int(xtab.shape[0])
    
    # Setting up "ytab"
    if ((not issubclass(type(ytab), numpy.ndarray)) or
        (not numpy.asarray(ytab).flags.f_contiguous) or
        (not (ytab.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'ytab' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        ytab = numpy.asarray(ytab, dtype=ctypes.c_float, order='F')
    ytab_dim_1 = ctypes.c_int(ytab.shape[0])
    
    # Setting up "mtab"
    if ((not issubclass(type(mtab), numpy.ndarray)) or
        (not numpy.asarray(mtab).flags.f_contiguous) or
        (not (mtab.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'mtab' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        mtab = numpy.asarray(mtab, dtype=ctypes.c_float, order='F')
    mtab_dim_1 = ctypes.c_int(mtab.shape[0])
    
    # Setting up "num"
    if (type(num) is not ctypes.c_int): num = ctypes.c_int(num)
    
    # Setting up "nume"
    if (type(nume) is not ctypes.c_int): nume = ctypes.c_int(nume)
    
    # Setting up "eps"
    if (type(eps) is not ctypes.c_float): eps = ctypes.c_float(eps)
    
    # Setting up "err"
    if (type(err) is not ctypes.c_int): err = ctypes.c_int(err)

    # Call C-accessible Fortran wrapper.
    clib.c_meval(ctypes.byref(xval_dim_1), ctypes.c_void_p(xval.ctypes.data), ctypes.byref(yval_dim_1), ctypes.c_void_p(yval.ctypes.data), ctypes.byref(xtab_dim_1), ctypes.c_void_p(xtab.ctypes.data), ctypes.byref(ytab_dim_1), ctypes.c_void_p(ytab.ctypes.data), ctypes.byref(mtab_dim_1), ctypes.c_void_p(mtab.ctypes.data), ctypes.byref(num), ctypes.byref(nume), ctypes.byref(eps), ctypes.byref(err))

    # Return final results, 'INTENT(OUT)' arguments only.
    return xval, yval, xtab, ytab, mtab, num.value, nume.value, eps.value, err.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine SEARCH

def search(xtab, num, s, lcn, fnd):
    '''!
!                                 SHAPE PRESERVING QUADRATIC SPLINES
!                                   BY D.F.MCALLISTER & J.A.ROULIER
!                                     CODED BY S.L.DODD & M.ROULIER
!                                       N.C. STATE UNIVERSITY
!
!
! SEARCH CONDUCTS A BINARY SEARCH FOR S.  SEARCH IS CALLED ONLY IF S IS
! BETWEEN XTAB(1) AND XTAB(NUM).
!
! ON INPUT--
!
!   XTAB CONTAINS THE ABSCISSAS OF THE DATA POINTS OF INTERPOLATION.
!
!   NUM IS THE DIMENSION OF XTAB.
!
!   S IS THE VALUE WHOSE RELATIVE POSITION IN XTAB IS LOCATED BY SEARCH.
!
!
! ON OUTPUT--
!
!   FND IS SET EQUAL TO 1 IF S IS FOUND IN XTAB AND IS SET EQUAL TO 0
!   OTHERWISE.
!
!   LCN IS THE INDEX OF THE LARGEST VALUE IN XTAB FOR WHICH XTAB(I)
!   .LT. S.
!
! AND
!
!   SEARCH DOES NOT ALTER XTAB,NUM,S.
!
!-----------------------------------------------------------------------'''
    
    # Setting up "xtab"
    if ((not issubclass(type(xtab), numpy.ndarray)) or
        (not numpy.asarray(xtab).flags.f_contiguous) or
        (not (xtab.dtype == numpy.dtype(ctypes.c_float)))):
        import warnings
        warnings.warn("The provided argument 'xtab' was not an f_contiguous NumPy array of type 'ctypes.c_float' (or equivalent). Automatically converting (probably creating a full copy).")
        xtab = numpy.asarray(xtab, dtype=ctypes.c_float, order='F')
    xtab_dim_1 = ctypes.c_int(xtab.shape[0])
    
    # Setting up "num"
    if (type(num) is not ctypes.c_int): num = ctypes.c_int(num)
    
    # Setting up "s"
    if (type(s) is not ctypes.c_float): s = ctypes.c_float(s)
    
    # Setting up "lcn"
    if (type(lcn) is not ctypes.c_int): lcn = ctypes.c_int(lcn)
    
    # Setting up "fnd"
    if (type(fnd) is not ctypes.c_int): fnd = ctypes.c_int(fnd)

    # Call C-accessible Fortran wrapper.
    clib.c_search(ctypes.byref(xtab_dim_1), ctypes.c_void_p(xtab.ctypes.data), ctypes.byref(num), ctypes.byref(s), ctypes.byref(lcn), ctypes.byref(fnd))

    # Return final results, 'INTENT(OUT)' arguments only.
    return xtab, num.value, s.value, lcn.value, fnd.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine CHOOSE

def choose(p1, p2, m1, m2, q1, q2, eps, ncase):
    '''!
!                                 SHAPE PRESERVING QUADRATIC SPLINES
!                                   BY D.F.MCALLISTER & J.A.ROULIER
!                                     CODED BY S.L.DODD & M.ROULIER
!
!
! CHOOSE DETERMINES THE CASE NEEDED FOR THE COMPUTATION OF THE PARAME-
! TERS OF THE QUADRATIC SPLINE AND RETURNS THE VALUE IN THE VARIABLE
! NCASE.
!
! ON INPUT--
!
!   (P1,P2) GIVES THE COORDINATES OF ONE OF THE POINTS OF INTERPOLATION.
!
!   M1 SPECIFIES THE DERIVATIVE CONDITION AT (P1,P2).
!
!   (Q1,Q2) GIVES THE COORDINATES OF ONE OF THE POINTS OF INTERPOLATION.
!
!   M2 SPECIFIES THE DERIVATIVE CONDITION AT (Q1,Q2).
!
!   EPS IS AN ERROR TOLERANCE USED TO DISTINGUISH CASES WHEN M1 OR M2 IS
!   RELATIVELY CLOSE TO THE SLOPE OR TWICE THE SLOPE OF THE LINE
!   SEGMENT JOINING (P1,P2) AND (Q1,Q2).  IF EPS IS NOT EQUAL TO ZERO,
!   THEN EPS SHOULD BE GREATER THAN OR EQUAL TO MACHINE EPSILON.
!
!
! ON OUTPUT--
!
!   NCASE CONTAINS THE VALUE WHICH CONTROLS HOW THE PARAMETERS OF THE
!   QUADRATIC SPLINE ARE EVALUATED.
!
! AND
!
!   CHOOSE DOES NOT ALTER P1,P2,Q1,Q2,M1,M2,EPS.
!
!-----------------------------------------------------------------------'''
    
    # Setting up "p1"
    if (type(p1) is not ctypes.c_float): p1 = ctypes.c_float(p1)
    
    # Setting up "p2"
    if (type(p2) is not ctypes.c_float): p2 = ctypes.c_float(p2)
    
    # Setting up "m1"
    if (type(m1) is not ctypes.c_float): m1 = ctypes.c_float(m1)
    
    # Setting up "m2"
    if (type(m2) is not ctypes.c_float): m2 = ctypes.c_float(m2)
    
    # Setting up "q1"
    if (type(q1) is not ctypes.c_float): q1 = ctypes.c_float(q1)
    
    # Setting up "q2"
    if (type(q2) is not ctypes.c_float): q2 = ctypes.c_float(q2)
    
    # Setting up "eps"
    if (type(eps) is not ctypes.c_float): eps = ctypes.c_float(eps)
    
    # Setting up "ncase"
    if (type(ncase) is not ctypes.c_int): ncase = ctypes.c_int(ncase)

    # Call C-accessible Fortran wrapper.
    clib.c_choose(ctypes.byref(p1), ctypes.byref(p2), ctypes.byref(m1), ctypes.byref(m2), ctypes.byref(q1), ctypes.byref(q2), ctypes.byref(eps), ctypes.byref(ncase))

    # Return final results, 'INTENT(OUT)' arguments only.
    return p1.value, p2.value, m1.value, m2.value, q1.value, q2.value, eps.value, ncase.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine CASES

def cases(p1, p2, m1, m2, q1, q2, e1, e2, v1, v2, w1, w2, z1, z2, y1, y2, ncase):
    '''!
!                                 SHAPE PRESERVING QUADRATIC SPLINES
!                                   BY D.F.MCALLISTER & J.A.ROULIER
!                                     CODED BY S.L.DODD & M.ROULIER
!                                       N.C. STATE UNIVERSITY
!
!
! CASES COMPUTES THE KNOTS AND OTHER PARAMETERS OF THE SPLINE ON THE
! INTERVAL (P1,Q1).
!
!
! ON INPUT--
!
!   (P1,P2) AND (Q1,Q2) ARE THE COORDINATES OF THE POINTS OF
!   INTERPOLATION.
!
!   M1 IS THE SLOPE AT (P1,P2).
!
!   M2 IS THE SLOPE AT (Q1,Q2)
!
!   NCASE CONTROLS THE NUMBER AND LOCATION OF THE KNOTS.
!
!
! ON OUTPUT--
!
!   (V1,V2),(W1,W2),(Z1,Z2), AND (E1,E2) ARE THE COORDINATES OF THE
!   KNOTS AND OTHER PARAMETERS OF THE SPLINE ON (P1,Q1).  (E1,E2)
!   AND (Y1,Y2) ARE USED ONLY IF NCASE=4.
!
! AND
!
!   CASES DOES NOT ALTER P1,P2,M1,M2,Q1,Q2.
!
!-----------------------------------------------------------------------'''
    
    # Setting up "p1"
    if (type(p1) is not ctypes.c_float): p1 = ctypes.c_float(p1)
    
    # Setting up "p2"
    if (type(p2) is not ctypes.c_float): p2 = ctypes.c_float(p2)
    
    # Setting up "m1"
    if (type(m1) is not ctypes.c_float): m1 = ctypes.c_float(m1)
    
    # Setting up "m2"
    if (type(m2) is not ctypes.c_float): m2 = ctypes.c_float(m2)
    
    # Setting up "q1"
    if (type(q1) is not ctypes.c_float): q1 = ctypes.c_float(q1)
    
    # Setting up "q2"
    if (type(q2) is not ctypes.c_float): q2 = ctypes.c_float(q2)
    
    # Setting up "e1"
    if (type(e1) is not ctypes.c_float): e1 = ctypes.c_float(e1)
    
    # Setting up "e2"
    if (type(e2) is not ctypes.c_float): e2 = ctypes.c_float(e2)
    
    # Setting up "v1"
    if (type(v1) is not ctypes.c_float): v1 = ctypes.c_float(v1)
    
    # Setting up "v2"
    if (type(v2) is not ctypes.c_float): v2 = ctypes.c_float(v2)
    
    # Setting up "w1"
    if (type(w1) is not ctypes.c_float): w1 = ctypes.c_float(w1)
    
    # Setting up "w2"
    if (type(w2) is not ctypes.c_float): w2 = ctypes.c_float(w2)
    
    # Setting up "z1"
    if (type(z1) is not ctypes.c_float): z1 = ctypes.c_float(z1)
    
    # Setting up "z2"
    if (type(z2) is not ctypes.c_float): z2 = ctypes.c_float(z2)
    
    # Setting up "y1"
    if (type(y1) is not ctypes.c_float): y1 = ctypes.c_float(y1)
    
    # Setting up "y2"
    if (type(y2) is not ctypes.c_float): y2 = ctypes.c_float(y2)
    
    # Setting up "ncase"
    if (type(ncase) is not ctypes.c_int): ncase = ctypes.c_int(ncase)

    # Call C-accessible Fortran wrapper.
    clib.c_cases(ctypes.byref(p1), ctypes.byref(p2), ctypes.byref(m1), ctypes.byref(m2), ctypes.byref(q1), ctypes.byref(q2), ctypes.byref(e1), ctypes.byref(e2), ctypes.byref(v1), ctypes.byref(v2), ctypes.byref(w1), ctypes.byref(w2), ctypes.byref(z1), ctypes.byref(z2), ctypes.byref(y1), ctypes.byref(y2), ctypes.byref(ncase))

    # Return final results, 'INTENT(OUT)' arguments only.
    return p1.value, p2.value, m1.value, m2.value, q1.value, q2.value, e1.value, e2.value, v1.value, v2.value, w1.value, w2.value, z1.value, z2.value, y1.value, y2.value, ncase.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine SPLINE

def spline(xvals, z1, z2, xtabs, ytabs, xtabs1, ytabs1, y1, y2, e2, w2, v2, ncase):
    '''!
!                                 SHAPE PRESERVING QUADRATIC SPLINES
!                                   BY D.F.MCALLISTER & J.A.ROULIER
!                                     CODED BY S.L.DODD & M.ROULIER
!                                       N.C. STATE UNIVERSITY
!
!
!   SPLINE FINDS THE IMAGE OF A POINT IN XVAL.
!
! ON INPUT--
!
!   XVALS CONTAINS THE VALUE AT WHICH THE SPLINE IS EVALUATED.
!
!   (XTABS,YTABS) ARE THE COORDINATES OF THE LEFT-HAND DATA POINT
!   USED IN THE EVALUATION OF XVALS.
!
!   (XTABS1,YTABS1) ARE THE COORDINATES OF THE RIGHT-HAND DATA POINT
!   USED IN THE EVALUATION OF XVALS.
!
!   Z1,Z2,Y1,Y2,E2,W2,V2 ARE THE PARAMETERS OF THE SPLINE.
!
!   NCASE CONTROLS THE EVALUATION OF THE SPLINE BY INDICATING WHETHER
!   ONE OR TWO KNOTS WERE PLACED IN THE INTERVAL (XTABS,XTABS1).
!
!
! ON OUTPUT--
!
!   SPLINE IS THE IMAGE OF XVALS.
!
! AND
!
!   SPLINE DOES NOT ALTER ANY OF THE INPUT PARAMETERS.
!
!-----------------------------------------------------------------------'''
    
    # Setting up "xvals"
    if (type(xvals) is not ctypes.c_float): xvals = ctypes.c_float(xvals)
    
    # Setting up "z1"
    if (type(z1) is not ctypes.c_float): z1 = ctypes.c_float(z1)
    
    # Setting up "z2"
    if (type(z2) is not ctypes.c_float): z2 = ctypes.c_float(z2)
    
    # Setting up "xtabs"
    if (type(xtabs) is not ctypes.c_float): xtabs = ctypes.c_float(xtabs)
    
    # Setting up "ytabs"
    if (type(ytabs) is not ctypes.c_float): ytabs = ctypes.c_float(ytabs)
    
    # Setting up "xtabs1"
    if (type(xtabs1) is not ctypes.c_float): xtabs1 = ctypes.c_float(xtabs1)
    
    # Setting up "ytabs1"
    if (type(ytabs1) is not ctypes.c_float): ytabs1 = ctypes.c_float(ytabs1)
    
    # Setting up "y1"
    if (type(y1) is not ctypes.c_float): y1 = ctypes.c_float(y1)
    
    # Setting up "y2"
    if (type(y2) is not ctypes.c_float): y2 = ctypes.c_float(y2)
    
    # Setting up "e2"
    if (type(e2) is not ctypes.c_float): e2 = ctypes.c_float(e2)
    
    # Setting up "w2"
    if (type(w2) is not ctypes.c_float): w2 = ctypes.c_float(w2)
    
    # Setting up "v2"
    if (type(v2) is not ctypes.c_float): v2 = ctypes.c_float(v2)
    
    # Setting up "ncase"
    if (type(ncase) is not ctypes.c_int): ncase = ctypes.c_int(ncase)
    
    # Setting up "spline_result"
    spline_result = ctypes.c_float()

    # Call C-accessible Fortran wrapper.
    clib.c_spline(ctypes.byref(xvals), ctypes.byref(z1), ctypes.byref(z2), ctypes.byref(xtabs), ctypes.byref(ytabs), ctypes.byref(xtabs1), ctypes.byref(ytabs1), ctypes.byref(y1), ctypes.byref(y2), ctypes.byref(e2), ctypes.byref(w2), ctypes.byref(v2), ctypes.byref(ncase), ctypes.byref(spline_result))

    # Return final results, 'INTENT(OUT)' arguments only.
    return xvals.value, z1.value, z2.value, xtabs.value, ytabs.value, xtabs1.value, ytabs1.value, y1.value, y2.value, e2.value, w2.value, v2.value, ncase.value, spline_result.value



# Given data points and values, construct and return a function that
# evaluates a shape preserving quadratic spline through the data.
def spline_fit(x, y, pts=1000):
    import numpy as np
    n = len(x)
    x = np.asarray(x, dtype=np.float32)
    x_min = x.min()
    x_max = x.max()
    y = np.asarray(y, dtype=np.float32)
    m = np.zeros(n, dtype=np.float32)
    # Get the slopes for the quadratic spline interpolant.
    slopes(x, y, m, n)
    # Define a function for evaluating the quadratic spline.
    def eval_quadratic(z, x=x, y=y, m=m):
        # Make sure "z" is an array.
        if (not issubclass(type(z), np.ndarray)): z = [z]
        # Clip input positions to be inside evaluated range.
        z = np.clip(z, min(x), max(x))
        # Initialize all arguments for calling TOMS 574 code.
        k = len(z)
        z = np.asarray(z, dtype=np.float32)
        z.sort()
        fz = np.zeros(k, dtype=np.float32)
        eps = 2**(-13)
        err = 0
        # Call TOMS 574 code and check for errors.
        err = meval(z, fz, x, y, m, n, k, eps, err)[-1]
        assert (err == 0), f"Nonzero error '{err}' when evaluating spline."
        # Return values.
        return fz
    def deriv_1(z, s=.001):
        try: return [deriv_1(v) for v in z]
        except:
            if min(abs(z - x_min), abs(z - x_max)) < 0.0001: return 0.0
            from tlux.math import fit_polynomial
            # Construct a local quadratic approximation.
            px = np.array([z-s, z, z+s])
            py = eval_quadratic(px)
            f = fit_polynomial(px, py)
            return f.derivative()(z)
    def deriv_2(z, s=.001):
        try: return [deriv_2(v) for v in z]
        except:
            if (min(abs(z - x_min), abs(z - x_max)) <= s): return 0.0
            from tlux.math import fit_polynomial
            # Construct a local quadratic approximation.
            # px = np.array([z-s, z-s/2, z, z+s/2, z+s])
            px = np.array([z-s, z, z+s])
            py = eval_quadratic(px)
            f = fit_polynomial(px, py)
            return f.derivative().derivative()(z)
    deriv_1.derivative = deriv_2
    eval_quadratic.derivative = deriv_1
    return eval_quadratic
