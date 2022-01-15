SUBROUTINE DBVSIS ( X , Y , NP , N , K , OPT , D0 , DNP , D20 , D2NP , CONSTR , EPS , BETA , BETAI , RHO , RHOI , KMAX , MAXSTP , XTAB , NTAB , SBOPT , Y0OPT , Y1OPT , Y2OPT , ERRC , ERRE , D , D2 , DIAGN , Y0TAB , Y1TAB , Y2TAB , WORK , NWORK )

!  DBVSIS is merely a support routine which calls DBVSSC and DBVSSE
!  for the computation of the needed parameters and for the evaluation
!  of a shape-preserving, C(k), k=1,2 , interpolating spline,
!  optionally subject to boundary conditions.
!  The use of DBVSIS is not recommended when more evaluations of the
!  same spline are required; in this case it is better to separately
!  call DBVSSC and then DBVSSE repeatedly.
!  For an explanation of input and output parameters, the user is
!  referred to the comments of DBVSSC and DBVSSE.



INTEGER NP , N , K , OPT , CONSTR ( 0 : NP - 1 ) , NTAB , KMAX , MAXSTP , SBOPT , Y0OPT , Y1OPT , Y2OPT , DIAGN ( 0 : NP - 1 ) , ERRC , ERRE , NWORK

REAL ( KIND ( 0.0D0 ) ) X ( 0 : NP ) , Y ( 0 : NP ) , D0 , DNP , D20 , D2NP , XTAB ( 0 : NTAB ) , EPS , BETA , BETAI , RHO , RHOI , D ( 0 : NP ) , D2 ( 0 : NP ) , Y0TAB ( 0 : NTAB ) , Y1TAB ( 0 : NTAB ) , Y2TAB ( 0 : NTAB ) , WORK ( 1 : NWORK )

END

!  ---------------------------------------------------------------------

SUBROUTINE DBVSSC ( X , Y , NP , N , K , OPT , D0 , DNP , D20 , D2NP , CONSTR , EPS , BETA , BETAI , RHO , RHOI , KMAX , MAXSTP , ERRC , D , D2 , DIAGN , WORK , NWORK )

!  -------------------------------------------------
!            Lines 49-549 are comment lines.
!            The actual code begins at line 555.
!  -------------------------------------------------

!  ABSTRACT:
!
!  DBVSSC is designed to compute the coefficients (first and, if
!  appropriate, second derivatives) of a shape-preserving spline, of
!  continuity class C(k), k=1,2 , which interpolates a set of data
!  points and, if required, satisfies additional boundary conditions.
!  DBVSSC furnishes the input parameters for DBVSSE, which, in turn,
!  provides to evaluate the spline and its derivatives at a set of
!  tabulation points.
!
!  The user is allowed to use the following options:
!
!  - to compute a spline subject to:
!        - no constraint,
!        - monotonicity constraints,
!        - convexity constraints,
!        - monotonicity and convexity constraints,
!        - one of the above constraints in each subinterval;
!
!  - to impose separable or non-separable boundary conditions on the
!    spline,
!
!  - to assign the first derivatives d(i), i=0,1,...,np , in input or to
!    compute them from the constraints only or as the best approximation
!    to a set of optimal values. Although the final sequence of
!    derivatives does in any case satisfy the imposed restrictions on
!    the shape, the resulting graphs may exhibit different behaviors.
!
!
!  REMARK:
!
!  In these comments variable and array names will be denoted with
!  capital letters, and their contents with small letters. Moreover:
!  .IN.   means belonging to;
!  .INT.  stands for intersection.
!
!
!  The code has the following structure:
!
!         DBVSSC
!              DBVC
!                   DSTINF
!                        DMSK1
!                        DMSK2
!                             DPRJ0
!                             DPRJ1
!                        DTDC
!                             DMNMOD
!                             DMDIAN
!                   DALG3
!                        DPRJ0
!                        DALG1
!                             DPRJ1
!                        DINTRS
!                        DTST
!                        DFPSVF
!                             DSL
!                             DALG1D
!                                  DPRJ1
!                        DAL2
!                             DPRJ2
!                        DAL2DP
!                             DMNIND
!                             DPRJ2
!                             DSL
!                        DSCDRC
!
!
!  CALLING SEQUENCE:
!
!       CALL DBVSSC (X,Y,NP,N,K,OPT,D0,DNP,D20,D2NP,CONSTR,EPS,BETA,
!    *               BETAI,RHO,RHOI,KMAX,MAXSTP,ERRC,D,D2,DIAGN,
!    *               WORK,NWORK)
!
!
!  INPUT PARAMETERS:
!
!  X       : floating array, of bounds 0:NP, containing the data
!            abscissas  x(i), i=0,1,...,np.
!            Restriction: x(i).LT.x(i+1), i=0,1,...,np.
!  Y       : floating array, of bounds 0:NP, containing the data
!            ordinates  y(i), i=0,1,...,np.
!  NP      : integer variable, defining the number of interpolation
!            points. Restriction: np.GE.2 .
!  N       : integer variable, containing the degree of s.
!            Restriction: n.GE.3 .
!  K       : integer variable, containing the class of continuity of s.
!            Restriction:  k.EQ.1  or  k.EQ.2  and  n.GE.3*k .
!  OPT     : integer variable, containing a control parameter. It is
!            a three-digit decimal of the form  pqr  (that is of
!            numerical value  p*100+q*10+r ) where:
!            r  controls the constraints on the shape.
!            q  controls the boundary conditions and
!            p  controls the computation of the derivatives,
!            More specifically:
!            r=0 (opt=pq0) : no constraint on the shape is required;
!            r=1 (opt=pq1) : monotonicity constraints are required;
!            r=2 (opt=pq2) : convexity constraints are required;
!            r=3 (opt=pq3) : monotonicity and convexity constraints are
!                            required;
!            r=4 (opt=pq4) : local constraints for any subinterval are
!                            supplied by the user (see the description
!                            of the array CONSTR);
!            q=1 (opt=p1r) : no boundary condition is imposed;
!            q=2 (opt=p2r) : non-separable boundary conditions are
!                            imposed (see the description of BETA,
!                            BETAI, RHO, RHOI);
!            q=3 (opt=p3r) : separable boundary conditions are imposed
!                            (see the description of D0, DNP, D20,
!                             D2NP);
!            p=1 (opt=1qr) : the sequence of first derivatives
!                            d(0),....,d(np)  is computed using the
!                            constraints only using subroutine DAL2;
!            p=2 (opt=2qr) : the sequence is computed as the constrained
!                            best approximation to Bessel derivatives
!                            using subroutine DAL2DP;
!            p=3 (opt=3qr) : the sequence is computed as the constrained
!                            best approximation to a set of third order
!                            accurate derivative estimates produced in
!                            subroutine DTDC using subroutine DAL2DP
!                            (since this estimates are inherently mono-
!                            tonicity preserving, it is not recommended
!                            to associate this option with the convexity
!                            constraints only);
!            p=4 (opt=4qr) : the sequence is computed as the constrained
!                            best approximation to a set of values given
!                            in input by the user using DAL2DP; note
!                            that opt.EQ.410 will provide the classical
!                            C(k) function interpolating the data and
!                            the derivatives.
!         Restriction: ( p.GE.1 .AND. p.LE.4 ) .AND.
!                      ( q.GE.1.AND. q.LE.3 ) .AND.
!                      ( r.GE.0 .AND. r.LE.4 ) .AND.
!                      .NOT. ( r.EQ.0 .AND. p.EQ.1 ) .
!  D0      : floating variable containing the left separable boundary
!            condition for the first derivative (d(0)=d0).
!            D0 is referenced only when  q=3 .
!  DNP     : floating variable containing the right separable boundary
!            condition for the first derivative (d(np)=dnp).
!            DNP is referenced only when  q=3 .
!  D20     : floating variable containing the left separable boundary
!            condition for the second derivative (d2(0)=d20).
!            D20 is referenced only when  q=3  and  k=2 .
!  D2NP    : floating variable containing the right separable boundary
!            condition for the second derivative (d2(np)=d2np).
!            D2NP is referenced only when  q=3  and  k=2 .
!  EPS     : floating variable, containing a value for computing the
!            relative tolerance of the method. It should be set greater
!            or equal to the machine precision. However, if eps.LE.0,
!            DBVSSC resets it to 0.0001 which has turned out to be a
!            good choice for graphical applications.
!  BETA    : user supplied function, which represents non-separable
!            boundary conditions for the first derivatives.
!            BETA is referenced only when  q=2 .
!  BETAI   : user supplied function, which is the inverse of BETA.
!            BETAI is referenced only when  q=2 .
!  RHO     : user supplied function, which represents non-separable
!            boundary conditions for the second derivatives.
!            RHO is referenced only when  q=2  and  k=2 .
!  RHOI    : user supplied function, which is the inverse of RHO.
!            RHOI is referenced only when  q=2  and  k=2 .
!  KMAX    : integer variable, containing the number of iterations
!            allowed for selecting the minimal set ASTAR described
!            below. If kmax.LE.0, DBVSSC resets it to 10 .
!            KMAX is referenced only when  q=2 .
!  MAXSTP  : integer variable, containing the number of steps allowed
!            to find dstar in the set of admissible values.
!            If maxstp.LE.0, DBVSSC resets it to 10 .
!            MAXSTP is referenced only when  q=2 .
!
!
!  INPUT / OUTPUT PARAMETERS:
!
!  CONSTR  : integer array, of bounds  0:NP , containing, in input the
!            desired constraints on the shape for each subinterval.
!            constr(i)=kind , kind=0,1,2,3 , means that none, monotoni-
!            city, convexity, monotonicity and convexity constraint is
!            imposed on the subinterval [x(i),x(i+1)]. If constr(i) is
!            not compatible with the data it is relaxed according to
!            their shape (see subroutine DMSK1 for details). So, on out-
!            put, CONSTR contains the constraints actually imposed.
!            For example, if the data are convex and on input we have
!            constr(i)=3 , the result in output will be  constr(i)=2.
!            Restriction: constr(i).GE.0 .AND. constr(i).LE.3 .
!            CONSTR is referenced only when  r=4 .
!  D       : floating array, of bounds 0:NP, containing the first
!            derivatives at  x(i), i=0,1,...,np . If  p=4 , d(i) is the
!            input value to be approximated by the computed derivative,
!            which is then stored in the same location.
!            On output, D is computed by the routine DAL2 if  p=1  and
!            is computed by the routine DAL2DP if  p=2  or  p=3 .
!
!
!  OUTPUT PARAMETERS
!
!  ERRC    : integer variable, containing an error flag which displays
!            the status of the code. The status is divided into: severe
!            error (error on the input data, no computation has been
!            done), error (some computation has been done and some
!            information or suggestions are available), warning (some
!            requirement is not fulfilled, but the spline's parameters
!            have been computed), success.
!            errc=0 : success, normal return of the code;
!            errc=1 : severe error, incorrect assignment for some of
!                     the values nwork, opt, np;
!            errc=2 : severe error, for some i the restriction
!                     0.LE.constr(i) .AND. constr(i).LE.3  is not
!                     fulfilled;
!            errc=3 : severe error, incorrect assignment for some of
!                     the values n,k;
!            errc=4 : severe error, the restriction x(i).LT.x(i+1) is
!                     not fulfilled for some i;
!            errc=5 : error, the problem does not have any solution
!                     because the set
!                     betai ( phi(a(0,k)) .INT. beta(a(0,k)) )
!                     is empty for some k. In other words the boundary
!                     conditions cannot be satisfied and the output
!                     parameters are meaningless.
!                     The user is suggested to increase the value of n.
!            errc=6 : warning; for some i, the constraints on the
!                     interval  [x(i),x(i+1)]  are too strong and they
!                     have not been considered. There is no guarantee
!                     that the spline is shape-preserving within all
!                     the intervals. More accurate diagnostic details
!                     can be found in the array DIAGN.
!                     The user is suggested to increase the value of n.
!            errc=7 : error, dstar such that beta(dstar).IN.phi(dstar)
!                     has not been found. The integer parameter maxstp
!                     should be increased.
!                     The output parameters are meaningless.
!            errc=8 : error, both situations described in errc=6 and
!                     errc=7  have occurred.
!            errc=9 : warning, one of the separable boundary conditions
!                     d(0)=d0  and/or  d(np)=dnp  are not compatible
!                     with the constraints in  [x(0),x(1)]  and/or
!                     [x(np-1),x(np)]  which have consequently been
!                     relaxed. The user is suggested to increase the
!                     value of n. More accurate diagnostic details can
!                     be found in the array DIAGN.
!            errc=10: warning, both situations described for errc=6 and
!                     errc=9 have occurred.
!            errc=11: warning, one of the separable boundary conditions
!                     d2(0)=d20  and/or  d2(np)=d2np  are not compatible
!                     with the constraints in  [x(0),x(1)]  and/or
!                     [x(np-1),x(np)] . The boundary conditions have
!                     consequently been approximated. The user is
!                     suggested to increase the value of n.
!            errc=12: warning, both situations described for errc=6 and
!                     errc=11 have occurred.
!            errc=13: warning, both situations described for errc=9 and
!                     errc=11 have occurred.
!            errc=14: warning, both situations described for errc=10 and
!                     errc=11 have occurred.
!            errc=15: warning, the non-separable boundary conditions
!                     d2(np)=rho(d2(0))  are not compatible with the
!                     constraints. The boundary conditions have
!                     consequently been approximated. The user is
!                     suggested to increase the value of n.
!            errc=16: warning, both situations described for errc=6 and
!                     errc=15 have occurred.
!            errc=17: warning, both situations described for errc=9 and
!                     errc=15 have occurred.
!            errc=18: warning, both situations described for errc=10 and
!                     errc=15 have occurred.
!  D2      : floating array of bounds 0:NP containing the second
!            derivatives at knots. D2 is computed in subroutine DCDERC .
!            D2 is referenced only when  k=2 .
!  DIAGN   : integer array of bounds 0:NP-1 containing further
!            diagnostic information:
!            diagn(i)=0 : the constraints in the interval [x(i),x(i+1)]
!                         have been satisfied;
!            diagn(i)=1 : the constraints in the interval [x(i),x(i+1)]
!                         have not been satisfied;
!
!
!
!  OTHER PARAMETERS:
!
!  WORK    : floating array, of bounds 1:NKORK, which is used as
!            a work area to store intermediate results.
!            The same array can be used to provide workspace for both
!            the main subroutines  DBVSSC and DBVSSE .
!  NWORK   : integer variable containing the size of the work area.
!            Restriction: nwork .GE. comm+(part+7)*np+(n*(n+11))/2+9
!                           that is
!                         nwork .GE. 5+(2+7)*np+(n*(n+11))/2+9
!
!
!  ------------------------------------------------
!
!  METHOD:
!
!  Let the integers n and k, such that k=1,2 ; n >= 3k , and the
!  sequence of points  (x(i),y(i)), i=0,1,...,np , with
!  x(0) < x(1) < ... <x(np) , be given; let us denote with  BS(n;k)
!  the set of the splines s of degree n and continuity k whose second
!  derivative, in the case k=2 , vanishes at the knots. We are
!  interested in the existence and construction, if possible, of a
!  shape-preserving interpolating spline s of BS(n;k) such that
!
!            s(x(i)) = y(i) , i=0,1,...,np                          (1)
!
!  and optionally subject to constraints on the shape in each interval
!  [x(i),x(i+1)] .
!
!  In the case k=2 the zero derivatives of the spline  s.IN.BS(n;k) are
!  then modified to assume non-zero values which are not in contrast
!  with the shape constraints and, if possible, satisfy the boundary
!  conditions eventually imposed by the user. For details we refer to
!  the comments in subroutine DCSDRC.
!
!  Since any s.IN.BS(n;k) is determined by its values and slopes at
!  x(i) , i=0,1,...,np , we can reformulate the problem as follows:
!  compute the values  d(i), i=0,1,...,np , such that the spline s,
!  satisfying (1) and
!
!            Ds(x(i)) := d(i) , i=0,1,...,np                        (2)
!
!  is shape-preserving.
!  Setting  delta(i) := (y(i+1)-y(i))/(x(i+1)-x(i)) , we have that s is
!  increasing (I) ( decreasing (D) ) in [x(i),x(i+1)] if and only if
!  (d(i),d(i+1))  belongs to
!
!    D(i) := { (u,v).IN.RxR : u >= 0, v >= 0, v =< -u+ n/k delta(i) }
!                                                                    (3)
!  ( D(i) := { (u,v).IN.RxR : u =< 0, v =< 0, v >= -u+ n/k delta(i) } )
!
!  s is convex (CVX) ( concave (CNC) ) if and only if (d(i),d(i+1))
!  belongs to
!
!    D(i) := { (u,v).IN.RxR : v >= - (k/(n-k)) u + (n/(n-k)) delta(i) ,
!                             v =< - ((n-k)/k) u + (n/k) delta(i) }
!                                                                    (4)
!  ( D(i) := { (u,v).IN.RxR : v =< - (k/(n-k)) u + (n/(n-k)) delta(i) ,
!                             v >= - ((n-k)/k) u + (n/k) delta(i) }  )
!
!  and that s is I (D) and CVX (CNC) if and only if (d(i),d(i+1))
!  belongs to
!
!             D(i) := { (u,v) satisfying (3) and (4) } .
!
!  So, if we choose the family of sets D(i) , i=0,1,...,np-1 , according
!  to the shape of the data, we have to solve:
!
!  PROBLEM P1. Does a sequence ( d(0), d(1), ..., d(np) ) such that
!              (d(i),d(i+1)) .IN. D(i) , i=0,1,...,np-1 , exist ?
!
!  PROBLEM P2. If P1 is feasible, how can a (the best) solution be
!              computed ?
!
!  Let DPRJ1: RxR -> R and DPRJ2: RxR -> R be, respectively, the
!  projection maps from uv-plane onto the u-axis and v-axis and let us
!  denote with  B(i) := DPRJ1(D(i)) :
!
!      ALGORITHM A1[B0].
!        1. Set A(0):=B(0); J:=np.
!        2. For i=1,2,...,np
!           2.1. Set A(i):= DPRJ2( D(i-1).INT.{ A(i-1) x B(i) } ) .
!           2.2. If A(i) is empty, set J:=i and stop.
!        3. Stop.
!
!  We have the following result:
!
!  THEOREM 1. P1 has a solution if, and only if, J=np, that is A(i) is
!             not empty , i=0,1,...,np . If ( d(0), d(1), ...,d(np) )
!             is a solution then  d(i).IN.A(i) , i=0,1,...,np .
!
!  A solution can be computed with the following algorithm:
!
!      ALGORITHM A2[A(np),B0].
!        1. Choose d(np).IN.A(np).
!        2. For i=np-1, np-2, ..., 0
!           2.1. Choose d(i).IN.DPRJ1( D(i).INT.{ A(i) x { d(i+1) }}).
!        3. Stop.
!
!  For more theoretical details about A1 and A2 see \1\ , and for
!  practical details see subprograms DALG1, DAL2, DAL2DP. In the latter
!  a dynamic programming scheme is used to find the best solution in
!  the feasible set. More specifically, it is possible to compute the
!  values  d(i),i=0,..,np which satisfy the constraints and are as close
!  as possible to another sequence which does not satisfy the
!  constraints but is, in some sense, optimal.
!
!  From a theoretical point of view, algs A1 and A2 give a complete
!  answer to problems P1 and P2. However, it could be pointed out that,
!  for practical applications, we would like to have the best possible
!  plot, whether or not P1 has solutions. Let us suppose that the
!  problem is solvable from 0 to j and from j to np, but that alg A1
!  applied to the whole family of sets  D(i), i=0,1,...,np-1  gives
!  J.eq.j.ne.np ; if we reset  D(j-1) := A(j-1) x B(j) , alg A1 applied
!  to this new family of sets will produce J=np . However, it must be
!  recalled that, in this way, we do not consider the constraints in the
!  interval [x(j-i),x(j)] and so there is no guarantee that the spline
!  is shape-preserving in this interval. Whenever this fact cannot be
!  accepted it is convenient to rerun the code with a larger value for
!  the degree n , because the domains of constraints enlarge as n
!  increases (see (3) and (4)).
!
!  It is immediate to see that separable boundary conditions of the form
!
!            d(0) := d0 ; d(np) := dnp
!
!  can be easily inserted with a reduction of the corresponding
!  admissible sets which does not modify the above theory:
!
!       D(0) := D(0).INT.{d(0)=d0} ; D(np) := D(np).INT.{d(np)=dnp}
!
!  In the case k=2 the corresponding conditions  d2(0) = d20 ,
!  d2(np) = d2np  are imposed only if not in contrast with the shape of
!  the data; otherwise the admissible values for  d2(0) and d2(np)
!  respectively closest to d20 and d2np are chosen.
!
!  Now, let beta be a continuous function from R to R, with continuous
!  inverse betai, we want to solve the following non-separable boundary
!  valued problem:
!
!  PROBLEM P3. Do sequences ( d(0), d(1), ..., d(np) ) , such that
!              (d(i),d(i+1)).IN.D(i), i=0,1,...,np-1    and
!              d(np) = beta ( d(0) ) , exist ?
!
!  It is obvious that a solution of this new problem, if it exists, can
!  be found amongst the solutions of P1. Let A(0), A(1),...,A(np) be the
!  sequence of sets given by alg A1 (we assume that A(i) is not empty,
!  i=0,1,...,np , that is P1 is solvable or, if this is not the case,
!  the constraints have been relaxed ), we can assume that
!  A(np) = phi(A(0)) , where  phi: R -> R is a set valued function
!  (see \1\ for details). It can be demonstrated that:
!
!  THEOREM 2. P1 is solvable if, and only if, there is  dstar.IN.A(0)
!             such that   beta(dstar).IN.phi({dstar}) .
!
!  It should be noted that if ( d(0), d(1), ..., d(np) ) satisfies P1,
!       d(0) .IN. betai(phi(A(0)).INT.beta(A(0))) =: gamma(A(0))
!  and, consequently, the set of admissible values is reduced. If we
!  repeat this procedure, we get a gradually diminishing admissible set
!  for d(0). We define
!     ASTAR := lim A(0,m)  where
!     A(0,0) := A(0)   and   A(0,m) := gamma(A(0,m-1)) ;
!  ASTAR is the minimal admissible set for dstar. We can now combine the
!  various theorems and algorithms and give the general algorithm to
!  solve P3:
!
!      ALGORITHM A3.
!        1. Set A(0,0) := B0 ; m:=1.
!        2. Use A1[A(0,0)] for computing phi (A(0,0)).
!        3. Set A(0,1) := gamma(A(0,0))
!                       = betai(phi(A(0,0)).INT.beta(A(0,0))).
!        4. If A(0,1) is empty, stop (P1 is unsolvable).
!        5. While ( convergence test not satisfied ) do
!           5.1. Use A1[A(0,m)] for computing A(np,m) = phi (A(0,m)).
!           5.2. Set A(0,m+1) := gamma(A(0,m)).
!           5.3. Set m:=m+1.
!        6. Set ASTAR := A(0,m).
!        7. Use A1[{d(0)}] to find dstar.IN.ASTAR such that
!           beta(dstar).IN.phi(dstar).
!        8. Use A2[beta(dstar),dstar] for computing a sequence
!           ( d(0), d(1), ..., d(np) )  which solves P1.
!
!  In the case k=2 the corresponding condition  d2(np) = beta2(d2(0))
!  is imposed only if not in contrast with the shape of
!  the data; otherwise the admissible values for  d2(0) and d2(np)
!  closest to the boundary condition are chosen.
!
!  References
!
!  \1\ P.Costantini: Boundary Valued Shape-Preserving Interpolating
!      Splines, ACM Trans. on Math. Softw., companion paper.
!  \2\ R.Bellman, S.Dreyfus: Applied Dynamic Programming, Princeton
!      University Press, New York, 1962.
!  \3\ H.T.Huynh: Accurate Monotone Cubic Interpolation, SIAM J. Num.
!      Anal., 30 (1993), 57-100.
!
!  The ideas involved in Algorithm A3 have been implemented in the code
!  in a general form. Since Algorithm A3 resembles closely the abstract
!  formulation it could, therefore, be used for several practical
!  problems. The particular case actually treated is reflected in the
!  contents of the information array INFO (see its description in
!  subroutine DSTINF) which contains all the data needed for the
!  construction of the operators DPRJ0, DPRJ1 and DPRJ2.
!
!  As a consequence, the user has the following options:
!
!  - to compute a Spline subject to:
!        - no constraint;
!        - monotonicity constraints,
!        - convexity constraints,
!        - monotonicity and convexity constraints,
!        - one of the above constraints in each subinterval, as
!          specified in the corresponding array CONSTR;
!
!  - to impose separable or non-separable boundary conditions on the
!    spline. In the latter case, the external functions BETA, BETAI,
!    RHO and RHOI must be supplied,
!
!  - to assign the first derivatives d(i), i=0,1,...,np , in input or to
!    compute them from the only constraints or as the best approximation
!    to a set of optimal values. Although the final sequence of
!    derivatives does in any case satisfy the imposed restrictions on
!    the shape, the resulting graphs may exhibit different behaviors.
!
!  See the description of the input parameter OPT for more details.

!  ------------------------------------------------
!            End of comments.
!  ------------------------------------------------

INTEGER COMM , PART

!  COMM contains the number of global data referring to the initial
!  points  (x(i),y(i)) stored in the array INFO, described in
!  subroutine DSTINF.
!  PART contains the number of particular data referring to each
!  interval  (x(i),x(i+1)) , i=0,1,...,np , stored in the array INFO.



INTEGER NP , N , K , OPT , CONSTR ( 0 : NP - 1 ) , KMAX , MAXSTP , ERRC , DIAGN ( 0 : NP - 1 ) , NWORK , I1 , I2 , I3 , I4 , I5 , I6 , I7 , I8 , I9 , I10

REAL ( KIND ( 0.0D0 ) ) X ( 0 : NP ) , Y ( 0 : NP ) , D0 , DNP , D20 , D2NP , EPS , BETA , BETAI , RHO , RHOI , D ( 0 : NP ) , D2 ( 0 : NP ) , WORK ( 1 : NWORK )


!  Assign the success value to the error flag.


!  Check the size of the work area.


!  Compute indices for the splitting of the work array WORK.


!  DBVSSC is essentially an interfacing routine which relieves the
!  user of a longer calling sequence. The structure of the method can
! be seen in DBVC and in the subroutines called. WORK ( I9 ) , WORK ( I10 ) , BETA , BETAI , RHO , RHOI , D , D2 , ERRC , DIAGN )

END

!  ---------------------------------------------------------------------

SUBROUTINE DBVSSE ( X , Y , NP , N , K , XTAB , NTAB , SBOPT , Y0OPT , Y1OPT , Y2OPT , ERRC , D , D2 , Y0TAB , Y1TAB , Y2TAB , ERRE , WORK , NWORK )

!  -------------------------------------------------
!            Lines 621-754 are comment lines.
!            The actual code begins at line 760.
!  -------------------------------------------------

!  ABSTRACT:
!
!  DBVSSE is designed to evaluate the interpolating, shape-preserving
!  spline computed in subroutine DBVSSC.
!
!
!  REMARK:
!
!  In these comments variable and array names will be denoted with
!  capital letters, and with small letters their contents.
!
!
!  METHOD:
!
!  Let a spline  s:=s(x)  of degree n and continuity k (k=1,2) ,
!  interpolating at the knots the point (x(i),y(i)) , i=0,1,...,np ,
!  be previously computed in subroutine DBVSSC. Then, given a set of
!  tabulation points  xtab(i) , i=0,1,...,ntab , DBVSSE computes the
!  values  y0tab(itab):=s(xtab(itab))  and/or
!  y1tab(itab):=Ds(xtab(itab))  and/or  y2tab(itab):=DDs(xtab(itab)) ,
!  using, under user selection, a sequential or binary search scheme.
!
!  The code has the following structure:
!
!         DBVSSE
!             DBVE
!                 DTRMB
!                 DSQTAB
!                     DLSPIS
!                     DBL
!                     DBL1
!                     DBL2
!                 DBNTAB
!                     DBSEAR
!                     DLSPIS
!                     DBL
!                     DBL1
!                     DBL2
!
!
!  CALLING SEQUENCE:
!
!        CALL DBVSSE (X,Y,NP,N,K,XTAB,NTAB,SBOPT,Y0OPT,Y1OPT,Y2OPT,
!    *                ERRC,D,D2,Y0TAB,Y1TAB,Y2TAB,ERRE,WORK,NWORK)
!
!
!  INPUT PARAMETERS:
!
!  X       : floating array, of bounds 0:NP, containing the data
!            abscissas  x(i), i=0,1,...,np. Restriction:
!            x(i).LT.x(i+1), i=0,1,...,np , checked in DBVSSC.
!  Y       : floating array, of bounds 0:NP, containing the data
!            ordinates  y(i), i=0,1,...,np.
!  NP      : integer variable, defining the number of interpolation
!            points. Restriction: np.GE.2 , checked in DBVSSC.
!  N       : integer variable, containing the degree of s.
!            Restriction: n.GE.3 , checked in DBVSSC
!  K       : integer variable, containing the class of continuity of s.
!            Restriction:  k.EQ.1  or  k.EQ.2  and  n.GE.3*k , checked
!            in DBVSSC.
!  XTAB    : floating array, of bounds 0:NTAB, containing the abscissas
!            of tabulation points.
!            Restriction: xtab(i).LE.xtab(i+1), i=0,1,...,ntab-1 .
!  NTAB    : integer variable, defining the number of tabulation points.
!            Restriction: ntab.GE.0 .
!  SBOPT   : integer variable, containing a control parameter.
!            If sbopt=1 then the sequential search is used for selecting
!            the interval of interpolation points in which xtab(i)
!            falls. If sbopt=2, binary search is used.
!            Restriction: sbopt.EQ.1 .OR. sbopt.EQ.2 .
!  Y0OPT   : integer variable, containing a control parameter.
!            If y0opt=1, the spline is evaluated at the points
!            xtab(i), i=0,1,...,ntab and the results are stored at the
!            array  Y0TAB.
!            Restriction: y0opt.EQ.0 .OR. y0opt.EQ.1 .
!  Y1OPT   : integer variable, containing a control parameter.
!            If y1opt=1 the first derivatives of the spline at points
!            xtab(i) i=0,1,...,ntab , are computed and the results are
!            stored in the array Y1TAB .
!            Restriction: y1opt.EQ.0 .OR. y1opt.EQ.1 .
!  Y2OPT   : integer variable, containing a control parameter.
!            If y2opt=1 the second derivatives of the spline at points
!            xtab(i), i=0,1,...,ntab  are computed and the results are
!            stored in the array Y2TAB.
!            Restriction: y2opt.EQ.0 .OR. y2opt.EQ.1 .
!  ERRC    : integer variable, containing the status of the last
!            execution of subroutine DBVSSC.
!  D       : floating array, of bounds 0:NP, containing the first
!            derivatives at the knots.
!  D2      : floating array of bounds 0:NP containing the second
!            derivatives at the knots.
!
!
!  OUTPUT PARAMETERS:
!
!
!  Y0TAB   : floating array, of bounds 0:NTAB, containing the values of
!            the spline at the tabulation points xtab(i) ,
!            i=0,1,...,ntab when the option  y0opt=1  is activated.
!  Y1TAB   : floating array, of bounds 0:NTAB, containing the values of
!            the first derivative of the spline at the tabulation points
!            xtab(i) , i=0,1,...ntab , when the option y1opt=1 is
!            activated.
!  Y2TAB   : floating array, of bounds 0:NTAB, containing the values of
!            the second derivative of the spline at the tabulation
!            points xtab(i) , i=0,1,...,ntab , when the option y2opt=1
!            is activated.
!  ERRE    : integer variable, containing an error flag which displays
!            the status of the code. DBVSSE has only two levels of error
!            (see DBVSSC for comparison): success and severe error,
!            which means that some incorrect assignment for input data
!            have been set.
!            ERRE=0:  success, normal return of the code;
!            ERRE=1:  severe error, the value errc gives a status of
!                     error, which means that the output of DBVSSC is
!                     meaningless. Check the input parameters of DBVSSC.
!            ERRE=2:  severe error, incorrect assignment for some of
!                     the values ntab, sbopt, y0opt, y1opt, y2opt ,
!                     nwork;
!            ERRE=3:  severe error, the restriction xtab(i).LT.xtab(i+1)
!                     is not fulfilled for some i when sequential search
!                     is required;
!
!
!  OTHER PARAMETERS:
!
!  WORK    : floating array, of bounds 1:NKORK, which is used as
!            a work area to store intermediate results.
!            The same array can be used to provide workspace for both
!            the main subroutines  DBVSSC and DBVSSE .
!  NWORK   : integer variable containing the size of the work area.
!            Restriction: nwork .GE. comm+(part+7)*np+(n*(n+11))/2+9
!                           that is
!                         nwork .GE. 3+(2+7)*np+(n*(n+11))/2+9

!  -------------------------------------------------
!            End of comments.
!  -------------------------------------------------

INTEGER COMM , PART

INTEGER NP , N , K , NTAB , SBOPT , Y0OPT , Y1OPT , Y2OPT , ERRC , ERRE , NWORK , I1 , I2 , I3 , I4 , I5 , I6 , I7 , I8 , I9 , I10

REAL ( KIND ( 0.0D0 ) ) X ( 0 : NP ) , Y ( 0 : NP ) , XTAB ( 0 : NTAB ) , Y0TAB ( 0 : NTAB ) , Y1TAB ( 0 : NTAB ) , Y2TAB ( 0 : NTAB ) , D ( 0 : NP ) , D2 ( 0 : NP ) , WORK ( 1 : NWORK )


!  Assign the success value to the error flag.


!  Check the size of the work area.


!  Compute indices for the splitting of the work array WORK.


!  DBVSSE is essentially an interfacing routine which relieves the
!  user of a longer calling sequence. The structure of the method can
! be seen in DBVE and in the subroutines called. WORK ( I7 ) , WORK ( I8 ) , Y0TAB , Y1TAB , Y2TAB , ERRE )

END


SUBROUTINE DALG1 ( A1 , NP , INFO , COMM , PART , EPS , A2 , ERRC , DIAGN )

!  DALG1 implements the algorithm A1[B(0)] described in subr. DBVSSC.
!
!  The input parameters NP,COMM,PART,EPS and the output parameters
!  ERRC, DIAGN are described in DBVSSC. The input parameter INFO is
!  described in DSTINF.
!
!  Items of possible interest are:
!
!  A1: floating array, of bounds 1:2, 0:NP, containing the sequence of
!      the sets  B(i), i=0,1,...,np (see the comments in DBVSSC).
!      More precisely,  B(i) = [a1(1,i),a1(2,i)] .
!
!  A2: floating array, of bounds 1:2, 0:NP, containing the sequence of
!      the sets  A(i), i=0,1,...,np (see the comments in DBVSSC).
!      More precisely, A(i) = [a2(1,i),a2(2,i)] .


INTEGER NP , COMM , PART , ERRC , DIAGN ( 0 : NP - 1 ) , ERRC1 , I

REAL ( KIND ( 0.0D0 ) ) A1 ( 1 : 2 , 0 : NP ) , INFO ( 1 : COMM + PART * NP + NP + 1 ) , EPS , A2 ( 1 : 2 , 0 : NP )

REAL ( KIND( 0.0D0 ) ) FL0


!  Step 1.


!  Step 2.


!  Step 2.1.

!  Ignore the constraints in  [x(i),x(i+1)]  when A(i) is empty.



END

!  ---------------------------------------------------------------------

SUBROUTINE DALG1D ( DSTAR , A1 , NP , INFO , COMM , PART , EPS , A2 , ERRC1 )

!  DALG1D computes the sequence of sets A(i), i=0,1,...,np, implementing
!  the algorithm A1[{dstar}], that is with A(0)={dstar} (see the com-
!  ments in subroutine DBVSSC for details).
!
!  The input parameters NP,COMM,PART,EPS are described in DBVSSC; the
!  input parameter INFO is described in DSTINF; the input parameters A1
!  and A2 are described in subprogram DALG1.
!
!  Item of possible interest is:
!
!  ERRC1  : Integer parameter, containing a control variable which is
!           then used in subr. DFPSVF
!           errc1 = 0 - success, normal return of the subprogram;
!           errc1 = 1 - A(i) is empty for some i.


INTEGER NP , COMM , PART , ERRC1 , I

REAL ( KIND ( 0.0D0 ) ) DSTAR , A1 ( 1 : 2 , 0 : NP ) , INFO ( 1 : COMM + PART * NP + NP + 1 ) , EPS , A2 ( 1 : 2 , 0 : NP )


!  Step 1.


!  Step 2.

END

!  ---------------------------------------------------------------------

SUBROUTINE DALG3 ( INFO , NP , COMM , PART , OPT , D0 , DNP , EPS , KMAX , MAXSTP , BETA , BETAI , A1 , A2 , D , ERRC , DIAGN )

!  DALG3 computes a sequence of slopes ( d(0), d(1), ..., d(np) ) which
!  can be used to compute a shape-preserving interpolating spline with
!  or without boundary conditions, as requested by the user. It is an
!  implementation of the algorithm A3 described in subroutine DBVSSC.
!
!  The input parameters NP,COMM,PART,OPT,EPS,KMAX,MAXSTP,BETA,BETAI,D
!  and the output parameter ERRC are described in subprogram DBVSSC.
!  The input parameter INFO is described in subprogram DSTINF.



INTEGER NP , COMM , PART , OPT , KMAX , MAXSTP , ERRC , DIAGN ( 0 : NP - 1 ) , I , K , P , Q

REAL ( KIND ( 0.0D0 ) ) INFO ( 1 : COMM + PART * NP + NP + 1 ) , D0 , DNP , EPS , BETA , BETAI , A1 ( 1 : 2 , 0 : NP ) , A2 ( 1 : 2 , 0 : NP ) , D ( 0 : NP ) , DSTAR , P1 , P2

LOGICAL DTST

INTEGER INK , INSTP

REAL ( KIND( 0.0D0 ) ) FL2


!  If kmax.LE.0 it is reset to ink.


!  If maxstp.LE.0 it is reset to instp.


!  Start step 1: store the sets  B(i), i=0,1,...,np , into the array A1.


!  Reset the first and the last interval if separable boundary condtions
!  are required


!  Start step 2. Call DALG1 to compute the array A2 containing the
!  sets A(i) , i=0,1,...,np.


!  Start step 3 (steps 3-7 are activated only if boundary conditions are
!  required).


! Compute betai ( phi ( A ( 0 ) .INT.beta ( A ( 0 ) ) ) . A2 ( 1 , NP ) , A2 ( 2 , NP ) , P1 , P2 )

!  Start step 4.


!  Start step 5 : initialization


!  Iteration. The loop is stopped if a convergence test is satisfied
!  or kmax iterations have already been done.

!  Step 5.1 .


! Step 5.2 . A2 ( 1 , NP ) , A2 ( 2 , NP ) , P1 , P2 )

!  If  gamma(A(0))  is empty for some k the problem does not have any
!  solution.




!  Start step 7.
!  Assign to dstar a suitable value


!  Check if dstar solves the problem, that is,  beta(dstar)  belongs to
!  phi(dstar); if it is not the case, another value for dstar
! is looked for. .AND.ERRC.NE.10 ) RETURN


!  Start step 8.


END

!  ---------------------------------------------------------------------

SUBROUTINE DAL2 ( A2 , NP , INFO , COMM , PART , D )

!  DAL2 computes a sequence of slopes (d(0),d(1),...,d(np)) implementing
!  alg. A2  described in subr. DBVSSC. Each d(i),i=0,1,...,np , is
!  chosen as the midpoint of the interval of all feasible values .
!
!  The input parameters NP,COMM,PART and the output parameter D are
!  described in DBVSSC; the input parameter INFO is described in DSTINF.
!
!  Item of possible interest is:
!
!  A2   : floating array, of bounds 1:2, 0:NP; [a2(1,i),a2(2,i)]
!         is the feasible interval for d(i) .


INTEGER NP , COMM , PART , I

REAL ( KIND ( 0.0D0 ) ) A2 ( 1 : 2 , 0 : NP ) , INFO ( 1 : COMM + PART * NP + NP + 1 ) , D ( 0 : NP ) , P1 , P2

REAL ( KIND( 0.0D0 ) ) FL1D2


END

!  ---------------------------------------------------------------------

SUBROUTINE DAL2DP ( A2 , NP , INFO , COMM , PART , D )

!  DAL2DP links algorithm A2 and a dynamic programming scheme
!  to select, among the set of all feasible solutions, the sequence
!  ( d(0),d(1), ..., d(np) ) which is the best 2-norm approximation to
!  a set of optimal values. More precisely, if (ds(0),ds(1), ...,ds(np))
!  is the sequence of optimal values, DAL2DP use the following dynamic
!  programming relations
!
!    psi(0;d(0)) := (d(0)-ds(0))**2
!    psi(j;d(j)) := (d(j)-ds(j))**2 + min(psi(j-1;d(j-1)))
!
!  for describing the objective function
!
!      SUM  ((d(j) - ds(j)) ** 2
!    j=0,np
!
!  For a complete comprehension of the algorithm see the book \2\
!  quoted in the references of subr. DBVSSC
!
!  The input parameters NP,COMM,PART and the output parameter D are
!  described in subprogram DBVSSC; the input parameter INFO is described
!  in subprogram DSTINF and the input parameter A2 is described in DAL2.
!  The constant NSUBD defined below is related to the discretization of
!  the admissible domain.


INTEGER NSUBD

INTEGER NP , COMM , PART , IND , I , J , JD0 , DMNIND

REAL ( KIND ( 0.0D0 ) ) A2 ( 1 : 2 , 0 : NP ) , INFO ( 1 : COMM + PART * NP + NP + 1 ) , D ( 0 : NP ) , PSI0 ( 0 : NSUBD + 1 ) , PSI1 ( 0 : NSUBD + 1 ) , PART0 ( 0 : NSUBD + 1 ) , PART1 ( 0 : NSUBD + 1 ) , H0 , H1 , PSI1MN , P1 , P2 , D0 , D1 , DSL

REAL ( KIND( 0.0D0 ) ) FL0 , FLE30



END

!  ---------------------------------------------------------------------

REAL ( KIND( 0.0D0 ) ) FUNCTION DBL ( X , N , L , X0 , XN , TB , FLAG , LAUX0 )

!  DBL computes the value assumed by the n-degree Bernstein polynomial
!  of a function  l  in the interval  (x0,xn)  at the point  x .
!  The evaluation is made using a Horner scheme, and the instructions
!  which do not depend upon  x  are executed under the control of
!  the variable  FLAG , for avoiding useless computations in
!  subsequent calls.
!  The degree  n  is supposed greater or equal to  3 .
!
!
!  INPUT PARAMETERS
!
!  X     : floating variable, containing the evaluation point.
!  N     : integer variable, containing the degree of Bernstein
!          polynomial.
!  L     : floating array, of bounds  0:N , containing the values
!          of the function  l .
!  X0    : floating variable, containing the left extreme of the
!          interval.
!  XN    : floating variable, containing the right extreme of the
!          interval.
!  TB    : floating array, of bounds  0:N , containing the binomial
!          terms used for computing the Bernstein polynomial.
!  FLAG  : integer variable, containing a control parameter.
!          In the case  flag=0  DBL  assumes to perform the first
!          evaluation of the polynomial, and computes the values
!          tb(i)*l(i) , i=0,1,...,n . In the case  flag=1  DBL
!          assumes to perform subsequent evaluations, and uses the
!          values previously computed.
!
!
!  OTHER PARAMETERS
!
!  LAUX0 : floating array, of bounds 0:N used as a work area to store
!          intermediate results.


INTEGER N , FLAG , I

REAL ( KIND ( 0.0D0 ) ) X , L ( 0 : N ) , X0 , XN , TB ( 0 : N ) , LAUX0 ( 0 : N ) , XNMX , XMX0 , AUX , FL1








END

!  ---------------------------------------------------------------------

REAL ( KIND( 0.0D0 ) ) FUNCTION DBL1 ( X , N , L , X0 , XN , TB , FLAG , LAUX1 )

!  DBL1 computes the value assumed by the first derivative of an
!  n-degree Bernstein polynomial of a function  l  in the interval
!  (x0,xn)  at the point  x .
!  The evaluation is made using a Horner scheme, and the instructions
!  which do not depend upon  x  are executed under the control of
!  the variable  FLAG , for avoiding useless computations in
!  subsequent calls.
!  The degree  n  is supposed greater or equal to  3 .
!
!  INPUT PARAMETERS
!
!  X     : floating variable, containing the evaluation point.
!  N     : integer variable, containing the degree of Bernstein
!          polynomial.
!  L     : floating array, of bounds  0:N , containing the values
!          of the function  l .
!  X0    : floating variable, containing the left extreme of the
!          interval.
!  XN    : floating variable, containing the right extreme of the
!          interval.
!  TB    : floating array, of bounds  0:N-1 , containing the binomial
!          terms used for computing the Bernstein polynomial.
!  FLAG  : integer variable, containing a control parameter.
!          In the case  flag=0  DBL1  assumes to perform the first
!          evaluation of the polynomial, and computes the values
!          tb(i)*(l(i+1)-l(i)) , i=0,1,...,n-1 . In the case  flag=1
!          DBL1 assumes to perform subsequent evaluations, and uses
!          the values previously computed.
!
!
!  OTHER PARAMETERS
!
!  LAUX1 : floating array, of bounds 0:N-1 used as a work area to store
!          intermediate results.


INTEGER N , FLAG , I

REAL ( KIND ( 0.0D0 ) ) X , L ( 0 : N ) , X0 , XN , TB ( 0 : N - 1 ) , LAUX1 ( 0 : N - 1 ) , XNMX , XMX0 , AUX , FL1








END

!  ---------------------------------------------------------------------

REAL ( KIND( 0.0D0 ) ) FUNCTION DBL2 ( X , N , L , X0 , XN , TB , FLAG , LAUX2 )

!  DBL2 computes the value assumed by the second derivative of an
!  n-degree Bernstein polynomial of a function  l  in the interval
!  (x0,xn)  at the point  x .
!  The evaluation is made using a Horner scheme, and the instructions
!  which do not depend upon  x  are executed under the control of
!  the variable  FLAG , for avoiding useless computations in
!  subsequent calls.
!  The degree  n  is supposed greater or equal to  3 .
!
!  INPUT PARAMETERS
!
!  X     : floating variable, containing the evaluation point.
!  N     : integer variable, containing the degree of Bernstein
!          polynomial.
!  L     : floating array, of bounds  0:N , containing the values
!          of the function  l .
!  X0    : floating variable, containing the left extreme of the
!          interval.
!  XN    : floating variable, containing the right extreme of the
!          interval.
!  TB    : floating array, of bounds  0:N-2 , containing the binomial
!          terms used for computing the Bernstein polynomial.
!  FLAG  : integer variable, containing a control parameter.
!          In the case  flag=0  DBL2  assumes to perform the first
!          evaluation of the polynomial, and computes the values
!          tb(i)*(l(i+2)-2*l(i+1)+l(i)) , i=0,1,...,n-2 .
!          In the case  flag=1  DBL2 assumes to perform subsequent
!          evaluations, and uses the values previously computed.
!
!
!  OTHER PARAMETERS
!
!  LAUX2 : floating array, of bounds 0:N-2 used as a work area to store
!          intermediate results.


INTEGER N , FLAG , I

REAL ( KIND ( 0.0D0 ) ) X , L ( 0 : N ) , X0 , XN , TB ( 0 : N - 2 ) , LAUX2 ( 0 : N - 2 ) , XNMX , XMX0 , AUX , FL1








END

!  ---------------------------------------------------------------------

SUBROUTINE DBNTAB ( X , Y , NP , XTAB , NTAB , Y0OPT , Y1OPT , Y2OPT , N , K , D , D2 , TB , L , LAUX0 , LAUX1 , LAUX2 , Y0TAB , Y1TAB , Y2TAB )

!  DBNTAB evaluates the spline and/or its first derivative and/or its
!  second derivative at the points  xtab(j) , j=0,1,...,ntab  using
!  a binary search for finding the interval  [x(i),x(i+1)] in which
!  the tabulation point falls. The input (X,Y,NP,XTAB,NTAB,Y0OPT,
!  Y1OPT,Y2OPT,N,K,D,D2,TB) and the output (Y0TAB,Y1TAB,Y2TAB)
!  parameters have been explained in subroutine DBVSSE. For the others
!  see subroutines DTRMB, DLSPIS.


INTEGER NP , NTAB , Y0OPT , Y1OPT , Y2OPT , N , K , IND , J

REAL ( KIND ( 0.0D0 ) ) X ( 0 : NP ) , Y ( 0 : NP ) , XTAB ( 0 : NTAB ) , D ( 0 : NP ) , D2 ( 0 : NP ) , TB ( 1 : N * ( N + 1 ) / 2 + N ) , L ( 0 : N ) , LAUX0 ( 0 : N ) , LAUX1 ( 0 : N ) , LAUX2 ( 0 : N ) , Y0TAB ( 0 : NTAB ) , Y1TAB ( 0 : NTAB ) , Y2TAB ( 0 : NTAB ) , DBL , DBL1 , DBL2


!  Call subprogram  DBSEAR  to compute the index  ind  such that
!       x(ind).LE.xtab(j).LT.x(ind+1) .


!  Call subprogram  DLSPIS  to compute the linear shape-preserving
!  interpolating spline  l  at
!      x(ind)+p*(x(ind+1)-x(ind))/n , p=0,1,...,n .



!  Evaluate the spline at  xtab(j) .


!  Evaluate the first derivative of the spline at  xtab(j) .


!  Evaluate the second derivative of the spline at  xtab(j) .


END

!  ---------------------------------------------------------------------

SUBROUTINE DBSEAR ( X , NP , XTAB , IND )

!  Given an ordered set of points  (x(i), i=0,1,...,np)  and the
!  point  xtab , DBSEAR finds the index  ind  such that
!
!              x(ind) .LE. xtab .LT. x(ind+1)
!
!  using a standard binary search. DBSEAR  sets  ind=0  or  ind=np-1
!  whenever  xtab.LT.x(0)  or  x(np).LE.xtab .
!
!
!  INPUT PARAMETERS
!
!  X     : floating array, of bounds  0:NP , containing the set of
!          ordered points.
!  XTAB  : floating variable, containing the point to be processed.
!  NP    : integer  variable, defining the number of points of the
!          ordered set.
!
!
!  OUTPUT PARAMETERS
!
!  IND   : integer variable, whose value selects the interval in
!          which the point  xtab  falls.


INTEGER NP , IND , I1 , I2 , MED

REAL ( KIND( 0.0D0 ) ) X ( 0 : NP ) , XTAB








END

!  ---------------------------------------------------------------------

SUBROUTINE DBVC ( X , Y , NP , N , K , OPT , D0 , DNP , D20 , D2NP , CONSTR , INFO , COMM , PART , EPS , KMAX , MAXSTP , A1 , A2 , DAUX2 , DAUX3 , BETA , BETAI , RHO , RHOI , D , D2 , ERRC , DIAGN )

!  DBVC checks input parameters and computes the required spline.
!
!  The input parameters X,Y,NP,N,K,OPT,D0,DNP,D20,D2NP,CONSTR,COMM,PART,
!  EPS,KMAX,MAXSTP,BETA,BETAI,RHO,RHOI and the output parameters
!  D,D2,ERRC,DIAGN are described in subroutine DBVSSC.
!  The other parameters are described in the called subprograms.



INTEGER NP , N , K , OPT , CONSTR ( 0 : NP - 1 ) , COMM , PART , KMAX , MAXSTP , ERRC , DIAGN ( 0 : NP - 1 ) , P , Q , R , I

REAL ( KIND ( 0.0D0 ) ) X ( 0 : NP ) , Y ( 0 : NP ) , D0 , DNP , D20 , D2NP , INFO ( 1 : COMM + PART * NP + NP + 1 ) , EPS , A1 ( 1 : 2 , 0 : NP ) , A2 ( 1 : 2 , 0 : NP ) , D ( 0 : NP ) , D2 ( 0 : NP ) , BETA , BETAI , RHO , RHOI , DAUX2 ( 1 : NP - 1 ) , DAUX3 ( 0 : NP - 1 )

!  Check the input parameters NP and OPT.

!  Check the array CONSTR.


!  Check the input parameters N and K.


!  Check the abscissas of the interpolation points.


!  Call subprogram DSTINF to set the information array INFO.

!  Initialize the array DIAGN.


!  Call subprogram DALG3 to compute the array D containing the first
!  derivative at initial points.


!  A  C(2) spline is required. Compute the sequence of second derivati-
!  ves d2(i), i=0,...,np , according to the shape constraints and, if
!  possible, to boundary conditions.

END

!  ---------------------------------------------------------------------

SUBROUTINE DBVE ( X , Y , NP , N , K , XTAB , NTAB , SBOPT , Y0OPT , Y1OPT , Y2OPT , D , D2 , ERRC , TB , L , LAUX0 , LAUX1 , LAUX2 , Y0TAB , Y1TAB , Y2TAB , ERRE )

!  DBVE checks input parameters and evaluates the required spline.
!
!  The input parameters X,Y,NP,N,K,XTAB,NTAB,SBOPT,Y0OPT,Y1OPT,Y2OPT,
!  D,D2,ERRC and the output parameters Y0TAB,Y1TAB,Y2TAB,ERRE are
!  described in subroutine DBVSSE. The others are used as work areas
!  and will be eventually described in the subsequent routines.


INTEGER NP , N , K , NTAB , SBOPT , Y0OPT , Y1OPT , Y2OPT , ERRC , ERRE , I

REAL ( KIND ( 0.0D0 ) ) X ( 0 : NP ) , Y ( 0 : NP ) , XTAB ( 0 : NTAB ) , D ( 0 : NP ) , D2 ( 0 : NP ) , L ( 0 : N ) , LAUX0 ( 0 : N ) , LAUX1 ( 0 : N ) , LAUX2 ( 0 : N ) , TB ( 1 : N * ( N + 1 ) / 2 + N ) , Y0TAB ( 0 : NTAB ) , Y1TAB ( 0 : NTAB ) , Y2TAB ( 0 : NTAB )

!  Check the correctness of input data, that is if subroutine DBVSSC
!  has correctly run.

! Check the input parameters NTAB , SBOPT , Y0OPT , Y1OPT , Y2OPT. .OR. ( Y2OPT.NE.0.AND.Y2OPT.NE.1 ) ) THEN


!  Check the abscissas of the tabulation points when the sequential
!  search is required.


!  Call subprogram DTRMB to compute the binomial terms needed
!  in the expression of Bernstein polynomials.



!  sbopt=1:  sequential search is required.

!  sbopt=2: binary search is required.

END

!  ---------------------------------------------------------------------

SUBROUTINE DFPSVF ( A1 , A2 , NP , INFO , COMM , PART , EPS , MAXSTP , BETA , ERRC , DSTAR )

!  DFPSVF finds, if possible, dstar.IN.[a1(1,0),a1(2,0)] such that
!               beta(dstar) .INT. phi(dstar)                     (1)
!
!  The input parameters NP,COMM,PART,EPS,MAXSTP,BETA, and the output
!  parameter ERRC are described in DBVSSC. The input parameters A1 and
!  A2 are described in  DALG1.



INTEGER NP , COMM , PART , MAXSTP , ERRC , STEP , SUBSTP , ERRC1 , I

REAL ( KIND ( 0.0D0 ) ) A1 ( 1 : 2 , 0 : NP ) , A2 ( 1 : 2 , 0 : NP ) , INFO ( 1 : COMM + PART * NP + NP + 1 ) , EPS , BETA , DSTAR , MID , H , DSL

REAL ( KIND( 0.0D0 ) ) FL1D2

!  If the optimum input value of dstar does not belong to
!  [a1(1,0),a1(2,0)] , the nearest extreme of this interval
!  replaces the old value of dstar.


!  Compute phi(dstar).


!  If phi(dstar) is not empty and dstar satisfies (1), it is the desired
!  value.

!  If it is not the case, look for another value. First, check
!  if the midpoint of the interval of all possible values satisfies
!  condition (1).

!  Second, check if any point of a tabulation of interval
!  [a1(1,0),a1(2,0)]  satisfies the condition (1). The tabulation
!  points are given by jumps of decreasing lenghts with alternate
! direction with respect to the middle of the interval. BETA ( DSTAR ) .LE.A2 ( 2 , NP ) + EPS ) ) RETURN A2 , ERRC1 ) BETA ( DSTAR ) .LE.A2 ( 2 , NP ) + EPS ) ) RETURN

!  Finally, check if condition (1) is satisfied by one of the
! [a1 ( 1 , 0 ) , a1 ( 2 , 0 ) ] extremes. BETA ( DSTAR ) .LE.A2 ( 2 , NP ) + EPS ) ) RETURN

!  If dstar satisfying (1) has not been found, send a message resetting
!  the error flag errc.


END

!  ---------------------------------------------------------------------

SUBROUTINE DINTRS ( A , B , C , D , P1 , P2 )

!  DINTRS computes the intersection of the two intervals  [a,b]
!  and [c,d]. [p1,p2] is the result. The output  p1.GT.p2 means that
!  the intervals are disjoint. DINTRS assumes  a.LE.b  and  c.LE.d .


REAL ( KIND( 0.0D0 ) ) A , B , C , D , P1 , P2


END

!  ---------------------------------------------------------------------

SUBROUTINE DLSPIS ( X , Y , D , D2 , NP , N , K , IND , L )

!  DLSPIS   evaluates the control points of the Bernstein-Bezier net
!  l:=l(x) of the interpolating spline  s:=s(x) , s.IN.BS(n;k) in the
!  interval  [x(ind),x(ind+1)] . For a description of the function  l
!  see the comments in subroutines  DBVSSC and DSCDRC. Here we only
!  recall that the structure of the net is different for k=1 or k=2 .
!
!  The input parameters  X,Y,D,D2,NP,N,K  are explained in subroutine
!  SPISE.
!
!  OTHER PARAMETERS
!
!  IND   : integer variable, used to select the knots interval.
!  L     : floating array, of bounds  0:N , containing the ordinates
!          of the control points.


INTEGER NP , N , K , IND , I

REAL ( KIND ( 0.0D0 ) ) X ( 0 : NP ) , Y ( 0 : NP ) , D ( 0 : NP ) , D2 ( 0 : NP ) , L ( 0 : N ) , H , ALPHA , Q1 , Q2 , FL1 , FL2 , FL4



!  Compute the net in the case  k=1 .





!  Compute the net in the case  k=2 .




END

!  ---------------------------------------------------------------------

REAL ( KIND( 0.0D0 ) ) FUNCTION DMDIAN ( A , B , C )

!  Given three numbers a,b,c , median  is the one which lies between
!  the other two.


REAL ( KIND( 0.0D0 ) ) A , B , C


END

!  ---------------------------------------------------------------------

INTEGER FUNCTION DMNIND ( D , PART )

!  DMNIND finds the index of the component of the array PART closest
!  to d .


INTEGER NSUBD

INTEGER J

REAL ( KIND( 0.0D0 ) ) D , PART ( 0 : NSUBD + 1 ) , AUX ( 0 : NSUBD + 1 ) , MINDIS

REAL ( KIND( 0.0D0 ) ) FL30




END

!  ---------------------------------------------------------------------

REAL ( KIND( 0.0D0 ) ) FUNCTION DMNMOD ( A , B )

!  Given two real numbers a and b, DMNMOD returns the number between
!  a and b which is closest to zero.


REAL ( KIND( 0.0D0 ) ) A , B , FL1 , FL2



END

!  ---------------------------------------------------------------------

SUBROUTINE DMSK1 ( INFO , CONSTR , COMM , PART , IND1 , NP )

!  DMSK1 compares the constraints required in input by the user and
!  stored in the array CONSTR with the shape of the data, stored by
!  DSTINF in the array INFO. If the required and real shapes do not
!  agree, DMSK1 resets both INFO and CONSTR with the 'intersection'
!  of the shapes. For example, if info(ind1+i)=11 , that is the data
!  are increasing and convex, and constr(i)=2 , that is only convexity
!  is required, then the output values will be  info(ind1+i)=10
!  (convexity) and constr(i)=2 (unchanged). If  info(ind1+i)=20
!  (concavity) and  constr(i)=1 (monotonicity) the output will be
!  info(ind1+i)=constr(i)=0 (no constraints). So, the computations made
!  in DALG3 will be based on these new values for selecting the domains
!  of admissible derivatives, and CONSTR will contain information on
!  the constraints effectively imposed.
!  Further details on the parameters INFO and IND1 can be found in sub-
!  routine DSTINF; CONSTR, COMM, PART, NP are explaained in subroutine
!  DBVSSC.


INTEGER NP
INTEGER CONSTR ( 0 : NP - 1 ) , COMM , PART , IND1 , I

REAL ( KIND( 0.0D0 ) ) INFO ( 1 : COMM + PART * NP + NP + 1 )




END

!  ---------------------------------------------------------------------

SUBROUTINE DMSK2 ( INFO , COMM , PART , IND1 , NP , D0 , DNP , EPS , ERRC , DIAGN )

!  This routine controls if the separable boundary conditions d(0)=d0
!  and d(np)=dnp are compatible with the first and the last domain of
!  constraints. The error flag is reset correspondingly.
!  Details on the parameters INFO, IND1 and COMM, PART, NP, D0, DNP,
!  EPS, ERRC, DIAGN can be found in subroutines
!  DSTINF and DBVSSC respectively.


INTEGER COMM , PART , IND1 , NP , ERRC , DIAGN ( 0 : NP - 1 )

REAL ( KIND( 0.0D0 ) ) INFO ( 1 : COMM + PART * NP + NP + 1 ) , D0 , DNP , EPS , P1 , P2 , A , B





END

!  ---------------------------------------------------------------------

SUBROUTINE DPRJ0 ( I , INFO , COMM , PART , NP , P1 , P2 )

!  Given the integer i , DPRJ0 computes the set B(i) performing the
!  projection of D(i) (subset of the (i-1)i-plane) onto the (i-1)-axis.
!
!  The input parameters COMM,PART,NP are described in DBVSSC; the input
!  parameter INFO is described in subroutine DSTINF.
!
!  OUTPUT PARAMETERS:
!
!  P1  : floating variable, containing the left extreme of the
!        resulting interval.
!
!  P2  : floating variable, containing the right extreme of the
!        resulting interval.


INTEGER I , COMM , PART , NP , KIND

REAL ( KIND( 0.0D0 ) ) INFO ( 1 : COMM + PART * NP + NP + 1 ) , P1 , P2 , N , K , DEL

REAL ( KIND( 0.0D0 ) ) FL0


!  No constraint


!  Increase constraints


!  Decrease constraints


!  Convexity constraints


!  Concavity constraints


!  Increase and convexity


!  Increase and concavity


!  Decrease and convexity


!  Decrease and concavity



END

!  ---------------------------------------------------------------------

SUBROUTINE DPRJ1 ( A , B , C , D , I , INFO , COMM , PART , NP , P1 , P2 )

!  Given the set S=[a,b]x[c,d] and the integer i , DPRJ1 performs the
!  intersection of S with the domain D(i) and the projection of the
!  resulting set (a subset of (i-1)i-plane) onto the i-axis .
!
!  The input parameters COMM,PART,NP are described in DBVSSC; the input
!  parameter INFO is described in DSTINF.
!
!  OUTPUT PARAMETERS:
!
!  P1  : floating variable, containing the left extreme of the
!        resulting interval.
!
!  P2  : floating variable, containing the right extreme of the
!        resulting interval.


INTEGER I , COMM , PART , NP , KIND

REAL ( KIND ( 0.0D0 ) ) A , B , C , D , INFO ( 1 : COMM + PART * NP + NP + 1 ) , P1 , P2 , N , K , DEL , F1 , F2 , F3 , X

REAL ( KIND( 0.0D0 ) ) FL0



!  No constraint


!  Increase constraints


!  Decrease constraints


!  Convexity constraints


!  Concavity constraints


!  Increase and convexity


!  Increase and concavity


!  Decrease and convexity


!  Decrease and concavity




END

!  ---------------------------------------------------------------------

SUBROUTINE DPRJ2 ( A , B , C , D , I , INFO , COMM , PART , NP , P1 , P2 )

!  Given the set s=[a,b]x[c,d] and the integer i, DPRJ2 performs the
!  intersection of S with the domain D(i) and the projection of the
!  resulting set (subset of (i-1)i-plane) onto the (i-1)-axis .
!
!  The input parameters COMM,PART,NP are described in DBVSSC; the input
!  parameter INFO is described in DSTINF.
!
!  OUTPUT PARAMETERS:
!
!  P1  : floating variable, containing the left extreme of the
!        resulting interval.
!
!  P2  : floating variable, containing the right extreme of the
!        resulting interval.


INTEGER I , COMM , PART , NP , KIND

REAL ( KIND ( 0.0D0 ) ) A , B , C , D , INFO ( 1 : COMM + PART * NP + NP + 1 ) , P1 , P2 , N , K , DEL , F1I , F2I , F3I , X

REAL ( KIND( 0.0D0 ) ) FL0



!  No constraints


!  Increase constraints


!  Decrease constraints


!  Convexity constraints


!  Concavity constraints


!  Increase and convexity


!  Increase and concavity


!  Decrease and convexity


!  Decrease and concavity




END

!  ---------------------------------------------------------------------

SUBROUTINE DSCDRC ( N , X , Y , D , OPT , NP , EPS , D20 , D2NP , RHO , RHOI , A1 , A2 , H , D2 , ERRC )

!  DSCDRC computes the sequence  d2(i) , i=0,1,...,np , of second
!  derivatives at the knots. The basic idea is that the vanishing second
!  derivatives (which are admissible by virtue of the theory involved in
!  the routines called previously) can be locally changed to non-zero
!  values without modifying the monotonicity and/or convexity.
!  Let us consider the restriction to the i-th subinterval of the
!  Bernstein-Bezier net for the C(2) spline with zero derivatives given
!  by subroutine DALG3. Let A, B and C be the second, third and
!  (int(n/2))-th point of the net, and let E, F, and G be given by a
!  symmetric construction.
!
!            B_______________C___G_______________F
!           /           .             .           \
!          /      .                         .      \
!         /  .D                                 H.  \
!        A                                           E
!       /                                             \
!      /                                               \
!     /                                                 \
!
!  Then the 'intermediate net' obtained inserting the straight lines
!  trough A-C and E-F is shape-preserving and we can take as the 'final
!  net' the union of two convex combination of A-B-C , A-D-C and H-F-G ,
!  E-H-D respectively. Expressing the net in term of the second
!  derivatives, the points D, B and H,F lead to restriction like
!  d2(i).IN.[a1(1,i),a1(2,i)] , d2(i+1).IN.[a2(1,i),a2(2,i)]
!  This construction must be repeated for all the subintervals and so
!  d2(i) .IN. [a2(1,i-1),a2(2,i-1)].INT.[a1(1,i),a1(2,i)] .
!
!  The input parameters N,X,Y,D,OPT,NP,EPS,D20,D2NP,RHO,RHOI and the
!  input ones D2,ERRC are documented in subroutine DBVSSC.



INTEGER NP , N , OPT , ERRC , I , Q

REAL ( KIND ( 0.0D0 ) ) X ( 0 : NP ) , Y ( 0 : NP ) , D20 , D2NP , EPS , D ( 0 : NP ) , D2 ( 0 : NP ) , H ( 0 : NP - 1 ) , A1 ( 1 : 2 , 0 : NP ) , A2 ( 1 : 2 , 0 : NP ) , A , B , C , DD , E , F , G , HH , ALPHA , GAMMA , DIFF2 , P1 , P2 , Q1 , Q2 , FL0 , FL1 , FL2 , FL4 , RHO , RHOI , DSL




!  Compute the points of the 'original' and 'intermediate' net.




!  Define the left and the right restriction for the second finite
!  difference of the net.




!  Take the intersection of the left and right restrictions for the
!  same second differences and translate it in terms of the second
!  derivatives.






!  The internal derivatives are defined as the admissible value closest
!  to the central second divided difference of the data.





!  No boundary condition is required. Take the first and last
!  derivative as the middle of admissible values.



!  Non-separable boundary conditions are required. Check if these can be
!  satisfied by admissible derivatives.



!  The boundary conditions cannot be satisfied. Set the error flag and
!  define the first and the last derivative as the nearest point to the
!  admissible and the boundary interval.




!  It is possible to satisfy the boundary conditions.




!  Separable boundary conditions are required. Check if they are
!  compatible with the admissible intervals and, if not, set the
!  error flag and take the admissible points nearest to the boundary
!  values. Otherwise take simply the boundary values.



END

!  ---------------------------------------------------------------------

REAL ( KIND( 0.0D0 ) ) FUNCTION DSL ( A , B , C )

!  Given the interval [a,b] and the number c, dsl is c if c belongs
!  to [a,b], otherwise, it is the nearest extreme to c.


REAL ( KIND( 0.0D0 ) ) A , B , C


END

!  ---------------------------------------------------------------------

SUBROUTINE DSQTAB ( X , Y , NP , XTAB , NTAB , Y0OPT , Y1OPT , Y2OPT , N , K , D , D2 , TB , L , LAUX0 , LAUX1 , LAUX2 , Y0TAB , Y1TAB , Y2TAB )

!  DSQTAB evaluates the spline and/or its first derivative and/or its
!  second derivative at the points  xtab(j) , j=0,1,...,ntab  using
!  a sequential search for finding the interval  [x(i),x(i+1)] in which
!  the tabulation point falls. The input (X,Y,NP,XTAB,NTAB,Y0OPT,
!  Y1OPT,Y2OPT,N,K,D,D2,TB) and the output (Y0TAB,Y1TAB,Y2TAB)
!  parameters have been explained in subroutine DBVSSE. For the others
!  see subroutines DTRMB, DLSPIS.


INTEGER NP , NTAB , Y0OPT , Y1OPT , Y2OPT , N , K , IND , IND1 , J , I

REAL ( KIND ( 0.0D0 ) ) X ( 0 : NP ) , Y ( 0 : NP ) , XTAB ( 0 : NTAB ) , D ( 0 : NP ) , D2 ( 0 : NP ) , TB ( 1 : N * ( N + 1 ) / 2 + N ) , L ( 0 : N ) , LAUX0 ( 0 : N ) , LAUX1 ( 0 : N ) , LAUX2 ( 0 : N ) , Y0TAB ( 0 : NTAB ) , Y1TAB ( 0 : NTAB ) , Y2TAB ( 0 : NTAB ) , DBL , DBL1 , DBL2



!  Compute the index  ind  such that  x(ind).LE.xtab(j).LT.x(ind+1) .




!  Check if  ind  selects a new subinterval.


!  Call subprogram  DLSPIS  to compute the linear shape-preserving
!  interpolating spline  l:=l(x)  at
!      x(ind)+p*(x(ind+1)-x(ind))/n , p=0,1,...,n .



!  Evaluate the spline at  xtab(j)  using new values of  l .


!  Evaluate the first derivative of the spline at  xtab(j)  using new
!  values of  l .
! TB ( ( N - 1 ) * N / 2 ) , 0 , LAUX1 )


!  Evaluate the second derivative of the spline at  xtab(j)  using new
!  values of  l .



!  Evaluate the spline at  xtab(j)  using old values of  l .


!  Evaluate the first derivative of the spline at  xtab(j)  using old
!  values of  l .


!  Evaluate the second derivative of the spline at  xtab(j)  using old
!  values of  l .




END

!  ---------------------------------------------------------------------

SUBROUTINE DSTINF ( OPT , D0 , DNP , CONSTR , N , K , X , Y , D , NP , COMM , PART , EPS , BETA , BETAI , DAUX2 , DAUX3 , INFO , ERRC , DIAGN )

!  DSTINF computes the information needed in the other parts of the
!  program using the data-dependent input parameters and stores it in
!  the output array INFO.
!
!  The parameters OPT,N,K,X,Y,D,NP,COMM,PART,EPS,BETA,BETAI,ERRC,DIAGN
!  are described in subroutine DBVSSC .
!
!  Items of possible interest are:
!
!  INFO  : floating array, of bounds 1:COMM+PART*NP+NP+1. It is composed
!          of four parts: the first, of bounds 1:comm, contains the
!          global information n, k , the maximum of the first divided
!          differences of initial points and the lower and upper bounds
!          for the derivatives, bounds which are used when no
!          constraints are imposed (see the parameter OPT described in
!          DBVSSC) or when the constraints must be relaxed; the second,
!          of bounds  comm+1:comm+np, contains information about
!          constraints in the interval (x(i),x(i+1)) , i=0,1,...,np-1 ;
!          if:
!          info((comm+1)+i)= 0 - no attribute;
!          info((comm+1)+i)= 1 - the data are increasing;
!          info((comm+1)+i)= 2 - the data are decreasing;
!          info((comm+1)+i)=10 - the data are convex;
!          info((comm+1)+i)=11 - the data are increasing and convex;
!          info((comm+1)+i)=12 - the data are decreasing and convex;
!          info((comm+1)+i)=20 - the data are concave;
!          info((comm+1)+i)=21 - the data are increasing and concave;
!          info((comm+1)+i)=22 - the data are decreasing and concave.
!          The third part, of bounds comm+np+1:comm+part*np, contains
!          the first divided differences of initial points
!              ( y(i+1)-y(i) ) / ( x(i+1)-x(i) ) ,  i=0,1,...,np-1 .
!          The fourth, of bounds comm+part*np+1:comm+part*np+np+1,
!          contains, eventually, the initial estimates of the first
!          derivatives which are then used to compute the constrained
!          best approximation (see the description of the input
!          parameter OPT  and of the array D in subr. DBVSSC). More
!          precisely, having defined  p := opt/100 , if p=2 it contains
!          the Bessel estimates, if p=3 it contains a set of third order
!          accurate estimates giving a co-monotone cubic Hermite
!          interpolant (see subr. DTDC described later), if p=4 it
!          contains a set of values given by the user; if p=1 this part
!          of INFO is not referenced.



INTEGER NP
INTEGER OPT , CONSTR ( 0 : NP - 1 ) , N , K , COMM , PART , ERRC , DIAGN ( 0 : NP - 1 ) , I , R , Q , P , IND1 , IND2 , IND3

REAL ( KIND ( 0.0D0 ) ) D0 , DNP , X ( 0 : NP ) , Y ( 0 : NP ) , D ( 0 : NP ) , EPS , BETA , BETAI , DAUX2 ( 1 : NP - 1 ) , DAUX3 ( 0 : NP - 1 ) , INFO ( 1 : COMM + PART * NP + NP + 1 ) , D2IM1 , D2I , IAUX0 , IAUXNP

REAL ( KIND( 0.0D0 ) ) FL0 , FLEM4 , FL2



!  Set the first and the second components of INFO to n and k
!  respectively.


!  Compute the first divided differences of the initial points and set
!  info(3) to their maximum.


!  Compute the lower and upper bounds for derivatives


!  If eps.LE.0 it is reset to flem4.


!  Compute the relative tollerance of the method.


!  Set the second part of INFO. Firstly, all the components are
!  initialized with 0.


!  Monotonicity is required: check if the initial points are increasing
!  or decreasing in each interval ( x(i), x(i+1) ) , i=0,1,...,np-1 .


!  Convexity is required: check if the initial points are concave or
! convex in each interval ( x ( i ) , x ( i + 1 ) ) , i = 1 , ... , np - 2 . ABS ( D2I ) .LE.EPS ) THEN .AND.D2I.LE. - EPS ) THEN

!  The convexity in the first and in the last interval is defined as the
!  second and the second to last, respectively.

!  In the case  r=4 , that is when the constraint selection
!  is made on any interval, we compare the kind given by the data with
!  those given by the array CONSTR


!  In the case q=3, the kind in the first and last subinterval
!  is compared with the boundary conditions

!  If p=2 the Bessel derivatives are stored in the fourth
!  part of INFO.


!  If no boundary condition is imposed, set the first and last
! derivatives using the standard formulas for Bessel interpolation. INFO ( IND2 + ( NP - 2 ) ) ) + ( X ( NP - 1 ) - X ( NP - 2 ) ) * INFO ( IND2 + ( NP - 1 ) ) ) / ( X ( NP ) - X ( NP - 2 ) )

!  Compute the first and last derivatives taking into account both the
!  slopes of the data and the restriction imposed by the boundary
! conditions INFO ( IND2 + ( NP - 2 ) ) ) + ( X ( NP - 1 ) - X ( NP - 2 ) ) * INFO ( IND2 + ( NP - 1 ) ) ) / ( X ( NP ) - X ( NP - 2 ) ) ( ( X ( 1 ) - X ( 0 ) ) + ( X ( NP ) - X ( NP - 1 ) ) ) ( X ( 1 ) - X ( 0 ) ) * IAUXNP ) / ( ( X ( 1 ) - X ( 0 ) ) + ( X ( NP ) - X ( NP - 1 ) ) )

!  If p=3 then the set of third order accurate estimates, computed by
!  subr. DTDC, is stored in the fourth part of INFO.


!  If p=4 then the set of values given by the user is stored in
!  the fourth part of INFO.


END

!  ---------------------------------------------------------------------

SUBROUTINE DTDC ( NP , X , COMM , PART , INFO , DD2 , DD3 )

!  Given the initial points ( x(i), y(i) ) , i=0,1,...,np , DTDC
!  computes a sequence  ds(0),ds(1),...,ds(np)  of estimates of the
!  function's derivatives which are third order accurate and furnish a
!  cubic Hermite interpolant preserving monotonicity.
!  The method is composed by the two following fundamental steps:
!  1 - compute an initial estimate of ds(i), i=0,1,...,np , which is
!      third or higher order accurate;
!  2 - refine it imposing the monotonicity constraint.
!  The computation of ds(i) needs the points x(i+j), j = -3,-2,...,3 ,
!  consequently, the boundary values ds(0), ds(1), ds(2) and ds(np-2),
!  ds(np-1), ds(np) are computed in an approximate way. Although they
!  are still third order accurate, may not preserve the monotonicity.
!  For more details see \3\ .
!
!  The input parameter NP,X,COMM,PART are described in subr. DBVSSC; the
!  parameter INFO is described in subr. DSTINF .
!
!  The computed values are stored in the last part of INFO.


INTEGER NP , I , IND , COMM , PART , IND1

REAL ( KIND ( 0.0D0 ) ) X ( 0 : NP ) , INFO ( 1 : COMM + PART * NP + NP + 1 ) , DD2 ( 1 : NP - 1 ) , DD3 ( 0 : NP - 1 ) , Q1 , Q2 , TIT , F , P1 , P2 , TI , TMAX , TMIN , SI , Q12 , Q32 , E1 , E2 , E3 , D1 , D2 , DMNMOD , DMDIAN

REAL ( KIND( 0.0D0 ) ) FL0 , FL2 , FL3


!  Compute the second divided differences of the initial points.


!  Compute the third divided differences of the initial points


!  Compute approximate values for  f[x(-1),x(0),x(1),x(2)]  and
!  f[x(np-2),x(np-1),x(np),x(np+1)] ; they are needed for the
! computation of ds ( 2 ) and ds ( np - 2 ) . ( X ( NP ) + X ( NP - 1 ) - X ( NP - 2 ) - X ( NP - 3 ) )


! ds ( i ) : initialization ( X ( I ) - X ( I - 1 ) ) , DD2 ( I + 1 ) + E3 * ( X ( I ) - X ( I + 2 ) ) )

!  Refinement


!  ds(1): initialization

!  refinement


! ds ( np - 1 ) : initialization DD3 ( NP - 1 ) * ( X ( NP - 1 ) - X ( NP - 2 ) ) * ( X ( NP - 1 ) - X ( NP ) ) DD3 ( NP - 2 ) * ( X ( NP - 1 ) - X ( NP - 2 ) ) * ( X ( NP - 1 ) - X ( NP ) )

!  Refinement


!  ds(0):


! ds ( np ) : DD3 ( NP - 2 ) * ( X ( NP ) - X ( NP - 2 ) ) * ( X ( NP ) - X ( NP - 1 ) )

END

!  ---------------------------------------------------------------------

SUBROUTINE DTRMB ( N , TB )

!  DTRMB    computes the binomial terms
!      i!/(k!*(i-k)!) , i=1,2,...,n , k=0,1,...,i .
!
!  INPUT PARAMETERS
!
!  N     : integer variable, containing the largest binomial term
!          needed.
!
!  OUTPUT PARAMETERS
!
!  TB    : floating array, of bounds  1:N*(N+1)/2+N  , containing
!          the values   i!/(k!*(i-k)!)  , k=0,1,...,i , in the
!          elements   TB(i*(i+1)/2),...,TB((i*(i+1)/2)+i) .


INTEGER N , I , K , IND , IND1
REAL ( KIND( 0.0D0 ) ) TB ( 1 : N * ( N + 1 ) / 2 + N ) , FL1






END

!  ---------------------------------------------------------------------

LOGICAL FUNCTION DTST ( A , B , C , D , EPS )

!  DTST checks if two intervals [a,b] and [c,d] differ less than eps.
!  DTST assumes  a.LE.b  and  c.LE.d .


REAL ( KIND( 0.0D0 ) ) A , B , C , D , EPS


END
