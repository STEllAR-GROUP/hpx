C==============================================================================
C
C     LIBRARY ROUTINES FOR SPECTRAL METHODS
C
C     March 1989
C
C     For questions, comments or suggestions, please contact:
C
C     Einar Malvin Ronquist
C     Room 3-243
C     Department of Mechanical Engineering
C     Massachusetts Institute of Technology
C     77 Massachusetts Avenue
C     Cambridge, MA 0299
C     U.S.A.
C
C------------------------------------------------------------------------------
C
C     ABBRIVIATIONS:
C
C     M   - Set of mesh points
C     Z   - Set of collocation/quadrature points
C     W   - Set of quadrature weights
C     H   - Lagrangian interpolant
C     D   - Derivative operator
C     I   - Interpolation operator
C     GL  - Gauss Legendre
C     GLL - Gauss-Lobatto Legendre
C     GJ  - Gauss Jacobi
C     GLJ - Gauss-Lobatto Jacobi
C
C
C     MAIN ROUTINES:
C
C     Points and weights:
C
C     ZWGL      Compute Gauss Legendre points and weights
C     ZWGLL     Compute Gauss-Lobatto Legendre points and weights
C     ZWGJ      Compute Gauss Jacobi points and weights (general)
C     ZWGLJ     Compute Gauss-Lobatto Jacobi points and weights (general)
C
C     Lagrangian interpolants:
C
C     HGL       Compute Gauss Legendre Lagrangian interpolant
C     HGLL      Compute Gauss-Lobatto Legendre Lagrangian interpolant
C     HGJ       Compute Gauss Jacobi Lagrangian interpolant (general)
C     HGLJ      Compute Gauss-Lobatto Jacobi Lagrangian interpolant (general)
C
C     Derivative operators:
C
C     DGLL      Compute Gauss-Lobatto Legendre derivative matrix
C     DGLLGL    Compute derivative matrix for a staggered mesh (GLL->GL)
C     DGJ       Compute Gauss Jacobi derivative matrix (general)
C     DGLJ      Compute Gauss-Lobatto Jacobi derivative matrix (general)
C     DGLJGJ    Compute derivative matrix for a staggered mesh (GLJ->GJ) (general)
C
C     Interpolation operators:
C
C     IGLM      Compute interpolation operator GL  -> M
C     IGLLM     Compute interpolation operator GLL -> M
C     IGJM      Compute interpolation operator GJ  -> M  (general)
C     IGLJM     Compute interpolation operator GLJ -> M  (general)
C
C     Other:
C
C     PNLEG     Compute Legendre polynomial of degree N
C     PNDLEG    Compute derivative of Legendre polynomial of degree N
C
C     Comments:
C
C     Note that many of the above routines exist in both single and
C     double precision. If the name of the single precision routine is
C     SUB, the double precision version is called SUBD. In most cases
C     all the "low-level" arithmetic is done in double precision, even
C     for the single precsion versions.
C
C     Useful references:
C
C [1] Gabor Szego: Orthogonal Polynomials, American Mathematical Society,
C     Providence, Rhode Island, 1939.
C [2] Abramowitz & Stegun: Handbook of Mathematical Functions,
C     Dover, New York, 1972.
C [3] Canuto, Hussaini, Quarteroni & Zang: Spectral Methods in Fluid
C     Dynamics, Springer-Verlag, 1988.
C
C
C==============================================================================
C
C--------------------------------------------------------------------
      SUBROUTINE ZWGL (Z,W,NP)
C--------------------------------------------------------------------
C
C     Generate NP Gauss Legendre points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha=0,beta=0).
C     The polynomial degree N=NP-1.
C     Z and W are in single precision, but all the arithmetic
C     operations are done in double precision.
C
C--------------------------------------------------------------------
      REAL Z(1),W(1)
      ALPHA = 0.
      BETA  = 0.
      CALL ZWGJ (Z,W,NP,ALPHA,BETA)
      RETURN
      END
C
      SUBROUTINE ZWGLL (Z,W,NP)
C--------------------------------------------------------------------
C
C     Generate NP Gauss-Lobatto Legendre points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha=0,beta=0).
C     The polynomial degree N=NP-1.
C     Z and W are in single precision, but all the arithmetic
C     operations are done in double precision.
C
C--------------------------------------------------------------------
      REAL Z(1),W(1)
      ALPHA = 0.
      BETA  = 0.
      CALL ZWGLJ (Z,W,NP,ALPHA,BETA)
      RETURN
      END
C
      SUBROUTINE ZWGJ (Z,W,NP,ALPHA,BETA)
C--------------------------------------------------------------------
C
C     Generate NP GAUSS JACOBI points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha>-1,beta>-1).
C     The polynomial degree N=NP-1.
C     Single precision version.
C
C--------------------------------------------------------------------
      PARAMETER (NMAX=84)
      PARAMETER (NZD = NMAX)
      REAL*8  ZD(NZD),WD(NZD)
      REAL Z(1),W(1),ALPHA,BETA
C
      NPMAX = NZD
      IF (NP.GT.NPMAX) THEN
         WRITE (6,*) 'Too large polynomial degree in ZWGJ'
         WRITE (6,*) 'Maximum polynomial degree is',NMAX
         WRITE (6,*) 'Here NP=',NP
         call exitt
      ENDIF
      ALPHAD = ALPHA
      BETAD  = BETA
      CALL ZWGJD (ZD,WD,NP,ALPHAD,BETAD)
      DO 100 I=1,NP
         Z(I) = ZD(I)
         W(I) = WD(I)
 100  CONTINUE
      RETURN
      END
C
      SUBROUTINE ZWGJD (Z,W,NP,ALPHA,BETA)
C--------------------------------------------------------------------
C
C     Generate NP GAUSS JACOBI points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha>-1,beta>-1).
C     The polynomial degree N=NP-1.
C     Double precision version.
C
C--------------------------------------------------------------------
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  Z(1),W(1),ALPHA,BETA
C
      N     = NP-1
      DN    = ((N))
      ONE   = 1.
      TWO   = 2.
      APB   = ALPHA+BETA
C
      IF (NP.LE.0) THEN
         WRITE (6,*) 'ZWGJD: Minimum number of Gauss points is 1',np
         call exitt
      ENDIF
      IF ((ALPHA.LE.-ONE).OR.(BETA.LE.-ONE)) THEN
         WRITE (6,*) 'ZWGJD: Alpha and Beta must be greater than -1'
         call exitt
      ENDIF
C
      IF (NP.EQ.1) THEN
         Z(1) = (BETA-ALPHA)/(APB+TWO)
         W(1) = GAMMAF(ALPHA+ONE)*GAMMAF(BETA+ONE)/GAMMAF(APB+TWO)
     $          * TWO**(APB+ONE)
         RETURN
      ENDIF
C
      CALL JACG (Z,NP,ALPHA,BETA)
C
      NP1   = N+1
      NP2   = N+2
      DNP1  = ((NP1))
      DNP2  = ((NP2))
      FAC1  = DNP1+ALPHA+BETA+ONE
      FAC2  = FAC1+DNP1
      FAC3  = FAC2+ONE
      FNORM = PNORMJ(NP1,ALPHA,BETA)
      RCOEF = (FNORM*FAC2*FAC3)/(TWO*FAC1*DNP2)
      DO 100 I=1,NP
         CALL JACOBF (P,PD,PM1,PDM1,PM2,PDM2,NP2,ALPHA,BETA,Z(I))
         W(I) = -RCOEF/(P*PDM1)
 100  CONTINUE
      RETURN
      END
C
      SUBROUTINE ZWGLJ (Z,W,NP,ALPHA,BETA)
C--------------------------------------------------------------------
C
C     Generate NP GAUSS LOBATTO JACOBI points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha>-1,beta>-1).
C     The polynomial degree N=NP-1.
C     Single precision version.
C
C--------------------------------------------------------------------
      PARAMETER (NMAX=84)
      PARAMETER (NZD = NMAX)
      REAL*8  ZD(NZD),WD(NZD)
      REAL Z(1),W(1),ALPHA,BETA
C
      NPMAX = NZD
      IF (NP.GT.NPMAX) THEN
         WRITE (6,*) 'Too large polynomial degree in ZWGLJ'
         WRITE (6,*) 'Maximum polynomial degree is',NMAX
         WRITE (6,*) 'Here NP=',NP
         call exitt
      ENDIF
      ALPHAD = ALPHA
      BETAD  = BETA
      CALL ZWGLJD (ZD,WD,NP,ALPHAD,BETAD)
      DO 100 I=1,NP
         Z(I) = ZD(I)
         W(I) = WD(I)
 100  CONTINUE
      RETURN
      END
C
      SUBROUTINE ZWGLJD (Z,W,NP,ALPHA,BETA)
C--------------------------------------------------------------------
C
C     Generate NP GAUSS LOBATTO JACOBI points (Z) and weights (W)
C     associated with Jacobi polynomial P(N)(alpha>-1,beta>-1).
C     The polynomial degree N=NP-1.
C     Double precision version.
C
C--------------------------------------------------------------------
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  Z(NP),W(NP),ALPHA,BETA
C
      N     = NP-1
      NM1   = N-1
      ONE   = 1.
      TWO   = 2.
C
      IF (NP.LE.1) THEN
       WRITE (6,*) 'ZWGLJD: Minimum number of Gauss-Lobatto points is 2'
       WRITE (6,*) 'ZWGLJD: alpha,beta:',alpha,beta,np
       call exitt
      ENDIF
      IF ((ALPHA.LE.-ONE).OR.(BETA.LE.-ONE)) THEN
         WRITE (6,*) 'ZWGLJD: Alpha and Beta must be greater than -1'
         call exitt
      ENDIF
C
      IF (NM1.GT.0) THEN
         ALPG  = ALPHA+ONE
         BETG  = BETA+ONE
         CALL ZWGJD (Z(2),W(2),NM1,ALPG,BETG)
      ENDIF
      Z(1)  = -ONE
      Z(NP) =  ONE
      DO 100  I=2,NP-1
         W(I) = W(I)/(ONE-Z(I)**2)
 100  CONTINUE
      CALL JACOBF (P,PD,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,Z(1))
      W(1)  = ENDW1 (N,ALPHA,BETA)/(TWO*PD)
      CALL JACOBF (P,PD,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,Z(NP))
      W(NP) = ENDW2 (N,ALPHA,BETA)/(TWO*PD)
C
      RETURN
      END
C
      REAL*8  FUNCTION ENDW1 (N,ALPHA,BETA)
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  ALPHA,BETA
      ZERO  = 0.
      ONE   = 1.
      TWO   = 2.
      THREE = 3.
      FOUR  = 4.
      APB   = ALPHA+BETA
      IF (N.EQ.0) THEN
         ENDW1 = ZERO
         RETURN
      ENDIF
      F1   = GAMMAF(ALPHA+TWO)*GAMMAF(BETA+ONE)/GAMMAF(APB+THREE)
      F1   = F1*(APB+TWO)*TWO**(APB+TWO)/TWO
      IF (N.EQ.1) THEN
         ENDW1 = F1
         RETURN
      ENDIF
      FINT1 = GAMMAF(ALPHA+TWO)*GAMMAF(BETA+ONE)/GAMMAF(APB+THREE)
      FINT1 = FINT1*TWO**(APB+TWO)
      FINT2 = GAMMAF(ALPHA+TWO)*GAMMAF(BETA+TWO)/GAMMAF(APB+FOUR)
      FINT2 = FINT2*TWO**(APB+THREE)
      F2    = (-TWO*(BETA+TWO)*FINT1 + (APB+FOUR)*FINT2)
     $        * (APB+THREE)/FOUR
      IF (N.EQ.2) THEN
         ENDW1 = F2
         RETURN
      ENDIF
      DO 100 I=3,N
         DI   = ((I-1))
         ABN  = ALPHA+BETA+DI
         ABNN = ABN+DI
         A1   = -(TWO*(DI+ALPHA)*(DI+BETA))/(ABN*ABNN*(ABNN+ONE))
         A2   =  (TWO*(ALPHA-BETA))/(ABNN*(ABNN+TWO))
         A3   =  (TWO*(ABN+ONE))/((ABNN+TWO)*(ABNN+ONE))
         F3   =  -(A2*F2+A1*F1)/A3
         F1   = F2
         F2   = F3
 100  CONTINUE
      ENDW1  = F3
      RETURN
      END
C
      REAL*8  FUNCTION ENDW2 (N,ALPHA,BETA)
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  ALPHA,BETA
      ZERO  = 0.
      ONE   = 1.
      TWO   = 2.
      THREE = 3.
      FOUR  = 4.
      APB   = ALPHA+BETA
      IF (N.EQ.0) THEN
         ENDW2 = ZERO
         RETURN
      ENDIF
      F1   = GAMMAF(ALPHA+ONE)*GAMMAF(BETA+TWO)/GAMMAF(APB+THREE)
      F1   = F1*(APB+TWO)*TWO**(APB+TWO)/TWO
      IF (N.EQ.1) THEN
         ENDW2 = F1
         RETURN
      ENDIF
      FINT1 = GAMMAF(ALPHA+ONE)*GAMMAF(BETA+TWO)/GAMMAF(APB+THREE)
      FINT1 = FINT1*TWO**(APB+TWO)
      FINT2 = GAMMAF(ALPHA+TWO)*GAMMAF(BETA+TWO)/GAMMAF(APB+FOUR)
      FINT2 = FINT2*TWO**(APB+THREE)
      F2    = (TWO*(ALPHA+TWO)*FINT1 - (APB+FOUR)*FINT2)
     $        * (APB+THREE)/FOUR
      IF (N.EQ.2) THEN
         ENDW2 = F2
         RETURN
      ENDIF
      DO 100 I=3,N
         DI   = ((I-1))
         ABN  = ALPHA+BETA+DI
         ABNN = ABN+DI
         A1   =  -(TWO*(DI+ALPHA)*(DI+BETA))/(ABN*ABNN*(ABNN+ONE))
         A2   =  (TWO*(ALPHA-BETA))/(ABNN*(ABNN+TWO))
         A3   =  (TWO*(ABN+ONE))/((ABNN+TWO)*(ABNN+ONE))
         F3   =  -(A2*F2+A1*F1)/A3
         F1   = F2
         F2   = F3
 100  CONTINUE
      ENDW2  = F3
      RETURN
      END
C
      REAL*8  FUNCTION GAMMAF (X)
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  X
      ZERO = 0.0
      HALF = 0.5
      ONE  = 1.0
      TWO  = 2.0
      FOUR = 4.0
      PI   = FOUR*ATAN(ONE)
      GAMMAF = ONE
      IF (X.EQ.-HALF) GAMMAF = -TWO*SQRT(PI)
      IF (X.EQ. HALF) GAMMAF =  SQRT(PI)
      IF (X.EQ. ONE ) GAMMAF =  ONE
      IF (X.EQ. TWO ) GAMMAF =  ONE
      IF (X.EQ. 1.5  ) GAMMAF =  SQRT(PI)/2.
      IF (X.EQ. 2.5) GAMMAF =  1.5*SQRT(PI)/2.
      IF (X.EQ. 3.5) GAMMAF =  0.5*(2.5*(1.5*SQRT(PI)))
      IF (X.EQ. 3. ) GAMMAF =  2.
      IF (X.EQ. 4. ) GAMMAF = 6.
      IF (X.EQ. 5. ) GAMMAF = 24.
      IF (X.EQ. 6. ) GAMMAF = 120.
      RETURN
      END
C
      REAL*8  FUNCTION PNORMJ (N,ALPHA,BETA)
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  ALPHA,BETA
      ONE   = 1.
      TWO   = 2.
      DN    = ((N))
      CONST = ALPHA+BETA+ONE
      IF (N.LE.1) THEN
         PROD   = GAMMAF(DN+ALPHA)*GAMMAF(DN+BETA)
         PROD   = PROD/(GAMMAF(DN)*GAMMAF(DN+ALPHA+BETA))
         PNORMJ = PROD * TWO**CONST/(TWO*DN+CONST)
         RETURN
      ENDIF
      PROD  = GAMMAF(ALPHA+ONE)*GAMMAF(BETA+ONE)
      PROD  = PROD/(TWO*(ONE+CONST)*GAMMAF(CONST+ONE))
      PROD  = PROD*(ONE+ALPHA)*(TWO+ALPHA)
      PROD  = PROD*(ONE+BETA)*(TWO+BETA)
      DO 100 I=3,N
         DINDX = ((I))
         FRAC  = (DINDX+ALPHA)*(DINDX+BETA)/(DINDX*(DINDX+ALPHA+BETA))
         PROD  = PROD*FRAC
 100  CONTINUE
      PNORMJ = PROD * TWO**CONST/(TWO*DN+CONST)
      RETURN
      END
C
      SUBROUTINE JACG (XJAC,NP,ALPHA,BETA)
C--------------------------------------------------------------------
C
C     Compute NP Gauss points XJAC, which are the zeros of the
C     Jacobi polynomial J(NP) with parameters ALPHA and BETA.
C     ALPHA and BETA determines the specific type of Gauss points.
C     Examples:
C     ALPHA = BETA =  0.0  ->  Legendre points
C     ALPHA = BETA = -0.5  ->  Chebyshev points
C
C--------------------------------------------------------------------
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  XJAC(1)
      DATA KSTOP /10/
      DATA EPS/1.0e-12/
      N   = NP-1
      one = 1.
      DTH = 4.*ATAN(one)/(2.*((N))+2.)
      DO 40 J=1,NP
         IF (J.EQ.1) THEN
            X = COS((2.*(((J))-1.)+1.)*DTH)
         ELSE
            X1 = COS((2.*(((J))-1.)+1.)*DTH)
            X2 = XLAST
            X  = (X1+X2)/2.
         ENDIF
         DO 30 K=1,KSTOP
            CALL JACOBF (P,PD,PM1,PDM1,PM2,PDM2,NP,ALPHA,BETA,X)
            RECSUM = 0.
            JM = J-1
            DO 29 I=1,JM
               RECSUM = RECSUM+1./(X-XJAC(NP-I+1))
 29         CONTINUE
            DELX = -P/(PD-RECSUM*P)
            X    = X+DELX
            IF (ABS(DELX) .LT. EPS) GOTO 31
 30      CONTINUE
 31      CONTINUE
         XJAC(NP-J+1) = X
         XLAST        = X
 40   CONTINUE
      DO 200 I=1,NP
         XMIN = 2.
         DO 100 J=I,NP
            IF (XJAC(J).LT.XMIN) THEN
               XMIN = XJAC(J)
               JMIN = J
            ENDIF
 100     CONTINUE
         IF (JMIN.NE.I) THEN
            SWAP = XJAC(I)
            XJAC(I) = XJAC(JMIN)
            XJAC(JMIN) = SWAP
         ENDIF
 200  CONTINUE
      RETURN
      END
C
      SUBROUTINE JACOBF (POLY,PDER,POLYM1,PDERM1,POLYM2,PDERM2,
     $                   N,ALP,BET,X)
C--------------------------------------------------------------------
C
C     Computes the Jacobi polynomial (POLY) and its derivative (PDER)
C     of degree N at X.
C
C--------------------------------------------------------------------
      IMPLICIT REAL*8  (A-H,O-Z)
      APB  = ALP+BET
      POLY = 1.
      PDER = 0.
      IF (N .EQ. 0) RETURN
      POLYL = POLY
      PDERL = PDER
      POLY  = (ALP-BET+(APB+2.)*X)/2.
      PDER  = (APB+2.)/2.
      IF (N .EQ. 1) RETURN
      DO 20 K=2,N
         DK = ((K))
         A1 = 2.*DK*(DK+APB)*(2.*DK+APB-2.)
         A2 = (2.*DK+APB-1.)*(ALP**2-BET**2)
         B3 = (2.*DK+APB-2.)
         A3 = B3*(B3+1.)*(B3+2.)
         A4 = 2.*(DK+ALP-1.)*(DK+BET-1.)*(2.*DK+APB)
         POLYN  = ((A2+A3*X)*POLY-A4*POLYL)/A1
         PDERN  = ((A2+A3*X)*PDER-A4*PDERL+A3*POLY)/A1
         PSAVE  = POLYL
         PDSAVE = PDERL
         POLYL  = POLY
         POLY   = POLYN
         PDERL  = PDER
         PDER   = PDERN
 20   CONTINUE
      POLYM1 = POLYL
      PDERM1 = PDERL
      POLYM2 = PSAVE
      PDERM2 = PDSAVE
      RETURN
      END
C
      REAL FUNCTION HGJ (II,Z,ZGJ,NP,ALPHA,BETA)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant HGJ through
C     the NP Gauss Jacobi points ZGJ at the point Z.
C     Single precision version.
C
C---------------------------------------------------------------------
      PARAMETER (NMAX=84)
      PARAMETER (NZD = NMAX)
      REAL*8  ZD,ZGJD(NZD)
      REAL Z,ZGJ(1),ALPHA,BETA
      NPMAX = NZD
      IF (NP.GT.NPMAX) THEN
         WRITE (6,*) 'Too large polynomial degree in HGJ'
         WRITE (6,*) 'Maximum polynomial degree is',NMAX
         WRITE (6,*) 'Here NP=',NP
         call exitt
      ENDIF
      ZD = Z
      DO 100 I=1,NP
         ZGJD(I) = ZGJ(I)
 100  CONTINUE
      ALPHAD = ALPHA
      BETAD  = BETA
      HGJ    = HGJD (II,ZD,ZGJD,NP,ALPHAD,BETAD)
      RETURN
      END
C
      REAL*8  FUNCTION HGJD (II,Z,ZGJ,NP,ALPHA,BETA)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant HGJD through
C     the NZ Gauss-Lobatto Jacobi points ZGJ at the point Z.
C     Double precision version.
C
C---------------------------------------------------------------------
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  Z,ZGJ(1),ALPHA,BETA
      EPS = 1.e-5
      ONE = 1.
      ZI  = ZGJ(II)
      DZ  = Z-ZI
      IF (ABS(DZ).LT.EPS) THEN
         HGJD = ONE
         RETURN
      ENDIF
      CALL JACOBF (PZI,PDZI,PM1,PDM1,PM2,PDM2,NP,ALPHA,BETA,ZI)
      CALL JACOBF (PZ,PDZ,PM1,PDM1,PM2,PDM2,NP,ALPHA,BETA,Z)
      HGJD  = PZ/(PDZI*(Z-ZI))
      RETURN
      END
C
      REAL FUNCTION HGLJ (II,Z,ZGLJ,NP,ALPHA,BETA)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant HGLJ through
C     the NZ Gauss-Lobatto Jacobi points ZGLJ at the point Z.
C     Single precision version.
C
C---------------------------------------------------------------------
      PARAMETER (NMAX=84)
      PARAMETER (NZD = NMAX)
      REAL*8  ZD,ZGLJD(NZD)
      REAL Z,ZGLJ(1),ALPHA,BETA
      NPMAX = NZD
      IF (NP.GT.NPMAX) THEN
         WRITE (6,*) 'Too large polynomial degree in HGLJ'
         WRITE (6,*) 'Maximum polynomial degree is',NMAX
         WRITE (6,*) 'Here NP=',NP
         call exitt
      ENDIF
      ZD = Z
      DO 100 I=1,NP
         ZGLJD(I) = ZGLJ(I)
 100  CONTINUE
      ALPHAD = ALPHA
      BETAD  = BETA
      HGLJ   = HGLJD (II,ZD,ZGLJD,NP,ALPHAD,BETAD)
      RETURN
      END
C
      REAL*8  FUNCTION HGLJD (I,Z,ZGLJ,NP,ALPHA,BETA)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant HGLJD through
C     the NZ Gauss-Lobatto Jacobi points ZJACL at the point Z.
C     Double precision version.
C
C---------------------------------------------------------------------
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  Z,ZGLJ(1),ALPHA,BETA
      EPS = 1.e-5
      ONE = 1.
      ZI  = ZGLJ(I)
      DZ  = Z-ZI
      IF (ABS(DZ).LT.EPS) THEN
         HGLJD = ONE
         RETURN
      ENDIF
      N      = NP-1
      DN     = ((N))
      EIGVAL = -DN*(DN+ALPHA+BETA+ONE)
      CALL JACOBF (PI,PDI,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,ZI)
      CONST  = EIGVAL*PI+ALPHA*(ONE+ZI)*PDI-BETA*(ONE-ZI)*PDI
      CALL JACOBF (P,PD,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,Z)
      HGLJD  = (ONE-Z**2)*PD/(CONST*(Z-ZI))
      RETURN
      END
C
      SUBROUTINE DGJ (D,DT,Z,NZ,NZD,ALPHA,BETA)
C-----------------------------------------------------------------
C
C     Compute the derivative matrix D and its transpose DT
C     associated with the Nth order Lagrangian interpolants
C     through the NZ Gauss Jacobi points Z.
C     Note: D and DT are square matrices.
C     Single precision version.
C
C-----------------------------------------------------------------
      PARAMETER (NMAX=84)
      PARAMETER (NZDD = NMAX)
      REAL*8  DD(NZDD,NZDD),DTD(NZDD,NZDD),ZD(NZDD)
      REAL D(NZD,NZD),DT(NZD,NZD),Z(1),ALPHA,BETA
C
      IF (NZ.LE.0) THEN
         WRITE (6,*) 'DGJ: Minimum number of Gauss points is 1'
         call exitt
      ENDIF
      IF (NZ .GT. NMAX) THEN
         WRITE (6,*) 'Too large polynomial degree in DGJ'
         WRITE (6,*) 'Maximum polynomial degree is',NMAX
         WRITE (6,*) 'Here Nz=',Nz
         call exitt
      ENDIF
      IF ((ALPHA.LE.-1.).OR.(BETA.LE.-1.)) THEN
         WRITE (6,*) 'DGJ: Alpha and Beta must be greater than -1'
         call exitt
      ENDIF
      ALPHAD = ALPHA
      BETAD  = BETA
      DO 100 I=1,NZ
         ZD(I) = Z(I)
 100  CONTINUE
      CALL DGJD (DD,DTD,ZD,NZ,NZDD,ALPHAD,BETAD)
      DO 200 I=1,NZ
      DO 200 J=1,NZ
         D(I,J)  = DD(I,J)
         DT(I,J) = DTD(I,J)
 200  CONTINUE
      RETURN
      END
C
      SUBROUTINE DGJD (D,DT,Z,NZ,NZD,ALPHA,BETA)
C-----------------------------------------------------------------
C
C     Compute the derivative matrix D and its transpose DT
C     associated with the Nth order Lagrangian interpolants
C     through the NZ Gauss Jacobi points Z.
C     Note: D and DT are square matrices.
C     Double precision version.
C
C-----------------------------------------------------------------
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  D(NZD,NZD),DT(NZD,NZD),Z(1),ALPHA,BETA
      N    = NZ-1
      DN   = ((N))
      ONE  = 1.
      TWO  = 2.
C
      IF (NZ.LE.1) THEN
       WRITE (6,*) 'DGJD: Minimum number of Gauss-Lobatto points is 2'
       call exitt
      ENDIF
      IF ((ALPHA.LE.-ONE).OR.(BETA.LE.-ONE)) THEN
         WRITE (6,*) 'DGJD: Alpha and Beta must be greater than -1'
         call exitt
      ENDIF
C
      DO 200 I=1,NZ
      DO 200 J=1,NZ
         CALL JACOBF (PI,PDI,PM1,PDM1,PM2,PDM2,NZ,ALPHA,BETA,Z(I))
         CALL JACOBF (PJ,PDJ,PM1,PDM1,PM2,PDM2,NZ,ALPHA,BETA,Z(J))
         IF (I.NE.J) D(I,J) = PDI/(PDJ*(Z(I)-Z(J)))
         IF (I.EQ.J) D(I,J) = ((ALPHA+BETA+TWO)*Z(I)+ALPHA-BETA)/
     $                        (TWO*(ONE-Z(I)**2))
         DT(J,I) = D(I,J)
 200  CONTINUE
      RETURN
      END
C
      SUBROUTINE DGLJ (D,DT,Z,NZ,NZD,ALPHA,BETA)
C-----------------------------------------------------------------
C
C     Compute the derivative matrix D and its transpose DT
C     associated with the Nth order Lagrangian interpolants
C     through the NZ Gauss-Lobatto Jacobi points Z.
C     Note: D and DT are square matrices.
C     Single precision version.
C
C-----------------------------------------------------------------
      PARAMETER (NMAX=84)
      PARAMETER (NZDD = NMAX)
      REAL*8  DD(NZDD,NZDD),DTD(NZDD,NZDD),ZD(NZDD)
      REAL D(NZD,NZD),DT(NZD,NZD),Z(1),ALPHA,BETA
C
      IF (NZ.LE.1) THEN
       WRITE (6,*) 'DGLJ: Minimum number of Gauss-Lobatto points is 2'
       call exitt
      ENDIF
      IF (NZ .GT. NMAX) THEN
         WRITE (6,*) 'Too large polynomial degree in DGLJ'
         WRITE (6,*) 'Maximum polynomial degree is',NMAX
         WRITE (6,*) 'Here NZ=',NZ
         call exitt
      ENDIF
      IF ((ALPHA.LE.-1.).OR.(BETA.LE.-1.)) THEN
         WRITE (6,*) 'DGLJ: Alpha and Beta must be greater than -1'
         call exitt
      ENDIF
      ALPHAD = ALPHA
      BETAD  = BETA
      DO 100 I=1,NZ
         ZD(I) = Z(I)
 100  CONTINUE
      CALL DGLJD (DD,DTD,ZD,NZ,NZDD,ALPHAD,BETAD)
      DO 200 I=1,NZ
      DO 200 J=1,NZ
         D(I,J)  = DD(I,J)
         DT(I,J) = DTD(I,J)
 200  CONTINUE
      RETURN
      END
C
      SUBROUTINE DGLJD (D,DT,Z,NZ,NZD,ALPHA,BETA)
C-----------------------------------------------------------------
C
C     Compute the derivative matrix D and its transpose DT
C     associated with the Nth order Lagrangian interpolants
C     through the NZ Gauss-Lobatto Jacobi points Z.
C     Note: D and DT are square matrices.
C     Double precision version.
C
C-----------------------------------------------------------------
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  D(NZD,NZD),DT(NZD,NZD),Z(1),ALPHA,BETA
      N    = NZ-1
      DN   = ((N))
      ONE  = 1.
      TWO  = 2.
      EIGVAL = -DN*(DN+ALPHA+BETA+ONE)
C
      IF (NZ.LE.1) THEN
       WRITE (6,*) 'DGLJD: Minimum number of Gauss-Lobatto points is 2'
       call exitt
      ENDIF
      IF ((ALPHA.LE.-ONE).OR.(BETA.LE.-ONE)) THEN
         WRITE (6,*) 'DGLJD: Alpha and Beta must be greater than -1'
         call exitt
      ENDIF
C
      DO 200 I=1,NZ
      DO 200 J=1,NZ
         CALL JACOBF (PI,PDI,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,Z(I))
         CALL JACOBF (PJ,PDJ,PM1,PDM1,PM2,PDM2,N,ALPHA,BETA,Z(J))
         CI = EIGVAL*PI-(BETA*(ONE-Z(I))-ALPHA*(ONE+Z(I)))*PDI
         CJ = EIGVAL*PJ-(BETA*(ONE-Z(J))-ALPHA*(ONE+Z(J)))*PDJ
         IF (I.NE.J) D(I,J) = CI/(CJ*(Z(I)-Z(J)))
         IF ((I.EQ.J).AND.(I.NE.1).AND.(I.NE.NZ))
     $   D(I,J) = (ALPHA*(ONE+Z(I))-BETA*(ONE-Z(I)))/
     $            (TWO*(ONE-Z(I)**2))
         IF ((I.EQ.J).AND.(I.EQ.1))
     $   D(I,J) =  (EIGVAL+ALPHA)/(TWO*(BETA+TWO))
         IF ((I.EQ.J).AND.(I.EQ.NZ))
     $   D(I,J) = -(EIGVAL+BETA)/(TWO*(ALPHA+TWO))
         DT(J,I) = D(I,J)
 200  CONTINUE
      RETURN
      END
C
      SUBROUTINE DGLL (D,DT,Z,NZ,NZD)
C-----------------------------------------------------------------
C
C     Compute the derivative matrix D and its transpose DT
C     associated with the Nth order Lagrangian interpolants
C     through the NZ Gauss-Lobatto Legendre points Z.
C     Note: D and DT are square matrices.
C
C-----------------------------------------------------------------
      PARAMETER (NMAX=84)
      REAL D(NZD,NZD),DT(NZD,NZD),Z(1)
      N  = NZ-1
      IF (NZ .GT. NMAX) THEN
         WRITE (6,*) 'Subroutine DGLL'
         WRITE (6,*) 'Maximum polynomial degree =',NMAX
         WRITE (6,*) 'Polynomial degree         =',NZ
      ENDIF
      IF (NZ .EQ. 1) THEN
         D(1,1) = 0.
         RETURN
      ENDIF
      FN = (N)
      d0 = FN*(FN+1.)/4.
      DO 200 I=1,NZ
      DO 200 J=1,NZ
         D(I,J) = 0.
         IF  (I.NE.J) D(I,J) = PNLEG(Z(I),N)/
     $                        (PNLEG(Z(J),N)*(Z(I)-Z(J)))
         IF ((I.EQ.J).AND.(I.EQ.1))  D(I,J) = -d0
         IF ((I.EQ.J).AND.(I.EQ.NZ)) D(I,J) =  d0
         DT(J,I) = D(I,J)
 200  CONTINUE
      RETURN
      END
C
      REAL FUNCTION HGLL (I,Z,ZGLL,NZ)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant L through
C     the NZ Gauss-Lobatto Legendre points ZGLL at the point Z.
C
C---------------------------------------------------------------------
      REAL ZGLL(1)
      EPS = 1.E-5
      DZ = Z - ZGLL(I)
      IF (ABS(DZ) .LT. EPS) THEN
         HGLL = 1.
         RETURN
      ENDIF
      N = NZ - 1
      ALFAN = (N)*((N)+1.)
      HGLL = - (1.-Z*Z)*PNDLEG(Z,N)/
     $         (ALFAN*PNLEG(ZGLL(I),N)*(Z-ZGLL(I)))
      RETURN
      END
C
      REAL FUNCTION HGL (I,Z,ZGL,NZ)
C---------------------------------------------------------------------
C
C     Compute the value of the Lagrangian interpolant HGL through
C     the NZ Gauss Legendre points ZGL at the point Z.
C
C---------------------------------------------------------------------
      REAL ZGL(1)
      EPS = 1.E-5
      DZ = Z - ZGL(I)
      IF (ABS(DZ) .LT. EPS) THEN
         HGL = 1.
         RETURN
      ENDIF
      N = NZ-1
      HGL = PNLEG(Z,NZ)/(PNDLEG(ZGL(I),NZ)*(Z-ZGL(I)))
      RETURN
      END
C
      REAL FUNCTION PNLEG (Z,N)
C---------------------------------------------------------------------
C
C     Compute the value of the Nth order Legendre polynomial at Z.
C     (Simpler than JACOBF)
C     Based on the recursion formula for the Legendre polynomials.
C
C---------------------------------------------------------------------
C
C     This next statement is to overcome the underflow bug in the i860.  
C     It can be removed at a later date.  11 Aug 1990   pff.
C
      IF(ABS(Z) .LT. 1.0E-25) Z = 0.0
C
      P1   = 1.
      IF (N.EQ.0) THEN
         PNLEG = P1
         RETURN
      ENDIF
      P2   = Z
      P3   = P2
      DO 10 K = 1, N-1
         FK  = (K)
         P3  = ((2.*FK+1.)*Z*P2 - FK*P1)/(FK+1.)
         P1  = P2
         P2  = P3
 10   CONTINUE
      PNLEG = P3
      if (n.eq.0) pnleg = 1.
      RETURN
      END
C
      REAL FUNCTION PNDLEG (Z,N)
C----------------------------------------------------------------------
C
C     Compute the derivative of the Nth order Legendre polynomial at Z.
C     (Simpler than JACOBF)
C     Based on the recursion formula for the Legendre polynomials.
C
C----------------------------------------------------------------------
      P1   = 1.
      P2   = Z
      P1D  = 0.
      P2D  = 1.
      P3D  = 1.
      DO 10 K = 1, N-1
         FK  = (K)
         P3  = ((2.*FK+1.)*Z*P2 - FK*P1)/(FK+1.)
         P3D = ((2.*FK+1.)*P2 + (2.*FK+1.)*Z*P2D - FK*P1D)/(FK+1.)
         P1  = P2
         P2  = P3
         P1D = P2D
         P2D = P3D
 10   CONTINUE
      PNDLEG = P3D
      IF (N.eq.0) pndleg = 0.
      RETURN
      END
C
      SUBROUTINE DGLLGL (D,DT,ZM1,ZM2,IM12,NZM1,NZM2,ND1,ND2)
C-----------------------------------------------------------------------
C
C     Compute the (one-dimensional) derivative matrix D and its
C     transpose DT associated with taking the derivative of a variable
C     expanded on a Gauss-Lobatto Legendre mesh (M1), and evaluate its
C     derivative on a Guass Legendre mesh (M2).
C     Need the one-dimensional interpolation operator IM12
C     (see subroutine IGLLGL).
C     Note: D and DT are rectangular matrices.
C
C-----------------------------------------------------------------------
      REAL D(ND2,ND1), DT(ND1,ND2), ZM1(ND1), ZM2(ND2), IM12(ND2,ND1)
      IF (NZM1.EQ.1) THEN
        D (1,1) = 0.
        DT(1,1) = 0.
        RETURN
      ENDIF
      EPS = 1.E-6
      NM1 = NZM1-1
      DO 10 IP = 1, NZM2
         DO 10 JQ = 1, NZM1
            ZP = ZM2(IP)
            ZQ = ZM1(JQ)
            IF ((ABS(ZP) .LT. EPS).AND.(ABS(ZQ) .LT. EPS)) THEN
                D(IP,JQ) = 0.
            ELSE
                D(IP,JQ) = (PNLEG(ZP,NM1)/PNLEG(ZQ,NM1)
     $                     -IM12(IP,JQ))/(ZP-ZQ)
            ENDIF
            DT(JQ,IP) = D(IP,JQ)
 10   CONTINUE
      RETURN
      END
C
      SUBROUTINE DGLJGJ (D,DT,ZGL,ZG,IGLG,NPGL,NPG,ND1,ND2,ALPHA,BETA)
C-----------------------------------------------------------------------
C
C     Compute the (one-dimensional) derivative matrix D and its
C     transpose DT associated with taking the derivative of a variable
C     expanded on a Gauss-Lobatto Jacobi mesh (M1), and evaluate its
C     derivative on a Guass Jacobi mesh (M2).
C     Need the one-dimensional interpolation operator IM12
C     (see subroutine IGLJGJ).
C     Note: D and DT are rectangular matrices.
C     Single precision version.
C
C-----------------------------------------------------------------------
      REAL D(ND2,ND1), DT(ND1,ND2), ZGL(ND1), ZG(ND2), IGLG(ND2,ND1)
      PARAMETER (NMAX=84)
      PARAMETER (NDD = NMAX)
      REAL*8  DD(NDD,NDD), DTD(NDD,NDD)
      REAL*8  ZGD(NDD), ZGLD(NDD), IGLGD(NDD,NDD)
      REAL*8  ALPHAD, BETAD
C
      IF (NPGL.LE.1) THEN
       WRITE(6,*) 'DGLJGJ: Minimum number of Gauss-Lobatto points is 2'
       call exitt
      ENDIF
      IF (NPGL.GT.NMAX) THEN
         WRITE(6,*) 'Polynomial degree too high in DGLJGJ'
         WRITE(6,*) 'Maximum polynomial degree is',NMAX
         WRITE(6,*) 'Here NPGL=',NPGL
         call exitt
      ENDIF
      IF ((ALPHA.LE.-1.).OR.(BETA.LE.-1.)) THEN
         WRITE(6,*) 'DGLJGJ: Alpha and Beta must be greater than -1'
         call exitt
      ENDIF
C
      ALPHAD = ALPHA
      BETAD  = BETA
      DO 100 I=1,NPG
         ZGD(I) = ZG(I)
         DO 100 J=1,NPGL
            IGLGD(I,J) = IGLG(I,J)
 100  CONTINUE
      DO 200 I=1,NPGL
         ZGLD(I) = ZGL(I)
 200  CONTINUE
      CALL DGLJGJD (DD,DTD,ZGLD,ZGD,IGLGD,NPGL,NPG,NDD,NDD,ALPHAD,BETAD)
      DO 300 I=1,NPG
      DO 300 J=1,NPGL
         D(I,J)  = DD(I,J)
         DT(J,I) = DTD(J,I)
 300  CONTINUE
      RETURN
      END
C
      SUBROUTINE DGLJGJD (D,DT,ZGL,ZG,IGLG,NPGL,NPG,ND1,ND2,ALPHA,BETA)
C-----------------------------------------------------------------------
C
C     Compute the (one-dimensional) derivative matrix D and its
C     transpose DT associated with taking the derivative of a variable
C     expanded on a Gauss-Lobatto Jacobi mesh (M1), and evaluate its
C     derivative on a Guass Jacobi mesh (M2).
C     Need the one-dimensional interpolation operator IM12
C     (see subroutine IGLJGJ).
C     Note: D and DT are rectangular matrices.
C     Double precision version.
C
C-----------------------------------------------------------------------
      IMPLICIT REAL*8  (A-H,O-Z)
      REAL*8  D(ND2,ND1), DT(ND1,ND2), ZGL(ND1), ZG(ND2)
      REAL*8  IGLG(ND2,ND1), ALPHA, BETA
C
      IF (NPGL.LE.1) THEN
       WRITE(6,*) 'DGLJGJD: Minimum number of Gauss-Lobatto points is 2'
       call exitt
      ENDIF
      IF ((ALPHA.LE.-1.).OR.(BETA.LE.-1.)) THEN
         WRITE(6,*) 'DGLJGJD: Alpha and Beta must be greater than -1'
         call exitt
      ENDIF
C
      EPS    = 1.e-6
      ONE    = 1.
      TWO    = 2.
      NGL    = NPGL-1
      DN     = ((NGL))
      EIGVAL = -DN*(DN+ALPHA+BETA+ONE)
C
      DO 100 I=1,NPG
      DO 100 J=1,NPGL
         DZ = ABS(ZG(I)-ZGL(J))
         IF (DZ.LT.EPS) THEN
            D(I,J) = (ALPHA*(ONE+ZG(I))-BETA*(ONE-ZG(I)))/
     $               (TWO*(ONE-ZG(I)**2))
         ELSE
            CALL JACOBF (PI,PDI,PM1,PDM1,PM2,PDM2,NGL,ALPHA,BETA,ZG(I))
            CALL JACOBF (PJ,PDJ,PM1,PDM1,PM2,PDM2,NGL,ALPHA,BETA,ZGL(J))
            FACI   = ALPHA*(ONE+ZG(I))-BETA*(ONE-ZG(I))
            FACJ   = ALPHA*(ONE+ZGL(J))-BETA*(ONE-ZGL(J))
            CONST  = EIGVAL*PJ+FACJ*PDJ
            D(I,J) = ((EIGVAL*PI+FACI*PDI)*(ZG(I)-ZGL(J))
     $               -(ONE-ZG(I)**2)*PDI)/(CONST*(ZG(I)-ZGL(J))**2)
         ENDIF
         DT(J,I) = D(I,J)
 100  CONTINUE
      RETURN
      END
C
      SUBROUTINE IGLM (I12,IT12,Z1,Z2,NZ1,NZ2,ND1,ND2)
C----------------------------------------------------------------------
C
C     Compute the one-dimensional interpolation operator (matrix) I12
C     ands its transpose IT12 for interpolating a variable from a
C     Gauss Legendre mesh (1) to a another mesh M (2).
C     Z1 : NZ1 Gauss Legendre points.
C     Z2 : NZ2 points on mesh M.
C
C--------------------------------------------------------------------
      REAL I12(ND2,ND1),IT12(ND1,ND2),Z1(ND1),Z2(ND2)
      IF (NZ1 .EQ. 1) THEN
         I12 (1,1) = 1.
         IT12(1,1) = 1.
         RETURN
      ENDIF
      DO 10 I=1,NZ2
         ZI = Z2(I)
         DO 10 J=1,NZ1
            I12 (I,J) = HGL(J,ZI,Z1,NZ1)
            IT12(J,I) = I12(I,J)
 10   CONTINUE
      RETURN
      END
c
      SUBROUTINE IGLLM (I12,IT12,Z1,Z2,NZ1,NZ2,ND1,ND2)
C----------------------------------------------------------------------
C
C     Compute the one-dimensional interpolation operator (matrix) I12
C     ands its transpose IT12 for interpolating a variable from a
C     Gauss-Lobatto Legendre mesh (1) to a another mesh M (2).
C     Z1 : NZ1 Gauss-Lobatto Legendre points.
C     Z2 : NZ2 points on mesh M.
C
C--------------------------------------------------------------------
      REAL I12(ND2,ND1),IT12(ND1,ND2),Z1(ND1),Z2(ND2)
      IF (NZ1 .EQ. 1) THEN
         I12 (1,1) = 1.
         IT12(1,1) = 1.
         RETURN
      ENDIF
      DO 10 I=1,NZ2
         ZI = Z2(I)
         DO 10 J=1,NZ1
            I12 (I,J) = HGLL(J,ZI,Z1,NZ1)
            IT12(J,I) = I12(I,J)
 10   CONTINUE
      RETURN
      END
C
      SUBROUTINE IGJM (I12,IT12,Z1,Z2,NZ1,NZ2,ND1,ND2,ALPHA,BETA)
C----------------------------------------------------------------------
C
C     Compute the one-dimensional interpolation operator (matrix) I12
C     ands its transpose IT12 for interpolating a variable from a
C     Gauss Jacobi mesh (1) to a another mesh M (2).
C     Z1 : NZ1 Gauss Jacobi points.
C     Z2 : NZ2 points on mesh M.
C     Single precision version.
C
C--------------------------------------------------------------------
      REAL I12(ND2,ND1),IT12(ND1,ND2),Z1(ND1),Z2(ND2)
      IF (NZ1 .EQ. 1) THEN
         I12 (1,1) = 1.
         IT12(1,1) = 1.
         RETURN
      ENDIF
      DO 10 I=1,NZ2
         ZI = Z2(I)
         DO 10 J=1,NZ1
            I12 (I,J) = HGJ(J,ZI,Z1,NZ1,ALPHA,BETA)
            IT12(J,I) = I12(I,J)
 10   CONTINUE
      RETURN
      END
c
      SUBROUTINE IGLJM (I12,IT12,Z1,Z2,NZ1,NZ2,ND1,ND2,ALPHA,BETA)
C----------------------------------------------------------------------
C
C     Compute the one-dimensional interpolation operator (matrix) I12
C     ands its transpose IT12 for interpolating a variable from a
C     Gauss-Lobatto Jacobi mesh (1) to a another mesh M (2).
C     Z1 : NZ1 Gauss-Lobatto Jacobi points.
C     Z2 : NZ2 points on mesh M.
C     Single precision version.
C
C--------------------------------------------------------------------
      REAL I12(ND2,ND1),IT12(ND1,ND2),Z1(ND1),Z2(ND2)
      IF (NZ1 .EQ. 1) THEN
         I12 (1,1) = 1.
         IT12(1,1) = 1.
         RETURN
      ENDIF
      DO 10 I=1,NZ2
         ZI = Z2(I)
         DO 10 J=1,NZ1
            I12 (I,J) = HGLJ(J,ZI,Z1,NZ1,ALPHA,BETA)
            IT12(J,I) = I12(I,J)
 10   CONTINUE
      RETURN
      END
