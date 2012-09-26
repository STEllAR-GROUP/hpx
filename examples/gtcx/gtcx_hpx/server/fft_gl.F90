subroutine fftr1d(isign,irank,scale,x,y,icount)
  use precision
  implicit none
  integer, intent(in) :: isign,irank,icount
  real(wp), intent(in) :: scale
  real(wp), intent(inout), dimension(0:irank-1) :: x
  complex(wp), intent(inout), dimension(0:irank/2) :: y

  if (icount.gt.0.and.icount.le.3) then
     if(isign==1)then
        call r2cfftgl(1,irank,scale,x,y)
     else
        call c2rfftgl(-1,irank,scale,y,x)
     endif
  endif
end subroutine fftr1d

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! only for parallel direction
subroutine fftc1d(isign,irank,scale,x)
  use precision
  implicit none
  integer, intent(in) :: isign,irank
  real(wp), intent(in) :: scale
  complex(wp), intent(inout), dimension(0:irank-1) :: x
  complex(wp),  dimension(0:irank-1) :: y

  call c2cfftgl(isign,irank,scale,x,y)
  x=y
end subroutine fftc1d

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Subroutine r2cfftgl(isign,n,scale,rin,cout)
! real to complex transform. n: array size, must be power of 2
! rin: real array to be transformed,cout=\int{rin*exp(-i*k*x)}dx
! cout: complex array of output for Fourier modes=[0,n/2+1], normalized by n

  use precision
  implicit none
  integer,intent(in) :: n,isign
  real(wp),intent(in) :: rin(n)
  real(wp),intent(in) :: scale
  complex(wp),intent(out) :: cout(n/2+1)  
  integer i
  complex(wp),dimension(n) :: fdata,dwork
  
  do i=1,n
     fdata(i)=cmplx(rin(i),0.0)
  enddo
  
  call spcfft(fdata,n,-isign,dwork,scale)
  
  do i=1,n/2+1
     cout(i)=fdata(i)
  enddo

end Subroutine r2cfftgl
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Subroutine c2rfftgl(isign,n,scale,cin,rout)
! complex to real transform. n: array size
! cin: complex array of Fourier modes=[0,n/2+1] to be transformed
! rout: real array for output, rout=\int{cin*exp(ikx)}dk

  use precision
  implicit none
  integer,intent(in) :: isign,n
  real(wp),intent(in) :: scale
  complex(wp),intent(in) :: cin(n/2+1)
  real(wp),intent(out) :: rout(n)  
  integer i
  complex(wp),dimension(n) :: fdata,dwork
  
  do i=1,n/2+1
     fdata(i)=cin(i)
  enddo
! For next n/2-1 values, for the complex conjugate of first n/2 values.
  do i=1,n/2-1
     fdata(n/2+1+i)=cmplx(real(cin(n/2+1-i)),-aimag(cin(n/2+1-i)))
  enddo
  
  call spcfft(fdata,n,-isign,dwork,scale)
  
! The 1/n factor for the inverse FFT is added inside spcfft
  do i=1,n
     rout(i)=real(fdata(i))
  enddo

end Subroutine c2rfftgl
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Subroutine c2cfftgl(isign,n,scale,cin,cout)
! complex to complex transform. isign: direction of transform
! n: array size
! cin: complex array to be transformed, cout=\int{cin*exp(ifft*i*k*x)}dx
! cout: complex array of output for Fourier modes=[0,n-1], normalized by n

  use precision
  implicit none
  integer,intent(in) :: isign,n
  complex(wp),intent(in),dimension(n) :: cin
  complex(wp),intent(out),dimension(n) :: cout
  complex(wp),dimension(n) :: dwork
  real(wp) :: cfac,scale
  integer i
  
  cout=cin
  call spcfft(cout,n,-isign,dwork,scale)
  
end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!Subroutine fft(isign,nn,fdata)

!=======================================================================
! Single Precision Complex Fast Fourier Transform
!
!  A subroutine to compute the discrete Fourier transform by the fastest
! available algorithm for any length input series.
!
! Reference:
!        Ferguson, W., (1979),   A Simple Derivation of Glassmans's
!          General N Fast Fourier Transform, MRC Tech. Summ. Rept. 2029,
!          Math. Res. Cent. U. of Wisconsin, Madison, Wis.
!
!  REFERENCES
!  ----------
!
! Routines called:
! SPCPFT
!
! Functions called:
! MOD
! FLOAT
!
! VAX extensions:
! DO WHILE
!=======================================================================

SUBROUTINE SPCFFT(U,N,ISIGN,WORK,INTERP)

  use precision

! VARIABLES
! ---------

  IMPLICIT NONE

  LOGICAL(1)  &
    INU,      &  ! Flag for calling SUBROUTINE SPCPFT( arguments ).
    SCALE        ! .TRUE.=inverse transform -- .FALSE.=forward transform

  INTEGER     &
    A,        &  ! After    |
    B,        &  ! Before   |>  Factors of N.
    C,        &  ! Current  |
    N,        &  ! Length of the array to be transformed.
    I,        &  ! DO loop index.
    ISIGN        ! sign of transform

  REAL(wp)    &
    INTERP       ! interpolation factor

  COMPLEX(wp) &
    U(*),     &  !  Vector to be transformed
    WORK(*)      !  Working storage.

!     Initialize parameters.

  A = 1
  B = N
  C = 1

  INU = .TRUE.

  IF (ISIGN.EQ.1) THEN

     SCALE = .TRUE.

  ELSE IF (ISIGN.EQ.-1) THEN

     SCALE = .FALSE.

  END IF

! Calculate Fourier transform by means of Glassman's algorithm

  DO WHILE ( B .GT. 1 )

     A = C * A

! Start of routine to get factors of N

     C = 2

! Increment C until it is an integer factor of B

     DO WHILE ( MOD(B,C) .NE. 0 )

        C = C + 1

     END DO

! Calculate largest factor of B

     B = B / C


! Call Glassman's Fast Fourier Transform routine

     IF ( INU ) THEN

        CALL SPCPFT (A,B,C,U,WORK,ISIGN)

     ELSE

        CALL SPCPFT (A,B,C,WORK,U,ISIGN)

     END IF

! Set flag to swap input & output arrays to SPCPFT

     INU = ( .NOT. INU )

  END DO

! If odd number of calls to SPCPFT swap WORK into U

  IF ( .NOT. INU ) THEN

     DO I = 1, N
        U(I) = WORK(I)
     END DO

  END IF

! Scale the output for inverse Fourier transforms.

  IF ( SCALE ) THEN

     DO I = 1, N
        U(I) = U(I) / (N/INTERP)
     END DO

  END IF

END


!=======================================================================
! Single Precision Complex Prime Factor Transform
!
!  REFERENCES
!  ----------
!
! Calling routines:
! SPCFFT
!
! Subroutines called:
! -none-
!
! Functions called:
! CMLPX
! COS
! SIN
! FLOAT
!=======================================================================

SUBROUTINE SPCPFT( A, B, C, UIN, UOUT, ISIGN )

  use precision

! VARIABLES
! ---------

  IMPLICIT NONE

  INTEGER    &
    ISIGN,   &      !  ISIGN of the Fourier transform.
    A,       &      !  After   |
    B,       &      !  Before  |>  Factors of N.
    C,       &      !  Current |
    IA,      &      !  |
    IB,      &      !  |>  DO loop indicies.
    IC,      &      !  |>
    JCR,     &      !  |
    JC              !  Dummy index.

  REAL(doubleprec)   ANGLE

  COMPLEX(wp)   &
    UIN(B,C,A), &   !  Input vector.
    UOUT(B,A,C),&   !  Output vector.
    DELTA,      &   !  Fourier transform kernel.
    OMEGA,      &   !  Multiples of exp( i TWOPI ).
    SUM             !  Dummy register for addition for UOUT(B,A,C)

! ALGORITHM
! ---------

! Initialize run time parameters.


  ANGLE =8.0_doubleprec*ATAN(1.0_doubleprec) / REAL( A * C, doubleprec )
  OMEGA = CMPLX( 1.0, 0.0 )

! Check the ISIGN of the transform.

  IF( ISIGN .EQ. 1 ) THEN
     DELTA = CMPLX( COS(ANGLE), SIN(ANGLE) )
  ELSE
     DELTA = CMPLX( COS(ANGLE), -SIN(ANGLE) )
  END IF


! Do the computations.

  DO IC = 1, C
     DO IA = 1, A
        DO IB = 1, B

           SUM = UIN( IB, C, IA )

           DO JCR = 2, C
              JC = C + 1 - JCR
              SUM = UIN( IB, JC, IA ) + OMEGA * SUM
           END DO

           UOUT( IB, IA, IC ) = SUM

        END DO
        OMEGA = DELTA * OMEGA
     END DO
  END DO

END SUBROUTINE SPCPFT

