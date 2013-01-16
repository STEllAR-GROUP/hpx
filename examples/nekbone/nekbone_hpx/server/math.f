c-----------------------------------------------------------------------
      function mod1(i,n)
C
C     Yields MOD(I,N) with the exception that if I=K*N, result is N.
C
      MOD1=0
      IF (I.EQ.0) THEN
         return
      ENDIF
      IF (N.EQ.0) THEN
         WRITE(6,*) 
     $  'WARNING:  Attempt to take MOD(I,0) in function mod1.'
         return
      ENDIF
      II = I+N-1
      MOD1 = MOD(II,N)+1
      return
      END

c-----------------------------------------------------------------------
      subroutine rone(a,n)
      DIMENSION  A(1)
      DO 100 I = 1, N
 100     A(I ) = 1.0
      return
      END
c-----------------------------------------------------------------------
      subroutine rzero(a,n)
      DIMENSION  A(1)
      DO 100 I = 1, N
 100     A(I ) = 0.0
      return
      END

c-----------------------------------------------------------------------
      subroutine copy(a,b,n)
      real a(1),b(1)

      do i=1,n
         a(i)=b(i)
      enddo

      return
      end

c-----------------------------------------------------------------------
      subroutine col2(a,b,n)
      real a(1),b(1)

!xbm* unroll (10)
      do i=1,n
         a(i)=a(i)*b(i)
      enddo

      return
      end
C----------------------------------------------------------------------------
C
C     Vector reduction routines which require communication 
C     on a parallel machine. These routines must be substituted with
C     appropriate routines which take into account the specific
C     architecture.
C
C----------------------------------------------------------------------------
      function glsc3(hpx_bti,a,b,mult,n,
c nekmpi
     &     nid_,np_,nekcomm,nekgroup,nekreal)

      use, intrinsic :: iso_c_binding, only : c_ptr
      interface ! {{{
      subroutine gop(hpx_bti, x, w, op, n,
c nekmpi
     &     nid_,np_,nekcomm,nekgroup,nekreal)
        use, intrinsic :: iso_c_binding, only : c_ptr
        TYPE(C_PTR), INTENT(IN), VALUE :: hpx_bti 
        real x(n), w(n)
        character*3 op 
ccccccccccccccccccccccccccccccccccc
c nekmpi
        integer nid_,np_,nekcomm,nekgroup,nekreal
ccccccccccccccccccccccccccccccccccc
      end subroutine gop
      end interface ! }}}
      TYPE(C_PTR), INTENT(IN), VALUE :: hpx_bti
      
ccccccccccccccccccccccccccccccccccc
c nekmpi
      integer nid_,np_,nekcomm,nekgroup,nekreal
ccccccccccccccccccccccccccccccccccc

C
C     Perform inner-product in double precision
C
      real a(1),b(1),mult(1)
      real tmp(1),work(1)

      tmp(1) = 0.0
      do 10 i=1,n
         tmp(1) = tmp(1) + a(i)*b(i)*mult(i)
 10   continue
      call gop(hpx_bti,tmp,work,'+  ',1,
c nekmpi
     &     nid_,np_,nekcomm,nekgroup,nekreal)
      glsc3 = tmp(1)
      return
      end
c-----------------------------------------------------------------------
      subroutine add2s1(a,b,c1,n)
      real a(1),b(1)

      DO 100 I=1,N
        A(I)=C1*A(I)+B(I)
  100 CONTINUE
      return
      END subroutine add2s1



