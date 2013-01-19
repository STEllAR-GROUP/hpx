      subroutine mxm(a,n1,b,n2,c,n3)
c
c     Compute matrix-matrix product C = A*B
c     for contiguously packed matrices A,B, and C.
c
      real a(n1,n2),b(n2,n3),c(n1,n3)
c
c
      integer aligned
      integer K10_mxm

#ifdef BLAS_MXM
      call dgemm('N','N',n1,n3,n2,1.0,a,n1,b,n2,0.0,c,n1)
      return
#endif
 
#ifdef BG
      call bg_aligned3(a,b,c,aligned)
      if (n2.eq.2) then
         call mxm44_2(a,n1,b,n2,c,n3)
      else if ((aligned.eq.1) .and.
     $         (n1.ge.8) .and. (n2.ge.8) .and. (n3.ge.8) .and.
     $         (modulo(n1,2).eq.0) .and. (modulo(n2,2).eq.0) ) then
         if (modulo(n3,4).eq.0) then
            call bg_mxm44(a,n1,b,n2,c,n3)
         else
            call bg_mxm44_uneven(a,n1,b,n2,c,n3)
         endif
      else if((aligned.eq.1) .and.
     $        (modulo(n1,6).eq.0) .and. (modulo(n3,6).eq.0) .and.
     $        (n2.ge.4) .and. (modulo(n2,2).eq.0) ) then
         call bg_mxm3(a,n1,b,n2,c,n3)
      else
         call mxm44_0(a,n1,b,n2,c,n3)
      endif
      return
#endif

#ifdef K10_MXM
      ! fow now only supported for lx1=8
      ! tuned for AMD K10
      ierr = K10_mxm(a,n1,b,n2,c,n3) 
      if (ierr.gt.0) call mxmf2(a,n1,b,n2,c,n3)
      return
#endif

      call mxmf2(a,n1,b,n2,c,n3)

      return
      end
