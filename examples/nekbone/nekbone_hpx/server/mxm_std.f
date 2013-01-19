      subroutine mxmf2(A,N1,B,N2,C,N3)
c
c     unrolled loop version 
c
      real a(n1,n2),b(n2,n3),c(n1,n3)

      if (n2.le.8) then
         if (n2.eq.1) then
            call mxf1(a,n1,b,n2,c,n3)
         elseif (n2.eq.2) then
            call mxf2(a,n1,b,n2,c,n3)
         elseif (n2.eq.3) then
            call mxf3(a,n1,b,n2,c,n3)
         elseif (n2.eq.4) then
            call mxf4(a,n1,b,n2,c,n3)
         elseif (n2.eq.5) then
            call mxf5(a,n1,b,n2,c,n3)
         elseif (n2.eq.6) then
            call mxf6(a,n1,b,n2,c,n3)
         elseif (n2.eq.7) then
            call mxf7(a,n1,b,n2,c,n3)
         else
            call mxf8(a,n1,b,n2,c,n3)
         endif
      elseif (n2.le.16) then
         if (n2.eq.9) then
            call mxf9(a,n1,b,n2,c,n3)
         elseif (n2.eq.10) then
            call mxf10(a,n1,b,n2,c,n3)
         elseif (n2.eq.11) then
            call mxf11(a,n1,b,n2,c,n3)
         elseif (n2.eq.12) then
            call mxf12(a,n1,b,n2,c,n3)
         elseif (n2.eq.13) then
            call mxf13(a,n1,b,n2,c,n3)
         elseif (n2.eq.14) then
            call mxf14(a,n1,b,n2,c,n3)
         elseif (n2.eq.15) then
            call mxf15(a,n1,b,n2,c,n3)
         else
            call mxf16(a,n1,b,n2,c,n3)
         endif
      elseif (n2.le.24) then
         if (n2.eq.17) then
            call mxf17(a,n1,b,n2,c,n3)
         elseif (n2.eq.18) then
            call mxf18(a,n1,b,n2,c,n3)
         elseif (n2.eq.19) then
            call mxf19(a,n1,b,n2,c,n3)
         elseif (n2.eq.20) then
            call mxf20(a,n1,b,n2,c,n3)
         elseif (n2.eq.21) then
            call mxf21(a,n1,b,n2,c,n3)
         elseif (n2.eq.22) then
            call mxf22(a,n1,b,n2,c,n3)
         elseif (n2.eq.23) then
            call mxf23(a,n1,b,n2,c,n3)
         elseif (n2.eq.24) then
            call mxf24(a,n1,b,n2,c,n3)
         endif
      else
         call mxm44_0(a,n1,b,n2,c,n3)
      endif
c
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf1(a,n1,b,n2,c,n3)
c
      real a(n1,1),b(1,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf2(a,n1,b,n2,c,n3)
c
      real a(n1,2),b(2,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf3(a,n1,b,n2,c,n3)
c
      real a(n1,3),b(3,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf4(a,n1,b,n2,c,n3)
c
      real a(n1,4),b(4,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf5(a,n1,b,n2,c,n3)
c
      real a(n1,5),b(5,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf6(a,n1,b,n2,c,n3)
c
      real a(n1,6),b(6,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf7(a,n1,b,n2,c,n3)
c
      real a(n1,7),b(7,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf8(a,n1,b,n2,c,n3)
c
      real a(n1,8),b(8,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf9(a,n1,b,n2,c,n3)
c
      real a(n1,9),b(9,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf10(a,n1,b,n2,c,n3)
c
      real a(n1,10),b(10,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf11(a,n1,b,n2,c,n3)
c
      real a(n1,11),b(11,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf12(a,n1,b,n2,c,n3)
c
      real a(n1,12),b(12,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf13(a,n1,b,n2,c,n3)
c
      real a(n1,13),b(13,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf14(a,n1,b,n2,c,n3)
c
      real a(n1,14),b(14,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf15(a,n1,b,n2,c,n3)
c
      real a(n1,15),b(15,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf16(a,n1,b,n2,c,n3)
c
      real a(n1,16),b(16,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf17(a,n1,b,n2,c,n3)
c
      real a(n1,17),b(17,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf18(a,n1,b,n2,c,n3)
c
      real a(n1,18),b(18,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf19(a,n1,b,n2,c,n3)
c
      real a(n1,19),b(19,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf20(a,n1,b,n2,c,n3)
c
      real a(n1,20),b(20,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf21(a,n1,b,n2,c,n3)
c
      real a(n1,21),b(21,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf22(a,n1,b,n2,c,n3)
c
      real a(n1,22),b(22,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
     $             + a(i,22)*b(22,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf23(a,n1,b,n2,c,n3)
c
      real a(n1,23),b(23,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
     $             + a(i,22)*b(22,j)
     $             + a(i,23)*b(23,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxf24(a,n1,b,n2,c,n3)
c
      real a(n1,24),b(24,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
     $             + a(i,22)*b(22,j)
     $             + a(i,23)*b(23,j)
     $             + a(i,24)*b(24,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxm44_0(a, m, b, k, c, n)
c
c matrix multiply with a 4x4 pencil 
c
      real a(m,k), b(k,n), c(m,n)
      real s11, s12, s13, s14, s21, s22, s23, s24
      real s31, s32, s33, s34, s41, s42, s43, s44

      mresid = iand(m,3) 
      nresid = iand(n,3) 
      m1 = m - mresid + 1
      n1 = n - nresid + 1

      do i=1,m-mresid,4
        do j=1,n-nresid,4
          s11 = 0.0d0
          s21 = 0.0d0
          s31 = 0.0d0
          s41 = 0.0d0
          s12 = 0.0d0
          s22 = 0.0d0
          s32 = 0.0d0
          s42 = 0.0d0
          s13 = 0.0d0
          s23 = 0.0d0
          s33 = 0.0d0
          s43 = 0.0d0
          s14 = 0.0d0
          s24 = 0.0d0
          s34 = 0.0d0
          s44 = 0.0d0
          do l=1,k
            s11 = s11 + a(i,l)*b(l,j)
            s12 = s12 + a(i,l)*b(l,j+1)
            s13 = s13 + a(i,l)*b(l,j+2)
            s14 = s14 + a(i,l)*b(l,j+3)

            s21 = s21 + a(i+1,l)*b(l,j)
            s22 = s22 + a(i+1,l)*b(l,j+1)
            s23 = s23 + a(i+1,l)*b(l,j+2)
            s24 = s24 + a(i+1,l)*b(l,j+3)

            s31 = s31 + a(i+2,l)*b(l,j)
            s32 = s32 + a(i+2,l)*b(l,j+1)
            s33 = s33 + a(i+2,l)*b(l,j+2)
            s34 = s34 + a(i+2,l)*b(l,j+3)

            s41 = s41 + a(i+3,l)*b(l,j)
            s42 = s42 + a(i+3,l)*b(l,j+1)
            s43 = s43 + a(i+3,l)*b(l,j+2)
            s44 = s44 + a(i+3,l)*b(l,j+3)
          enddo
          c(i,j)     = s11 
          c(i,j+1)   = s12 
          c(i,j+2)   = s13
          c(i,j+3)   = s14

          c(i+1,j)   = s21 
          c(i+2,j)   = s31 
          c(i+3,j)   = s41 

          c(i+1,j+1) = s22
          c(i+2,j+1) = s32
          c(i+3,j+1) = s42

          c(i+1,j+2) = s23
          c(i+2,j+2) = s33
          c(i+3,j+2) = s43

          c(i+1,j+3) = s24
          c(i+2,j+3) = s34
          c(i+3,j+3) = s44
        enddo
* Residual when n is not multiple of 4
        if (nresid .ne. 0) then
          if (nresid .eq. 1) then
            s11 = 0.0d0
            s21 = 0.0d0
            s31 = 0.0d0
            s41 = 0.0d0
            do l=1,k
              s11 = s11 + a(i,l)*b(l,n)
              s21 = s21 + a(i+1,l)*b(l,n)
              s31 = s31 + a(i+2,l)*b(l,n)
              s41 = s41 + a(i+3,l)*b(l,n)
            enddo
            c(i,n)     = s11 
            c(i+1,n)   = s21 
            c(i+2,n)   = s31 
            c(i+3,n)   = s41 
          elseif (nresid .eq. 2) then
            s11 = 0.0d0
            s21 = 0.0d0
            s31 = 0.0d0
            s41 = 0.0d0
            s12 = 0.0d0
            s22 = 0.0d0
            s32 = 0.0d0
            s42 = 0.0d0
            do l=1,k
              s11 = s11 + a(i,l)*b(l,j)
              s12 = s12 + a(i,l)*b(l,j+1)

              s21 = s21 + a(i+1,l)*b(l,j)
              s22 = s22 + a(i+1,l)*b(l,j+1)

              s31 = s31 + a(i+2,l)*b(l,j)
              s32 = s32 + a(i+2,l)*b(l,j+1)

              s41 = s41 + a(i+3,l)*b(l,j)
              s42 = s42 + a(i+3,l)*b(l,j+1)
            enddo
            c(i,j)     = s11 
            c(i,j+1)   = s12

            c(i+1,j)   = s21 
            c(i+2,j)   = s31 
            c(i+3,j)   = s41 

            c(i+1,j+1) = s22
            c(i+2,j+1) = s32
            c(i+3,j+1) = s42
          else
            s11 = 0.0d0
            s21 = 0.0d0
            s31 = 0.0d0
            s41 = 0.0d0
            s12 = 0.0d0
            s22 = 0.0d0
            s32 = 0.0d0
            s42 = 0.0d0
            s13 = 0.0d0
            s23 = 0.0d0
            s33 = 0.0d0
            s43 = 0.0d0
            do l=1,k
              s11 = s11 + a(i,l)*b(l,j)
              s12 = s12 + a(i,l)*b(l,j+1)
              s13 = s13 + a(i,l)*b(l,j+2)

              s21 = s21 + a(i+1,l)*b(l,j)
              s22 = s22 + a(i+1,l)*b(l,j+1)
              s23 = s23 + a(i+1,l)*b(l,j+2)

              s31 = s31 + a(i+2,l)*b(l,j)
              s32 = s32 + a(i+2,l)*b(l,j+1)
              s33 = s33 + a(i+2,l)*b(l,j+2)

              s41 = s41 + a(i+3,l)*b(l,j)
              s42 = s42 + a(i+3,l)*b(l,j+1)
              s43 = s43 + a(i+3,l)*b(l,j+2)
            enddo
            c(i,j)     = s11 
            c(i+1,j)   = s21 
            c(i+2,j)   = s31 
            c(i+3,j)   = s41 
            c(i,j+1)   = s12 
            c(i+1,j+1) = s22
            c(i+2,j+1) = s32
            c(i+3,j+1) = s42
            c(i,j+2)   = s13
            c(i+1,j+2) = s23
            c(i+2,j+2) = s33
            c(i+3,j+2) = s43
          endif
        endif
      enddo

* Residual when m is not multiple of 4
      if (mresid .eq. 0) then
        return
      elseif (mresid .eq. 1) then
        do j=1,n-nresid,4
          s11 = 0.0d0
          s12 = 0.0d0
          s13 = 0.0d0
          s14 = 0.0d0
          do l=1,k
            s11 = s11 + a(m,l)*b(l,j)
            s12 = s12 + a(m,l)*b(l,j+1)
            s13 = s13 + a(m,l)*b(l,j+2)
            s14 = s14 + a(m,l)*b(l,j+3)
          enddo
          c(m,j)     = s11 
          c(m,j+1)   = s12 
          c(m,j+2)   = s13
          c(m,j+3)   = s14
        enddo
* mresid is 1, check nresid
        if (nresid .eq. 0) then
          return
        elseif (nresid .eq. 1) then
          s11 = 0.0d0
          do l=1,k
            s11 = s11 + a(m,l)*b(l,n)
          enddo
          c(m,n) = s11
          return
        elseif (nresid .eq. 2) then
          s11 = 0.0d0
          s12 = 0.0d0
          do l=1,k
            s11 = s11 + a(m,l)*b(l,n-1)
            s12 = s12 + a(m,l)*b(l,n)
          enddo
          c(m,n-1) = s11
          c(m,n) = s12
          return
        else
          s11 = 0.0d0
          s12 = 0.0d0
          s13 = 0.0d0
          do l=1,k
            s11 = s11 + a(m,l)*b(l,n-2)
            s12 = s12 + a(m,l)*b(l,n-1)
            s13 = s13 + a(m,l)*b(l,n)
          enddo
          c(m,n-2) = s11
          c(m,n-1) = s12
          c(m,n) = s13
          return
        endif          
      elseif (mresid .eq. 2) then
        do j=1,n-nresid,4
          s11 = 0.0d0
          s12 = 0.0d0
          s13 = 0.0d0
          s14 = 0.0d0
          s21 = 0.0d0
          s22 = 0.0d0
          s23 = 0.0d0
          s24 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-1,l)*b(l,j)
            s12 = s12 + a(m-1,l)*b(l,j+1)
            s13 = s13 + a(m-1,l)*b(l,j+2)
            s14 = s14 + a(m-1,l)*b(l,j+3)

            s21 = s21 + a(m,l)*b(l,j)
            s22 = s22 + a(m,l)*b(l,j+1)
            s23 = s23 + a(m,l)*b(l,j+2)
            s24 = s24 + a(m,l)*b(l,j+3)
          enddo
          c(m-1,j)   = s11 
          c(m-1,j+1) = s12 
          c(m-1,j+2) = s13
          c(m-1,j+3) = s14
          c(m,j)     = s21
          c(m,j+1)   = s22 
          c(m,j+2)   = s23
          c(m,j+3)   = s24
        enddo
* mresid is 2, check nresid
        if (nresid .eq. 0) then
          return
        elseif (nresid .eq. 1) then
          s11 = 0.0d0
          s21 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-1,l)*b(l,n)
            s21 = s21 + a(m,l)*b(l,n)
          enddo
          c(m-1,n) = s11
          c(m,n) = s21
          return
        elseif (nresid .eq. 2) then
          s11 = 0.0d0
          s21 = 0.0d0
          s12 = 0.0d0
          s22 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-1,l)*b(l,n-1)
            s12 = s12 + a(m-1,l)*b(l,n)
            s21 = s21 + a(m,l)*b(l,n-1)
            s22 = s22 + a(m,l)*b(l,n)
          enddo
          c(m-1,n-1) = s11
          c(m-1,n)   = s12
          c(m,n-1)   = s21
          c(m,n)     = s22
          return
        else
          s11 = 0.0d0
          s21 = 0.0d0
          s12 = 0.0d0
          s22 = 0.0d0
          s13 = 0.0d0
          s23 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-1,l)*b(l,n-2)
            s12 = s12 + a(m-1,l)*b(l,n-1)
            s13 = s13 + a(m-1,l)*b(l,n)
            s21 = s21 + a(m,l)*b(l,n-2)
            s22 = s22 + a(m,l)*b(l,n-1)
            s23 = s23 + a(m,l)*b(l,n)
          enddo
          c(m-1,n-2) = s11
          c(m-1,n-1) = s12
          c(m-1,n)   = s13
          c(m,n-2)   = s21
          c(m,n-1)   = s22
          c(m,n)     = s23
          return
        endif
      else
* mresid is 3
        do j=1,n-nresid,4
          s11 = 0.0d0
          s21 = 0.0d0
          s31 = 0.0d0

          s12 = 0.0d0
          s22 = 0.0d0
          s32 = 0.0d0

          s13 = 0.0d0
          s23 = 0.0d0
          s33 = 0.0d0

          s14 = 0.0d0
          s24 = 0.0d0
          s34 = 0.0d0

          do l=1,k
            s11 = s11 + a(m-2,l)*b(l,j)
            s12 = s12 + a(m-2,l)*b(l,j+1)
            s13 = s13 + a(m-2,l)*b(l,j+2)
            s14 = s14 + a(m-2,l)*b(l,j+3)

            s21 = s21 + a(m-1,l)*b(l,j)
            s22 = s22 + a(m-1,l)*b(l,j+1)
            s23 = s23 + a(m-1,l)*b(l,j+2)
            s24 = s24 + a(m-1,l)*b(l,j+3)

            s31 = s31 + a(m,l)*b(l,j)
            s32 = s32 + a(m,l)*b(l,j+1)
            s33 = s33 + a(m,l)*b(l,j+2)
            s34 = s34 + a(m,l)*b(l,j+3)
          enddo
          c(m-2,j)   = s11 
          c(m-2,j+1) = s12 
          c(m-2,j+2) = s13
          c(m-2,j+3) = s14

          c(m-1,j)   = s21 
          c(m-1,j+1) = s22
          c(m-1,j+2) = s23
          c(m-1,j+3) = s24

          c(m,j)     = s31 
          c(m,j+1)   = s32
          c(m,j+2)   = s33
          c(m,j+3)   = s34
        enddo
* mresid is 3, check nresid
        if (nresid .eq. 0) then
          return
        elseif (nresid .eq. 1) then
          s11 = 0.0d0
          s21 = 0.0d0
          s31 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-2,l)*b(l,n)
            s21 = s21 + a(m-1,l)*b(l,n)
            s31 = s31 + a(m,l)*b(l,n)
          enddo
          c(m-2,n) = s11
          c(m-1,n) = s21
          c(m,n) = s31
          return
        elseif (nresid .eq. 2) then
          s11 = 0.0d0
          s21 = 0.0d0
          s31 = 0.0d0
          s12 = 0.0d0
          s22 = 0.0d0
          s32 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-2,l)*b(l,n-1)
            s12 = s12 + a(m-2,l)*b(l,n)
            s21 = s21 + a(m-1,l)*b(l,n-1)
            s22 = s22 + a(m-1,l)*b(l,n)
            s31 = s31 + a(m,l)*b(l,n-1)
            s32 = s32 + a(m,l)*b(l,n)
          enddo
          c(m-2,n-1) = s11
          c(m-2,n)   = s12
          c(m-1,n-1) = s21
          c(m-1,n)   = s22
          c(m,n-1)   = s31
          c(m,n)     = s32
          return
        else
          s11 = 0.0d0
          s21 = 0.0d0
          s31 = 0.0d0
          s12 = 0.0d0
          s22 = 0.0d0
          s32 = 0.0d0
          s13 = 0.0d0
          s23 = 0.0d0
          s33 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-2,l)*b(l,n-2)
            s12 = s12 + a(m-2,l)*b(l,n-1)
            s13 = s13 + a(m-2,l)*b(l,n)
            s21 = s21 + a(m-1,l)*b(l,n-2)
            s22 = s22 + a(m-1,l)*b(l,n-1)
            s23 = s23 + a(m-1,l)*b(l,n)
            s31 = s31 + a(m,l)*b(l,n-2)
            s32 = s32 + a(m,l)*b(l,n-1)
            s33 = s33 + a(m,l)*b(l,n)
          enddo
          c(m-2,n-2) = s11
          c(m-2,n-1) = s12
          c(m-2,n)   = s13
          c(m-1,n-2) = s21
          c(m-1,n-1) = s22
          c(m-1,n)   = s23
          c(m,n-2)   = s31
          c(m,n-1)   = s32
          c(m,n)     = s33
          return
        endif
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine mxm44_2(a, m, b, k, c, n)
      real a(m,2), b(2,n), c(m,n)

      nresid = iand(n,3) 
      n1 = n - nresid + 1

      do j=1,n-nresid,4
         do i=1,m
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
            c(i,j+1) = a(i,1)*b(1,j+1)
     $             + a(i,2)*b(2,j+1)
            c(i,j+2) = a(i,1)*b(1,j+2)
     $             + a(i,2)*b(2,j+2)
            c(i,j+3) = a(i,1)*b(1,j+3)
     $             + a(i,2)*b(2,j+3)
         enddo
      enddo
      if (nresid .eq. 0) then
        return
      elseif (nresid .eq. 1) then
         do i=1,m
            c(i,n) = a(i,1)*b(1,n)
     $             + a(i,2)*b(2,n)
         enddo
      elseif (nresid .eq. 2) then
         do i=1,m
            c(i,n-1) = a(i,1)*b(1,n-1)
     $             + a(i,2)*b(2,n-1)
            c(i,n) = a(i,1)*b(1,n)
     $             + a(i,2)*b(2,n)
         enddo
      else
         do i=1,m
            c(i,n-2) = a(i,1)*b(1,n-2)
     $             + a(i,2)*b(2,n-2)
            c(i,n-1) = a(i,1)*b(1,n-1)
     $             + a(i,2)*b(2,n-1)
            c(i,n) = a(i,1)*b(1,n)
     $             + a(i,2)*b(2,n)
         enddo
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine initab(a,b,n)
      real a(1),b(1)
      do i=1,n-1
         x  = i
         k = mod(i,19) + 2
         l = mod(i,17) + 5
         m = mod(i,31) + 3
         a(i) = -.25*(a(i)+a(i+1)) + (x*x + k + l)/(x*x+m)
         b(i) = -.25*(b(i)+b(i+1)) + (x*x + k + m)/(x*x+l)
      enddo
      a(n) = -.25*(a(n)+a(n)) + (x*x + k + l)/(x*x+m)
      b(n) = -.25*(b(n)+b(n)) + (x*x + k + m)/(x*x+l)
      return
      end
c-----------------------------------------------------------------------
      subroutine mxms(a,n1,b,n2,c,n3)
C----------------------------------------------------------------------
C
C     Matrix-vector product routine. 
C     NOTE: Use assembly coded routine if available.
C
C---------------------------------------------------------------------
      REAL A(N1,N2),B(N2,N3),C(N1,N3)
C
         N0=N1*N3
         DO 10 I=1,N0
            C(I,1)=0.
 10      CONTINUE
         DO 100 J=1,N3
         DO 100 K=1,N2
         BB=B(K,J)
         DO 100 I=1,N1
            C(I,J)=C(I,J)+A(I,K)*BB
 100     CONTINUE
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmu4(a,n1,b,n2,c,n3)
C----------------------------------------------------------------------
C
C     Matrix-vector product routine. 
C     NOTE: Use assembly coded routine if available.
C
C---------------------------------------------------------------------
      REAL A(N1,N2),B(N2,N3),C(N1,N3)
C
         N0=N1*N3
         DO 10 I=1,N0
            C(I,1)=0.
 10      CONTINUE
         i1 = n1 - mod(n1,4) + 1
            DO 100 J=1,N3
            DO 100 K=1,N2
            BB=B(K,J)
               DO I=1,N1-3,4
                  C(I  ,J)=C(I  ,J)+A(I  ,K)*BB
                  C(I+1,J)=C(I+1,J)+A(I+1,K)*BB
                  C(I+2,J)=C(I+2,J)+A(I+2,K)*BB
                  C(I+3,J)=C(I+3,J)+A(I+3,K)*BB
               enddo
               DO i=i1,N1
                  C(I  ,J)=C(I  ,J)+A(I  ,K)*BB
               enddo
 100        CONTINUE
      return
      end
c-----------------------------------------------------------------------
      subroutine madd (a,n1,b,n2,c,n3)
c
      real a(n1,n2),b(n2,n3),c(n1,n3)
c
      do j=1,n3
      do i=1,n1
         c(i,j) = a(i,j)+b(i,j)
      enddo
      enddo
c
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmUR2(a,n1,b,n2,c,n3)
C----------------------------------------------------------------------
C
C     Matrix-vector product routine. 
C     NOTE: Use assembly coded routine if available.
C
C---------------------------------------------------------------------
      REAL A(N1,N2),B(N2,N3),C(N1,N3)
C
      if (n2.le.8) then
         if (n2.eq.1) then
            call mxmur2_1(a,n1,b,n2,c,n3)
         elseif (n2.eq.2) then
            call mxmur2_2(a,n1,b,n2,c,n3)
         elseif (n2.eq.3) then
            call mxmur2_3(a,n1,b,n2,c,n3)
         elseif (n2.eq.4) then
            call mxmur2_4(a,n1,b,n2,c,n3)
         elseif (n2.eq.5) then
            call mxmur2_5(a,n1,b,n2,c,n3)
         elseif (n2.eq.6) then
            call mxmur2_6(a,n1,b,n2,c,n3)
         elseif (n2.eq.7) then
            call mxmur2_7(a,n1,b,n2,c,n3)
         else
            call mxmur2_8(a,n1,b,n2,c,n3)
         endif
      elseif (n2.le.16) then
         if (n2.eq.9) then
            call mxmur2_9(a,n1,b,n2,c,n3)
         elseif (n2.eq.10) then
            call mxmur2_10(a,n1,b,n2,c,n3)
         elseif (n2.eq.11) then
            call mxmur2_11(a,n1,b,n2,c,n3)
         elseif (n2.eq.12) then
            call mxmur2_12(a,n1,b,n2,c,n3)
         elseif (n2.eq.13) then
            call mxmur2_13(a,n1,b,n2,c,n3)
         elseif (n2.eq.14) then
            call mxmur2_14(a,n1,b,n2,c,n3)
         elseif (n2.eq.15) then
            call mxmur2_15(a,n1,b,n2,c,n3)
         else
            call mxmur2_16(a,n1,b,n2,c,n3)
         endif
      else
         N0=N1*N3
         DO 10 I=1,N0
            C(I,1)=0.
 10      CONTINUE
         DO 100 J=1,N3
         DO 100 K=1,N2
         BB=B(K,J)
         DO 100 I=1,N1
            C(I,J)=C(I,J)+A(I,K)*BB
 100     CONTINUE
      endif
      return
      end
c
      subroutine mxmur2_1(a,n1,b,n2,c,n3)
c
      real a(n1,1),b(1,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_2(a,n1,b,n2,c,n3)
c
      real a(n1,2),b(2,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_3(a,n1,b,n2,c,n3)
c
      real a(n1,3),b(3,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_4(a,n1,b,n2,c,n3)
c
      real a(n1,4),b(4,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_5(a,n1,b,n2,c,n3)
c
      real a(n1,5),b(5,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_6(a,n1,b,n2,c,n3)
c
      real a(n1,6),b(6,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_7(a,n1,b,n2,c,n3)
c
      real a(n1,7),b(7,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_8(a,n1,b,n2,c,n3)
c
      real a(n1,8),b(8,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_9(a,n1,b,n2,c,n3)
c
      real a(n1,9),b(9,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_10(a,n1,b,n2,c,n3)
c
      real a(n1,10),b(10,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_11(a,n1,b,n2,c,n3)
c
      real a(n1,11),b(11,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_12(a,n1,b,n2,c,n3)
c
      real a(n1,12),b(12,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_13(a,n1,b,n2,c,n3)
c
      real a(n1,13),b(13,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_14(a,n1,b,n2,c,n3)
c
      real a(n1,14),b(14,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_15(a,n1,b,n2,c,n3)
c
      real a(n1,15),b(15,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
         enddo
      enddo
      return
      end
      subroutine mxmur2_16(a,n1,b,n2,c,n3)
c
      real a(n1,16),b(16,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmUR3(a,n1,b,n2,c,n3)
C----------------------------------------------------------------------
C
C     Matrix-vector product routine. 
C     NOTE: Use assembly coded routine if available.
C
C---------------------------------------------------------------------
      REAL A(N1,N2),B(N2,N3),C(N1,N3)
C
      N0=N1*N3
      DO 10 I=1,N0
         C(I,1)=0.
 10   CONTINUE
      if (n3.le.8) then
         if (n3.eq.1) then
            call mxmur3_1(a,n1,b,n2,c,n3)
         elseif (n3.eq.2) then
            call mxmur3_2(a,n1,b,n2,c,n3)
         elseif (n3.eq.3) then
            call mxmur3_3(a,n1,b,n2,c,n3)
         elseif (n3.eq.4) then
            call mxmur3_4(a,n1,b,n2,c,n3)
         elseif (n3.eq.5) then
            call mxmur3_5(a,n1,b,n2,c,n3)
         elseif (n3.eq.6) then
            call mxmur3_6(a,n1,b,n2,c,n3)
         elseif (n3.eq.7) then
            call mxmur3_7(a,n1,b,n2,c,n3)
         else
            call mxmur3_8(a,n1,b,n2,c,n3)
         endif
      elseif (n3.le.16) then
         if (n3.eq.9) then
            call mxmur3_9(a,n1,b,n2,c,n3)
         elseif (n3.eq.10) then
            call mxmur3_10(a,n1,b,n2,c,n3)
         elseif (n3.eq.11) then
            call mxmur3_11(a,n1,b,n2,c,n3)
         elseif (n3.eq.12) then
            call mxmur3_12(a,n1,b,n2,c,n3)
         elseif (n3.eq.13) then
            call mxmur3_13(a,n1,b,n2,c,n3)
         elseif (n3.eq.14) then
            call mxmur3_14(a,n1,b,n2,c,n3)
         elseif (n3.eq.15) then
            call mxmur3_15(a,n1,b,n2,c,n3)
         else
            call mxmur3_16(a,n1,b,n2,c,n3)
         endif
      else
         DO 100 J=1,N3
         DO 100 K=1,N2
         BB=B(K,J)
         DO 100 I=1,N1
            C(I,J)=C(I,J)+A(I,K)*BB
 100     CONTINUE
      endif
      return
      end
c
      subroutine mxmur3_16(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,16),c(n1,16)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         tmp6  =  b(k, 6)
         tmp7  =  b(k, 7)
         tmp8  =  b(k, 8)
         tmp9  =  b(k, 9)
         tmp10 =  b(k,10)
         tmp11 =  b(k,11)
         tmp12 =  b(k,12)
         tmp13 =  b(k,13)
         tmp14 =  b(k,14)
         tmp15 =  b(k,15)
         tmp16 =  b(k,16)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
            c(i, 6)  =  c(i, 6) + a(i,k) * tmp6
            c(i, 7)  =  c(i, 7) + a(i,k) * tmp7
            c(i, 8)  =  c(i, 8) + a(i,k) * tmp8
            c(i, 9)  =  c(i, 9) + a(i,k) * tmp9
            c(i,10)  =  c(i,10) + a(i,k) * tmp10
            c(i,11)  =  c(i,11) + a(i,k) * tmp11
            c(i,12)  =  c(i,12) + a(i,k) * tmp12
            c(i,13)  =  c(i,13) + a(i,k) * tmp13
            c(i,14)  =  c(i,14) + a(i,k) * tmp14
            c(i,15)  =  c(i,15) + a(i,k) * tmp15
            c(i,16)  =  c(i,16) + a(i,k) * tmp16
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_15(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,15),c(n1,15)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         tmp6  =  b(k, 6)
         tmp7  =  b(k, 7)
         tmp8  =  b(k, 8)
         tmp9  =  b(k, 9)
         tmp10 =  b(k,10)
         tmp11 =  b(k,11)
         tmp12 =  b(k,12)
         tmp13 =  b(k,13)
         tmp14 =  b(k,14)
         tmp15 =  b(k,15)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
            c(i, 6)  =  c(i, 6) + a(i,k) * tmp6
            c(i, 7)  =  c(i, 7) + a(i,k) * tmp7
            c(i, 8)  =  c(i, 8) + a(i,k) * tmp8
            c(i, 9)  =  c(i, 9) + a(i,k) * tmp9
            c(i,10)  =  c(i,10) + a(i,k) * tmp10
            c(i,11)  =  c(i,11) + a(i,k) * tmp11
            c(i,12)  =  c(i,12) + a(i,k) * tmp12
            c(i,13)  =  c(i,13) + a(i,k) * tmp13
            c(i,14)  =  c(i,14) + a(i,k) * tmp14
            c(i,15)  =  c(i,15) + a(i,k) * tmp15
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_14(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,14),c(n1,14)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         tmp6  =  b(k, 6)
         tmp7  =  b(k, 7)
         tmp8  =  b(k, 8)
         tmp9  =  b(k, 9)
         tmp10 =  b(k,10)
         tmp11 =  b(k,11)
         tmp12 =  b(k,12)
         tmp13 =  b(k,13)
         tmp14 =  b(k,14)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
            c(i, 6)  =  c(i, 6) + a(i,k) * tmp6
            c(i, 7)  =  c(i, 7) + a(i,k) * tmp7
            c(i, 8)  =  c(i, 8) + a(i,k) * tmp8
            c(i, 9)  =  c(i, 9) + a(i,k) * tmp9
            c(i,10)  =  c(i,10) + a(i,k) * tmp10
            c(i,11)  =  c(i,11) + a(i,k) * tmp11
            c(i,12)  =  c(i,12) + a(i,k) * tmp12
            c(i,13)  =  c(i,13) + a(i,k) * tmp13
            c(i,14)  =  c(i,14) + a(i,k) * tmp14
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_13(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,13),c(n1,13)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         tmp6  =  b(k, 6)
         tmp7  =  b(k, 7)
         tmp8  =  b(k, 8)
         tmp9  =  b(k, 9)
         tmp10 =  b(k,10)
         tmp11 =  b(k,11)
         tmp12 =  b(k,12)
         tmp13 =  b(k,13)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
            c(i, 6)  =  c(i, 6) + a(i,k) * tmp6
            c(i, 7)  =  c(i, 7) + a(i,k) * tmp7
            c(i, 8)  =  c(i, 8) + a(i,k) * tmp8
            c(i, 9)  =  c(i, 9) + a(i,k) * tmp9
            c(i,10)  =  c(i,10) + a(i,k) * tmp10
            c(i,11)  =  c(i,11) + a(i,k) * tmp11
            c(i,12)  =  c(i,12) + a(i,k) * tmp12
            c(i,13)  =  c(i,13) + a(i,k) * tmp13
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_12(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,12),c(n1,12)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         tmp6  =  b(k, 6)
         tmp7  =  b(k, 7)
         tmp8  =  b(k, 8)
         tmp9  =  b(k, 9)
         tmp10 =  b(k,10)
         tmp11 =  b(k,11)
         tmp12 =  b(k,12)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
            c(i, 6)  =  c(i, 6) + a(i,k) * tmp6
            c(i, 7)  =  c(i, 7) + a(i,k) * tmp7
            c(i, 8)  =  c(i, 8) + a(i,k) * tmp8
            c(i, 9)  =  c(i, 9) + a(i,k) * tmp9
            c(i,10)  =  c(i,10) + a(i,k) * tmp10
            c(i,11)  =  c(i,11) + a(i,k) * tmp11
            c(i,12)  =  c(i,12) + a(i,k) * tmp12
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_11(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,11),c(n1,11)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         tmp6  =  b(k, 6)
         tmp7  =  b(k, 7)
         tmp8  =  b(k, 8)
         tmp9  =  b(k, 9)
         tmp10 =  b(k,10)
         tmp11 =  b(k,11)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
            c(i, 6)  =  c(i, 6) + a(i,k) * tmp6
            c(i, 7)  =  c(i, 7) + a(i,k) * tmp7
            c(i, 8)  =  c(i, 8) + a(i,k) * tmp8
            c(i, 9)  =  c(i, 9) + a(i,k) * tmp9
            c(i,10)  =  c(i,10) + a(i,k) * tmp10
            c(i,11)  =  c(i,11) + a(i,k) * tmp11
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_10(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,10),c(n1,10)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         tmp6  =  b(k, 6)
         tmp7  =  b(k, 7)
         tmp8  =  b(k, 8)
         tmp9  =  b(k, 9)
         tmp10 =  b(k,10)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
            c(i, 6)  =  c(i, 6) + a(i,k) * tmp6
            c(i, 7)  =  c(i, 7) + a(i,k) * tmp7
            c(i, 8)  =  c(i, 8) + a(i,k) * tmp8
            c(i, 9)  =  c(i, 9) + a(i,k) * tmp9
            c(i,10)  =  c(i,10) + a(i,k) * tmp10
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_9(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,9),c(n1,9)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         tmp6  =  b(k, 6)
         tmp7  =  b(k, 7)
         tmp8  =  b(k, 8)
         tmp9  =  b(k, 9)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
            c(i, 6)  =  c(i, 6) + a(i,k) * tmp6
            c(i, 7)  =  c(i, 7) + a(i,k) * tmp7
            c(i, 8)  =  c(i, 8) + a(i,k) * tmp8
            c(i, 9)  =  c(i, 9) + a(i,k) * tmp9
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_8(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,8),c(n1,8)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         tmp6  =  b(k, 6)
         tmp7  =  b(k, 7)
         tmp8  =  b(k, 8)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
            c(i, 6)  =  c(i, 6) + a(i,k) * tmp6
            c(i, 7)  =  c(i, 7) + a(i,k) * tmp7
            c(i, 8)  =  c(i, 8) + a(i,k) * tmp8
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_7(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,7),c(n1,7)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         tmp6  =  b(k, 6)
         tmp7  =  b(k, 7)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
            c(i, 6)  =  c(i, 6) + a(i,k) * tmp6
            c(i, 7)  =  c(i, 7) + a(i,k) * tmp7
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_6(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,6),c(n1,6)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         tmp6  =  b(k, 6)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
            c(i, 6)  =  c(i, 6) + a(i,k) * tmp6
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_5(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,5),c(n1,5)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         tmp5  =  b(k, 5)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
            c(i, 5)  =  c(i, 5) + a(i,k) * tmp5
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_4(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,4),c(n1,4)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         tmp4  =  b(k, 4)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
            c(i, 4)  =  c(i, 4) + a(i,k) * tmp4
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_3(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,3),c(n1,3)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         tmp3  =  b(k, 3)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
            c(i, 3)  =  c(i, 3) + a(i,k) * tmp3
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_2(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,2),c(n1,2)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         tmp2  =  b(k, 2)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
            c(i, 2)  =  c(i, 2) + a(i,k) * tmp2
         enddo
c
      enddo
c
      return
      end
      subroutine mxmur3_1(a,n1,b,n2,c,n3)
      real a(n1,n2),b(n2,1),c(n1,1)
c
      do k=1,n2
         tmp1  =  b(k, 1)
         do i=1,n1
            c(i, 1)  =  c(i, 1) + a(i,k) * tmp1
         enddo
      enddo
c
      return
      end
C----------------------------------------------------------------------
      subroutine mxmd(a,n1,b,n2,c,n3)
C
C     Matrix-vector product routine. 
C     NOTE: Use assembly coded routine if available.
C
C---------------------------------------------------------------------
      REAL A(N1,N2),B(N2,N3),C(N1,N3)
      REAL ONE,ZERO,EPS
C
C
C
      one=1.0
      zero=0.0
      call dgemm( 'N','N',n1,n3,n2,ONE,A,N1,B,N2,ZERO,C,N1)
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb(a,n1,b,n2,c,n3)
C-----------------------------------------------------------------------
C
C     Matrix-vector product routine. 
C     NOTE: Use assembly coded routine if available.
C
C----------------------------------------------------------------------
      REAL A(N1,N2),B(N2,N3),C(N1,N3)
C
      integer wdsize
      save    wdsize
      data    wdsize/0/
c
c     First call: determine word size for dgemm/sgemm discrimination, below.
c
      if (wdsize.eq.0) then
         one = 1.0
         eps = 1.e-12
         wdsize = 8
         if (one+eps.eq.1.0) wdsize = 4
      endif
c
      if (n2.le.8) then
         if (n2.eq.1) then
            call mxmfb_1(a,n1,b,n2,c,n3)
         elseif (n2.eq.2) then
            call mxmfb_2(a,n1,b,n2,c,n3)
         elseif (n2.eq.3) then
            call mxmfb_3(a,n1,b,n2,c,n3)
         elseif (n2.eq.4) then
            call mxmfb_4(a,n1,b,n2,c,n3)
         elseif (n2.eq.5) then
            call mxmfb_5(a,n1,b,n2,c,n3)
         elseif (n2.eq.6) then
            call mxmfb_6(a,n1,b,n2,c,n3)
         elseif (n2.eq.7) then
            call mxmfb_7(a,n1,b,n2,c,n3)
         else
            call mxmfb_8(a,n1,b,n2,c,n3)
         endif
      elseif (n2.le.16) then
         if (n2.eq.9) then
            call mxmfb_9(a,n1,b,n2,c,n3)
         elseif (n2.eq.10) then
            call mxmfb_10(a,n1,b,n2,c,n3)
         elseif (n2.eq.11) then
            call mxmfb_11(a,n1,b,n2,c,n3)
         elseif (n2.eq.12) then
            call mxmfb_12(a,n1,b,n2,c,n3)
         elseif (n2.eq.13) then
            call mxmfb_13(a,n1,b,n2,c,n3)
         elseif (n2.eq.14) then
            call mxmfb_14(a,n1,b,n2,c,n3)
         elseif (n2.eq.15) then
            call mxmfb_15(a,n1,b,n2,c,n3)
         else
            call mxmfb_16(a,n1,b,n2,c,n3)
         endif
      elseif (n2.le.24) then
         if (n2.eq.17) then
            call mxmfb_17(a,n1,b,n2,c,n3)
         elseif (n2.eq.18) then
            call mxmfb_18(a,n1,b,n2,c,n3)
         elseif (n2.eq.19) then
            call mxmfb_19(a,n1,b,n2,c,n3)
         elseif (n2.eq.20) then
            call mxmfb_20(a,n1,b,n2,c,n3)
         elseif (n2.eq.21) then
            call mxmfb_21(a,n1,b,n2,c,n3)
         elseif (n2.eq.22) then
            call mxmfb_22(a,n1,b,n2,c,n3)
         elseif (n2.eq.23) then
            call mxmfb_23(a,n1,b,n2,c,n3)
         elseif (n2.eq.24) then
            call mxmfb_24(a,n1,b,n2,c,n3)
         endif
      else
c
         one=1.0
         zero=0.0
         if (wdsize.eq.4) then
            call sgemm( 'N','N',n1,n3,n2,ONE,A,N1,B,N2,ZERO,C,N1)
         else
            call dgemm( 'N','N',n1,n3,n2,ONE,A,N1,B,N2,ZERO,C,N1)
         endif
 
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_1(a,n1,b,n2,c,n3)
c
      real a(n1,1),b(1,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_2(a,n1,b,n2,c,n3)
c
      real a(n1,2),b(2,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_3(a,n1,b,n2,c,n3)
c
      real a(n1,3),b(3,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_4(a,n1,b,n2,c,n3)
c
      real a(n1,4),b(4,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_5(a,n1,b,n2,c,n3)
c
      real a(n1,5),b(5,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_6(a,n1,b,n2,c,n3)
c
      real a(n1,6),b(6,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_7(a,n1,b,n2,c,n3)
c
      real a(n1,7),b(7,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_8(a,n1,b,n2,c,n3)
c
      real a(n1,8),b(8,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_9(a,n1,b,n2,c,n3)
c
      real a(n1,9),b(9,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_10(a,n1,b,n2,c,n3)
c
      real a(n1,10),b(10,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_11(a,n1,b,n2,c,n3)
c
      real a(n1,11),b(11,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_12(a,n1,b,n2,c,n3)
c
      real a(n1,12),b(12,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_13(a,n1,b,n2,c,n3)
c
      real a(n1,13),b(13,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_14(a,n1,b,n2,c,n3)
c
      real a(n1,14),b(14,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_15(a,n1,b,n2,c,n3)
c
      real a(n1,15),b(15,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_16(a,n1,b,n2,c,n3)
c
      real a(n1,16),b(16,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_17(a,n1,b,n2,c,n3)
c
      real a(n1,17),b(17,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_18(a,n1,b,n2,c,n3)
c
      real a(n1,18),b(18,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_19(a,n1,b,n2,c,n3)
c
      real a(n1,19),b(19,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_20(a,n1,b,n2,c,n3)
c
      real a(n1,20),b(20,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_21(a,n1,b,n2,c,n3)
c
      real a(n1,21),b(21,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_22(a,n1,b,n2,c,n3)
c
      real a(n1,22),b(22,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
     $             + a(i,22)*b(22,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_23(a,n1,b,n2,c,n3)
c
      real a(n1,23),b(23,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
     $             + a(i,22)*b(22,j)
     $             + a(i,23)*b(23,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmfb_24(a,n1,b,n2,c,n3)
c
      real a(n1,24),b(24,n3),c(n1,n3)
c
      do j=1,n3
         do i=1,n1
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
     $             + a(i,22)*b(22,j)
     $             + a(i,23)*b(23,j)
     $             + a(i,24)*b(24,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3(a,n1,b,n2,c,n3)
C-----------------------------------------------------------------------
C
C     Matrix-vector product routine. 
C     NOTE: Use assembly coded routine if available.
C
C----------------------------------------------------------------------
      REAL A(N1,N2),B(N2,N3),C(N1,N3)
C
      integer wdsize
      save    wdsize
      data    wdsize/0/
c
c     First call: determine word size for dgemm/sgemm discrimination, below.
c
      if (wdsize.eq.0) then
         one = 1.0
         eps = 1.e-12
         wdsize = 8
         if (one+eps.eq.1.0) wdsize = 4
      endif
c
      if (n2.le.8) then
         if (n2.eq.1) then
            call mxmf3_1(a,n1,b,n2,c,n3)
         elseif (n2.eq.2) then
            call mxmf3_2(a,n1,b,n2,c,n3)
         elseif (n2.eq.3) then
            call mxmf3_3(a,n1,b,n2,c,n3)
         elseif (n2.eq.4) then
            call mxmf3_4(a,n1,b,n2,c,n3)
         elseif (n2.eq.5) then
            call mxmf3_5(a,n1,b,n2,c,n3)
         elseif (n2.eq.6) then
            call mxmf3_6(a,n1,b,n2,c,n3)
         elseif (n2.eq.7) then
            call mxmf3_7(a,n1,b,n2,c,n3)
         else
            call mxmf3_8(a,n1,b,n2,c,n3)
         endif
      elseif (n2.le.16) then
         if (n2.eq.9) then
            call mxmf3_9(a,n1,b,n2,c,n3)
         elseif (n2.eq.10) then
            call mxmf3_10(a,n1,b,n2,c,n3)
         elseif (n2.eq.11) then
            call mxmf3_11(a,n1,b,n2,c,n3)
         elseif (n2.eq.12) then
            call mxmf3_12(a,n1,b,n2,c,n3)
         elseif (n2.eq.13) then
            call mxmf3_13(a,n1,b,n2,c,n3)
         elseif (n2.eq.14) then
            call mxmf3_14(a,n1,b,n2,c,n3)
         elseif (n2.eq.15) then
            call mxmf3_15(a,n1,b,n2,c,n3)
         else
            call mxmf3_16(a,n1,b,n2,c,n3)
         endif
      elseif (n2.le.24) then
         if (n2.eq.17) then
            call mxmf3_17(a,n1,b,n2,c,n3)
         elseif (n2.eq.18) then
            call mxmf3_18(a,n1,b,n2,c,n3)
         elseif (n2.eq.19) then
            call mxmf3_19(a,n1,b,n2,c,n3)
         elseif (n2.eq.20) then
            call mxmf3_20(a,n1,b,n2,c,n3)
         elseif (n2.eq.21) then
            call mxmf3_21(a,n1,b,n2,c,n3)
         elseif (n2.eq.22) then
            call mxmf3_22(a,n1,b,n2,c,n3)
         elseif (n2.eq.23) then
            call mxmf3_23(a,n1,b,n2,c,n3)
         elseif (n2.eq.24) then
            call mxmf3_24(a,n1,b,n2,c,n3)
         endif
      else
c
         one=1.0
         zero=0.0
         if (wdsize.eq.4) then
            call sgemm( 'N','N',n1,n3,n2,ONE,A,N1,B,N2,ZERO,C,N1)
         else
            call dgemm( 'N','N',n1,n3,n2,ONE,A,N1,B,N2,ZERO,C,N1)
         endif
c
c        N0=N1*N3
c        DO 10 I=1,N0
c           C(I,1)=0.
c  10    CONTINUE
c        DO 100 J=1,N3
c        DO 100 K=1,N2
c        BB=B(K,J)
c        DO 100 I=1,N1
c           C(I,J)=C(I,J)+A(I,K)*BB
c 100    CONTINUE
 
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_1(a,n1,b,n2,c,n3)
c
      real a(n1,1),b(1,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_2(a,n1,b,n2,c,n3)
c
      real a(n1,2),b(2,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_3(a,n1,b,n2,c,n3)
c
      real a(n1,3),b(3,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_4(a,n1,b,n2,c,n3)
c
      real a(n1,4),b(4,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_5(a,n1,b,n2,c,n3)
c
      real a(n1,5),b(5,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_6(a,n1,b,n2,c,n3)
c
      real a(n1,6),b(6,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_7(a,n1,b,n2,c,n3)
c
      real a(n1,7),b(7,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_8(a,n1,b,n2,c,n3)
c
      real a(n1,8),b(8,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_9(a,n1,b,n2,c,n3)
c
      real a(n1,9),b(9,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_10(a,n1,b,n2,c,n3)
c
      real a(n1,10),b(10,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_11(a,n1,b,n2,c,n3)
c
      real a(n1,11),b(11,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_12(a,n1,b,n2,c,n3)
c
      real a(n1,12),b(12,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_13(a,n1,b,n2,c,n3)
c
      real a(n1,13),b(13,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_14(a,n1,b,n2,c,n3)
c
      real a(n1,14),b(14,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_15(a,n1,b,n2,c,n3)
c
      real a(n1,15),b(15,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_16(a,n1,b,n2,c,n3)
c
      real a(n1,16),b(16,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_17(a,n1,b,n2,c,n3)
c
      real a(n1,17),b(17,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_18(a,n1,b,n2,c,n3)
c
      real a(n1,18),b(18,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_19(a,n1,b,n2,c,n3)
c
      real a(n1,19),b(19,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_20(a,n1,b,n2,c,n3)
c
      real a(n1,20),b(20,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_21(a,n1,b,n2,c,n3)
c
      real a(n1,21),b(21,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_22(a,n1,b,n2,c,n3)
c
      real a(n1,22),b(22,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
     $             + a(i,22)*b(22,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_23(a,n1,b,n2,c,n3)
c
      real a(n1,23),b(23,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
     $             + a(i,22)*b(22,j)
     $             + a(i,23)*b(23,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxmf3_24(a,n1,b,n2,c,n3)
c
      real a(n1,24),b(24,n3),c(n1,n3)
c
      do i=1,n1
         do j=1,n3
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
     $             + a(i,3)*b(3,j)
     $             + a(i,4)*b(4,j)
     $             + a(i,5)*b(5,j)
     $             + a(i,6)*b(6,j)
     $             + a(i,7)*b(7,j)
     $             + a(i,8)*b(8,j)
     $             + a(i,9)*b(9,j)
     $             + a(i,10)*b(10,j)
     $             + a(i,11)*b(11,j)
     $             + a(i,12)*b(12,j)
     $             + a(i,13)*b(13,j)
     $             + a(i,14)*b(14,j)
     $             + a(i,15)*b(15,j)
     $             + a(i,16)*b(16,j)
     $             + a(i,17)*b(17,j)
     $             + a(i,18)*b(18,j)
     $             + a(i,19)*b(19,j)
     $             + a(i,20)*b(20,j)
     $             + a(i,21)*b(21,j)
     $             + a(i,22)*b(22,j)
     $             + a(i,23)*b(23,j)
     $             + a(i,24)*b(24,j)
         enddo
      enddo
      return
      end
c-----------------------------------------------------------------------
      subroutine mxm44(a,n1,b,n2,c,n3)
C-----------------------------------------------------------------------
C
C     NOTE -- this code has been set up with the "mxmf3" routine
c             referenced in memtime.f.   On most machines, the f2
c             and f3 versions give the same performance (f2 is the
c             nekton standard).  On the t3e, f3 is noticeably faster.
c             pff  10/5/98
C
C
C     Matrix-vector product routine. 
C     NOTE: Use assembly coded routine if available.
C
C----------------------------------------------------------------------
      REAL A(N1,N2),B(N2,N3),C(N1,N3)
c
      if (n2.eq.1) then
         call mxm44_2_t(a,n1,b,2,c,n3)
      elseif (n2.eq.2) then
         call mxm44_2_t(a,n1,b,n2,c,n3)
      else
         call mxm44_0_t(a,n1,b,n2,c,n3)
      endif
c
      return
      end
c
c-----------------------------------------------------------------------
      subroutine mxm44_0_t(a, m, b, k, c, n)
*      subroutine matmul44(m, n, k, a, lda, b, ldb, c, ldc)
*      real*8 a(lda,k), b(ldb,n), c(ldc,n)
      real a(m,k), b(k,n), c(m,n)
      real s11, s12, s13, s14, s21, s22, s23, s24
      real s31, s32, s33, s34, s41, s42, s43, s44
c
c matrix multiply with a 4x4 pencil 
c

      mresid = iand(m,3) 
      nresid = iand(n,3) 
      m1 = m - mresid + 1
      n1 = n - nresid + 1

      do i=1,m-mresid,4
        do j=1,n-nresid,4
          s11 = 0.0d0
          s21 = 0.0d0
          s31 = 0.0d0
          s41 = 0.0d0
          s12 = 0.0d0
          s22 = 0.0d0
          s32 = 0.0d0
          s42 = 0.0d0
          s13 = 0.0d0
          s23 = 0.0d0
          s33 = 0.0d0
          s43 = 0.0d0
          s14 = 0.0d0
          s24 = 0.0d0
          s34 = 0.0d0
          s44 = 0.0d0
          do l=1,k
            s11 = s11 + a(i,l)*b(l,j)
            s12 = s12 + a(i,l)*b(l,j+1)
            s13 = s13 + a(i,l)*b(l,j+2)
            s14 = s14 + a(i,l)*b(l,j+3)

            s21 = s21 + a(i+1,l)*b(l,j)
            s22 = s22 + a(i+1,l)*b(l,j+1)
            s23 = s23 + a(i+1,l)*b(l,j+2)
            s24 = s24 + a(i+1,l)*b(l,j+3)

            s31 = s31 + a(i+2,l)*b(l,j)
            s32 = s32 + a(i+2,l)*b(l,j+1)
            s33 = s33 + a(i+2,l)*b(l,j+2)
            s34 = s34 + a(i+2,l)*b(l,j+3)

            s41 = s41 + a(i+3,l)*b(l,j)
            s42 = s42 + a(i+3,l)*b(l,j+1)
            s43 = s43 + a(i+3,l)*b(l,j+2)
            s44 = s44 + a(i+3,l)*b(l,j+3)
          enddo
          c(i,j)     = s11 
          c(i,j+1)   = s12 
          c(i,j+2)   = s13
          c(i,j+3)   = s14

          c(i+1,j)   = s21 
          c(i+2,j)   = s31 
          c(i+3,j)   = s41 

          c(i+1,j+1) = s22
          c(i+2,j+1) = s32
          c(i+3,j+1) = s42

          c(i+1,j+2) = s23
          c(i+2,j+2) = s33
          c(i+3,j+2) = s43

          c(i+1,j+3) = s24
          c(i+2,j+3) = s34
          c(i+3,j+3) = s44
        enddo
* Residual when n is not multiple of 4
        if (nresid .ne. 0) then
          if (nresid .eq. 1) then
            s11 = 0.0d0
            s21 = 0.0d0
            s31 = 0.0d0
            s41 = 0.0d0
            do l=1,k
              s11 = s11 + a(i,l)*b(l,n)
              s21 = s21 + a(i+1,l)*b(l,n)
              s31 = s31 + a(i+2,l)*b(l,n)
              s41 = s41 + a(i+3,l)*b(l,n)
            enddo
            c(i,n)     = s11 
            c(i+1,n)   = s21 
            c(i+2,n)   = s31 
            c(i+3,n)   = s41 
          elseif (nresid .eq. 2) then
            s11 = 0.0d0
            s21 = 0.0d0
            s31 = 0.0d0
            s41 = 0.0d0
            s12 = 0.0d0
            s22 = 0.0d0
            s32 = 0.0d0
            s42 = 0.0d0
            do l=1,k
              s11 = s11 + a(i,l)*b(l,j)
              s12 = s12 + a(i,l)*b(l,j+1)

              s21 = s21 + a(i+1,l)*b(l,j)
              s22 = s22 + a(i+1,l)*b(l,j+1)

              s31 = s31 + a(i+2,l)*b(l,j)
              s32 = s32 + a(i+2,l)*b(l,j+1)

              s41 = s41 + a(i+3,l)*b(l,j)
              s42 = s42 + a(i+3,l)*b(l,j+1)
            enddo
            c(i,j)     = s11 
            c(i,j+1)   = s12

            c(i+1,j)   = s21 
            c(i+2,j)   = s31 
            c(i+3,j)   = s41 

            c(i+1,j+1) = s22
            c(i+2,j+1) = s32
            c(i+3,j+1) = s42
          else
            s11 = 0.0d0
            s21 = 0.0d0
            s31 = 0.0d0
            s41 = 0.0d0
            s12 = 0.0d0
            s22 = 0.0d0
            s32 = 0.0d0
            s42 = 0.0d0
            s13 = 0.0d0
            s23 = 0.0d0
            s33 = 0.0d0
            s43 = 0.0d0
            do l=1,k
              s11 = s11 + a(i,l)*b(l,j)
              s12 = s12 + a(i,l)*b(l,j+1)
              s13 = s13 + a(i,l)*b(l,j+2)

              s21 = s21 + a(i+1,l)*b(l,j)
              s22 = s22 + a(i+1,l)*b(l,j+1)
              s23 = s23 + a(i+1,l)*b(l,j+2)

              s31 = s31 + a(i+2,l)*b(l,j)
              s32 = s32 + a(i+2,l)*b(l,j+1)
              s33 = s33 + a(i+2,l)*b(l,j+2)

              s41 = s41 + a(i+3,l)*b(l,j)
              s42 = s42 + a(i+3,l)*b(l,j+1)
              s43 = s43 + a(i+3,l)*b(l,j+2)
            enddo
            c(i,j)     = s11 
            c(i+1,j)   = s21 
            c(i+2,j)   = s31 
            c(i+3,j)   = s41 
            c(i,j+1)   = s12 
            c(i+1,j+1) = s22
            c(i+2,j+1) = s32
            c(i+3,j+1) = s42
            c(i,j+2)   = s13
            c(i+1,j+2) = s23
            c(i+2,j+2) = s33
            c(i+3,j+2) = s43
          endif
        endif
      enddo

* Residual when m is not multiple of 4
      if (mresid .eq. 0) then
        return
      elseif (mresid .eq. 1) then
        do j=1,n-nresid,4
          s11 = 0.0d0
          s12 = 0.0d0
          s13 = 0.0d0
          s14 = 0.0d0
          do l=1,k
            s11 = s11 + a(m,l)*b(l,j)
            s12 = s12 + a(m,l)*b(l,j+1)
            s13 = s13 + a(m,l)*b(l,j+2)
            s14 = s14 + a(m,l)*b(l,j+3)
          enddo
          c(m,j)     = s11 
          c(m,j+1)   = s12 
          c(m,j+2)   = s13
          c(m,j+3)   = s14
        enddo
* mresid is 1, check nresid
        if (nresid .eq. 0) then
          return
        elseif (nresid .eq. 1) then
          s11 = 0.0d0
          do l=1,k
            s11 = s11 + a(m,l)*b(l,n)
          enddo
          c(m,n) = s11
          return
        elseif (nresid .eq. 2) then
          s11 = 0.0d0
          s12 = 0.0d0
          do l=1,k
            s11 = s11 + a(m,l)*b(l,n-1)
            s12 = s12 + a(m,l)*b(l,n)
          enddo
          c(m,n-1) = s11
          c(m,n) = s12
          return
        else
          s11 = 0.0d0
          s12 = 0.0d0
          s13 = 0.0d0
          do l=1,k
            s11 = s11 + a(m,l)*b(l,n-2)
            s12 = s12 + a(m,l)*b(l,n-1)
            s13 = s13 + a(m,l)*b(l,n)
          enddo
          c(m,n-2) = s11
          c(m,n-1) = s12
          c(m,n) = s13
          return
        endif          
      elseif (mresid .eq. 2) then
        do j=1,n-nresid,4
          s11 = 0.0d0
          s12 = 0.0d0
          s13 = 0.0d0
          s14 = 0.0d0
          s21 = 0.0d0
          s22 = 0.0d0
          s23 = 0.0d0
          s24 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-1,l)*b(l,j)
            s12 = s12 + a(m-1,l)*b(l,j+1)
            s13 = s13 + a(m-1,l)*b(l,j+2)
            s14 = s14 + a(m-1,l)*b(l,j+3)

            s21 = s21 + a(m,l)*b(l,j)
            s22 = s22 + a(m,l)*b(l,j+1)
            s23 = s23 + a(m,l)*b(l,j+2)
            s24 = s24 + a(m,l)*b(l,j+3)
          enddo
          c(m-1,j)   = s11 
          c(m-1,j+1) = s12 
          c(m-1,j+2) = s13
          c(m-1,j+3) = s14
          c(m,j)     = s21
          c(m,j+1)   = s22 
          c(m,j+2)   = s23
          c(m,j+3)   = s24
        enddo
* mresid is 2, check nresid
        if (nresid .eq. 0) then
          return
        elseif (nresid .eq. 1) then
          s11 = 0.0d0
          s21 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-1,l)*b(l,n)
            s21 = s21 + a(m,l)*b(l,n)
          enddo
          c(m-1,n) = s11
          c(m,n) = s21
          return
        elseif (nresid .eq. 2) then
          s11 = 0.0d0
          s21 = 0.0d0
          s12 = 0.0d0
          s22 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-1,l)*b(l,n-1)
            s12 = s12 + a(m-1,l)*b(l,n)
            s21 = s21 + a(m,l)*b(l,n-1)
            s22 = s22 + a(m,l)*b(l,n)
          enddo
          c(m-1,n-1) = s11
          c(m-1,n)   = s12
          c(m,n-1)   = s21
          c(m,n)     = s22
          return
        else
          s11 = 0.0d0
          s21 = 0.0d0
          s12 = 0.0d0
          s22 = 0.0d0
          s13 = 0.0d0
          s23 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-1,l)*b(l,n-2)
            s12 = s12 + a(m-1,l)*b(l,n-1)
            s13 = s13 + a(m-1,l)*b(l,n)
            s21 = s21 + a(m,l)*b(l,n-2)
            s22 = s22 + a(m,l)*b(l,n-1)
            s23 = s23 + a(m,l)*b(l,n)
          enddo
          c(m-1,n-2) = s11
          c(m-1,n-1) = s12
          c(m-1,n)   = s13
          c(m,n-2)   = s21
          c(m,n-1)   = s22
          c(m,n)     = s23
          return
        endif
      else
* mresid is 3
        do j=1,n-nresid,4
          s11 = 0.0d0
          s21 = 0.0d0
          s31 = 0.0d0

          s12 = 0.0d0
          s22 = 0.0d0
          s32 = 0.0d0

          s13 = 0.0d0
          s23 = 0.0d0
          s33 = 0.0d0

          s14 = 0.0d0
          s24 = 0.0d0
          s34 = 0.0d0

          do l=1,k
            s11 = s11 + a(m-2,l)*b(l,j)
            s12 = s12 + a(m-2,l)*b(l,j+1)
            s13 = s13 + a(m-2,l)*b(l,j+2)
            s14 = s14 + a(m-2,l)*b(l,j+3)

            s21 = s21 + a(m-1,l)*b(l,j)
            s22 = s22 + a(m-1,l)*b(l,j+1)
            s23 = s23 + a(m-1,l)*b(l,j+2)
            s24 = s24 + a(m-1,l)*b(l,j+3)

            s31 = s31 + a(m,l)*b(l,j)
            s32 = s32 + a(m,l)*b(l,j+1)
            s33 = s33 + a(m,l)*b(l,j+2)
            s34 = s34 + a(m,l)*b(l,j+3)
          enddo
          c(m-2,j)   = s11 
          c(m-2,j+1) = s12 
          c(m-2,j+2) = s13
          c(m-2,j+3) = s14

          c(m-1,j)   = s21 
          c(m-1,j+1) = s22
          c(m-1,j+2) = s23
          c(m-1,j+3) = s24

          c(m,j)     = s31 
          c(m,j+1)   = s32
          c(m,j+2)   = s33
          c(m,j+3)   = s34
        enddo
* mresid is 3, check nresid
        if (nresid .eq. 0) then
          return
        elseif (nresid .eq. 1) then
          s11 = 0.0d0
          s21 = 0.0d0
          s31 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-2,l)*b(l,n)
            s21 = s21 + a(m-1,l)*b(l,n)
            s31 = s31 + a(m,l)*b(l,n)
          enddo
          c(m-2,n) = s11
          c(m-1,n) = s21
          c(m,n) = s31
          return
        elseif (nresid .eq. 2) then
          s11 = 0.0d0
          s21 = 0.0d0
          s31 = 0.0d0
          s12 = 0.0d0
          s22 = 0.0d0
          s32 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-2,l)*b(l,n-1)
            s12 = s12 + a(m-2,l)*b(l,n)
            s21 = s21 + a(m-1,l)*b(l,n-1)
            s22 = s22 + a(m-1,l)*b(l,n)
            s31 = s31 + a(m,l)*b(l,n-1)
            s32 = s32 + a(m,l)*b(l,n)
          enddo
          c(m-2,n-1) = s11
          c(m-2,n)   = s12
          c(m-1,n-1) = s21
          c(m-1,n)   = s22
          c(m,n-1)   = s31
          c(m,n)     = s32
          return
        else
          s11 = 0.0d0
          s21 = 0.0d0
          s31 = 0.0d0
          s12 = 0.0d0
          s22 = 0.0d0
          s32 = 0.0d0
          s13 = 0.0d0
          s23 = 0.0d0
          s33 = 0.0d0
          do l=1,k
            s11 = s11 + a(m-2,l)*b(l,n-2)
            s12 = s12 + a(m-2,l)*b(l,n-1)
            s13 = s13 + a(m-2,l)*b(l,n)
            s21 = s21 + a(m-1,l)*b(l,n-2)
            s22 = s22 + a(m-1,l)*b(l,n-1)
            s23 = s23 + a(m-1,l)*b(l,n)
            s31 = s31 + a(m,l)*b(l,n-2)
            s32 = s32 + a(m,l)*b(l,n-1)
            s33 = s33 + a(m,l)*b(l,n)
          enddo
          c(m-2,n-2) = s11
          c(m-2,n-1) = s12
          c(m-2,n)   = s13
          c(m-1,n-2) = s21
          c(m-1,n-1) = s22
          c(m-1,n)   = s23
          c(m,n-2)   = s31
          c(m,n-1)   = s32
          c(m,n)     = s33
          return
        endif
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine mxm44_2_t(a, m, b, k, c, n)
      real a(m,2), b(2,n), c(m,n)

      nresid = iand(n,3) 
      n1 = n - nresid + 1

      do j=1,n-nresid,4
         do i=1,m
            c(i,j) = a(i,1)*b(1,j)
     $             + a(i,2)*b(2,j)
            c(i,j+1) = a(i,1)*b(1,j+1)
     $             + a(i,2)*b(2,j+1)
            c(i,j+2) = a(i,1)*b(1,j+2)
     $             + a(i,2)*b(2,j+2)
            c(i,j+3) = a(i,1)*b(1,j+3)
     $             + a(i,2)*b(2,j+3)
         enddo
      enddo
      if (nresid .eq. 0) then
        return
      elseif (nresid .eq. 1) then
         do i=1,m
            c(i,n) = a(i,1)*b(1,n)
     $             + a(i,2)*b(2,n)
         enddo
      elseif (nresid .eq. 2) then
         do i=1,m
            c(i,n-1) = a(i,1)*b(1,n-1)
     $             + a(i,2)*b(2,n-1)
            c(i,n) = a(i,1)*b(1,n)
     $             + a(i,2)*b(2,n)
         enddo
      else
         do i=1,m
            c(i,n-2) = a(i,1)*b(1,n-2)
     $             + a(i,2)*b(2,n-2)
            c(i,n-1) = a(i,1)*b(1,n-1)
     $             + a(i,2)*b(2,n-1)
            c(i,n) = a(i,1)*b(1,n)
     $             + a(i,2)*b(2,n)
         enddo
      endif

      return
      end
