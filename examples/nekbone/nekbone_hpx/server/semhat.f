c-----------------------------------------------------------------------
      subroutine semhat(a,b,c,d,z,w,n)
c
c     Generate matrices for single element, 1D operators:
c
c        a    = Laplacian
c        b    = diagonal mass matrix    (GLL weights)
c        c    = convection operator b*d
c        d    = derivative matrix
c        z    = GLL points

      real a(0:n,0:n),b(0:n),c(0:n,0:n),d(0:n,0:n),z(0:n)
      real w(0:2*n)
 
      np = n+1
 
      call zwgll (z,b,np)
 
      do i=0,n
         call fd_weights_full(z(i),z,n,1,w)
         do j=0,n
            d(i,j) = w(j+np)                   !  Derivative matrix
         enddo
      enddo

      call rzero(a,np*np)
      do j=0,n
      do i=0,n
         do k=0,n
            a(i,j) = a(i,j) + d(k,i)*b(k)*d(k,j)
         enddo
         c(i,j) = b(i)*d(i,j)
      enddo
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine fd_weights_full(xx,x,n,m,c)
c
c     This routine evaluates the derivative based on all points
c     in the stencils.  It is more memory efficient than "fd_weights"
c
c     This set of routines comes from the appendix of 
c     A Practical Guide to Pseudospectral Methods, B. Fornberg
c     Cambridge Univ. Press, 1996.   (pff)
c
c     Input parameters:
c       xx -- point at wich the approximations are to be accurate
c       x  -- array of x-ordinates:   x(0:n)
c       n  -- polynomial degree of interpolant (# of points := n+1)
c       m  -- highest order of derivative to be approxxmated at xi
c
c     Output:
c       c  -- set of coefficients c(0:n,0:m).
c             c(j,k) is to be applied at x(j) when
c             the kth derivative is approxxmated by a 
c             stencil extending over x(0),x(1),...x(n).
c
c
      real x(0:n),c(0:n,0:m)
 
      c1       = 1.
      c4       = x(0) - xx
 
      do k=0,m
      do j=0,n
         c(j,k) = 0.
      enddo
      enddo
      c(0,0) = 1.
 
      do i=1,n
         mn = min(i,m)
         c2 = 1.
         c5 = c4
         c4 = x(i)-xx
         do j=0,i-1
            c3 = x(i)-x(j)
            c2 = c2*c3
            do k=mn,1,-1
               c(i,k) = c1*(k*c(i-1,k-1)-c5*c(i-1,k))/c2
            enddo
            c(i,0) = -c1*c5*c(i-1,0)/c2
            do k=mn,1,-1
               c(j,k) = (c4*c(j,k)-k*c(j,k-1))/c3
            enddo
            c(j,0) = c4*c(j,0)/c3
         enddo
         c1 = c2
      enddo
      return
      end
c-----------------------------------------------------------------------
