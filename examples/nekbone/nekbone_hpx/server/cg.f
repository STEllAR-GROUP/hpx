c-----------------------------------------------------------------------
      subroutine cg(hpx_bti,x,f,g,c,r,w,p,z,n,niter,
c SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid,
c DXYZ
     &     dxm1,dxtm1,
c INPUT
     &     xc,yc,zc,bc,ccurve,cbc,
c MASS
     &     bm1,binvm1,volvm1,
c PARALLEL
     &     node,pid,np,nullpid,node0
     &     nelgf,lglel,gllel,gllnid,nelgv,nelgt,
     &     nvtot,ifgprnt,wdsize,isize,lsize,csize,
     &     ifdblas,cr_h,gsh,gsh_fld,xxth,
c WZ
     &     zgm1,wxm1,wym1,wzm1,w3m1,
c nekmpi
     &     nid_,np_,nekcomm,nekgroup,nekreal)
      use, intrinsic :: iso_c_binding, only : c_ptr
      interface ! {{{
      function glsc3(hpx_bti,a,b,mult,n,
c nekmpi
     &     nid_,np_,nekcomm,nekgroup,nekreal)
        use, intrinsic :: iso_c_binding, only : c_ptr
        TYPE(C_PTR), INTENT(IN), VALUE :: hpx_bti
        real a(1),b(1),mult(1)
        integer nid_,np_,nekcomm,nekgroup,nekreal
      end function glsc3
      end interface ! }}}
      TYPE(C_PTR), INTENT(IN), VALUE :: hpx_bti
ccccccccccccccccccccccccccccccccccc
c SIZE
c     These numbers come from example1 
      parameter (ldim=3)
      parameter (lx1=10,ly1=lx1,lz1=lx1)
      parameter (lxd=10,lyd=lxd,lzd=1)

      parameter (lp = 2)
      parameter (lelt=1)

      parameter (lelg=lelt*lp)
      parameter (lelx=lelg,lely=1,lelz=1)

      parameter (ldimt=1,ldimt1=ldimt+1)

c      common /dimn/ nelx,nely,nelz,nelt,nelg
c     $            , nx1,ny1,nz1,ndim,nfield,nid
      integer nelx,nely,nelz,nelt,nelg
     &            , nx1,ny1,nz1,ndim,nfield,nid
ccccccccccccccccccccccccccccccccccc
c DXYZ
      real dxm1(lx1,lx1),  dxtm1(lx1,lx1)
ccccccccccccccccccccccccccccccccccc
c INPUT
      real xc(8,lelt),yc(8,lelt),zc(8,lelt),
     &       bc(5,6,lelt,0:ldimt1)
      character*1     ccurve(12,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
ccccccccccccccccccccccccccccccccccc
c MASS
      real bm1   (lx1,ly1,lz1,lelt)
     &      ,binvm1(lx1,ly1,lz1,lelt)
     &      ,volvm1
ccccccccccccccccccccccccccccccccccc
c PARALLEL
      integer node,pid,np,nullpid,node0
c     Maximum number of elements (limited to 2**31/12, at least
c      for now)
      parameter(nelgt_max = 178956970)
      integer nelgf(0:ldimt1)
     &              ,lglel(lelt)
     &              ,gllel(lelg)
     &              ,gllnid(lelg)
     &              ,nelgv,nelgt

      integer*8      nvtot

      logical ifgprnt
      integer wdsize,isize,lsize,csize
      logical ifdblas
C
C     crystal-router, gather-scatter, and xxt handles (xxt=csr grid
C     solve)
C
      integer cr_h, gsh, gsh_fld(0:ldimt1), xxth(ldimt1)
ccccccccccccccccccccccccccccccccccc
c WZ
c     Gauss-Labotto and Gauss points
      real zgm1(lx1,3)

c     Weights
      real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)
ccccccccccccccccccccccccccccccccccc
c nekmpi
      integer nid_,np_,nekcomm,nekgroup,nekreal
ccccccccccccccccccccccccccccccccccc

c     Solve Ax=f where A is SPD and is invoked by ax()
c
c     Output:  x - vector of length n
c
c     Input:   f - vector of length n
c     Input:   g - geometric factors for SEM operator
c     Input:   c - inverse of the counting matrix
c
c     Work arrays:   r,w,p,z  - vectors of length n
c
c     User-provided ax(w,z,n) returns  w := Az,  
c
c     User-provided solveM(z,r,n) ) returns  z := M^-1 r,  
c
      parameter (lxyz=lx1*ly1*lz1)
      real ur(lxyz),us(lxyz),ut(lxyz),wk(lxyz)

      real x(n),f(n),r(n),w(n),p(n),z(n),g(1),c(n)

      character*1 ans

      pap = 0.0

c     set machine tolerances
      one = 1.
      eps = 1.e-20
      if (one+eps .eq. one) eps = 1.e-14
      if (one+eps .eq. one) eps = 1.e-7

      rtz1=1.0

      call rzero(x,n)
      call copy (r,f,n)
      call mask (r,
c SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid)   ! Zero out Dirichlet conditions


      rnorm = sqrt(glsc3(hpx_bti,r,c,r,n,
c nekmpi
     &     nid_,np_,nekcomm,nekgroup,nekreal))
      iter = 0
      if (nid.eq.0)write(6,6) iter,rnorm
c
c      miter = niter
c      do iter=1,miter
c         call solveM(z,r,n)    ! preconditioner here
c
c         rtz2=rtz1
c         rtz1=glsc3(r,c,z,n)   ! parallel weighted inner product r^T C z
c
c         beta = rtz1/rtz2
c         if (iter.eq.1) beta=0.0
c         call add2s1(p,z,beta,n)
c
c         call ax(w,p,g,ur,us,ut,wk,n)
c         pap=glsc3(w,c,p,n)
c
c         alpha=rtz1/pap
c         alphm=-alpha
c         call add2s2(x,p,alpha,n)
c         call add2s2(r,w,alphm,n)
c
c         rtr = glsc3(r,c,r,n)
c         if (iter.eq.1) rlim2 = rtr*eps**2
c         if (iter.eq.1) rtr0  = rtr
c         rnorm = sqrt(rtr)
c         if (nid.eq.0) write(6,6) iter,rnorm,alpha,beta,pap
    6    format('cg:',i4,1p4e12.4)
c         if (rtr.le.rlim2) goto 1001
c
c      enddo

 1001 continue
c     write(6,6) iter,rnorm,rlim2,rtr


      return
      end

c-----------------------------------------------------------------------
      subroutine mask(w,
c SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid)   ! Zero out Dirichlet conditions

ccccccccccccccccccccccccccccccccccc
c SIZE
c     These numbers come from example1 
      parameter (ldim=3)
      parameter (lx1=10,ly1=lx1,lz1=lx1)
      parameter (lxd=10,lyd=lxd,lzd=1)

      parameter (lp = 2)
      parameter (lelt=1)

      parameter (lelg=lelt*lp)
      parameter (lelx=lelg,lely=1,lelz=1)

      parameter (ldimt=1,ldimt1=ldimt+1)

c      common /dimn/ nelx,nely,nelz,nelt,nelg
c     $            , nx1,ny1,nz1,ndim,nfield,nid
      integer nelx,nely,nelz,nelt,nelg
     &            , nx1,ny1,nz1,ndim,nfield,nid
ccccccccccccccccccccccccccccccccccc
      real w(1)

      if (nid.eq.0) w(1) = 0.  ! suitable for solvability

      return
      end

