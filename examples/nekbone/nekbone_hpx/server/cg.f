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
      subroutine ax(hpx_bti,w,u,gxyz,ur,us,ut,wk,n, 
! Matrix-vector product: w=A*u
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

      parameter (lxyz=lx1*ly1*lz1)
      real w(lxyz,lelt),u(lxyz,lelt),gxyz(2*ldim,lxyz,lelt)
      real ur(lxyz),us(lxyz),ut(lxyz),wk(lxyz)

      end subroutine ax
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

      miter = niter
      do iter=1,miter
         call solveM(z,r,n)    ! preconditioner here

         rtz2=rtz1
         rtz1=glsc3(hpx_bti,r,c,z,n,   ! parallel weighted inner product r^T C z
c nekmpi
     &     nid_,np_,nekcomm,nekgroup,nekreal)

         beta = rtz1/rtz2
         if (iter.eq.1) beta=0.0
         call add2s1(p,z,beta,n)

         call ax(hpx_bti,w,p,g,ur,us,ut,wk,n,
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

         pap=glsc3(hpx_bti,w,c,p,n,
c nekmpi
     &     nid_,np_,nekcomm,nekgroup,nekreal)

         alpha=rtz1/pap
         alphm=-alpha
         call add2s2(x,p,alpha,n)
         call add2s2(r,w,alphm,n)

         rtr = glsc3(hpx_bti,r,c,r,n,
c nekmpi
     &     nid_,np_,nekcomm,nekgroup,nekreal)
         if (iter.eq.1) rlim2 = rtr*eps**2
         if (iter.eq.1) rtr0  = rtr
         rnorm = sqrt(rtr)
         if (nid.eq.0) write(6,6) iter,rnorm,alpha,beta,pap
    6    format('cg:',i4,1p4e12.4)
         if (rtr.le.rlim2) goto 1001

      enddo

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
      end subroutine mask

c-----------------------------------------------------------------------
      subroutine solveM(z,r,n)
      real z(n),r(n)

      call copy(z,r,n)

      return
      end subroutine solveM
c-----------------------------------------------------------------------
      subroutine ax(hpx_bti,w,u,gxyz,ur,us,ut,wk,n, 
! Matrix-vector product: w=A*u
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

      parameter (lxyz=lx1*ly1*lz1)
      real w(lxyz,lelt),u(lxyz,lelt),gxyz(2*ldim,lxyz,lelt)
      real ur(lxyz),us(lxyz),ut(lxyz),wk(lxyz)

      integer e

      do e=1,nelt
        call ax_e( w(1,e),u(1,e),gxyz(1,1,e)    ! w   = A  u
     $                             ,ur,us,ut,wk, !  L     L  L
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
     &     zgm1,wxm1,wym1,wzm1,w3m1)
      enddo                                      ! 
      call fgs_op(gsh,w,1,1,0)  ! Gather-scatter operation  ! w   = QQ  w
                                                           !            L
      call mask(w,             ! Zero out Dirichlet conditions
c SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid) 

      return
      end subroutine ax
c-----------------------------------------------------------------------
      subroutine ax_e(w,u,g,ur,us,ut,wk, ! Local matrix-vector product
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
     &     zgm1,wxm1,wym1,wzm1,w3m1)
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

      parameter (lxyz=lx1*ly1*lz1)
      real w(lxyz),u(lxyz),g(2*ldim,lxyz)

      real ur(lxyz),us(lxyz),ut(lxyz),wk(lxyz)

      nxyz = nx1*ny1*nz1
      n    = nx1-1

      call local_grad3(ur,us,ut,u,n,dxm1,dxtm1)

      do i=1,nxyz
         wr = g(1,i)*ur(i) + g(2,i)*us(i) + g(3,i)*ut(i)
         ws = g(2,i)*ur(i) + g(4,i)*us(i) + g(5,i)*ut(i)
         wt = g(3,i)*ur(i) + g(5,i)*us(i) + g(6,i)*ut(i)
         ur(i) = wr
         us(i) = ws
         ut(i) = wt
      enddo

      call local_grad3_t(w,ur,us,ut,n,dxm1,dxtm1,wk)

      return
      end subroutine ax_e
c-------------------------------------------------------------------------
      subroutine local_grad3(ur,us,ut,u,n,D,Dt)
c     Output: ur,us,ut         Input:u,n,D,Dt
      real ur(0:n,0:n,0:n),us(0:n,0:n,0:n),ut(0:n,0:n,0:n)
      real u (0:n,0:n,0:n)
      real D (0:n,0:n),Dt(0:n,0:n)
      integer e

      m1 = n+1
      m2 = m1*m1

      call mxm(D ,m1,u,m1,ur,m2)
      do k=0,n
         call mxm(u(0,0,k),m1,Dt,m1,us(0,0,k),m1)
      enddo
      call mxm(u,m2,Dt,m1,ut,m1)

      return
      end subroutine local_grad3
c-------------------------------------------------------------------------
      subroutine local_grad3_t(u,ur,us,ut,N,D,Dt,w)
c     Output: ur,us,ut         Input:u,N,D,Dt
      real u (0:N,0:N,0:N)
      real ur(0:N,0:N,0:N),us(0:N,0:N,0:N),ut(0:N,0:N,0:N)
      real D (0:N,0:N),Dt(0:N,0:N)
      real w (0:N,0:N,0:N)
      integer e

      m1 = N+1
      m2 = m1*m1
      m3 = m1*m1*m1

      call mxm(Dt,m1,ur,m1,u,m2)

      do k=0,N
         call mxm(us(0,0,k),m1,D ,m1,w(0,0,k),m1)
      enddo
      call add2(u,w,m3)

      call mxm(ut,m2,D ,m1,w,m1)
      call add2(u,w,m3)

      return
      end subroutine local_grad3_t
