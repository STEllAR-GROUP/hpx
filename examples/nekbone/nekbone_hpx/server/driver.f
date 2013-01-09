c Matthew Anderson
c interface to nekbone driver
c notice that nekbone was written with implicits
c variables starting with i -- n are integers, all others are reals
c-----------------------------------------------------------------------
      subroutine nekproxy(hpx_bti,hpx_mype,hpx_numberpe) ! {{{
      use, intrinsic :: iso_c_binding, only : c_ptr

      interface ! {{{
        subroutine init_mesh(
     &     hpx_bti,
c   SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid,
c   DXYZ
     &     dxm1,dxtm1,
c   INPUT
     &     xc,yc,zc,bc,ccurve,cbc,
c   MASS
     &     bm1,binvm1,volvm1,
c   PARALLEL
     &     node,pid,np,nullpid,node0
     &     nelgf,lglel,gllel,gllnid,nelgv,nelgt, 
     &     nvtot,ifgprnt,wdsize,isize,lsize,csize,
     &     ifdblas,cr_h,gsh,gsh_fld,xxth,
c   WZ
     &     zgm1,wxm1,wym1,wzm1,w3m1)

        use, intrinsic :: iso_c_binding, only : c_ptr
        TYPE(C_PTR), INTENT(IN), VALUE :: hpx_bti
ccccccccccccccccccccccccccccccccccc
c   SIZE
c       These numbers come from example1 
        parameter (ldim=3)
        parameter (lx1=10,ly1=lx1,lz1=lx1)
        parameter (lxd=10,lyd=lxd,lzd=1)

        parameter (lp = 2)
        parameter (lelt=1)

        parameter (lelg=lelt*lp)
        parameter (lelx=lelg,lely=1,lelz=1)

        parameter (ldimt=1,ldimt1=ldimt+1)

c        common /dimn/ nelx,nely,nelz,nelt,nelg
c       $            , nx1,ny1,nz1,ndim,nfield,nid
        integer nelx,nely,nelz,nelt,nelg
     &            , nx1,ny1,nz1,ndim,nfield,nid

ccccccccccccccccccccccccccccccccccc
c   DXYZ
        real dxm1(lx1,lx1),  dxtm1(lx1,lx1)
ccccccccccccccccccccccccccccccccccc
c   INPUT
        real xc(8,lelt),yc(8,lelt),zc(8,lelt),
     &       bc(5,6,lelt,0:ldimt1)
        character*1     ccurve(12,lelt)
        character*3     cbc(6,lelt,0:ldimt1)
ccccccccccccccccccccccccccccccccccc
c   MASS
        real bm1   (lx1,ly1,lz1,lelt)
     &      ,binvm1(lx1,ly1,lz1,lelt)
     &      ,volvm1
ccccccccccccccccccccccccccccccccccc
c   PARALLEL
        integer node,pid,np,nullpid,node0
c       Maximum number of elements (limited to 2**31/12, at least
c        for now)
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
C       crystal-router, gather-scatter, and xxt handles (xxt=csr grid
C       solve)
C
        integer cr_h, gsh, gsh_fld(0:ldimt1), xxth(ldimt1)
ccccccccccccccccccccccccccccccccccc
c   WZ
c       Gauss-Labotto and Gauss points
        real zgm1(lx1,3)

c       Weights
        real wxm1(lx1), wym1(ly1), wzm1(lz1), w3m1(lx1,ly1,lz1)      
ccccccccccccccccccccccccccccccccccc
        end subroutine init_mesh
        subroutine proxy_setupds( 
     &     hpx_bti,gs_handle,
c SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid,
c INPUT
     &     xc,yc,zc,bc,ccurve,cbc,
c PARALLEL
     &     node,pid,np,nullpid,node0
     &     nelgf,lglel,gllel,gllnid,nelgv,nelgt, 
     &     nvtot,ifgprnt,wdsize,isize,lsize,csize,
     &     ifdblas,cr_h,gsh,gsh_fld,xxth,
c nekmpi
     &     mid,mp,nekcomm,nekgroup,nekreal)
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
c INPUT
      real xc(8,lelt),yc(8,lelt),zc(8,lelt),
     &       bc(5,6,lelt,0:ldimt1)
      character*1     ccurve(12,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
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
c nekmpi
      integer mid,mp,nekcomm,nekgroup,nekreal

      integer gs_handle

        end subroutine proxy_setupds 
      end interface ! }}}

      integer hpx_mype,hpx_numberpe
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

c     local variables
      parameter (lxyz = lx1*ly1*lz1)
      parameter (lt=lxyz*lelt)

      real ah(lx1*lx1),bh(lx1),ch(lx1*lx1),dh(lx1*lx1)
     $    ,zpts(2*lx1),wght(2*lx1)

      real x(lt),f(lt),r(lt),w(lt),p(lt),z(lt),c(lt)
      real g(6,lt)      

      call iniproc(hpx_bti,hpx_mype,hpx_numberpe,
     &   nid_,np_,nekcomm,nekgroup,nekreal,
c  SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid,
c  PARALLEL
     &     node,pid,np,nullpid,node0,
     &     nelgf,lglel,gllel,gllnid,
     &     nelgv,nelgt,nvtot,ifgprnt,
     &     wdsize,isize,lsize,csize,ifdblas,
     &     cr_h,gsh,gsh_fld,xxth)

      call init_dim(
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid,
     &     xc,yc,zc,bc,ccurve,cbc)

      call init_mesh( 
     &     hpx_bti,
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

      call proxy_setupds( 
     &     hpx_bti,gsh,
c SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid,
c INPUT
     &     xc,yc,zc,bc,ccurve,cbc,
c PARALLEL
     &     node,pid,np,nullpid,node0
     &     nelgf,lglel,gllel,gllnid,nelgv,nelgt, 
     &     nvtot,ifgprnt,wdsize,isize,lsize,csize,
     &     ifdblas,cr_h,gsh,gsh_fld,xxth,
c nekmpi
     &     nid_,np_,nekcomm,nekgroup,nekreal)

        return
      end subroutine nekproxy ! }}}

      subroutine init_dim( ! {{{
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid,
     &     xc,yc,zc,bc,ccurve,cbc)
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

      real xc(8,lelt),yc(8,lelt),zc(8,lelt),
     &       bc(5,6,lelt,0:ldimt1)
      character*1     ccurve(12,lelt)
      character*3     cbc(6,lelt,0:ldimt1)

      nx1=lx1
      ny1=ly1
      nz1=lz1

c      nx2=lx2
c      ny2=ly2
c      nz2=lz2
c
c      nx3=lx3
c      ny3=ly3
c      nz3=lz3

      nxd=lxd
      nyd=lyd
      nzd=lzd


      ndim=ldim

      end subroutine init_dim ! }}}

      subroutine init_mesh( ! {{{
     &     hpx_bti,
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
c local variables
      logical ifbrick
      integer e,eg,offs

      !open .rea
      if(nid.eq.0) then
         open(unit=9,file='data.rea',status='old')
         read(9,*,err=100) ifbrick
         read(9,*,err=100) nelt
         close(9)
      endif
      call broadcast_int_cmm(hpx_bti,ifbrick)
      call broadcast_int_cmm(hpx_bti,nelt)

      if(.not.ifbrick) then   ! A 1-D array of elements of length P*lelt
  10     continue
         nelx = nelt*np
         nely = 1
         nelz = 1

         do e=1,nelt
            eg = e + nid*nelt
            lglel(e) = eg
         enddo
      else    ! A 3-D block of elements 
        call cubic(npx,npy,npz,np)  !xyz distribution of total proc
        call cubic(mx,my,mz,nelt)   !xyz distribution of elements per proc 

        if(mx.eq.nelt) goto 10

        nelx = mx*npx
        nely = my*npy
        nelz = mz*npz

        e = 1
        offs = nid*mx + (my*mx)*(nid/npx)
        do k = 0,mz-1
        do j = 0,my-1
        do i = 0,mx-1
           eg = offs+i+(j*nelx)+(k*nelx*nely)+1
           lglel(e) = eg
           e        = e+1
        enddo
        enddo
        enddo
      endif

      return

  100 continue
      write(6,*) "ERROR READING data.rea....ABORT"

      return
      end subroutine init_mesh ! }}}

      subroutine cubic(mx,my,mz,np) ! {{{

        rp = np**(1./3.)
        mz = rp*(1.01)

        do iz=mz,1,-1
           myx = np/iz
           nrem = np-myx*iz
           if (nrem.eq.0) goto 10
        enddo
   10   mz = iz
        rq = myx**(1./2.)
        my = rq*(1.01)
        do iy=my,1,-1
           mx = myx/iy
           nrem = myx-mx*iy
           if (nrem.eq.0) goto 20
        enddo
   20   my = iy

        mx = np/(mz*my)

      return
      end subroutine cubic ! }}}

      subroutine proxy_setupds( ! {{{
     &     hpx_bti,gs_handle,
c SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid,
c INPUT
     &     xc,yc,zc,bc,ccurve,cbc,
c PARALLEL
     &     node,pid,np,nullpid,node0
     &     nelgf,lglel,gllel,gllnid,nelgv,nelgt, 
     &     nvtot,ifgprnt,wdsize,isize,lsize,csize,
     &     ifdblas,cr_h,gsh,gsh_fld,xxth,
c nekmpi
     &     mid,mp,nekcomm,nekgroup,nekreal)
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
c INPUT
      real xc(8,lelt),yc(8,lelt),zc(8,lelt),
     &       bc(5,6,lelt,0:ldimt1)
      character*1     ccurve(12,lelt)
      character*3     cbc(6,lelt,0:ldimt1)
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
c nekmpi
      integer mid,mp,nekcomm,nekgroup,nekreal

c other variables
      integer gs_handle, dof
      integer*8 glo_num(lx1*ly1*lz1*lelt)
    
c N/A
      !t0 = dnekclock()
  
      call set_vert_box(glo_num, ! Set global-to-local map
c SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid,
c PARALLEL
     &     node,pid,np,nullpid,node0
     &     nelgf,lglel,gllel,gllnid,nelgv,nelgt, 
     &     nvtot,ifgprnt,wdsize,isize,lsize,csize,
     &     ifdblas,cr_h,gsh,gsh_fld,xxth)

      ntot      = nx1*ny1*nz1*nelt
      call fgs_setup(gs_handle,glo_num,ntot,nekcomm,mp,mid) ! Initialize gather-scatter

      return
      end subroutine proxy_setupds ! }}}

      subroutine set_vert_box( ! {{{
     &     glo_num,
c SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid,
c PARALLEL
     &     node,pid,np,nullpid,node0
     &     nelgf,lglel,gllel,gllnid,nelgv,nelgt, 
     &     nvtot,ifgprnt,wdsize,isize,lsize,csize,
     &     ifdblas,cr_h,gsh,gsh_fld,xxth)

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

c other variables
      integer*8 glo_num(1),ii,kg,jg,ig ! The latter 3 for proper promotion

      integer e,ex,ey,ez,eg

      nn = nx1-1  ! nn := polynomial order

      do e=1,nelt
        eg = lglel(e)
        call get_exyz(ex,ey,ez,eg,nelx,nely,nelz)
        do k=0,nn
        do j=0,nn
        do i=0,nn
           kg = nn*(ez-1) + k
           jg = nn*(ey-1) + j
           ig = nn*(ex-1) + i
           ii = 1 + ig + jg*(nn*nelx+1) + kg*(nn*nelx+1)*(nn*nely+1)
           ll = 1 + i + nx1*j + nx1*ny1*k + nx1*ny1*nz1*(e-1)
           glo_num(ll) = ii
        enddo
        enddo
        enddo
      enddo

      return
      end subroutine set_vert_box ! }}}

      subroutine get_exyz(ex,ey,ez,eg,nelx,nely,nelz) ! {{{
      integer ex,ey,ez,eg

      nelxy = nelx*nely

      ez = 1 +  (eg-1)/nelxy
      ey = mod1 (eg,nelxy)
      ey = 1 +  (ey-1)/nelx
      ex = mod1 (eg,nelx)

      return
      end ! }}}

