c Matthew Anderson
c 3 Jan 2013
c hpx version of comm_mpi.f
c-----------------------------------------------------------------------
      subroutine iniproc(hpx_bti,hpx_mype,hpx_numberpe,
     &     nid_,np_,nekcomm,nekgroup,nekreal,
c  SIZE
     &     nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid,
c  PARALLEL
     &     node,pid,np,nullpid,node0,
     &     nelgf,lglel,gllel,gllnid,
     &     nelgv,nelgt,nvtot,ifgprnt,
     &     wdsize,isize,lsize,csize,ifdblas,
     &     cr_h,gsh,gsh_fld,xxth)
      use, intrinsic :: iso_c_binding, only : c_ptr

      integer hpx_mype,hpx_numberpe
      TYPE(C_PTR), INTENT(IN), VALUE :: hpx_bti

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

      integer nelx,nely,nelz,nelt,nelg,
     &     nx1,ny1,nz1,ndim,nfield,nid

c PARALLEL
      parameter(nelgt_max = 178956970)
      integer node,pid,np,nullpid,node0,
     &     nelgf(0:ldimt1),lglel(lelt),
     &     gllel(lelg),gllnid(lelg),
     &     nelgv,nelgt
      integer*8      nvtot
      logical ifgprnt
      integer wdsize,isize,lsize,csize
      logical ifdblas
      integer cr_h, gsh, gsh_fld(0:ldimt1), xxth(ldimt1)

c nekmpi
      integer nid_,np_,nekcomm,nekgroup,nekreal

      logical flag

      ! Unnecessary
      !call mpi_initialized(mpi_is_initialized, ierr) !  Initialize MPI
      !if ( mpi_is_initialized .eq. 0 ) then
      !   call mpi_init (ierr)
      !endif 

      ! create communicator
      !call init_nek_comm(intracomm)
      nid_ = hpx_mype
      np_ = hpx_numberpe
      
      np  = np_
      nid = nid_

c N/A
      !if(nid.eq.0) call printHeader

c N/A
      ! check upper tag size limit
      !call mpi_attr_get(MPI_COMM_WORLD,MPI_TAG_UB,nval,flag,ierr)
      !if (nval.lt.(10000+max(lp,lelg))) then
      !   if(nid.eq.0) write(6,*) 'ABORT: MPI_TAG_UB too small!'
      !   call exitt
      !endif

c N/A
c      IF (NP.GT.LP) THEN
c         WRITE(6,*)
c     $   'ERROR: Code compiled for a max of',LP,' processors.'
c         WRITE(6,*)
c     $   'Recompile with LP =',NP,' or run with fewer processors.'
c         WRITE(6,*)
c     $   'Aborting in routine INIPROC.'
c         call exitt
c      endif
 
! set word size for REAL
      wdsize=4
      eps=1.0e-12
      oneeps = 1.0+eps
      if (oneeps.ne.1.0) then
         wdsize=8
      else
         if(nid.eq.0)
     &     write(6,*) 'ABORT: single precision mode not supported!'
        ! call exitt
      endif

      nekreal = 2
      if (wdsize.eq.8) nekreal = 3

      ifdblas = .false.
      if (wdsize.eq.8) ifdblas = .true.

      ! set word size for INTEGER
      ! HARDCODED since there is no secure way to detect an int overflow
      isize = 4

      ! set word size for LOGICAL
      lsize = 4

      ! set word size for CHARACTER
      csize = 1
c
c N/A
      !PID = 0
      !NULLPID=0
      !NODE0=0
      !NODE= NID+1

      if (nid.eq.0) then
         write(6,*) 'Number of processors:',np
         WRITE(6,*) 'REAL    wdsize      :',WDSIZE
         WRITE(6,*) 'INTEGER wdsize      :',ISIZE
      endif

      call fcrystal_setup(cr_h,nekcomm,np,nid)

      return
      end subroutine iniproc

      subroutine exitt
        print*, 'EXIT'
      end subroutine exitt
c-----------------------------------------------------------------------
      subroutine gop(hpx_bti, x, w, op, n,
c nekmpi
     &     nid,np,nekcomm,nekgroup,nekreal)
      use, intrinsic :: iso_c_binding, only : c_ptr
      TYPE(C_PTR), INTENT(IN), VALUE :: hpx_bti
c
c     Global vector commutative operation
c
c      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
c
      real x(n), w(n)
      character*3 op
ccccccccccccccccccccccccccccccccccc
c nekmpi
      integer nid,np,nekcomm,nekgroup,nekreal
ccccccccccccccccccccccccccccccccccc
cc
      if (op.eq.'+  ') then
c         call mpi_allreduce(x,w,n,nekreal,mpi_sum ,nekcomm,ierr)
         call double_mpi_allreduce_cmm(hpx_bti,x,w,n,ierr)
      elseif (op.EQ.'M  ') then
c         call mpi_allreduce (x,w,n,nekreal,mpi_max ,nekcomm,ierr)
      elseif (op.EQ.'m  ') then
c         call mpi_allreduce (x,w,n,nekreal,mpi_min ,nekcomm,ierr)
      elseif (op.EQ.'*  ') then
c         call mpi_allreduce (x,w,n,nekreal,mpi_prod,nekcomm,ierr)
      else
         write(6,*) nid,' OP ',op,' not supported.  ABORT in GOP.'
         call exitt
      endif
c
c      call copy(x,w,n)

      return
      end

