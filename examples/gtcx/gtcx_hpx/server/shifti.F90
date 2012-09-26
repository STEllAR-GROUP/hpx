subroutine shifti(hpx4_bti,&
             t_gids, p_gids,&
! global parameters
                       ihistory,snapout,maxmpsi,&
          mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile,&
           nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron,&
                                       mtheta, &
                                       deltat, &
           do_collision, &
! particle array
                                      kzion,kzelectron,jtelectron0,jtelectron1,&
                                        jtion0,jtion1,&
                                       wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1, &
                                        wpion,wtion0,wtion1,&
                                         zion,zion0,zelectron,&
        zelectron0,zelectron1, &
! particle decomp
              ntoroidal,npartdom,&
              partd_comm,nproc_partd,myrank_partd,&
              toroidal_comm,nproc_toroidal,myrank_toroidal,&
              left_pe,right_pe,&
              toroidal_domain_location,particle_domain_location)

  !use global_parameters
  !use particle_array
  !use particle_decomp
  use, intrinsic :: iso_c_binding, only : c_ptr
  use precision
  implicit none

  TYPE(C_PTR), INTENT(IN), VALUE :: hpx4_bti
  integer,dimension(:),allocatable :: t_gids
  integer,dimension(:),allocatable :: p_gids

!  global parameters
  integer :: ihistory,snapout,maxmpsi
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

! particle array
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,&
        zelectron0,zelectron1

! particle decomp
  integer  :: ntoroidal,npartdom
  integer  :: partd_comm,nproc_partd,myrank_partd
  integer  :: toroidal_comm,nproc_toroidal,myrank_toroidal
  integer  :: left_pe,right_pe
  integer  :: toroidal_domain_location,particle_domain_location
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
  real(wp),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(wp) zetaright,zetaleft,pi_inv
  character(len=8) cdum

  if(numberpe==1)return

  nzion=2*nparam   ! nzion=14 if track_particles=1, =12 otherwise
  pi_inv=1.0/pi
  m0=1
  iteration=0
  
100 iteration=iteration+1
  if(iteration>ntoroidal)then
     write(0,*)'endless particle sorting loop at PE=',mype
     !call MPI_ABORT(MPI_COMM_WORLD,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= mi)then
     do m=m0,mi
        kzi(m)=0
     enddo

!  This section of code replaces the section above when the compilation does
!  NOT include the OpenMP support option. Temporary arrays msleft and msright
!  are not needed as well as the extra code for thread work distribution.

     do m=m0,mi
        zetaright=min(2.0*pi,zion(3,m))-zetamax
        zetaleft=zion(3,m)-zetamin

        if( zetaright*zetaleft > 0 )then
           zetaright=zetaright*0.5*pi_inv
           zetaright=zetaright-real(floor(zetaright))
           msend=msend+1
           kzi(msend)=m

           if( zetaright < 0.5 )then
! # of particle to move right
              msendright(1)=msendright(1)+1
              iright(msendright(1))=m
! keep track of tracer
              if( nhybrid == 0 .and. m == ntracer )then
                 msendright(2)=msendright(1)
                 ntracer=0
              endif

! # of particle to move left
           else
              msendleft(1)=msendleft(1)+1
              ileft(msendleft(1))=m
              if( nhybrid == 0 .and. m == ntracer )then
                 msendleft(2)=msendleft(1)
                 ntracer=0
              endif
           endif
        endif
     enddo

  endif

  if(iteration>1)then
! total # of particles to be shifted
     mrecv=0
     msend=msendleft(1)+msendright(1)

     !call MPI_ALLREDUCE(msend,mrecv,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierror)
      call int_comm_allreduce_cmm(hpx4_bti,msend,mrecv,1)

! no particle to be shifted, return
     if ( mrecv == 0 ) then
        return
     endif
  endif

! an extra space to prevent zero size when msendright(1)=msendleft(1)=0
  allocate(sendright(nzion,max(msendright(1),1)),sendleft(nzion,max(msendleft(1),1)))

! pack particle to move right
     do m=1,msendright(1)
        sendright(1:nparam,m)=zion(1:nparam,iright(m))
        sendright(nparam+1:nzion,m)=zion0(1:nparam,iright(m))
     enddo

! pack particle to move left
     do m=1,msendleft(1)    
        sendleft(1:nparam,m)=zion(1:nparam,ileft(m))
        sendleft(nparam+1:nzion,m)=zion0(1:nparam,ileft(m))
     enddo

     mtop=mi
! # of particles remain on local PE
     mi=mi-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzi(i)
        if (m > mi) exit  !Break out of the DO loop if m > mi
        do while(mtop == kzi(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( nhybrid == 0 .and. mtop == ntracer )ntracer=m
        zion(1:nparam,m)=zion(1:nparam,mtop)
        zion0(1:nparam,m)=zion0(1:nparam,mtop)
        mtop=mtop-1
        if (mtop == mi) exit  !Break out of the DO loop
     enddo
!  endif

! send # of particle to move right to neighboring PEs of same particle
! domain.
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call MPI_SENDRECV(msendright,2,MPI_INTEGER,idest,isendtag,&
  !     mrecvleft,2,MPI_INTEGER,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndrecv_toroidal_cmm(hpx4_bti,msendright,2, &
                                mrecvleft,2,idest)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call MPI_SENDRECV(sendright,isendcount,mpi_Rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_Rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndrecv_toroidal_cmm(hpx4_bti,sendright,isendcount,&
                             mrecvleft,irecvcount,idest)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call MPI_SENDRECV(msendleft,2,MPI_INTEGER,idest,isendtag,&
  !     mrecvright,2,MPI_INTEGER,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndrecv_toroidal_cmm(hpx4_bti,msendleft,2, &
                                mrecvright,2,idest)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call MPI_SENDRECV(sendleft,isendcount,mpi_Rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_Rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndrecv_toroidal_cmm(hpx4_bti,sendleft,isendcount, &
                            recvright,irecvcount,idest)
  
! tracer particle
  if( nhybrid == 0 .and. mrecvleft(2) > 0 )then
     ntracer=mi+mrecvleft(2)
  elseif( nhybrid == 0 .and. mrecvright(2) > 0 )then
     ntracer=mi+mrecvleft(1)+mrecvright(2)
  endif

! need extra particle array
  if(mi+mrecvleft(1)+mrecvright(1) > mimax)then
     write(0,*)"need bigger particle array",mimax,mi+mrecvleft(1)+mrecvright(1)
     !call MPI_ABORT(MPI_COMM_WORLD,1,ierror)
! open disk file
!     if(mype < 10)then
!        write(cdum,'("TEMP.00",i1)')mype
!     elseif(mype < 100)then
!        write(cdum,'("TEMP.0",i2)')mype
!     else
!        write(cdum,'("TEMP.",i3)')mype
!     endif
!     open(111,file=cdum,status='replace',form='unformatted')
     
! record particle information to file
!     write(111)zion(1:nparam,1:mi)
!     write(111)zion0(1:nparam,1:mi)
     
! make bigger array
!     deallocate(zion,zion0)
!     mimax=2*(mi+mrecvleft(1)+mrecvright(1))-mimax
!     allocate(zion(nparam,mimax),zion0(nparam,mimax))
     
! read back particle information
!     rewind(111)
!     read(111)zion(1:nparam,1:mi)
!     read(111)zion0(1:nparam,1:mi)
!     close(111)
  endif

! unpack particle, particle moved from left
  do m=1,mrecvleft(1)
     zion(1:nparam,m+mi)=recvleft(1:nparam,m)
     zion0(1:nparam,m+mi)=recvleft(nparam+1:nzion,m)
  enddo

! particle moved from right
  do m=1,mrecvright(1)
     zion(1:nparam,m+mi+mrecvleft(1))=recvright(1:nparam,m)
     zion0(1:nparam,m+mi+mrecvleft(1))=recvright(nparam+1:nzion,m)
  enddo

  mi=mi+mrecvleft(1)+mrecvright(1)

  deallocate(sendleft,sendright,recvleft,recvright)
  m0=mi-mrecvright(1)-mrecvleft(1)+1
  goto 100
  
end subroutine shifti


