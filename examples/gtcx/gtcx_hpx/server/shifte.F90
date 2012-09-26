subroutine shifte(hpx4_bti,&
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

  integer,dimension(:),allocatable :: t_gids
  integer,dimension(:),allocatable :: p_gids
  TYPE(C_PTR), INTENT(IN), VALUE :: hpx4_bti

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
       m0,msend,mrecv,&
       idest,isource,isendtag,irecvtag,nzelectron,iright(memax),ileft(memax),&
       isendcount,irecvcount,ierror,iteration,lasth
  real(wp),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(wp) zetaright,zetaleft,pi_inv
  character(len=8) cdum

  nzelectron=12
  pi_inv=1.0/pi
  m0=1
  iteration=0
  
100 iteration=iteration+1
  if(iteration>numberpe)then
     write(stdout,*)'endless particle sorting loop at PE=',mype
     !call MPI_ABORT(MPI_COMM_WORLD,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= me)then
     do m=1,me
        kzelectron(m)=0
     enddo
! NOTE: The vector kzelectron is also used in chargee and pushe to temporarily
!    store the closest zeta grid point to the electron "m". S.Ethier 10/7/04

!  This section of code replaces the section above when the compilation does
!  NOT include the OpenMP support option. Temporary arrays msleft and msright
!  are not needed as well as the extra code for thread work distribution.

     do m=m0,me
        zetaright=min(2.0*pi,zelectron(3,m))-zetamax
        zetaleft=zelectron(3,m)-zetamin

        if( zetaright*zetaleft > 0 )then
           zetaright=zetaright*0.5*pi_inv
           zetaright=zetaright-real(floor(zetaright))
           msend=msend+1
           kzelectron(msend)=m

           if( zetaright < 0.5 )then
! # of particle to move right
              msendright(1)=msendright(1)+1
              iright(msendright(1))=m
! keep track of tracer
              if( m == ntracer )then
                 msendright(2)=msendright(1)
                 ntracer=0
              endif

! # of particle to move left
           else
              msendleft(1)=msendleft(1)+1
              ileft(msendleft(1))=m
              if( m == ntracer )then
                 msendleft(2)=msendleft(1)
                 ntracer=0
              endif
           endif
        endif
     enddo

  endif

  if (msend /= (msendleft(1)+msendright(1))) then
     write(*,*)'mype=',mype,'  msend NOT equal to msendleft+msendright'
     msend=msendleft(1)+msendright(1)
  endif

  if(iteration>1)then
! total # of particles to be shifted
     mrecv=0

     !call MPI_ALLREDUCE(msend,mrecv,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierror)
     call int_comm_allreduce_cmm(hpx4_bti,msend,mrecv,1)

! no particle to be shifted, return
     if ( mrecv == 0 ) then
!        write(0,*)istep,irk,mype,me,m0,iteration
        return
     endif
  endif

! an extra space to prevent zero size when msendright(1)=msendleft(1)=0
  allocate(sendright(12,max(1,msendright(1))),sendleft(12,max(1,msendleft(1))))

! pack particle to move right
     do m=1,msendright(1)
        sendright(1:6,m)=zelectron(1:6,iright(m))
        sendright(7:12,m)=zelectron0(1:6,iright(m))
     enddo

! pack particle to move left
     do m=1,msendleft(1)    
        sendleft(1:6,m)=zelectron(1:6,ileft(m))
        sendleft(7:12,m)=zelectron0(1:6,ileft(m))
     enddo

     mtop=me
! # of particles remain on local PE
     me=me-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzelectron(i)
        if (m > me) exit  !Break out of the DO loop if m > me
        do while(mtop == kzelectron(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( mtop == ntracer )ntracer=m
        zelectron(1:6,m)=zelectron(1:6,mtop)
        zelectron0(1:6,m)=zelectron0(1:6,mtop)
        mtop=mtop-1
        if (mtop == me) exit  !Break out of the DO loop if mtop=me
     enddo
!  endif

! send # of particle to move right
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call MPI_SENDRECV(msendright,2,MPI_INTEGER,idest,isendtag,&
  !     mrecvleft,2,MPI_INTEGER,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndrecv_toroidal_cmm(hpx4_bti,msendright,2, &
                       mrecvleft,2,idest)
  
  allocate(recvleft(12,max(1,mrecvleft(1))))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=max(1,msendright(1))*nzelectron
  irecvcount=max(1,mrecvleft(1))*nzelectron
  !call MPI_SENDRECV(sendright,isendcount,mpi_Rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_Rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndrecv_toroidal_cmm(hpx4_bti,sendright,isendcount, &
                              recvleft,irecvcount,idest)
  
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
  
  allocate(recvright(12,max(1,mrecvright(1))))

! send particle to left and receive from right
  recvright=0.0
  isendcount=max(1,msendleft(1))*nzelectron
  irecvcount=max(1,mrecvright(1))*nzelectron
  !call MPI_SENDRECV(sendleft,isendcount,mpi_Rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_Rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndrecv_toroidal_cmm(hpx4_bti,sendleft,isendcount, &
                          recvright,irecvcount,idest)
  
! tracer particle
  if( mrecvleft(2) > 0 )then
     ntracer=me+mrecvleft(2)
  elseif( mrecvright(2) > 0 )then
     ntracer=me+mrecvleft(1)+mrecvright(2)
  endif

! need extra particle array
  if(me+mrecvleft(1)+mrecvright(1) > memax)then
     write(*,*)"need bigger particle array",mype,memax,me+mrecvleft(1)+mrecvright(1)
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
!     write(111)zelectron(1:6,1:me)
!     write(111)zelectron0(1:6,1:me)
     
! make bigger array
!     deallocate(zelectron,zelectron0)
!     memax=2*(me+mrecvleft(1)+mrecvright(1))-memax
!     allocate(zelectron(6,memax),zelectron0(6,memax))
     
! read back particle information
!     rewind(111)
!     read(111)zelectron(1:6,1:me)
!     read(111)zelectron0(1:6,1:me)
!     close(111)
  endif

! unpack particle, particle moved from left
  do m=1,mrecvleft(1)
     zelectron(1:6,m+me)=recvleft(1:6,m)
     zelectron0(1:6,m+me)=recvleft(7:12,m)
  enddo

! particle moved from right
  do m=1,mrecvright(1)
     zelectron(1:6,m+me+mrecvleft(1))=recvright(1:6,m)
     zelectron0(1:6,m+me+mrecvleft(1))=recvright(7:12,m)
  enddo

  me=me+mrecvleft(1)+mrecvright(1)
  
  deallocate(sendleft,sendright,recvleft,recvright)
  m0=me-mrecvright(1)-mrecvleft(1)+1
  goto 100
  
end subroutine shifte


