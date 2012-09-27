subroutine shifti(ptr)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters
  use particle_array
  use particle_decomp
  implicit none
  TYPE(C_PTR), INTENT(IN), VALUE :: ptr
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
       !istatus(MPI_STATUS_SIZE)
  real(kind=wp),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(kind=wp) zetaright,zetaleft,pi_inv
  character(len=8) cdum
#ifdef _OPENMP
  integer msleft(32,0:15),msright(32,0:15)
  integer nthreads,gnthreads,iam,delm,mbeg,mend,omp_get_num_threads,&
       omp_get_thread_num
#endif

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
!$omp parallel do private(m)
     do m=m0,mi
        kzi(m)=0
     enddo

#ifdef _OPENMP
! This section of code (down to #else) is included by the preprocessor when
! the compilation is performed with OpenMP support. We must then use a few
! temporary arrays and add some work distribution code.
  msleft=0
  msright=0

! First we start the parallel region with !$omp parallel

!$omp parallel private(nthreads,iam,delm,i,mbeg,mend,m,zetaright,zetaleft) &
!$omp& shared(gnthreads)
  nthreads=omp_get_num_threads()    !Get the number of threads ready to work
  iam=omp_get_thread_num()          !Get my thread number (position)
  delm=(mi-m0+1)/nthreads       !Calculate the number of steps per thread
  i=mod((mi-m0+1),nthreads)
!$omp single              !Put nthread in global variable for later use.
  gnthreads=nthreads      !nthread is the same for all threads so only one
!$omp end single nowait   !of them needs to copy the value in gnthreads

! We now distribute the work between the threads. The loop over the particles
! is distributed equally (as much as possible) between them.
  mbeg=m0+min(iam,i)*(delm+1)+max(0,(iam-i))*delm
  mend=mbeg+delm+(min((iam+1),i)/(iam+1))-1

! label particle to be moved 
     do m=mbeg,mend
        zetaright=min(2.0*pi,zion(3,m))-zetamax
        zetaleft=zion(3,m)-zetamin
        
        if( zetaright*zetaleft > 0 )then
           zetaright=zetaright*0.5*pi_inv
           zetaright=zetaright-real(floor(zetaright))
           msright(3,iam)=msright(3,iam)+1
           kzi(mbeg+msright(3,iam)-1)=m
           
           if( zetaright < 0.5 )then
! particle to move right               
              msright(1,iam)=msright(1,iam)+1
              iright(mbeg+msright(1,iam)-1)=m
! keep track of tracer
              if( nhybrid == 0 .and. m == ntracer )then
                 msright(2,iam)=msright(1,iam)
                 ntracer=0
              endif

! particle to move left
           else
              msleft(1,iam)=msleft(1,iam)+1
              ileft(mbeg+msleft(1,iam)-1)=m
              if( nhybrid == 0 .and. m == ntracer )then
                 msleft(2,iam)=msleft(1,iam)
                 ntracer=0
              endif
           endif
        endif
     enddo
! End of the OpenMP parallel region
!$omp end parallel

! Now that we are out of the parallel region we need to gather and rearrange
! the results of the multi-threaded calculation. We need to end up with the
! same arrays as for the sequential (single-threaded) calculation.
     do m=0,gnthreads-1
        delm=(mi-m0+1)/gnthreads
        i=mod((mi-m0+1),gnthreads)
        mbeg=m0+min(m,i)*(delm+1)+max(0,(m-i))*delm
        if( msleft(2,m) /= 0 )msendleft(2)=msendleft(1)+msleft(2,m)
        do i=1,msleft(1,m)
           ileft(msendleft(1)+i)=ileft(mbeg+i-1)
        enddo
        msendleft(1)=msendleft(1)+msleft(1,m)
        if( msright(2,m) /= 0 )msendright(2)=msendright(1)+msright(2,m)
        do i=1,msright(1,m)
           iright(msendright(1)+i)=iright(mbeg+i-1)
        enddo
        msendright(1)=msendright(1)+msright(1,m)
        do i=1,msright(3,m)
           kzi(msend+i)=kzi(mbeg+i-1)
        enddo
        msend=msend+msright(3,m)
     enddo

#else
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

#endif

  endif

  if(iteration>1)then
! total # of particles to be shifted
     mrecv=0
     msend=msendleft(1)+msendright(1)

     !call MPI_ALLREDUCE(msend,mrecv,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierror)
     call int_comm_allreduce_cmm(ptr,msend,mrecv,1)

! no particle to be shifted, return
     if ( mrecv == 0 ) then
        return
     endif
  endif

! an extra space to prevent zero size when msendright(1)=msendleft(1)=0
  allocate(sendright(nzion,max(msendright(1),1)),sendleft(nzion,max(msendleft(1),1)))

! pack particle to move right
!$omp parallel do private(m)
     do m=1,msendright(1)
        sendright(1:nparam,m)=zion(1:nparam,iright(m))
        sendright(nparam+1:nzion,m)=zion0(1:nparam,iright(m))
     enddo

! pack particle to move left
!$omp parallel do private(m)
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
  call int_sndright_toroidal_cmm(ptr,msendright,2)
  call int_rcvleft_toroidal_cmm(ptr,mrecvleft)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call MPI_SENDRECV(sendright,isendcount,mpi_Rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_Rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendright,isendcount)
  call rcvleft_toroidal_cmm(ptr,recvleft)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call MPI_SENDRECV(msendleft,2,MPI_INTEGER,idest,isendtag,&
  !     mrecvright,2,MPI_INTEGER,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndleft_toroidal_cmm(ptr,msendleft,2)
  call int_rcvright_toroidal_cmm(ptr,mrecvright)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call MPI_SENDRECV(sendleft,isendcount,mpi_Rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_Rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendleft,isendcount)
  call rcvright_toroidal_cmm(ptr,recvright)
  
! tracer particle
  if( nhybrid == 0 .and. mrecvleft(2) > 0 )then
     ntracer=mi+mrecvleft(2)
  elseif( nhybrid == 0 .and. mrecvright(2) > 0 )then
     ntracer=mi+mrecvleft(1)+mrecvright(2)
  endif

! need extra particle array
  if(mi+mrecvleft(1)+mrecvright(1) > mimax)then
     write(0,*)"need bigger particle array",mimax,mi+mrecvleft(1)+mrecvright(1)
    ! call MPI_ABORT(MPI_COMM_WORLD,1,ierror)
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
!$omp parallel do private(m)
  do m=1,mrecvleft(1)
     zion(1:nparam,m+mi)=recvleft(1:nparam,m)
     zion0(1:nparam,m+mi)=recvleft(nparam+1:nzion,m)
  enddo

! particle moved from right
!$omp parallel do private(m)
  do m=1,mrecvright(1)
     zion(1:nparam,m+mi+mrecvleft(1))=recvright(1:nparam,m)
     zion0(1:nparam,m+mi+mrecvleft(1))=recvright(nparam+1:nzion,m)
  enddo

  mi=mi+mrecvleft(1)+mrecvright(1)

  deallocate(sendleft,sendright,recvleft,recvright)
  m0=mi-mrecvright(1)-mrecvleft(1)+1
  goto 100
  
end subroutine shifti


