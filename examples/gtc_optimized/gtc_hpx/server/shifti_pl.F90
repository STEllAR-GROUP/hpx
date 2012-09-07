subroutine shifti_0(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_0
  use particle_array_0
  use particle_decomp_0
  implicit none
  type(c_ptr), intent(in), value :: ptr
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
       !istatus(mpi_status_size)
  real(8),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(8) zetaright,zetaleft,pi_inv
  character(len=8) cdum
#ifdef _openmp
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
     write(0,*)'endless particle sorting loop at pe=',mype
     !call mpi_abort(mpi_comm_world,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= mi)then
!$omp parallel do private(m)
     do m=m0,mi
        kzi(m)=0
     enddo

#ifdef _openmp
! this section of code (down to #else) is included by the preprocessor when
! the compilation is performed with openmp support. we must then use a few
! temporary arrays and add some work distribution code.
  msleft=0
  msright=0

! first we start the parallel region with !$omp parallel

!$omp parallel private(nthreads,iam,delm,i,mbeg,mend,m,zetaright,zetaleft) &
!$omp& shared(gnthreads)
  nthreads=omp_get_num_threads()    !get the number of threads ready to work
  iam=omp_get_thread_num()          !get my thread number (position)
  delm=(mi-m0+1)/nthreads       !calculate the number of steps per thread
  i=mod((mi-m0+1),nthreads)
!$omp single              !put nthread in global variable for later use.
  gnthreads=nthreads      !nthread is the same for all threads so only one
!$omp end single nowait   !of them needs to copy the value in gnthreads

! we now distribute the work between the threads. the loop over the particles
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
! end of the openmp parallel region
!$omp end parallel

! now that we are out of the parallel region we need to gather and rearrange
! the results of the multi-threaded calculation. we need to end up with the
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
!  this section of code replaces the section above when the compilation does
!  not include the openmp support option. temporary arrays msleft and msright
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

     !call mpi_allreduce(msend,mrecv,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
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
! # of particles remain on local pe
     mi=mi-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzi(i)
        if (m > mi) exit  !break out of the do loop if m > mi
        do while(mtop == kzi(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( nhybrid == 0 .and. mtop == ntracer )ntracer=m
        zion(1:nparam,m)=zion(1:nparam,mtop)
        zion0(1:nparam,m)=zion0(1:nparam,mtop)
        mtop=mtop-1
        if (mtop == mi) exit  !break out of the do loop
     enddo
!  endif

! send # of particle to move right to neighboring pes of same particle
! domain.
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendright,2,mpi_integer,idest,isendtag,&
  !     mrecvleft,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndright_toroidal_cmm(ptr,msendright,2)
  call int_rcvleft_toroidal_cmm(ptr,mrecvleft)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call mpi_sendrecv(sendright,isendcount,mpi_rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendright,isendcount)
  call rcvleft_toroidal_cmm(ptr,recvleft)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendleft,2,mpi_integer,idest,isendtag,&
  !     mrecvright,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndleft_toroidal_cmm(ptr,msendleft,2)
  call int_rcvright_toroidal_cmm(ptr,mrecvright)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call mpi_sendrecv(sendleft,isendcount,mpi_rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
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
    ! call mpi_abort(mpi_comm_world,1,ierror)
! open disk file
!     if(mype < 10)then
!        write(cdum,'("temp.00",i1)')mype
!     elseif(mype < 100)then
!        write(cdum,'("temp.0",i2)')mype
!     else
!        write(cdum,'("temp.",i3)')mype
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
  
end subroutine shifti_0


subroutine shifti_1(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_1
  use particle_array_1
  use particle_decomp_1
  implicit none
  type(c_ptr), intent(in), value :: ptr
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
       !istatus(mpi_status_size)
  real(8),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(8) zetaright,zetaleft,pi_inv
  character(len=8) cdum
#ifdef _openmp
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
     write(0,*)'endless particle sorting loop at pe=',mype
     !call mpi_abort(mpi_comm_world,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= mi)then
!$omp parallel do private(m)
     do m=m0,mi
        kzi(m)=0
     enddo

#ifdef _openmp
! this section of code (down to #else) is included by the preprocessor when
! the compilation is performed with openmp support. we must then use a few
! temporary arrays and add some work distribution code.
  msleft=0
  msright=0

! first we start the parallel region with !$omp parallel

!$omp parallel private(nthreads,iam,delm,i,mbeg,mend,m,zetaright,zetaleft) &
!$omp& shared(gnthreads)
  nthreads=omp_get_num_threads()    !get the number of threads ready to work
  iam=omp_get_thread_num()          !get my thread number (position)
  delm=(mi-m0+1)/nthreads       !calculate the number of steps per thread
  i=mod((mi-m0+1),nthreads)
!$omp single              !put nthread in global variable for later use.
  gnthreads=nthreads      !nthread is the same for all threads so only one
!$omp end single nowait   !of them needs to copy the value in gnthreads

! we now distribute the work between the threads. the loop over the particles
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
! end of the openmp parallel region
!$omp end parallel

! now that we are out of the parallel region we need to gather and rearrange
! the results of the multi-threaded calculation. we need to end up with the
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
!  this section of code replaces the section above when the compilation does
!  not include the openmp support option. temporary arrays msleft and msright
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

     !call mpi_allreduce(msend,mrecv,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
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
! # of particles remain on local pe
     mi=mi-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzi(i)
        if (m > mi) exit  !break out of the do loop if m > mi
        do while(mtop == kzi(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( nhybrid == 0 .and. mtop == ntracer )ntracer=m
        zion(1:nparam,m)=zion(1:nparam,mtop)
        zion0(1:nparam,m)=zion0(1:nparam,mtop)
        mtop=mtop-1
        if (mtop == mi) exit  !break out of the do loop
     enddo
!  endif

! send # of particle to move right to neighboring pes of same particle
! domain.
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendright,2,mpi_integer,idest,isendtag,&
  !     mrecvleft,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndright_toroidal_cmm(ptr,msendright,2)
  call int_rcvleft_toroidal_cmm(ptr,mrecvleft)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call mpi_sendrecv(sendright,isendcount,mpi_rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendright,isendcount)
  call rcvleft_toroidal_cmm(ptr,recvleft)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendleft,2,mpi_integer,idest,isendtag,&
  !     mrecvright,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndleft_toroidal_cmm(ptr,msendleft,2)
  call int_rcvright_toroidal_cmm(ptr,mrecvright)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call mpi_sendrecv(sendleft,isendcount,mpi_rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
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
    ! call mpi_abort(mpi_comm_world,1,ierror)
! open disk file
!     if(mype < 10)then
!        write(cdum,'("temp.00",i1)')mype
!     elseif(mype < 100)then
!        write(cdum,'("temp.0",i2)')mype
!     else
!        write(cdum,'("temp.",i3)')mype
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
  
end subroutine shifti_1


subroutine shifti_2(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_2
  use particle_array_2
  use particle_decomp_2
  implicit none
  type(c_ptr), intent(in), value :: ptr
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
       !istatus(mpi_status_size)
  real(8),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(8) zetaright,zetaleft,pi_inv
  character(len=8) cdum
#ifdef _openmp
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
     write(0,*)'endless particle sorting loop at pe=',mype
     !call mpi_abort(mpi_comm_world,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= mi)then
!$omp parallel do private(m)
     do m=m0,mi
        kzi(m)=0
     enddo

#ifdef _openmp
! this section of code (down to #else) is included by the preprocessor when
! the compilation is performed with openmp support. we must then use a few
! temporary arrays and add some work distribution code.
  msleft=0
  msright=0

! first we start the parallel region with !$omp parallel

!$omp parallel private(nthreads,iam,delm,i,mbeg,mend,m,zetaright,zetaleft) &
!$omp& shared(gnthreads)
  nthreads=omp_get_num_threads()    !get the number of threads ready to work
  iam=omp_get_thread_num()          !get my thread number (position)
  delm=(mi-m0+1)/nthreads       !calculate the number of steps per thread
  i=mod((mi-m0+1),nthreads)
!$omp single              !put nthread in global variable for later use.
  gnthreads=nthreads      !nthread is the same for all threads so only one
!$omp end single nowait   !of them needs to copy the value in gnthreads

! we now distribute the work between the threads. the loop over the particles
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
! end of the openmp parallel region
!$omp end parallel

! now that we are out of the parallel region we need to gather and rearrange
! the results of the multi-threaded calculation. we need to end up with the
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
!  this section of code replaces the section above when the compilation does
!  not include the openmp support option. temporary arrays msleft and msright
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

     !call mpi_allreduce(msend,mrecv,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
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
! # of particles remain on local pe
     mi=mi-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzi(i)
        if (m > mi) exit  !break out of the do loop if m > mi
        do while(mtop == kzi(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( nhybrid == 0 .and. mtop == ntracer )ntracer=m
        zion(1:nparam,m)=zion(1:nparam,mtop)
        zion0(1:nparam,m)=zion0(1:nparam,mtop)
        mtop=mtop-1
        if (mtop == mi) exit  !break out of the do loop
     enddo
!  endif

! send # of particle to move right to neighboring pes of same particle
! domain.
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendright,2,mpi_integer,idest,isendtag,&
  !     mrecvleft,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndright_toroidal_cmm(ptr,msendright,2)
  call int_rcvleft_toroidal_cmm(ptr,mrecvleft)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call mpi_sendrecv(sendright,isendcount,mpi_rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendright,isendcount)
  call rcvleft_toroidal_cmm(ptr,recvleft)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendleft,2,mpi_integer,idest,isendtag,&
  !     mrecvright,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndleft_toroidal_cmm(ptr,msendleft,2)
  call int_rcvright_toroidal_cmm(ptr,mrecvright)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call mpi_sendrecv(sendleft,isendcount,mpi_rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
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
    ! call mpi_abort(mpi_comm_world,1,ierror)
! open disk file
!     if(mype < 10)then
!        write(cdum,'("temp.00",i1)')mype
!     elseif(mype < 100)then
!        write(cdum,'("temp.0",i2)')mype
!     else
!        write(cdum,'("temp.",i3)')mype
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
  
end subroutine shifti_2


subroutine shifti_3(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_3
  use particle_array_3
  use particle_decomp_3
  implicit none
  type(c_ptr), intent(in), value :: ptr
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
       !istatus(mpi_status_size)
  real(8),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(8) zetaright,zetaleft,pi_inv
  character(len=8) cdum
#ifdef _openmp
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
     write(0,*)'endless particle sorting loop at pe=',mype
     !call mpi_abort(mpi_comm_world,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= mi)then
!$omp parallel do private(m)
     do m=m0,mi
        kzi(m)=0
     enddo

#ifdef _openmp
! this section of code (down to #else) is included by the preprocessor when
! the compilation is performed with openmp support. we must then use a few
! temporary arrays and add some work distribution code.
  msleft=0
  msright=0

! first we start the parallel region with !$omp parallel

!$omp parallel private(nthreads,iam,delm,i,mbeg,mend,m,zetaright,zetaleft) &
!$omp& shared(gnthreads)
  nthreads=omp_get_num_threads()    !get the number of threads ready to work
  iam=omp_get_thread_num()          !get my thread number (position)
  delm=(mi-m0+1)/nthreads       !calculate the number of steps per thread
  i=mod((mi-m0+1),nthreads)
!$omp single              !put nthread in global variable for later use.
  gnthreads=nthreads      !nthread is the same for all threads so only one
!$omp end single nowait   !of them needs to copy the value in gnthreads

! we now distribute the work between the threads. the loop over the particles
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
! end of the openmp parallel region
!$omp end parallel

! now that we are out of the parallel region we need to gather and rearrange
! the results of the multi-threaded calculation. we need to end up with the
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
!  this section of code replaces the section above when the compilation does
!  not include the openmp support option. temporary arrays msleft and msright
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

     !call mpi_allreduce(msend,mrecv,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
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
! # of particles remain on local pe
     mi=mi-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzi(i)
        if (m > mi) exit  !break out of the do loop if m > mi
        do while(mtop == kzi(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( nhybrid == 0 .and. mtop == ntracer )ntracer=m
        zion(1:nparam,m)=zion(1:nparam,mtop)
        zion0(1:nparam,m)=zion0(1:nparam,mtop)
        mtop=mtop-1
        if (mtop == mi) exit  !break out of the do loop
     enddo
!  endif

! send # of particle to move right to neighboring pes of same particle
! domain.
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendright,2,mpi_integer,idest,isendtag,&
  !     mrecvleft,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndright_toroidal_cmm(ptr,msendright,2)
  call int_rcvleft_toroidal_cmm(ptr,mrecvleft)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call mpi_sendrecv(sendright,isendcount,mpi_rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendright,isendcount)
  call rcvleft_toroidal_cmm(ptr,recvleft)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendleft,2,mpi_integer,idest,isendtag,&
  !     mrecvright,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndleft_toroidal_cmm(ptr,msendleft,2)
  call int_rcvright_toroidal_cmm(ptr,mrecvright)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call mpi_sendrecv(sendleft,isendcount,mpi_rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
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
    ! call mpi_abort(mpi_comm_world,1,ierror)
! open disk file
!     if(mype < 10)then
!        write(cdum,'("temp.00",i1)')mype
!     elseif(mype < 100)then
!        write(cdum,'("temp.0",i2)')mype
!     else
!        write(cdum,'("temp.",i3)')mype
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
  
end subroutine shifti_3


subroutine shifti_4(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_4
  use particle_array_4
  use particle_decomp_4
  implicit none
  type(c_ptr), intent(in), value :: ptr
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
       !istatus(mpi_status_size)
  real(8),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(8) zetaright,zetaleft,pi_inv
  character(len=8) cdum
#ifdef _openmp
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
     write(0,*)'endless particle sorting loop at pe=',mype
     !call mpi_abort(mpi_comm_world,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= mi)then
!$omp parallel do private(m)
     do m=m0,mi
        kzi(m)=0
     enddo

#ifdef _openmp
! this section of code (down to #else) is included by the preprocessor when
! the compilation is performed with openmp support. we must then use a few
! temporary arrays and add some work distribution code.
  msleft=0
  msright=0

! first we start the parallel region with !$omp parallel

!$omp parallel private(nthreads,iam,delm,i,mbeg,mend,m,zetaright,zetaleft) &
!$omp& shared(gnthreads)
  nthreads=omp_get_num_threads()    !get the number of threads ready to work
  iam=omp_get_thread_num()          !get my thread number (position)
  delm=(mi-m0+1)/nthreads       !calculate the number of steps per thread
  i=mod((mi-m0+1),nthreads)
!$omp single              !put nthread in global variable for later use.
  gnthreads=nthreads      !nthread is the same for all threads so only one
!$omp end single nowait   !of them needs to copy the value in gnthreads

! we now distribute the work between the threads. the loop over the particles
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
! end of the openmp parallel region
!$omp end parallel

! now that we are out of the parallel region we need to gather and rearrange
! the results of the multi-threaded calculation. we need to end up with the
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
!  this section of code replaces the section above when the compilation does
!  not include the openmp support option. temporary arrays msleft and msright
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

     !call mpi_allreduce(msend,mrecv,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
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
! # of particles remain on local pe
     mi=mi-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzi(i)
        if (m > mi) exit  !break out of the do loop if m > mi
        do while(mtop == kzi(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( nhybrid == 0 .and. mtop == ntracer )ntracer=m
        zion(1:nparam,m)=zion(1:nparam,mtop)
        zion0(1:nparam,m)=zion0(1:nparam,mtop)
        mtop=mtop-1
        if (mtop == mi) exit  !break out of the do loop
     enddo
!  endif

! send # of particle to move right to neighboring pes of same particle
! domain.
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendright,2,mpi_integer,idest,isendtag,&
  !     mrecvleft,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndright_toroidal_cmm(ptr,msendright,2)
  call int_rcvleft_toroidal_cmm(ptr,mrecvleft)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call mpi_sendrecv(sendright,isendcount,mpi_rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendright,isendcount)
  call rcvleft_toroidal_cmm(ptr,recvleft)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendleft,2,mpi_integer,idest,isendtag,&
  !     mrecvright,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndleft_toroidal_cmm(ptr,msendleft,2)
  call int_rcvright_toroidal_cmm(ptr,mrecvright)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call mpi_sendrecv(sendleft,isendcount,mpi_rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
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
    ! call mpi_abort(mpi_comm_world,1,ierror)
! open disk file
!     if(mype < 10)then
!        write(cdum,'("temp.00",i1)')mype
!     elseif(mype < 100)then
!        write(cdum,'("temp.0",i2)')mype
!     else
!        write(cdum,'("temp.",i3)')mype
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
  
end subroutine shifti_4


subroutine shifti_5(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_5
  use particle_array_5
  use particle_decomp_5
  implicit none
  type(c_ptr), intent(in), value :: ptr
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
       !istatus(mpi_status_size)
  real(8),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(8) zetaright,zetaleft,pi_inv
  character(len=8) cdum
#ifdef _openmp
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
     write(0,*)'endless particle sorting loop at pe=',mype
     !call mpi_abort(mpi_comm_world,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= mi)then
!$omp parallel do private(m)
     do m=m0,mi
        kzi(m)=0
     enddo

#ifdef _openmp
! this section of code (down to #else) is included by the preprocessor when
! the compilation is performed with openmp support. we must then use a few
! temporary arrays and add some work distribution code.
  msleft=0
  msright=0

! first we start the parallel region with !$omp parallel

!$omp parallel private(nthreads,iam,delm,i,mbeg,mend,m,zetaright,zetaleft) &
!$omp& shared(gnthreads)
  nthreads=omp_get_num_threads()    !get the number of threads ready to work
  iam=omp_get_thread_num()          !get my thread number (position)
  delm=(mi-m0+1)/nthreads       !calculate the number of steps per thread
  i=mod((mi-m0+1),nthreads)
!$omp single              !put nthread in global variable for later use.
  gnthreads=nthreads      !nthread is the same for all threads so only one
!$omp end single nowait   !of them needs to copy the value in gnthreads

! we now distribute the work between the threads. the loop over the particles
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
! end of the openmp parallel region
!$omp end parallel

! now that we are out of the parallel region we need to gather and rearrange
! the results of the multi-threaded calculation. we need to end up with the
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
!  this section of code replaces the section above when the compilation does
!  not include the openmp support option. temporary arrays msleft and msright
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

     !call mpi_allreduce(msend,mrecv,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
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
! # of particles remain on local pe
     mi=mi-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzi(i)
        if (m > mi) exit  !break out of the do loop if m > mi
        do while(mtop == kzi(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( nhybrid == 0 .and. mtop == ntracer )ntracer=m
        zion(1:nparam,m)=zion(1:nparam,mtop)
        zion0(1:nparam,m)=zion0(1:nparam,mtop)
        mtop=mtop-1
        if (mtop == mi) exit  !break out of the do loop
     enddo
!  endif

! send # of particle to move right to neighboring pes of same particle
! domain.
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendright,2,mpi_integer,idest,isendtag,&
  !     mrecvleft,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndright_toroidal_cmm(ptr,msendright,2)
  call int_rcvleft_toroidal_cmm(ptr,mrecvleft)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call mpi_sendrecv(sendright,isendcount,mpi_rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendright,isendcount)
  call rcvleft_toroidal_cmm(ptr,recvleft)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendleft,2,mpi_integer,idest,isendtag,&
  !     mrecvright,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndleft_toroidal_cmm(ptr,msendleft,2)
  call int_rcvright_toroidal_cmm(ptr,mrecvright)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call mpi_sendrecv(sendleft,isendcount,mpi_rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
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
    ! call mpi_abort(mpi_comm_world,1,ierror)
! open disk file
!     if(mype < 10)then
!        write(cdum,'("temp.00",i1)')mype
!     elseif(mype < 100)then
!        write(cdum,'("temp.0",i2)')mype
!     else
!        write(cdum,'("temp.",i3)')mype
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
  
end subroutine shifti_5


subroutine shifti_6(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_6
  use particle_array_6
  use particle_decomp_6
  implicit none
  type(c_ptr), intent(in), value :: ptr
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
       !istatus(mpi_status_size)
  real(8),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(8) zetaright,zetaleft,pi_inv
  character(len=8) cdum
#ifdef _openmp
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
     write(0,*)'endless particle sorting loop at pe=',mype
     !call mpi_abort(mpi_comm_world,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= mi)then
!$omp parallel do private(m)
     do m=m0,mi
        kzi(m)=0
     enddo

#ifdef _openmp
! this section of code (down to #else) is included by the preprocessor when
! the compilation is performed with openmp support. we must then use a few
! temporary arrays and add some work distribution code.
  msleft=0
  msright=0

! first we start the parallel region with !$omp parallel

!$omp parallel private(nthreads,iam,delm,i,mbeg,mend,m,zetaright,zetaleft) &
!$omp& shared(gnthreads)
  nthreads=omp_get_num_threads()    !get the number of threads ready to work
  iam=omp_get_thread_num()          !get my thread number (position)
  delm=(mi-m0+1)/nthreads       !calculate the number of steps per thread
  i=mod((mi-m0+1),nthreads)
!$omp single              !put nthread in global variable for later use.
  gnthreads=nthreads      !nthread is the same for all threads so only one
!$omp end single nowait   !of them needs to copy the value in gnthreads

! we now distribute the work between the threads. the loop over the particles
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
! end of the openmp parallel region
!$omp end parallel

! now that we are out of the parallel region we need to gather and rearrange
! the results of the multi-threaded calculation. we need to end up with the
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
!  this section of code replaces the section above when the compilation does
!  not include the openmp support option. temporary arrays msleft and msright
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

     !call mpi_allreduce(msend,mrecv,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
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
! # of particles remain on local pe
     mi=mi-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzi(i)
        if (m > mi) exit  !break out of the do loop if m > mi
        do while(mtop == kzi(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( nhybrid == 0 .and. mtop == ntracer )ntracer=m
        zion(1:nparam,m)=zion(1:nparam,mtop)
        zion0(1:nparam,m)=zion0(1:nparam,mtop)
        mtop=mtop-1
        if (mtop == mi) exit  !break out of the do loop
     enddo
!  endif

! send # of particle to move right to neighboring pes of same particle
! domain.
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendright,2,mpi_integer,idest,isendtag,&
  !     mrecvleft,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndright_toroidal_cmm(ptr,msendright,2)
  call int_rcvleft_toroidal_cmm(ptr,mrecvleft)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call mpi_sendrecv(sendright,isendcount,mpi_rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendright,isendcount)
  call rcvleft_toroidal_cmm(ptr,recvleft)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendleft,2,mpi_integer,idest,isendtag,&
  !     mrecvright,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndleft_toroidal_cmm(ptr,msendleft,2)
  call int_rcvright_toroidal_cmm(ptr,mrecvright)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call mpi_sendrecv(sendleft,isendcount,mpi_rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
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
    ! call mpi_abort(mpi_comm_world,1,ierror)
! open disk file
!     if(mype < 10)then
!        write(cdum,'("temp.00",i1)')mype
!     elseif(mype < 100)then
!        write(cdum,'("temp.0",i2)')mype
!     else
!        write(cdum,'("temp.",i3)')mype
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
  
end subroutine shifti_6


subroutine shifti_7(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_7
  use particle_array_7
  use particle_decomp_7
  implicit none
  type(c_ptr), intent(in), value :: ptr
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
       !istatus(mpi_status_size)
  real(8),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(8) zetaright,zetaleft,pi_inv
  character(len=8) cdum
#ifdef _openmp
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
     write(0,*)'endless particle sorting loop at pe=',mype
     !call mpi_abort(mpi_comm_world,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= mi)then
!$omp parallel do private(m)
     do m=m0,mi
        kzi(m)=0
     enddo

#ifdef _openmp
! this section of code (down to #else) is included by the preprocessor when
! the compilation is performed with openmp support. we must then use a few
! temporary arrays and add some work distribution code.
  msleft=0
  msright=0

! first we start the parallel region with !$omp parallel

!$omp parallel private(nthreads,iam,delm,i,mbeg,mend,m,zetaright,zetaleft) &
!$omp& shared(gnthreads)
  nthreads=omp_get_num_threads()    !get the number of threads ready to work
  iam=omp_get_thread_num()          !get my thread number (position)
  delm=(mi-m0+1)/nthreads       !calculate the number of steps per thread
  i=mod((mi-m0+1),nthreads)
!$omp single              !put nthread in global variable for later use.
  gnthreads=nthreads      !nthread is the same for all threads so only one
!$omp end single nowait   !of them needs to copy the value in gnthreads

! we now distribute the work between the threads. the loop over the particles
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
! end of the openmp parallel region
!$omp end parallel

! now that we are out of the parallel region we need to gather and rearrange
! the results of the multi-threaded calculation. we need to end up with the
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
!  this section of code replaces the section above when the compilation does
!  not include the openmp support option. temporary arrays msleft and msright
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

     !call mpi_allreduce(msend,mrecv,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
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
! # of particles remain on local pe
     mi=mi-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzi(i)
        if (m > mi) exit  !break out of the do loop if m > mi
        do while(mtop == kzi(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( nhybrid == 0 .and. mtop == ntracer )ntracer=m
        zion(1:nparam,m)=zion(1:nparam,mtop)
        zion0(1:nparam,m)=zion0(1:nparam,mtop)
        mtop=mtop-1
        if (mtop == mi) exit  !break out of the do loop
     enddo
!  endif

! send # of particle to move right to neighboring pes of same particle
! domain.
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendright,2,mpi_integer,idest,isendtag,&
  !     mrecvleft,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndright_toroidal_cmm(ptr,msendright,2)
  call int_rcvleft_toroidal_cmm(ptr,mrecvleft)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call mpi_sendrecv(sendright,isendcount,mpi_rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendright,isendcount)
  call rcvleft_toroidal_cmm(ptr,recvleft)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendleft,2,mpi_integer,idest,isendtag,&
  !     mrecvright,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndleft_toroidal_cmm(ptr,msendleft,2)
  call int_rcvright_toroidal_cmm(ptr,mrecvright)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call mpi_sendrecv(sendleft,isendcount,mpi_rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
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
    ! call mpi_abort(mpi_comm_world,1,ierror)
! open disk file
!     if(mype < 10)then
!        write(cdum,'("temp.00",i1)')mype
!     elseif(mype < 100)then
!        write(cdum,'("temp.0",i2)')mype
!     else
!        write(cdum,'("temp.",i3)')mype
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
  
end subroutine shifti_7


subroutine shifti_8(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_8
  use particle_array_8
  use particle_decomp_8
  implicit none
  type(c_ptr), intent(in), value :: ptr
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
       !istatus(mpi_status_size)
  real(8),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(8) zetaright,zetaleft,pi_inv
  character(len=8) cdum
#ifdef _openmp
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
     write(0,*)'endless particle sorting loop at pe=',mype
     !call mpi_abort(mpi_comm_world,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= mi)then
!$omp parallel do private(m)
     do m=m0,mi
        kzi(m)=0
     enddo

#ifdef _openmp
! this section of code (down to #else) is included by the preprocessor when
! the compilation is performed with openmp support. we must then use a few
! temporary arrays and add some work distribution code.
  msleft=0
  msright=0

! first we start the parallel region with !$omp parallel

!$omp parallel private(nthreads,iam,delm,i,mbeg,mend,m,zetaright,zetaleft) &
!$omp& shared(gnthreads)
  nthreads=omp_get_num_threads()    !get the number of threads ready to work
  iam=omp_get_thread_num()          !get my thread number (position)
  delm=(mi-m0+1)/nthreads       !calculate the number of steps per thread
  i=mod((mi-m0+1),nthreads)
!$omp single              !put nthread in global variable for later use.
  gnthreads=nthreads      !nthread is the same for all threads so only one
!$omp end single nowait   !of them needs to copy the value in gnthreads

! we now distribute the work between the threads. the loop over the particles
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
! end of the openmp parallel region
!$omp end parallel

! now that we are out of the parallel region we need to gather and rearrange
! the results of the multi-threaded calculation. we need to end up with the
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
!  this section of code replaces the section above when the compilation does
!  not include the openmp support option. temporary arrays msleft and msright
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

     !call mpi_allreduce(msend,mrecv,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
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
! # of particles remain on local pe
     mi=mi-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzi(i)
        if (m > mi) exit  !break out of the do loop if m > mi
        do while(mtop == kzi(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( nhybrid == 0 .and. mtop == ntracer )ntracer=m
        zion(1:nparam,m)=zion(1:nparam,mtop)
        zion0(1:nparam,m)=zion0(1:nparam,mtop)
        mtop=mtop-1
        if (mtop == mi) exit  !break out of the do loop
     enddo
!  endif

! send # of particle to move right to neighboring pes of same particle
! domain.
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendright,2,mpi_integer,idest,isendtag,&
  !     mrecvleft,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndright_toroidal_cmm(ptr,msendright,2)
  call int_rcvleft_toroidal_cmm(ptr,mrecvleft)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call mpi_sendrecv(sendright,isendcount,mpi_rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendright,isendcount)
  call rcvleft_toroidal_cmm(ptr,recvleft)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendleft,2,mpi_integer,idest,isendtag,&
  !     mrecvright,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndleft_toroidal_cmm(ptr,msendleft,2)
  call int_rcvright_toroidal_cmm(ptr,mrecvright)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call mpi_sendrecv(sendleft,isendcount,mpi_rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
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
    ! call mpi_abort(mpi_comm_world,1,ierror)
! open disk file
!     if(mype < 10)then
!        write(cdum,'("temp.00",i1)')mype
!     elseif(mype < 100)then
!        write(cdum,'("temp.0",i2)')mype
!     else
!        write(cdum,'("temp.",i3)')mype
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
  
end subroutine shifti_8


subroutine shifti_9(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_9
  use particle_array_9
  use particle_decomp_9
  implicit none
  type(c_ptr), intent(in), value :: ptr
  
  integer i,m,msendleft(2),msendright(2),mrecvleft(2),mrecvright(2),mtop,&
       kzi(mimax),m0,msend,mrecv,idest,isource,isendtag,irecvtag,nzion,&
       iright(mimax),ileft(mimax),isendcount,irecvcount,&
       ierror,iteration,lasth
       !istatus(mpi_status_size)
  real(8),dimension(:,:),allocatable :: recvleft,recvright,sendleft,sendright
  real(8) zetaright,zetaleft,pi_inv
  character(len=8) cdum
#ifdef _openmp
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
     write(0,*)'endless particle sorting loop at pe=',mype
     !call mpi_abort(mpi_comm_world,1,ierror)
  endif

  msend=0
  msendright=0
  msendleft=0

  if(m0 <= mi)then
!$omp parallel do private(m)
     do m=m0,mi
        kzi(m)=0
     enddo

#ifdef _openmp
! this section of code (down to #else) is included by the preprocessor when
! the compilation is performed with openmp support. we must then use a few
! temporary arrays and add some work distribution code.
  msleft=0
  msright=0

! first we start the parallel region with !$omp parallel

!$omp parallel private(nthreads,iam,delm,i,mbeg,mend,m,zetaright,zetaleft) &
!$omp& shared(gnthreads)
  nthreads=omp_get_num_threads()    !get the number of threads ready to work
  iam=omp_get_thread_num()          !get my thread number (position)
  delm=(mi-m0+1)/nthreads       !calculate the number of steps per thread
  i=mod((mi-m0+1),nthreads)
!$omp single              !put nthread in global variable for later use.
  gnthreads=nthreads      !nthread is the same for all threads so only one
!$omp end single nowait   !of them needs to copy the value in gnthreads

! we now distribute the work between the threads. the loop over the particles
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
! end of the openmp parallel region
!$omp end parallel

! now that we are out of the parallel region we need to gather and rearrange
! the results of the multi-threaded calculation. we need to end up with the
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
!  this section of code replaces the section above when the compilation does
!  not include the openmp support option. temporary arrays msleft and msright
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

     !call mpi_allreduce(msend,mrecv,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
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
! # of particles remain on local pe
     mi=mi-msendleft(1)-msendright(1)
! fill the hole
     lasth=msend
     do i=1,msend
        m=kzi(i)
        if (m > mi) exit  !break out of the do loop if m > mi
        do while(mtop == kzi(lasth))
           mtop=mtop-1
           lasth=lasth-1
        enddo
        if( nhybrid == 0 .and. mtop == ntracer )ntracer=m
        zion(1:nparam,m)=zion(1:nparam,mtop)
        zion0(1:nparam,m)=zion0(1:nparam,mtop)
        mtop=mtop-1
        if (mtop == mi) exit  !break out of the do loop
     enddo
!  endif

! send # of particle to move right to neighboring pes of same particle
! domain.
  mrecvleft=0
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendright,2,mpi_integer,idest,isendtag,&
  !     mrecvleft,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndright_toroidal_cmm(ptr,msendright,2)
  call int_rcvleft_toroidal_cmm(ptr,mrecvleft)
  
  allocate(recvleft(nzion,max(mrecvleft(1),1)))
 
! send particle to right and receive from left
  recvleft=0.0
  isendcount=msendright(1)*nzion
  irecvcount=mrecvleft(1)*nzion
  !call mpi_sendrecv(sendright,isendcount,mpi_rsize,idest,isendtag,recvleft,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendright,isendcount)
  call rcvleft_toroidal_cmm(ptr,recvleft)
  
! send # of particle to move left
  mrecvright=0
  idest=left_pe
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(msendleft,2,mpi_integer,idest,isendtag,&
  !     mrecvright,2,mpi_integer,isource,irecvtag,toroidal_comm,istatus,ierror)
  call int_sndleft_toroidal_cmm(ptr,msendleft,2)
  call int_rcvright_toroidal_cmm(ptr,mrecvright)
  
  allocate(recvright(nzion,max(mrecvright(1),1)))

! send particle to left and receive from right
  recvright=0.0
  isendcount=msendleft(1)*nzion
  irecvcount=mrecvright(1)*nzion
  !call mpi_sendrecv(sendleft,isendcount,mpi_rsize,idest,isendtag,recvright,&
  !     irecvcount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
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
    ! call mpi_abort(mpi_comm_world,1,ierror)
! open disk file
!     if(mype < 10)then
!        write(cdum,'("temp.00",i1)')mype
!     elseif(mype < 100)then
!        write(cdum,'("temp.0",i2)')mype
!     else
!        write(cdum,'("temp.",i3)')mype
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
  
end subroutine shifti_9


