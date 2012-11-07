! Matthew Anderson
! 6 Nov 2012

      program sendrecv_compare

      implicit none
      include 'mpif.h'

      real*8,dimension(:),allocatable :: send,recv

      integer right_pe,i,j
      integer mgrid,icount,idest
      integer ierror
      integer mype,numberpe,left_pe
      integer isource,isendtag,irecvtag,istatus(MPI_STATUS_SIZE)
      integer repeats
      real*8 NOWTIME,PASTTIME,TIME,MAXTIME

      call mpi_init(ierror)
      PASTTIME = MPI_WTIME()
      call mpi_comm_size(mpi_comm_world,numberpe,ierror)
      call mpi_comm_rank(mpi_comm_world,mype,ierror)

      mgrid = 1000
      repeats = 1000
  

      allocate(send(mgrid),recv(mgrid))

      right_pe = mype + 1
      if ( right_pe .ge. numberpe ) then
        right_pe = 0
      endif
      left_pe = mype - 1
      if ( left_pe .lt. 0 ) then
        left_pe = numberpe-1
      endif

      recv=0.0
      icount=mgrid
      idest= right_pe
      isource = left_pe
      isendtag=mype
      irecvtag=isource

      do j=1,mgrid
        send(j) = mype + j*(j*(mype+1))**(1.0d0/(mype+1))
      enddo

      do i=1,repeats
        call MPI_SENDRECV(send,icount,MPI_DOUBLE_PRECISION,&
                        idest,isendtag,&
                        recv,icount,MPI_DOUBLE_PRECISION,&
                       isource,irecvtag,mpi_comm_world,istatus,ierror)
      enddo
      NOWTIME = MPI_WTIME()
      TIME = NOWTIME-PASTTIME
      icount = 1
      call MPI_BARRIER(mpi_comm_world,ierror)
      call MPI_REDUCE(TIME,MAXTIME,icount,MPI_DOUBLE_PRECISION,&
                              MPI_MAX,0, & 
                              MPI_COMM_WORLD,ierror)
      if ( mype .eq. 0 ) then
        print*,' TIME ', MAXTIME
      endif
      !print*,' TIME ',NOWTIME-PASTTIME,' mype ', mype
      deallocate(send,recv)

      end program sendrecv_compare
