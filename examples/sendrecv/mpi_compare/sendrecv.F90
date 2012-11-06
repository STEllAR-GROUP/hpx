! Matthew Anderson
! 6 Nov 2012

      program sendrecv_compare

      implicit none
      include 'mpif.h'

      real*8,dimension(:),allocatable :: send,recv

      integer right_pe,i,j
      integer mgrid,icount,idest
      integer repeats
      integer ierror
      integer mype,numberpe,left_pe
      integer isource,isendtag,irecvtag,istatus(MPI_STATUS_SIZE)

      call mpi_init(ierror)
      call mpi_comm_size(mpi_comm_world,numberpe,ierror)
      call mpi_comm_rank(mpi_comm_world,mype,ierror)

      mgrid = 10000
      repeats = 10000

      allocate(send(mgrid),recv(mgrid))
      do i=1,mgrid
        send(i) = mype + i*(mype*i)**(1.0d0/mype)
      enddo

      right_pe = mype + 1
      if ( right_pe .ge. numberpe ) then
        right_pe = 0
      endif
      left_pe = mype - 1
      if ( left_pe .le. 0 ) then
        left_pe = numberpe-1
      endif

      recv=0.0
      icount=mgrid
      idest= right_pe
      isource = left_pe
      isendtag=mype
      irecvtag=isource

      do i=1,repeats
        do j=1,mgrid
          send(j) = mype + j*(mype*i*repeats)**(1.0d0/mype)
        enddo
        call MPI_SENDRECV(send,icount,MPI_DOUBLE_PRECISION,&
                          idest,isendtag,&
                          recv,icount,MPI_DOUBLE_PRECISION,&
                         isource,irecvtag,mpi_comm_world,istatus,ierror)
      enddo

      deallocate(send,recv)

      end program sendrecv_compare
