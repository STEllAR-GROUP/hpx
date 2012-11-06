!========================================================================

    Subroutine loop(hpx4_bti,hpx4_mype,hpx4_numberpe)

!========================================================================

  use, intrinsic :: iso_c_binding, only : c_ptr
  implicit none

  integer hpx4_mype,hpx4_numberpe
  TYPE(C_PTR), INTENT(IN), VALUE :: hpx4_bti
! hjw
  real*8,dimension(:),allocatable :: send,recv

  integer right_pe,i,j
  integer mgrid,icount,idest  
  integer repeats

  mgrid = 10000
  repeats = 10000  

  allocate(send(mgrid),recv(mgrid))
  do i=1,mgrid
    send(i) = hpx4_mype + i*(hpx4_mype*i)**(1.0d0/hpx4_mype)
  enddo  

  right_pe = hpx4_mype + 1
  if ( right_pe .ge. hpx4_numberpe ) then
    right_pe = 0
  endif 

  recv=0.0
  icount=mgrid
  idest= right_pe

  do i=1,repeats 
    do j=1,mgrid
      send(j) = hpx4_mype + j*(hpx4_mype*i*repeats)**(1.0d0/hpx4_mype)
    enddo  
    call sndrecv_toroidal_cmm(hpx4_bti,send,icount,recv,icount,idest)
  enddo

  deallocate(send,recv)
end subroutine loop
