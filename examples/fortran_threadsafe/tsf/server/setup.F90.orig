!========================================================================

module global_parameters
  integer  :: mype
end module global_parameters

!========================================================================

    Subroutine setup(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters
  implicit none
  TYPE(C_PTR), INTENT(IN), VALUE :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

! total # of PE and rank of PE
  mype = hpx_mype
  print*,' TEST mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' TEST A mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' TEST B mype ', mype
  !if (mype/=0) then
  ! unpack
  !  print*, ' TEST mype ', mype, &
  !' <= should be a number between 0 and 9; numbers should not be duplicated'
  !endif

end subroutine setup
