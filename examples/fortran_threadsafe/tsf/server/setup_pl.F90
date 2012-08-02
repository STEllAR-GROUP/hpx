! WARNING:  THIS FILE IS GENERATED FROM A PERL SCRIPT:  DO NOT MODIFY
! SINCE YOUR CHANGES WILL NOT PERSIST BETWEEN BUILDS.
!                                                    
!========================================================================

module global_parameters_0
  integer  :: mype
end module global_parameters_0
module global_parameters_1
  integer  :: mype
end module global_parameters_1
module global_parameters_2
  integer  :: mype
end module global_parameters_2
module global_parameters_3
  integer  :: mype
end module global_parameters_3
module global_parameters_4
  integer  :: mype
end module global_parameters_4
module global_parameters_5
  integer  :: mype
end module global_parameters_5
module global_parameters_6
  integer  :: mype
end module global_parameters_6
module global_parameters_7
  integer  :: mype
end module global_parameters_7
module global_parameters_8
  integer  :: mype
end module global_parameters_8
module global_parameters_9
  integer  :: mype
end module global_parameters_9
module global_parameters_10
  integer  :: mype
end module global_parameters_10
module global_parameters_11
  integer  :: mype
end module global_parameters_11

!========================================================================

    subroutine setup_0(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_0
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_0
    subroutine setup_1(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_1
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_1
    subroutine setup_2(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_2
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_2
    subroutine setup_3(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_3
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_3
    subroutine setup_4(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_4
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_4
    subroutine setup_5(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_5
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_5
    subroutine setup_6(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_6
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_6
    subroutine setup_7(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_7
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_7
    subroutine setup_8(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_8
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_8
    subroutine setup_9(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_9
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_9
    subroutine setup_10(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_10
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_10
    subroutine setup_11(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)


  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_11
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer,parameter:: n_integers = 20,n_reals=28
  integer  :: integer_params(n_integers)
  real*8   :: real_params(n_reals)
  integer  :: i
  integer hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal

  mype = hpx_mype
  print*,' test mype ', mype

  if (mype == 0) then
    do i=0,n_integers
      integer_params(i)= i
    end do
    do i=0,n_reals
      real_params(i)= i*10.0d0
    end do
  endif

  print*, ' test a mype ', mype

  call broadcast_parameters_cmm(ptr,integer_params,real_params,&
                                n_integers,n_reals); 

  print*, ' test b mype ', mype

end subroutine setup_11
