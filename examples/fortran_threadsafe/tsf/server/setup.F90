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

!========================================================================

    Subroutine setup_0(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters_0
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

end subroutine setup_0

!========================================================================

    Subroutine setup_1(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters_1
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

end subroutine setup_1
!========================================================================

    Subroutine setup_2(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters_2
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

end subroutine setup_2
!========================================================================

    Subroutine setup_3(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters_3
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

end subroutine setup_3
!========================================================================

    Subroutine setup_4(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters_4
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

end subroutine setup_4
!========================================================================

    Subroutine setup_5(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters_5
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

end subroutine setup_5
!========================================================================

    Subroutine setup_6(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters_6
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

end subroutine setup_6
!========================================================================

    Subroutine setup_7(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters_7
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

end subroutine setup_7
!========================================================================

    Subroutine setup_8(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters_8
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

end subroutine setup_8
!========================================================================

    Subroutine setup_9(ptr, &
                     hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters_9
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

end subroutine setup_9
