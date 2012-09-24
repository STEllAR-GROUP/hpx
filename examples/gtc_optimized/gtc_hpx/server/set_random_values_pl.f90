module rng_gtc_0
  use rng1_0
! declarations and parameters for the portable random number generator
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state
end module rng_gtc_0

subroutine rand_num_gen_init_0
  use global_parameters_0
  use rng_gtc_0
  implicit none

  integer i,m,nsize
  integer,dimension(:),allocatable :: mget,mput

  if(rng_control==0)then
   ! **** use the intrinsic f90 random number generator *****
   ! initialize f90 random number generator
     call random_seed
     call random_seed(size=nsize)
     allocate(mget(nsize),mput(nsize))
     call random_seed(get=mget)
     if(mype==0)write(stdout,*)"random_seed=",nsize,mget
     do i=1,nsize 
        call system_clock(m)   !random initialization for collision
        if(irun==0)m=1         !same initialization
        mput(i)=111111*(mype+1)+m+mget(i)
     enddo
     call random_seed(put=mput)
     deallocate(mget,mput)

  ! ****** use charles karney's portable random number generator *****
  elseif(rng_control>0)then
   ! all the processors start with the same seed.
     call rng_set_seed(seed,(rng_control-1))  !set seed to (rng_control-1)
     if(mype==0)write(0,*)'seed is set to ',rng_print_seed_0(seed)
   ! initialize the random number generator
     call rng_init_0(seed,state)

  else
   ! set seed to current time
     call rng_set_seed(seed)
   ! advance seed according to process number
     call rng_step_seed_0(seed,mype)
   ! initialize random number generator
     call rng_init_0(seed,state)
  endif

end subroutine rand_num_gen_init_0

subroutine set_random_zion_0
  use global_parameters_0
  use particle_array_0
  use rng_gtc_0
  implicit none

  integer :: i,ierr

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(zion(2,1:mi))
     call random_number(zion(3,1:mi))
     call random_number(zion(4,1:mi))
     call random_number(zion(5,1:mi))
     call random_number(zion(6,1:mi))

  elseif(rng_control>0)then
   ! the following calls to rng_number insure that the series of random
   ! numbers generated will be the same for a given seed no matter how
   ! many processors are used. this is useful to test the reproducibility
   ! of the results on different platforms and for general testing.
!hjw do i=1,mype+1
!hjw    call rng_number(state,zion(2:6,1:mi))
!hjw enddo
   ! we now force the processes to wait for each other since the preceding
   ! loop will take an increasing amount of time 
!hjw call mpi_barrier(mpi_comm_world,ierr)
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))

  else
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))
  endif

! debug statements
!  do i=1,mi
!     write(mype+50,*)(i+mi*mype),zion(2:6,i)
!  enddo
!  close(mype+50)

end subroutine set_random_zion_0

function get_random_number_0()
  use global_parameters_0
  use rng_gtc_0
  implicit none
  real(8) :: get_random_number_0

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(get_random_number_0)
  else
     call rng_number(state,get_random_number_0)
  endif

end function get_random_number_0
module rng_gtc_1
  use rng1_1
! declarations and parameters for the portable random number generator
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state
end module rng_gtc_1

subroutine rand_num_gen_init_1
  use global_parameters_1
  use rng_gtc_1
  implicit none

  integer i,m,nsize
  integer,dimension(:),allocatable :: mget,mput

  if(rng_control==0)then
   ! **** use the intrinsic f90 random number generator *****
   ! initialize f90 random number generator
     call random_seed
     call random_seed(size=nsize)
     allocate(mget(nsize),mput(nsize))
     call random_seed(get=mget)
     if(mype==0)write(stdout,*)"random_seed=",nsize,mget
     do i=1,nsize 
        call system_clock(m)   !random initialization for collision
        if(irun==0)m=1         !same initialization
        mput(i)=111111*(mype+1)+m+mget(i)
     enddo
     call random_seed(put=mput)
     deallocate(mget,mput)

  ! ****** use charles karney's portable random number generator *****
  elseif(rng_control>0)then
   ! all the processors start with the same seed.
     call rng_set_seed(seed,(rng_control-1))  !set seed to (rng_control-1)
     if(mype==0)write(0,*)'seed is set to ',rng_print_seed_1(seed)
   ! initialize the random number generator
     call rng_init_1(seed,state)

  else
   ! set seed to current time
     call rng_set_seed(seed)
   ! advance seed according to process number
     call rng_step_seed_1(seed,mype)
   ! initialize random number generator
     call rng_init_1(seed,state)
  endif

end subroutine rand_num_gen_init_1

subroutine set_random_zion_1
  use global_parameters_1
  use particle_array_1
  use rng_gtc_1
  implicit none

  integer :: i,ierr

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(zion(2,1:mi))
     call random_number(zion(3,1:mi))
     call random_number(zion(4,1:mi))
     call random_number(zion(5,1:mi))
     call random_number(zion(6,1:mi))

  elseif(rng_control>0)then
   ! the following calls to rng_number insure that the series of random
   ! numbers generated will be the same for a given seed no matter how
   ! many processors are used. this is useful to test the reproducibility
   ! of the results on different platforms and for general testing.
!hjw do i=1,mype+1
!hjw    call rng_number(state,zion(2:6,1:mi))
!hjw enddo
   ! we now force the processes to wait for each other since the preceding
   ! loop will take an increasing amount of time 
!hjw call mpi_barrier(mpi_comm_world,ierr)
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))

  else
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))
  endif

! debug statements
!  do i=1,mi
!     write(mype+50,*)(i+mi*mype),zion(2:6,i)
!  enddo
!  close(mype+50)

end subroutine set_random_zion_1

function get_random_number_1()
  use global_parameters_1
  use rng_gtc_1
  implicit none
  real(8) :: get_random_number_1

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(get_random_number_1)
  else
     call rng_number(state,get_random_number_1)
  endif

end function get_random_number_1
module rng_gtc_2
  use rng1_2
! declarations and parameters for the portable random number generator
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state
end module rng_gtc_2

subroutine rand_num_gen_init_2
  use global_parameters_2
  use rng_gtc_2
  implicit none

  integer i,m,nsize
  integer,dimension(:),allocatable :: mget,mput

  if(rng_control==0)then
   ! **** use the intrinsic f90 random number generator *****
   ! initialize f90 random number generator
     call random_seed
     call random_seed(size=nsize)
     allocate(mget(nsize),mput(nsize))
     call random_seed(get=mget)
     if(mype==0)write(stdout,*)"random_seed=",nsize,mget
     do i=1,nsize 
        call system_clock(m)   !random initialization for collision
        if(irun==0)m=1         !same initialization
        mput(i)=111111*(mype+1)+m+mget(i)
     enddo
     call random_seed(put=mput)
     deallocate(mget,mput)

  ! ****** use charles karney's portable random number generator *****
  elseif(rng_control>0)then
   ! all the processors start with the same seed.
     call rng_set_seed(seed,(rng_control-1))  !set seed to (rng_control-1)
     if(mype==0)write(0,*)'seed is set to ',rng_print_seed_2(seed)
   ! initialize the random number generator
     call rng_init_2(seed,state)

  else
   ! set seed to current time
     call rng_set_seed(seed)
   ! advance seed according to process number
     call rng_step_seed_2(seed,mype)
   ! initialize random number generator
     call rng_init_2(seed,state)
  endif

end subroutine rand_num_gen_init_2

subroutine set_random_zion_2
  use global_parameters_2
  use particle_array_2
  use rng_gtc_2
  implicit none

  integer :: i,ierr

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(zion(2,1:mi))
     call random_number(zion(3,1:mi))
     call random_number(zion(4,1:mi))
     call random_number(zion(5,1:mi))
     call random_number(zion(6,1:mi))

  elseif(rng_control>0)then
   ! the following calls to rng_number insure that the series of random
   ! numbers generated will be the same for a given seed no matter how
   ! many processors are used. this is useful to test the reproducibility
   ! of the results on different platforms and for general testing.
!hjw do i=1,mype+1
!hjw    call rng_number(state,zion(2:6,1:mi))
!hjw enddo
   ! we now force the processes to wait for each other since the preceding
   ! loop will take an increasing amount of time 
!hjw call mpi_barrier(mpi_comm_world,ierr)
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))

  else
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))
  endif

! debug statements
!  do i=1,mi
!     write(mype+50,*)(i+mi*mype),zion(2:6,i)
!  enddo
!  close(mype+50)

end subroutine set_random_zion_2

function get_random_number_2()
  use global_parameters_2
  use rng_gtc_2
  implicit none
  real(8) :: get_random_number_2

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(get_random_number_2)
  else
     call rng_number(state,get_random_number_2)
  endif

end function get_random_number_2
module rng_gtc_3
  use rng1_3
! declarations and parameters for the portable random number generator
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state
end module rng_gtc_3

subroutine rand_num_gen_init_3
  use global_parameters_3
  use rng_gtc_3
  implicit none

  integer i,m,nsize
  integer,dimension(:),allocatable :: mget,mput

  if(rng_control==0)then
   ! **** use the intrinsic f90 random number generator *****
   ! initialize f90 random number generator
     call random_seed
     call random_seed(size=nsize)
     allocate(mget(nsize),mput(nsize))
     call random_seed(get=mget)
     if(mype==0)write(stdout,*)"random_seed=",nsize,mget
     do i=1,nsize 
        call system_clock(m)   !random initialization for collision
        if(irun==0)m=1         !same initialization
        mput(i)=111111*(mype+1)+m+mget(i)
     enddo
     call random_seed(put=mput)
     deallocate(mget,mput)

  ! ****** use charles karney's portable random number generator *****
  elseif(rng_control>0)then
   ! all the processors start with the same seed.
     call rng_set_seed(seed,(rng_control-1))  !set seed to (rng_control-1)
     if(mype==0)write(0,*)'seed is set to ',rng_print_seed_3(seed)
   ! initialize the random number generator
     call rng_init_3(seed,state)

  else
   ! set seed to current time
     call rng_set_seed(seed)
   ! advance seed according to process number
     call rng_step_seed_3(seed,mype)
   ! initialize random number generator
     call rng_init_3(seed,state)
  endif

end subroutine rand_num_gen_init_3

subroutine set_random_zion_3
  use global_parameters_3
  use particle_array_3
  use rng_gtc_3
  implicit none

  integer :: i,ierr

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(zion(2,1:mi))
     call random_number(zion(3,1:mi))
     call random_number(zion(4,1:mi))
     call random_number(zion(5,1:mi))
     call random_number(zion(6,1:mi))

  elseif(rng_control>0)then
   ! the following calls to rng_number insure that the series of random
   ! numbers generated will be the same for a given seed no matter how
   ! many processors are used. this is useful to test the reproducibility
   ! of the results on different platforms and for general testing.
!hjw do i=1,mype+1
!hjw    call rng_number(state,zion(2:6,1:mi))
!hjw enddo
   ! we now force the processes to wait for each other since the preceding
   ! loop will take an increasing amount of time 
!hjw call mpi_barrier(mpi_comm_world,ierr)
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))

  else
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))
  endif

! debug statements
!  do i=1,mi
!     write(mype+50,*)(i+mi*mype),zion(2:6,i)
!  enddo
!  close(mype+50)

end subroutine set_random_zion_3

function get_random_number_3()
  use global_parameters_3
  use rng_gtc_3
  implicit none
  real(8) :: get_random_number_3

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(get_random_number_3)
  else
     call rng_number(state,get_random_number_3)
  endif

end function get_random_number_3
module rng_gtc_4
  use rng1_4
! declarations and parameters for the portable random number generator
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state
end module rng_gtc_4

subroutine rand_num_gen_init_4
  use global_parameters_4
  use rng_gtc_4
  implicit none

  integer i,m,nsize
  integer,dimension(:),allocatable :: mget,mput

  if(rng_control==0)then
   ! **** use the intrinsic f90 random number generator *****
   ! initialize f90 random number generator
     call random_seed
     call random_seed(size=nsize)
     allocate(mget(nsize),mput(nsize))
     call random_seed(get=mget)
     if(mype==0)write(stdout,*)"random_seed=",nsize,mget
     do i=1,nsize 
        call system_clock(m)   !random initialization for collision
        if(irun==0)m=1         !same initialization
        mput(i)=111111*(mype+1)+m+mget(i)
     enddo
     call random_seed(put=mput)
     deallocate(mget,mput)

  ! ****** use charles karney's portable random number generator *****
  elseif(rng_control>0)then
   ! all the processors start with the same seed.
     call rng_set_seed(seed,(rng_control-1))  !set seed to (rng_control-1)
     if(mype==0)write(0,*)'seed is set to ',rng_print_seed_4(seed)
   ! initialize the random number generator
     call rng_init_4(seed,state)

  else
   ! set seed to current time
     call rng_set_seed(seed)
   ! advance seed according to process number
     call rng_step_seed_4(seed,mype)
   ! initialize random number generator
     call rng_init_4(seed,state)
  endif

end subroutine rand_num_gen_init_4

subroutine set_random_zion_4
  use global_parameters_4
  use particle_array_4
  use rng_gtc_4
  implicit none

  integer :: i,ierr

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(zion(2,1:mi))
     call random_number(zion(3,1:mi))
     call random_number(zion(4,1:mi))
     call random_number(zion(5,1:mi))
     call random_number(zion(6,1:mi))

  elseif(rng_control>0)then
   ! the following calls to rng_number insure that the series of random
   ! numbers generated will be the same for a given seed no matter how
   ! many processors are used. this is useful to test the reproducibility
   ! of the results on different platforms and for general testing.
!hjw do i=1,mype+1
!hjw    call rng_number(state,zion(2:6,1:mi))
!hjw enddo
   ! we now force the processes to wait for each other since the preceding
   ! loop will take an increasing amount of time 
!hjw call mpi_barrier(mpi_comm_world,ierr)
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))

  else
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))
  endif

! debug statements
!  do i=1,mi
!     write(mype+50,*)(i+mi*mype),zion(2:6,i)
!  enddo
!  close(mype+50)

end subroutine set_random_zion_4

function get_random_number_4()
  use global_parameters_4
  use rng_gtc_4
  implicit none
  real(8) :: get_random_number_4

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(get_random_number_4)
  else
     call rng_number(state,get_random_number_4)
  endif

end function get_random_number_4
module rng_gtc_5
  use rng1_5
! declarations and parameters for the portable random number generator
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state
end module rng_gtc_5

subroutine rand_num_gen_init_5
  use global_parameters_5
  use rng_gtc_5
  implicit none

  integer i,m,nsize
  integer,dimension(:),allocatable :: mget,mput

  if(rng_control==0)then
   ! **** use the intrinsic f90 random number generator *****
   ! initialize f90 random number generator
     call random_seed
     call random_seed(size=nsize)
     allocate(mget(nsize),mput(nsize))
     call random_seed(get=mget)
     if(mype==0)write(stdout,*)"random_seed=",nsize,mget
     do i=1,nsize 
        call system_clock(m)   !random initialization for collision
        if(irun==0)m=1         !same initialization
        mput(i)=111111*(mype+1)+m+mget(i)
     enddo
     call random_seed(put=mput)
     deallocate(mget,mput)

  ! ****** use charles karney's portable random number generator *****
  elseif(rng_control>0)then
   ! all the processors start with the same seed.
     call rng_set_seed(seed,(rng_control-1))  !set seed to (rng_control-1)
     if(mype==0)write(0,*)'seed is set to ',rng_print_seed_5(seed)
   ! initialize the random number generator
     call rng_init_5(seed,state)

  else
   ! set seed to current time
     call rng_set_seed(seed)
   ! advance seed according to process number
     call rng_step_seed_5(seed,mype)
   ! initialize random number generator
     call rng_init_5(seed,state)
  endif

end subroutine rand_num_gen_init_5

subroutine set_random_zion_5
  use global_parameters_5
  use particle_array_5
  use rng_gtc_5
  implicit none

  integer :: i,ierr

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(zion(2,1:mi))
     call random_number(zion(3,1:mi))
     call random_number(zion(4,1:mi))
     call random_number(zion(5,1:mi))
     call random_number(zion(6,1:mi))

  elseif(rng_control>0)then
   ! the following calls to rng_number insure that the series of random
   ! numbers generated will be the same for a given seed no matter how
   ! many processors are used. this is useful to test the reproducibility
   ! of the results on different platforms and for general testing.
!hjw do i=1,mype+1
!hjw    call rng_number(state,zion(2:6,1:mi))
!hjw enddo
   ! we now force the processes to wait for each other since the preceding
   ! loop will take an increasing amount of time 
!hjw call mpi_barrier(mpi_comm_world,ierr)
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))

  else
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))
  endif

! debug statements
!  do i=1,mi
!     write(mype+50,*)(i+mi*mype),zion(2:6,i)
!  enddo
!  close(mype+50)

end subroutine set_random_zion_5

function get_random_number_5()
  use global_parameters_5
  use rng_gtc_5
  implicit none
  real(8) :: get_random_number_5

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(get_random_number_5)
  else
     call rng_number(state,get_random_number_5)
  endif

end function get_random_number_5
module rng_gtc_6
  use rng1_6
! declarations and parameters for the portable random number generator
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state
end module rng_gtc_6

subroutine rand_num_gen_init_6
  use global_parameters_6
  use rng_gtc_6
  implicit none

  integer i,m,nsize
  integer,dimension(:),allocatable :: mget,mput

  if(rng_control==0)then
   ! **** use the intrinsic f90 random number generator *****
   ! initialize f90 random number generator
     call random_seed
     call random_seed(size=nsize)
     allocate(mget(nsize),mput(nsize))
     call random_seed(get=mget)
     if(mype==0)write(stdout,*)"random_seed=",nsize,mget
     do i=1,nsize 
        call system_clock(m)   !random initialization for collision
        if(irun==0)m=1         !same initialization
        mput(i)=111111*(mype+1)+m+mget(i)
     enddo
     call random_seed(put=mput)
     deallocate(mget,mput)

  ! ****** use charles karney's portable random number generator *****
  elseif(rng_control>0)then
   ! all the processors start with the same seed.
     call rng_set_seed(seed,(rng_control-1))  !set seed to (rng_control-1)
     if(mype==0)write(0,*)'seed is set to ',rng_print_seed_6(seed)
   ! initialize the random number generator
     call rng_init_6(seed,state)

  else
   ! set seed to current time
     call rng_set_seed(seed)
   ! advance seed according to process number
     call rng_step_seed_6(seed,mype)
   ! initialize random number generator
     call rng_init_6(seed,state)
  endif

end subroutine rand_num_gen_init_6

subroutine set_random_zion_6
  use global_parameters_6
  use particle_array_6
  use rng_gtc_6
  implicit none

  integer :: i,ierr

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(zion(2,1:mi))
     call random_number(zion(3,1:mi))
     call random_number(zion(4,1:mi))
     call random_number(zion(5,1:mi))
     call random_number(zion(6,1:mi))

  elseif(rng_control>0)then
   ! the following calls to rng_number insure that the series of random
   ! numbers generated will be the same for a given seed no matter how
   ! many processors are used. this is useful to test the reproducibility
   ! of the results on different platforms and for general testing.
!hjw do i=1,mype+1
!hjw    call rng_number(state,zion(2:6,1:mi))
!hjw enddo
   ! we now force the processes to wait for each other since the preceding
   ! loop will take an increasing amount of time 
!hjw call mpi_barrier(mpi_comm_world,ierr)
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))

  else
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))
  endif

! debug statements
!  do i=1,mi
!     write(mype+50,*)(i+mi*mype),zion(2:6,i)
!  enddo
!  close(mype+50)

end subroutine set_random_zion_6

function get_random_number_6()
  use global_parameters_6
  use rng_gtc_6
  implicit none
  real(8) :: get_random_number_6

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(get_random_number_6)
  else
     call rng_number(state,get_random_number_6)
  endif

end function get_random_number_6
module rng_gtc_7
  use rng1_7
! declarations and parameters for the portable random number generator
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state
end module rng_gtc_7

subroutine rand_num_gen_init_7
  use global_parameters_7
  use rng_gtc_7
  implicit none

  integer i,m,nsize
  integer,dimension(:),allocatable :: mget,mput

  if(rng_control==0)then
   ! **** use the intrinsic f90 random number generator *****
   ! initialize f90 random number generator
     call random_seed
     call random_seed(size=nsize)
     allocate(mget(nsize),mput(nsize))
     call random_seed(get=mget)
     if(mype==0)write(stdout,*)"random_seed=",nsize,mget
     do i=1,nsize 
        call system_clock(m)   !random initialization for collision
        if(irun==0)m=1         !same initialization
        mput(i)=111111*(mype+1)+m+mget(i)
     enddo
     call random_seed(put=mput)
     deallocate(mget,mput)

  ! ****** use charles karney's portable random number generator *****
  elseif(rng_control>0)then
   ! all the processors start with the same seed.
     call rng_set_seed(seed,(rng_control-1))  !set seed to (rng_control-1)
     if(mype==0)write(0,*)'seed is set to ',rng_print_seed_7(seed)
   ! initialize the random number generator
     call rng_init_7(seed,state)

  else
   ! set seed to current time
     call rng_set_seed(seed)
   ! advance seed according to process number
     call rng_step_seed_7(seed,mype)
   ! initialize random number generator
     call rng_init_7(seed,state)
  endif

end subroutine rand_num_gen_init_7

subroutine set_random_zion_7
  use global_parameters_7
  use particle_array_7
  use rng_gtc_7
  implicit none

  integer :: i,ierr

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(zion(2,1:mi))
     call random_number(zion(3,1:mi))
     call random_number(zion(4,1:mi))
     call random_number(zion(5,1:mi))
     call random_number(zion(6,1:mi))

  elseif(rng_control>0)then
   ! the following calls to rng_number insure that the series of random
   ! numbers generated will be the same for a given seed no matter how
   ! many processors are used. this is useful to test the reproducibility
   ! of the results on different platforms and for general testing.
!hjw do i=1,mype+1
!hjw    call rng_number(state,zion(2:6,1:mi))
!hjw enddo
   ! we now force the processes to wait for each other since the preceding
   ! loop will take an increasing amount of time 
!hjw call mpi_barrier(mpi_comm_world,ierr)
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))

  else
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))
  endif

! debug statements
!  do i=1,mi
!     write(mype+50,*)(i+mi*mype),zion(2:6,i)
!  enddo
!  close(mype+50)

end subroutine set_random_zion_7

function get_random_number_7()
  use global_parameters_7
  use rng_gtc_7
  implicit none
  real(8) :: get_random_number_7

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(get_random_number_7)
  else
     call rng_number(state,get_random_number_7)
  endif

end function get_random_number_7
module rng_gtc_8
  use rng1_8
! declarations and parameters for the portable random number generator
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state
end module rng_gtc_8

subroutine rand_num_gen_init_8
  use global_parameters_8
  use rng_gtc_8
  implicit none

  integer i,m,nsize
  integer,dimension(:),allocatable :: mget,mput

  if(rng_control==0)then
   ! **** use the intrinsic f90 random number generator *****
   ! initialize f90 random number generator
     call random_seed
     call random_seed(size=nsize)
     allocate(mget(nsize),mput(nsize))
     call random_seed(get=mget)
     if(mype==0)write(stdout,*)"random_seed=",nsize,mget
     do i=1,nsize 
        call system_clock(m)   !random initialization for collision
        if(irun==0)m=1         !same initialization
        mput(i)=111111*(mype+1)+m+mget(i)
     enddo
     call random_seed(put=mput)
     deallocate(mget,mput)

  ! ****** use charles karney's portable random number generator *****
  elseif(rng_control>0)then
   ! all the processors start with the same seed.
     call rng_set_seed(seed,(rng_control-1))  !set seed to (rng_control-1)
     if(mype==0)write(0,*)'seed is set to ',rng_print_seed_8(seed)
   ! initialize the random number generator
     call rng_init_8(seed,state)

  else
   ! set seed to current time
     call rng_set_seed(seed)
   ! advance seed according to process number
     call rng_step_seed_8(seed,mype)
   ! initialize random number generator
     call rng_init_8(seed,state)
  endif

end subroutine rand_num_gen_init_8

subroutine set_random_zion_8
  use global_parameters_8
  use particle_array_8
  use rng_gtc_8
  implicit none

  integer :: i,ierr

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(zion(2,1:mi))
     call random_number(zion(3,1:mi))
     call random_number(zion(4,1:mi))
     call random_number(zion(5,1:mi))
     call random_number(zion(6,1:mi))

  elseif(rng_control>0)then
   ! the following calls to rng_number insure that the series of random
   ! numbers generated will be the same for a given seed no matter how
   ! many processors are used. this is useful to test the reproducibility
   ! of the results on different platforms and for general testing.
!hjw do i=1,mype+1
!hjw    call rng_number(state,zion(2:6,1:mi))
!hjw enddo
   ! we now force the processes to wait for each other since the preceding
   ! loop will take an increasing amount of time 
!hjw call mpi_barrier(mpi_comm_world,ierr)
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))

  else
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))
  endif

! debug statements
!  do i=1,mi
!     write(mype+50,*)(i+mi*mype),zion(2:6,i)
!  enddo
!  close(mype+50)

end subroutine set_random_zion_8

function get_random_number_8()
  use global_parameters_8
  use rng_gtc_8
  implicit none
  real(8) :: get_random_number_8

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(get_random_number_8)
  else
     call rng_number(state,get_random_number_8)
  endif

end function get_random_number_8
module rng_gtc_9
  use rng1_9
! declarations and parameters for the portable random number generator
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state
end module rng_gtc_9

subroutine rand_num_gen_init_9
  use global_parameters_9
  use rng_gtc_9
  implicit none

  integer i,m,nsize
  integer,dimension(:),allocatable :: mget,mput

  if(rng_control==0)then
   ! **** use the intrinsic f90 random number generator *****
   ! initialize f90 random number generator
     call random_seed
     call random_seed(size=nsize)
     allocate(mget(nsize),mput(nsize))
     call random_seed(get=mget)
     if(mype==0)write(stdout,*)"random_seed=",nsize,mget
     do i=1,nsize 
        call system_clock(m)   !random initialization for collision
        if(irun==0)m=1         !same initialization
        mput(i)=111111*(mype+1)+m+mget(i)
     enddo
     call random_seed(put=mput)
     deallocate(mget,mput)

  ! ****** use charles karney's portable random number generator *****
  elseif(rng_control>0)then
   ! all the processors start with the same seed.
     call rng_set_seed(seed,(rng_control-1))  !set seed to (rng_control-1)
     if(mype==0)write(0,*)'seed is set to ',rng_print_seed_9(seed)
   ! initialize the random number generator
     call rng_init_9(seed,state)

  else
   ! set seed to current time
     call rng_set_seed(seed)
   ! advance seed according to process number
     call rng_step_seed_9(seed,mype)
   ! initialize random number generator
     call rng_init_9(seed,state)
  endif

end subroutine rand_num_gen_init_9

subroutine set_random_zion_9
  use global_parameters_9
  use particle_array_9
  use rng_gtc_9
  implicit none

  integer :: i,ierr

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(zion(2,1:mi))
     call random_number(zion(3,1:mi))
     call random_number(zion(4,1:mi))
     call random_number(zion(5,1:mi))
     call random_number(zion(6,1:mi))

  elseif(rng_control>0)then
   ! the following calls to rng_number insure that the series of random
   ! numbers generated will be the same for a given seed no matter how
   ! many processors are used. this is useful to test the reproducibility
   ! of the results on different platforms and for general testing.
!hjw do i=1,mype+1
!hjw    call rng_number(state,zion(2:6,1:mi))
!hjw enddo
   ! we now force the processes to wait for each other since the preceding
   ! loop will take an increasing amount of time 
!hjw call mpi_barrier(mpi_comm_world,ierr)
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))

  else
     call rng_number(state,zion(2,1:mi))
     call rng_number(state,zion(3,1:mi))
     call rng_number(state,zion(4,1:mi))
     call rng_number(state,zion(5,1:mi))
     call rng_number(state,zion(6,1:mi))
  endif

! debug statements
!  do i=1,mi
!     write(mype+50,*)(i+mi*mype),zion(2:6,i)
!  enddo
!  close(mype+50)

end subroutine set_random_zion_9

function get_random_number_9()
  use global_parameters_9
  use rng_gtc_9
  implicit none
  real(8) :: get_random_number_9

  if(rng_control==0)then
   ! use fortran's intrinsic random number generator
     call random_number(get_random_number_9)
  else
     call rng_number(state,get_random_number_9)
  endif

end function get_random_number_9
