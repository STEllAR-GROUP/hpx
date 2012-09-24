subroutine rand_num_gen_init(rng_control,mype,irun,stdout)
  use precision
  use rng
  implicit none
  integer mype,rng_control,irun,stdout
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state

  integer i,m,nsize
  integer,dimension(:),allocatable :: mget,mput

  if(rng_control==0)then
   ! **** Use the intrinsic F90 random number generator *****
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

  ! ****** Use Charles Karney's portable random number generator *****
  elseif(rng_control>0)then
   ! All the processors start with the same seed.
     call rng_set_seed(seed,(rng_control-1))  !Set seed to (rng_control-1)
     if(mype==0)write(0,*)'Seed is set to ',rng_print_seed(seed)
   ! Initialize the random number generator
     call rng_init(seed,state)

  else
   ! Set seed to current time
     call rng_set_seed(seed)
   ! Advance seed according to process number
     call rng_step_seed(seed,mype)
   ! Initialize random number generator
     call rng_init(seed,state)
  endif

end subroutine rand_num_gen_init

subroutine set_random_zion(mi,rng_control,zion)
  use precision
  use rng
  implicit none

  integer mi,rng_control
  real(wp),dimension(:,:),allocatable :: zion
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state

  ! FIXME MATT
  ! for the moment, there are still threadsafe issues with rng_number.  
  ! so use the intrinsic generator for now
  call random_number(zion(2,1:mi))
  call random_number(zion(3,1:mi))
  call random_number(zion(4,1:mi))
  call random_number(zion(5,1:mi))
  call random_number(zion(6,1:mi))
  return

  if(rng_control==0)then
   ! Use Fortran's intrinsic random number generator
     call random_number(zion(2,1:mi))
     call random_number(zion(3,1:mi))
     call random_number(zion(4,1:mi))
     call random_number(zion(5,1:mi))
     call random_number(zion(6,1:mi))

  elseif(rng_control>0)then
   ! The following calls to rng_number insure that the series of random
   ! numbers generated will be the same for a given seed no matter how
   ! many processors are used. This is useful to test the reproducibility
   ! of the results on different platforms and for general testing.
!hjw do i=1,mype+1
!hjw    call rng_number(state,zion(2:6,1:mi))
!hjw enddo
   ! We now force the processes to wait for each other since the preceding
   ! loop will take an increasing amount of time 
!hjw call MPI_BARRIER(MPI_COMM_WORLD,ierr)
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

! Debug statements
!  do i=1,mi
!     write(mype+50,*)(i+mi*mype),zion(2:6,i)
!  enddo
!  close(mype+50)

end subroutine set_random_zion

function get_random_number(rng_control)
  use precision
  use rng
  implicit none
  real(wp) :: get_random_number
  integer rng_control
  integer,dimension(rng_s) :: seed
  type(rng_state) :: state

  if(rng_control==0)then
   ! Use Fortran's intrinsic random number generator
     call random_number(get_random_number)
  else
     call rng_number(state,get_random_number)
  endif

end function get_random_number
