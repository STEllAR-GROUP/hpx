! f90 interface to random number routine
! charles karney <karney@pppl.gov> 1999-09-30 16:58 -0400

 



 














module rngdef_0
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=kind(0.0)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(kind=double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef_0

module rng1_0
  ! the main module 
  use rngdef_0, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0_0
     subroutine rng_number_d1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_0
     subroutine rng_number_d2_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_0
     subroutine rng_number_d3_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3_0

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_number_s0_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0_0
     subroutine rng_number_s1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_0
     subroutine rng_number_s2_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_0
     subroutine rng_number_s3_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3_0

  end interface
  interface rng_gauss
     ! gaussian random numbers
     subroutine rng_gauss_d0_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0_0
     subroutine rng_gauss_d1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_0
     subroutine rng_gauss_d2_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_0
     subroutine rng_gauss_d3_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3_0

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_gauss_s0_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0_0
     subroutine rng_gauss_s1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_0
     subroutine rng_gauss_s2_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_0
     subroutine rng_gauss_s3_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3_0

  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time_0(seed,time)
       use rngdef_0
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time_0
     subroutine rng_set_seed_int_0(seed,n)
       use rngdef_0
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int_0
     subroutine rng_set_seed_string_0(seed,string)
       use rngdef_0
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string_0
  end interface

  interface
     function rng_print_seed_0(seed)
       ! return the seed in a string representation
       use rngdef_0
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed_0
     end function rng_print_seed_0
     subroutine rng_init_0(seed,state)
       ! initialize the state from the seed
       use rngdef_0
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init_0
     subroutine rng_step_seed_0(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef_0
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed_0
  end interface
end module rng1_0
 



 













module rngf77_0
  ! interface for the original f77 routines
  use rngdef_0
  implicit none
  interface
     function random1_0(ri,ra)
       use rngdef_0
       implicit none
       real(kind=double) :: random1_0
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random1_0

     function srandom1_0_0(ri,ra)
       use rngdef_0
       implicit none
       real(kind=single) :: srandom1_0_0
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom1_0_0

     subroutine random_array_0(y,n,ri,ra)
       use rngdef_0
       implicit none
       integer, intent(in) :: n
       real(kind=double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array_0

     subroutine srandom_array_0_0(y,n,ri,ra)
       use rngdef_0
       implicit none
       integer, intent(in) :: n
       real(kind=single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array_0_0

     subroutine rand_batch_0(ri,ra)
       use rngdef_0
       implicit none
       integer, intent(inout) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch_0
     subroutine random_init_0(seed,ri,ra)
       use rngdef_0
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init_0
     subroutine decimal_to_seed_0(decimal,seed)
       use rngdef_0
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed_0
     subroutine string_to_seed_0(string,seed)
       use rngdef_0
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed_0
     subroutine set_random_seed_0(time,seed)
       use rngdef_0
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed_0
     subroutine seed_to_decimal_0(seed,decimal)
       use rngdef_0
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal_0
     subroutine next_seed_03_0(n0,n1,n2,seed)
       use rngdef_0
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_03_0
     subroutine next_seed_0(n0,seed)
       use rngdef_0
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_0
  end interface
end module rngf77_0

subroutine rng_init_0(seed,state)
  use rngf77_0
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init_0(seed,state%index,state%array)
  return
end subroutine rng_init_0

subroutine rng_step_seed_0(seed,n0,n1,n2)
  use rngf77_0
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed_0(1,seed)
  else
     if (present(n0)) then
        m0=n0
     else
        m0=0
     endif
     if (present(n1)) then
        m1=n1
     else
        m1=0
     endif
     if (present(n2)) then
        m2=n2
     else
        m2=0
     endif
     call next_seed_03_0(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed_0

function rng_print_seed_0(seed)
  use rngf77_0
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed_0
  call seed_to_decimal_0(seed,rng_print_seed_0)
  return
end function rng_print_seed_0

subroutine rng_number_d0_0(state,x)
  use rngf77_0
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch_0(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0_0


subroutine rng_number_s0_0(state,x)
  use rngf77_0
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch_0(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0_0



subroutine rng_number_d1_0(state,x)
  use rngf77_0
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=state%array(i+state%index)+ulp2
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_0(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1_0


subroutine rng_number_s1_0(state,x)
  use rngf77_0
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=(int(mult*state%array(i+state%index))+0.5)*ulps
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_0(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1_0


subroutine rng_number_d2_0(state,x)
  use rngdef_0
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_0
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1_0(state,x(:,i))
  end do
end subroutine rng_number_d2_0


subroutine rng_number_s2_0(state,x)
  use rngdef_0
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_0
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1_0(state,x(:,i))
  end do
end subroutine rng_number_s2_0


subroutine rng_number_d3_0(state,x)
  use rngdef_0
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_0
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2_0(state,x(:,:,i))
  end do
end subroutine rng_number_d3_0


subroutine rng_number_s3_0(state,x)
  use rngdef_0
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_0
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2_0(state,x(:,:,i))
  end do
end subroutine rng_number_s3_0


subroutine rng_gauss_d1_0(state,x)
  use rngf77_0
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_0
  end interface
  integer :: i
  real(kind=double), parameter :: pi= 3.14159265358979323846264338328d0 
  real(kind=double) :: theta,z
  call rng_number_d1_0(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0d0 *x(i)- 1.0d0 )
     z=sqrt(- 2.0d0 *log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0d0 *x(ubound(x,1))- 1.0d0 )
  z=sqrt(- 2.0d0 *log(random1_0(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1_0

subroutine rng_gauss_d0_0(state,x)
  use rngdef_0
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_0
  end interface
  real(kind=double), dimension(1) :: y
  call rng_gauss_d1_0(state, y)
  x=y(1)
end subroutine rng_gauss_d0_0


subroutine rng_gauss_s1_0(state,x)
  use rngf77_0
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_0
  end interface
  integer :: i
  real(kind=single), parameter :: pi= 3.14159265358979323846264338328 
  real(kind=single) :: theta,z
  call rng_number_s1_0(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0*x(i)-1.0)
     z=sqrt(-2.0*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0*x(ubound(x,1))-1.0)
  z=sqrt(-2.0*log(srandom1_0_0(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1_0

subroutine rng_gauss_s0_0(state,x)
  use rngdef_0
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_0
  end interface
  real(kind=single), dimension(1) :: y
  call rng_gauss_s1_0(state, y)
  x=y(1)
end subroutine rng_gauss_s0_0


subroutine rng_gauss_d2_0(state,x)
  use rngdef_0
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_0
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1_0(state,x(:,i))
  end do
end subroutine rng_gauss_d2_0


subroutine rng_gauss_s2_0(state,x)
  use rngdef_0
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_0
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1_0(state,x(:,i))
  end do
end subroutine rng_gauss_s2_0


subroutine rng_gauss_d3_0(state,x)
  use rngdef_0
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2_0(state,x)
       use rngdef_0
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_0
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2_0(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3_0


subroutine rng_gauss_s3_0(state,x)
  use rngdef_0
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2_0(state,x)
       use rngdef_0
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_0
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2_0(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3_0


subroutine rng_set_seed_time_0(seed,time)
  use rngf77_0
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed_0(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed_0(ltime,seed)
  end if
end subroutine rng_set_seed_time_0

subroutine rng_set_seed_int_0(seed,n)
  use rngdef_0
  implicit none
  integer, dimension(0:rng_s-1), intent(out) :: seed
  integer, intent(in) :: n
  integer, parameter :: b=2**14
  integer :: m,i,sign
  if (n.lt.0) then
     seed=b-1
     m=-n-1
     sign=-1
  else
     seed=0
     m=n
     sign=1
  end if
  i=0
10 continue
  if (m.eq.0) return
  seed(i)=seed(i)+sign*mod(m,b)
  m=int(m/b)
  i=i+1
  goto 10
end subroutine rng_set_seed_int_0

subroutine rng_set_seed_string_0(seed,string)
  use rngf77_0
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed_0(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed_0(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string_0
 



 













! version 2.0 of random number routines
! author: charles karney <karney@princeton.edu>
! date: 1999-10-05 14:33 -0400
!
      function random1_0(ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      double precision  random1_0
      real srandom1_0_0
      integer ri
      double precision  ra(0:100-1)
      external rand_batch_0

      if(ri.ge.100)then
      call rand_batch_0(ri,ra(0))
      end if
      random1_0=ra(ri)+ulp2
      ri=ri+1
      return

      entry srandom1_0_0(ri,ra)
      if(ri.ge.100)then
      call rand_batch_0(ri,ra(0))
      end if

      srandom1_0_0=(int(mult*ra(ri))+0.5)*ulps

      ri=ri+1
      return
      end
!
      subroutine random_array_0(y,n,ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      integer n
      double precision  y(0:n-1)

      real ys(0:n-1)

      integer ri
      double precision  ra(0:100-1)
      integer i,k,j
      external rand_batch_0

      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      y(i)=ra(i+ri)+ulp2
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_0(ri,ra(0))
      do i=j,min(j+100,n)-1
      y(i)=ra(i-j+ri)+ulp2

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      entry srandom_array_0_0(ys,n,ri,ra)
      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      ys(i)=(int(mult*ra(i+ri))+0.5)*ulps
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_0(ri,ra(0))
      do i=j,min(j+100,n)-1
      ys(i)=(int(mult*ra(i-j+ri))+0.5)*ulps

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      end
!
      subroutine rand_batch_0(ri,ra)
      implicit none
      integer ri
      double precision  ra(0:100-1)
      integer i
      double precision  w(0:1009-100-1)
      double precision  tmp
      do i=0,63-1
      tmp=ra(i)+ra(i+100-63)

      w(i)=tmp-int(tmp)

      end do
      do i=63,100-1
      tmp=ra(i)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=100,1009-100-1
      tmp=w(i-100)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=1009-100,1009-100+63-1
      tmp=w(i-100)+w(i-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      do i=1009-100+63,1009-1
      tmp=w(i-100)+ra(i-1009+100-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      ri=0
      return
      end
!
      subroutine random_init_0(seed,ri,ra)
      implicit none
      integer b
      double precision  del,ulp
      parameter(b=2**14,del= 2.0d0 **(-14),ulp= 2.0d0 **(-47))
      integer a0,a1,a2,a3,a4,a5,a6,c0
      parameter(a0=15661,a1=678,a2=724,a3=5245,a4=13656,a5=11852,a6=29)
      parameter(c0=1)
      integer seed(0:8-1)
      integer ri
      double precision  ra(0:100-1)
      integer i,j,s(0:8-1)
      logical odd
      integer z(0:8-1),t
      do i=0,8-1
      s(i)=seed(i)
      end do
      odd=mod(s(7),2).ne.0
      ra(0)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      do j=1,100-1
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      odd=odd.or.(mod(s(7),2).ne.0)
      ra(j)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      end do
      ri=100
      if(odd)return
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      j=int((s(8-1)*100)/b)
      ra(j)=ra(j)+(ulp)
      return
      end
!
      subroutine decimal_to_seed_0(decimal,seed)
      implicit none
      character*(*)decimal
      integer seed(0:8-1)
      external rand_axc_0
      integer i,ten(0:8-1),c(0:8-1),ch
      data ten/10,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(decimal)
      ch=ichar(decimal(i:i))
      if(ch.ge.ichar('0').and.ch.le.ichar('9'))then
      c(0)=ch-ichar('0')
      call rand_axc_0(ten,seed,c)
      end if
      end do
      return
      end
!
      subroutine string_to_seed_0(string,seed)
      implicit none
      integer b
      parameter(b=2**14)
      character*(*)string
      integer seed(0:8-1)
      external rand_axc_0
      integer t,i,k,unity(0:8-1),c(0:8-1),ch
      data unity/1,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(string)
      ch=ichar(string(i:i))
      if(ch.gt.ichar(' ').and.ch.lt.127)then
      t=mod(seed(0),2)*(b/2)
      do k=0,8-1
      seed(k)=int(seed(k)/2)
      if(k.lt.8-1)then
      seed(k)=seed(k)+(mod(seed(k+1),2)*(b/2))
      else
      seed(k)=seed(k)+(t)
      end if
      end do
      c(0)=ch
      call rand_axc_0(unity,seed,c)
      end if
      end do
      return
      end
!
      subroutine set_random_seed_0(time,seed)
      implicit none
      integer time(8)
      integer seed(0:8-1)
      character*21 c
      external decimal_to_seed_0
      c=' '
      write(c(1:8),'(i4.4,2i2.2)')time(1),time(2),time(3)
      write(c(9:12),'(i1.1,i3.3)') (1-sign(1,time(4)))/2,abs(time(4))
      write(c(13:21),'(3i2.2,i3.3)')time(5),time(6),time(7),time(8)
      call decimal_to_seed_0(c,seed)
      return
      end
!
      subroutine seed_to_decimal_0(seed,decimal)
      implicit none
      integer pow,decbase,b
      parameter(pow=4,decbase=10**pow,b=2**14)
      character*(*)decimal
      integer seed(0:8-1)
      integer z(0:8-1),i,t,j,k
      character*36 str
      k=-1
      do i=0,8-1
      z(i)=seed(i)
      if(z(i).gt.0)k=i
      end do
      str=' '
      i=9
90000 continue
      i=i-1
      t=0
      do j=k,0,-1
      z(j)=z(j)+t*b
      t=mod(z(j),decbase)
      z(j)=int(z(j)/decbase)
      end do
      if(z(max(0,k)).eq.0)k=k-1
      j=pow*(i+1)
      if(k.ge.0)then
      str(j-(pow-1):j)='0000'
      else
      str(j-(pow-1):j)='   0'
      end if
90001 continue
      if(t.eq.0)goto 90010
      str(j:j)=char(ichar('0')+mod(t,10))
      j=j-1
      t=int(t/10)
      goto 90001
90010 continue
      if(k.ge.0)goto 90000
      if(len(decimal).gt.len(str))then
      decimal(:len(decimal)-len(str))=' '
      decimal(len(decimal)-len(str)+1:)=str
      else
      decimal=str(len(str)-len(decimal)+1:)
      end if
      return
      end
!
      subroutine rand_next_seed_0(n,ax,cx,y)
      implicit none
      integer n,ax(0:8-1),cx(0:8-1)
      integer y(0:8-1)
      external rand_axc_0
      integer a(0:8-1),c(0:8-1),z(0:8-1),t(0:8-1),m,i
      data z/8*0/
      if(n.eq.0)return
      m=n
      do i=0,8-1
      a(i)=ax(i)
      c(i)=cx(i)
      end do
90000 continue
      if(mod(m,2).gt.0)then
      call rand_axc_0(a,y,c)
      end if
      m=int(m/2)
      if(m.eq.0)return
      do i=0,8-1
      t(i)=c(i)
      end do
      call rand_axc_0(a,c,t)
      do i=0,8-1
      t(i)=a(i)
      end do
      call rand_axc_0(t,a,z)
      goto 90000
      end
!
      subroutine next_seed_03_0(n0,n1,n2,seed)
      implicit none
      integer n0,n1,n2
      integer seed(0:8-1)
      external rand_next_seed_0
      integer af0(0:8-1),cf0(0:8-1)
      integer ab0(0:8-1),cb0(0:8-1)
      integer af1(0:8-1),cf1(0:8-1)
      integer ab1(0:8-1),cb1(0:8-1)
      integer af2(0:8-1),cf2(0:8-1)
      integer ab2(0:8-1),cb2(0:8-1)
      data af0/15741,8689,9280,4732,12011,7130,6824,12302/
      data cf0/16317,10266,1198,331,10769,8310,2779,13880/
      data ab0/9173,9894,15203,15379,7981,2280,8071,429/
      data cb0/8383,3616,597,12724,15663,9639,187,4866/
      data af1/8405,4808,3603,6718,13766,9243,10375,12108/
      data cf1/13951,7170,9039,11206,8706,14101,1864,15191/
      data ab1/6269,3240,9759,7130,15320,14399,3675,1380/
      data cb1/15357,5843,6205,16275,8838,12132,2198,10330/
      data af2/445,10754,1869,6593,385,12498,14501,7383/
      data cf2/2285,8057,3864,10235,1805,10614,9615,15522/
      data ab2/405,4903,2746,1477,3263,13564,8139,2362/
      data cb2/8463,575,5876,2220,4924,1701,9060,5639/
      if(n2.gt.0)then
      call rand_next_seed_0(n2,af2,cf2,seed)
      else if(n2.lt.0)then
      call rand_next_seed_0(-n2,ab2,cb2,seed)
      end if
      if(n1.gt.0)then
      call rand_next_seed_0(n1,af1,cf1,seed)
      else if(n1.lt.0)then
      call rand_next_seed_0(-n1,ab1,cb1,seed)
      end if
      entry next_seed_0(n0,seed)
      if(n0.gt.0)then
      call rand_next_seed_0(n0,af0,cf0,seed)
      else if(n0.lt.0)then
      call rand_next_seed_0(-n0,ab0,cb0,seed)
      end if
      return
      end
!
      subroutine rand_axc_0(a,x,c)
      implicit none
      integer b
      parameter(b=2**14)
      integer a(0:8-1),c(0:8-1)
      integer x(0:8-1)
      integer z(0:8-1),i,j,t
      do i=0,8-1
      z(i)=c(i)
      end do
      do j=0,8-1
      do i=j,8-1
      z(i)=z(i)+(a(j)*x(i-j))
      end do
      end do
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      x(i)=mod(t,b)
      end do
      return
      end
!
! f90 interface to random number routine
! charles karney <karney@pppl.gov> 1999-09-30 16:58 -0400

 



 














module rngdef_1
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=kind(0.0)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(kind=double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef_1

module rng1_1
  ! the main module 
  use rngdef_1, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0_1
     subroutine rng_number_d1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_1
     subroutine rng_number_d2_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_1
     subroutine rng_number_d3_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3_1

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_number_s0_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0_1
     subroutine rng_number_s1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_1
     subroutine rng_number_s2_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_1
     subroutine rng_number_s3_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3_1

  end interface
  interface rng_gauss
     ! gaussian random numbers
     subroutine rng_gauss_d0_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0_1
     subroutine rng_gauss_d1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_1
     subroutine rng_gauss_d2_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_1
     subroutine rng_gauss_d3_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3_1

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_gauss_s0_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0_1
     subroutine rng_gauss_s1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_1
     subroutine rng_gauss_s2_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_1
     subroutine rng_gauss_s3_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3_1

  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time_1(seed,time)
       use rngdef_1
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time_1
     subroutine rng_set_seed_int_1(seed,n)
       use rngdef_1
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int_1
     subroutine rng_set_seed_string_1(seed,string)
       use rngdef_1
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string_1
  end interface

  interface
     function rng_print_seed_1(seed)
       ! return the seed in a string representation
       use rngdef_1
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed_1
     end function rng_print_seed_1
     subroutine rng_init_1(seed,state)
       ! initialize the state from the seed
       use rngdef_1
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init_1
     subroutine rng_step_seed_1(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef_1
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed_1
  end interface
end module rng1_1
 



 













module rngf77_1
  ! interface for the original f77 routines
  use rngdef_1
  implicit none
  interface
     function random1_1(ri,ra)
       use rngdef_1
       implicit none
       real(kind=double) :: random1_1
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random1_1

     function srandom1_1_1(ri,ra)
       use rngdef_1
       implicit none
       real(kind=single) :: srandom1_1_1
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom1_1_1

     subroutine random_array_1(y,n,ri,ra)
       use rngdef_1
       implicit none
       integer, intent(in) :: n
       real(kind=double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array_1

     subroutine srandom_array_1_1(y,n,ri,ra)
       use rngdef_1
       implicit none
       integer, intent(in) :: n
       real(kind=single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array_1_1

     subroutine rand_batch_1(ri,ra)
       use rngdef_1
       implicit none
       integer, intent(inout) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch_1
     subroutine random_init_1(seed,ri,ra)
       use rngdef_1
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init_1
     subroutine decimal_to_seed_1(decimal,seed)
       use rngdef_1
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed_1
     subroutine string_to_seed_1(string,seed)
       use rngdef_1
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed_1
     subroutine set_random_seed_1(time,seed)
       use rngdef_1
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed_1
     subroutine seed_to_decimal_1(seed,decimal)
       use rngdef_1
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal_1
     subroutine next_seed_13_1(n0,n1,n2,seed)
       use rngdef_1
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_13_1
     subroutine next_seed_1(n0,seed)
       use rngdef_1
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_1
  end interface
end module rngf77_1

subroutine rng_init_1(seed,state)
  use rngf77_1
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init_1(seed,state%index,state%array)
  return
end subroutine rng_init_1

subroutine rng_step_seed_1(seed,n0,n1,n2)
  use rngf77_1
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed_1(1,seed)
  else
     if (present(n0)) then
        m0=n0
     else
        m0=0
     endif
     if (present(n1)) then
        m1=n1
     else
        m1=0
     endif
     if (present(n2)) then
        m2=n2
     else
        m2=0
     endif
     call next_seed_13_1(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed_1

function rng_print_seed_1(seed)
  use rngf77_1
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed_1
  call seed_to_decimal_1(seed,rng_print_seed_1)
  return
end function rng_print_seed_1

subroutine rng_number_d0_1(state,x)
  use rngf77_1
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch_1(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0_1


subroutine rng_number_s0_1(state,x)
  use rngf77_1
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch_1(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0_1



subroutine rng_number_d1_1(state,x)
  use rngf77_1
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=state%array(i+state%index)+ulp2
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_1(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1_1


subroutine rng_number_s1_1(state,x)
  use rngf77_1
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=(int(mult*state%array(i+state%index))+0.5)*ulps
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_1(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1_1


subroutine rng_number_d2_1(state,x)
  use rngdef_1
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1_1(state,x(:,i))
  end do
end subroutine rng_number_d2_1


subroutine rng_number_s2_1(state,x)
  use rngdef_1
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1_1(state,x(:,i))
  end do
end subroutine rng_number_s2_1


subroutine rng_number_d3_1(state,x)
  use rngdef_1
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_1
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2_1(state,x(:,:,i))
  end do
end subroutine rng_number_d3_1


subroutine rng_number_s3_1(state,x)
  use rngdef_1
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_1
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2_1(state,x(:,:,i))
  end do
end subroutine rng_number_s3_1


subroutine rng_gauss_d1_1(state,x)
  use rngf77_1
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_1
  end interface
  integer :: i
  real(kind=double), parameter :: pi= 3.14159265358979323846264338328d0 
  real(kind=double) :: theta,z
  call rng_number_d1_1(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0d0 *x(i)- 1.0d0 )
     z=sqrt(- 2.0d0 *log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0d0 *x(ubound(x,1))- 1.0d0 )
  z=sqrt(- 2.0d0 *log(random1_1(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1_1

subroutine rng_gauss_d0_1(state,x)
  use rngdef_1
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_1
  end interface
  real(kind=double), dimension(1) :: y
  call rng_gauss_d1_1(state, y)
  x=y(1)
end subroutine rng_gauss_d0_1


subroutine rng_gauss_s1_1(state,x)
  use rngf77_1
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_1
  end interface
  integer :: i
  real(kind=single), parameter :: pi= 3.14159265358979323846264338328 
  real(kind=single) :: theta,z
  call rng_number_s1_1(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0*x(i)-1.0)
     z=sqrt(-2.0*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0*x(ubound(x,1))-1.0)
  z=sqrt(-2.0*log(srandom1_1_1(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1_1

subroutine rng_gauss_s0_1(state,x)
  use rngdef_1
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_1
  end interface
  real(kind=single), dimension(1) :: y
  call rng_gauss_s1_1(state, y)
  x=y(1)
end subroutine rng_gauss_s0_1


subroutine rng_gauss_d2_1(state,x)
  use rngdef_1
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1_1(state,x(:,i))
  end do
end subroutine rng_gauss_d2_1


subroutine rng_gauss_s2_1(state,x)
  use rngdef_1
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1_1(state,x(:,i))
  end do
end subroutine rng_gauss_s2_1


subroutine rng_gauss_d3_1(state,x)
  use rngdef_1
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2_1(state,x)
       use rngdef_1
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_1
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2_1(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3_1


subroutine rng_gauss_s3_1(state,x)
  use rngdef_1
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2_1(state,x)
       use rngdef_1
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_1
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2_1(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3_1


subroutine rng_set_seed_time_1(seed,time)
  use rngf77_1
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed_1(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed_1(ltime,seed)
  end if
end subroutine rng_set_seed_time_1

subroutine rng_set_seed_int_1(seed,n)
  use rngdef_1
  implicit none
  integer, dimension(0:rng_s-1), intent(out) :: seed
  integer, intent(in) :: n
  integer, parameter :: b=2**14
  integer :: m,i,sign
  if (n.lt.0) then
     seed=b-1
     m=-n-1
     sign=-1
  else
     seed=0
     m=n
     sign=1
  end if
  i=0
10 continue
  if (m.eq.0) return
  seed(i)=seed(i)+sign*mod(m,b)
  m=int(m/b)
  i=i+1
  goto 10
end subroutine rng_set_seed_int_1

subroutine rng_set_seed_string_1(seed,string)
  use rngf77_1
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed_1(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed_1(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string_1
 



 













! version 2.0 of random number routines
! author: charles karney <karney@princeton.edu>
! date: 1999-10-05 14:33 -0400
!
      function random1_1(ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      double precision  random1_1
      real srandom1_1_1
      integer ri
      double precision  ra(0:100-1)
      external rand_batch_1

      if(ri.ge.100)then
      call rand_batch_1(ri,ra(0))
      end if
      random1_1=ra(ri)+ulp2
      ri=ri+1
      return

      entry srandom1_1_1(ri,ra)
      if(ri.ge.100)then
      call rand_batch_1(ri,ra(0))
      end if

      srandom1_1_1=(int(mult*ra(ri))+0.5)*ulps

      ri=ri+1
      return
      end
!
      subroutine random_array_1(y,n,ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      integer n
      double precision  y(0:n-1)

      real ys(0:n-1)

      integer ri
      double precision  ra(0:100-1)
      integer i,k,j
      external rand_batch_1

      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      y(i)=ra(i+ri)+ulp2
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_1(ri,ra(0))
      do i=j,min(j+100,n)-1
      y(i)=ra(i-j+ri)+ulp2

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      entry srandom_array_1_1(ys,n,ri,ra)
      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      ys(i)=(int(mult*ra(i+ri))+0.5)*ulps
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_1(ri,ra(0))
      do i=j,min(j+100,n)-1
      ys(i)=(int(mult*ra(i-j+ri))+0.5)*ulps

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      end
!
      subroutine rand_batch_1(ri,ra)
      implicit none
      integer ri
      double precision  ra(0:100-1)
      integer i
      double precision  w(0:1009-100-1)
      double precision  tmp
      do i=0,63-1
      tmp=ra(i)+ra(i+100-63)

      w(i)=tmp-int(tmp)

      end do
      do i=63,100-1
      tmp=ra(i)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=100,1009-100-1
      tmp=w(i-100)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=1009-100,1009-100+63-1
      tmp=w(i-100)+w(i-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      do i=1009-100+63,1009-1
      tmp=w(i-100)+ra(i-1009+100-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      ri=0
      return
      end
!
      subroutine random_init_1(seed,ri,ra)
      implicit none
      integer b
      double precision  del,ulp
      parameter(b=2**14,del= 2.0d0 **(-14),ulp= 2.0d0 **(-47))
      integer a0,a1,a2,a3,a4,a5,a6,c0
      parameter(a0=15661,a1=678,a2=724,a3=5245,a4=13656,a5=11852,a6=29)
      parameter(c0=1)
      integer seed(0:8-1)
      integer ri
      double precision  ra(0:100-1)
      integer i,j,s(0:8-1)
      logical odd
      integer z(0:8-1),t
      do i=0,8-1
      s(i)=seed(i)
      end do
      odd=mod(s(7),2).ne.0
      ra(0)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      do j=1,100-1
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      odd=odd.or.(mod(s(7),2).ne.0)
      ra(j)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      end do
      ri=100
      if(odd)return
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      j=int((s(8-1)*100)/b)
      ra(j)=ra(j)+(ulp)
      return
      end
!
      subroutine decimal_to_seed_1(decimal,seed)
      implicit none
      character*(*)decimal
      integer seed(0:8-1)
      external rand_axc_1
      integer i,ten(0:8-1),c(0:8-1),ch
      data ten/10,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(decimal)
      ch=ichar(decimal(i:i))
      if(ch.ge.ichar('0').and.ch.le.ichar('9'))then
      c(0)=ch-ichar('0')
      call rand_axc_1(ten,seed,c)
      end if
      end do
      return
      end
!
      subroutine string_to_seed_1(string,seed)
      implicit none
      integer b
      parameter(b=2**14)
      character*(*)string
      integer seed(0:8-1)
      external rand_axc_1
      integer t,i,k,unity(0:8-1),c(0:8-1),ch
      data unity/1,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(string)
      ch=ichar(string(i:i))
      if(ch.gt.ichar(' ').and.ch.lt.127)then
      t=mod(seed(0),2)*(b/2)
      do k=0,8-1
      seed(k)=int(seed(k)/2)
      if(k.lt.8-1)then
      seed(k)=seed(k)+(mod(seed(k+1),2)*(b/2))
      else
      seed(k)=seed(k)+(t)
      end if
      end do
      c(0)=ch
      call rand_axc_1(unity,seed,c)
      end if
      end do
      return
      end
!
      subroutine set_random_seed_1(time,seed)
      implicit none
      integer time(8)
      integer seed(0:8-1)
      character*21 c
      external decimal_to_seed_1
      c=' '
      write(c(1:8),'(i4.4,2i2.2)')time(1),time(2),time(3)
      write(c(9:12),'(i1.1,i3.3)') (1-sign(1,time(4)))/2,abs(time(4))
      write(c(13:21),'(3i2.2,i3.3)')time(5),time(6),time(7),time(8)
      call decimal_to_seed_1(c,seed)
      return
      end
!
      subroutine seed_to_decimal_1(seed,decimal)
      implicit none
      integer pow,decbase,b
      parameter(pow=4,decbase=10**pow,b=2**14)
      character*(*)decimal
      integer seed(0:8-1)
      integer z(0:8-1),i,t,j,k
      character*36 str
      k=-1
      do i=0,8-1
      z(i)=seed(i)
      if(z(i).gt.0)k=i
      end do
      str=' '
      i=9
90000 continue
      i=i-1
      t=0
      do j=k,0,-1
      z(j)=z(j)+t*b
      t=mod(z(j),decbase)
      z(j)=int(z(j)/decbase)
      end do
      if(z(max(0,k)).eq.0)k=k-1
      j=pow*(i+1)
      if(k.ge.0)then
      str(j-(pow-1):j)='0000'
      else
      str(j-(pow-1):j)='   0'
      end if
90001 continue
      if(t.eq.0)goto 90010
      str(j:j)=char(ichar('0')+mod(t,10))
      j=j-1
      t=int(t/10)
      goto 90001
90010 continue
      if(k.ge.0)goto 90000
      if(len(decimal).gt.len(str))then
      decimal(:len(decimal)-len(str))=' '
      decimal(len(decimal)-len(str)+1:)=str
      else
      decimal=str(len(str)-len(decimal)+1:)
      end if
      return
      end
!
      subroutine rand_next_seed_1(n,ax,cx,y)
      implicit none
      integer n,ax(0:8-1),cx(0:8-1)
      integer y(0:8-1)
      external rand_axc_1
      integer a(0:8-1),c(0:8-1),z(0:8-1),t(0:8-1),m,i
      data z/8*0/
      if(n.eq.0)return
      m=n
      do i=0,8-1
      a(i)=ax(i)
      c(i)=cx(i)
      end do
90000 continue
      if(mod(m,2).gt.0)then
      call rand_axc_1(a,y,c)
      end if
      m=int(m/2)
      if(m.eq.0)return
      do i=0,8-1
      t(i)=c(i)
      end do
      call rand_axc_1(a,c,t)
      do i=0,8-1
      t(i)=a(i)
      end do
      call rand_axc_1(t,a,z)
      goto 90000
      end
!
      subroutine next_seed_13_1(n0,n1,n2,seed)
      implicit none
      integer n0,n1,n2
      integer seed(0:8-1)
      external rand_next_seed_1
      integer af0(0:8-1),cf0(0:8-1)
      integer ab0(0:8-1),cb0(0:8-1)
      integer af1(0:8-1),cf1(0:8-1)
      integer ab1(0:8-1),cb1(0:8-1)
      integer af2(0:8-1),cf2(0:8-1)
      integer ab2(0:8-1),cb2(0:8-1)
      data af0/15741,8689,9280,4732,12011,7130,6824,12302/
      data cf0/16317,10266,1198,331,10769,8310,2779,13880/
      data ab0/9173,9894,15203,15379,7981,2280,8071,429/
      data cb0/8383,3616,597,12724,15663,9639,187,4866/
      data af1/8405,4808,3603,6718,13766,9243,10375,12108/
      data cf1/13951,7170,9039,11206,8706,14101,1864,15191/
      data ab1/6269,3240,9759,7130,15320,14399,3675,1380/
      data cb1/15357,5843,6205,16275,8838,12132,2198,10330/
      data af2/445,10754,1869,6593,385,12498,14501,7383/
      data cf2/2285,8057,3864,10235,1805,10614,9615,15522/
      data ab2/405,4903,2746,1477,3263,13564,8139,2362/
      data cb2/8463,575,5876,2220,4924,1701,9060,5639/
      if(n2.gt.0)then
      call rand_next_seed_1(n2,af2,cf2,seed)
      else if(n2.lt.0)then
      call rand_next_seed_1(-n2,ab2,cb2,seed)
      end if
      if(n1.gt.0)then
      call rand_next_seed_1(n1,af1,cf1,seed)
      else if(n1.lt.0)then
      call rand_next_seed_1(-n1,ab1,cb1,seed)
      end if
      entry next_seed_1(n0,seed)
      if(n0.gt.0)then
      call rand_next_seed_1(n0,af0,cf0,seed)
      else if(n0.lt.0)then
      call rand_next_seed_1(-n0,ab0,cb0,seed)
      end if
      return
      end
!
      subroutine rand_axc_1(a,x,c)
      implicit none
      integer b
      parameter(b=2**14)
      integer a(0:8-1),c(0:8-1)
      integer x(0:8-1)
      integer z(0:8-1),i,j,t
      do i=0,8-1
      z(i)=c(i)
      end do
      do j=0,8-1
      do i=j,8-1
      z(i)=z(i)+(a(j)*x(i-j))
      end do
      end do
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      x(i)=mod(t,b)
      end do
      return
      end
!
! f90 interface to random number routine
! charles karney <karney@pppl.gov> 1999-09-30 16:58 -0400

 



 














module rngdef_2
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=kind(0.0)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(kind=double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef_2

module rng1_2
  ! the main module 
  use rngdef_2, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0_2
     subroutine rng_number_d1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_2
     subroutine rng_number_d2_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_2
     subroutine rng_number_d3_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3_2

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_number_s0_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0_2
     subroutine rng_number_s1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_2
     subroutine rng_number_s2_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_2
     subroutine rng_number_s3_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3_2

  end interface
  interface rng_gauss
     ! gaussian random numbers
     subroutine rng_gauss_d0_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0_2
     subroutine rng_gauss_d1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_2
     subroutine rng_gauss_d2_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_2
     subroutine rng_gauss_d3_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3_2

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_gauss_s0_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0_2
     subroutine rng_gauss_s1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_2
     subroutine rng_gauss_s2_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_2
     subroutine rng_gauss_s3_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3_2

  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time_2(seed,time)
       use rngdef_2
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time_2
     subroutine rng_set_seed_int_2(seed,n)
       use rngdef_2
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int_2
     subroutine rng_set_seed_string_2(seed,string)
       use rngdef_2
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string_2
  end interface

  interface
     function rng_print_seed_2(seed)
       ! return the seed in a string representation
       use rngdef_2
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed_2
     end function rng_print_seed_2
     subroutine rng_init_2(seed,state)
       ! initialize the state from the seed
       use rngdef_2
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init_2
     subroutine rng_step_seed_2(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef_2
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed_2
  end interface
end module rng1_2
 



 













module rngf77_2
  ! interface for the original f77 routines
  use rngdef_2
  implicit none
  interface
     function random1_2(ri,ra)
       use rngdef_2
       implicit none
       real(kind=double) :: random1_2
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random1_2

     function srandom1_2_2(ri,ra)
       use rngdef_2
       implicit none
       real(kind=single) :: srandom1_2_2
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom1_2_2

     subroutine random_array_2(y,n,ri,ra)
       use rngdef_2
       implicit none
       integer, intent(in) :: n
       real(kind=double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array_2

     subroutine srandom_array_2_2(y,n,ri,ra)
       use rngdef_2
       implicit none
       integer, intent(in) :: n
       real(kind=single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array_2_2

     subroutine rand_batch_2(ri,ra)
       use rngdef_2
       implicit none
       integer, intent(inout) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch_2
     subroutine random_init_2(seed,ri,ra)
       use rngdef_2
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init_2
     subroutine decimal_to_seed_2(decimal,seed)
       use rngdef_2
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed_2
     subroutine string_to_seed_2(string,seed)
       use rngdef_2
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed_2
     subroutine set_random_seed_2(time,seed)
       use rngdef_2
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed_2
     subroutine seed_to_decimal_2(seed,decimal)
       use rngdef_2
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal_2
     subroutine next_seed_23_2(n0,n1,n2,seed)
       use rngdef_2
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_23_2
     subroutine next_seed_2(n0,seed)
       use rngdef_2
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_2
  end interface
end module rngf77_2

subroutine rng_init_2(seed,state)
  use rngf77_2
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init_2(seed,state%index,state%array)
  return
end subroutine rng_init_2

subroutine rng_step_seed_2(seed,n0,n1,n2)
  use rngf77_2
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed_2(1,seed)
  else
     if (present(n0)) then
        m0=n0
     else
        m0=0
     endif
     if (present(n1)) then
        m1=n1
     else
        m1=0
     endif
     if (present(n2)) then
        m2=n2
     else
        m2=0
     endif
     call next_seed_23_2(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed_2

function rng_print_seed_2(seed)
  use rngf77_2
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed_2
  call seed_to_decimal_2(seed,rng_print_seed_2)
  return
end function rng_print_seed_2

subroutine rng_number_d0_2(state,x)
  use rngf77_2
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch_2(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0_2


subroutine rng_number_s0_2(state,x)
  use rngf77_2
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch_2(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0_2



subroutine rng_number_d1_2(state,x)
  use rngf77_2
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=state%array(i+state%index)+ulp2
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_2(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1_2


subroutine rng_number_s1_2(state,x)
  use rngf77_2
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=(int(mult*state%array(i+state%index))+0.5)*ulps
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_2(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1_2


subroutine rng_number_d2_2(state,x)
  use rngdef_2
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_2
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1_2(state,x(:,i))
  end do
end subroutine rng_number_d2_2


subroutine rng_number_s2_2(state,x)
  use rngdef_2
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_2
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1_2(state,x(:,i))
  end do
end subroutine rng_number_s2_2


subroutine rng_number_d3_2(state,x)
  use rngdef_2
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2_2(state,x(:,:,i))
  end do
end subroutine rng_number_d3_2


subroutine rng_number_s3_2(state,x)
  use rngdef_2
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2_2(state,x(:,:,i))
  end do
end subroutine rng_number_s3_2


subroutine rng_gauss_d1_2(state,x)
  use rngf77_2
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_2
  end interface
  integer :: i
  real(kind=double), parameter :: pi= 3.14159265358979323846264338328d0 
  real(kind=double) :: theta,z
  call rng_number_d1_2(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0d0 *x(i)- 1.0d0 )
     z=sqrt(- 2.0d0 *log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0d0 *x(ubound(x,1))- 1.0d0 )
  z=sqrt(- 2.0d0 *log(random1_2(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1_2

subroutine rng_gauss_d0_2(state,x)
  use rngdef_2
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_2
  end interface
  real(kind=double), dimension(1) :: y
  call rng_gauss_d1_2(state, y)
  x=y(1)
end subroutine rng_gauss_d0_2


subroutine rng_gauss_s1_2(state,x)
  use rngf77_2
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_2
  end interface
  integer :: i
  real(kind=single), parameter :: pi= 3.14159265358979323846264338328 
  real(kind=single) :: theta,z
  call rng_number_s1_2(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0*x(i)-1.0)
     z=sqrt(-2.0*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0*x(ubound(x,1))-1.0)
  z=sqrt(-2.0*log(srandom1_2_2(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1_2

subroutine rng_gauss_s0_2(state,x)
  use rngdef_2
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_2
  end interface
  real(kind=single), dimension(1) :: y
  call rng_gauss_s1_2(state, y)
  x=y(1)
end subroutine rng_gauss_s0_2


subroutine rng_gauss_d2_2(state,x)
  use rngdef_2
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_2
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1_2(state,x(:,i))
  end do
end subroutine rng_gauss_d2_2


subroutine rng_gauss_s2_2(state,x)
  use rngdef_2
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_2
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1_2(state,x(:,i))
  end do
end subroutine rng_gauss_s2_2


subroutine rng_gauss_d3_2(state,x)
  use rngdef_2
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2_2(state,x)
       use rngdef_2
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2_2(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3_2


subroutine rng_gauss_s3_2(state,x)
  use rngdef_2
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2_2(state,x)
       use rngdef_2
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2_2(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3_2


subroutine rng_set_seed_time_2(seed,time)
  use rngf77_2
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed_2(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed_2(ltime,seed)
  end if
end subroutine rng_set_seed_time_2

subroutine rng_set_seed_int_2(seed,n)
  use rngdef_2
  implicit none
  integer, dimension(0:rng_s-1), intent(out) :: seed
  integer, intent(in) :: n
  integer, parameter :: b=2**14
  integer :: m,i,sign
  if (n.lt.0) then
     seed=b-1
     m=-n-1
     sign=-1
  else
     seed=0
     m=n
     sign=1
  end if
  i=0
10 continue
  if (m.eq.0) return
  seed(i)=seed(i)+sign*mod(m,b)
  m=int(m/b)
  i=i+1
  goto 10
end subroutine rng_set_seed_int_2

subroutine rng_set_seed_string_2(seed,string)
  use rngf77_2
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed_2(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed_2(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string_2
 



 













! version 2.0 of random number routines
! author: charles karney <karney@princeton.edu>
! date: 1999-10-05 14:33 -0400
!
      function random1_2(ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      double precision  random1_2
      real srandom1_2_2
      integer ri
      double precision  ra(0:100-1)
      external rand_batch_2

      if(ri.ge.100)then
      call rand_batch_2(ri,ra(0))
      end if
      random1_2=ra(ri)+ulp2
      ri=ri+1
      return

      entry srandom1_2_2(ri,ra)
      if(ri.ge.100)then
      call rand_batch_2(ri,ra(0))
      end if

      srandom1_2_2=(int(mult*ra(ri))+0.5)*ulps

      ri=ri+1
      return
      end
!
      subroutine random_array_2(y,n,ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      integer n
      double precision  y(0:n-1)

      real ys(0:n-1)

      integer ri
      double precision  ra(0:100-1)
      integer i,k,j
      external rand_batch_2

      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      y(i)=ra(i+ri)+ulp2
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_2(ri,ra(0))
      do i=j,min(j+100,n)-1
      y(i)=ra(i-j+ri)+ulp2

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      entry srandom_array_2_2(ys,n,ri,ra)
      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      ys(i)=(int(mult*ra(i+ri))+0.5)*ulps
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_2(ri,ra(0))
      do i=j,min(j+100,n)-1
      ys(i)=(int(mult*ra(i-j+ri))+0.5)*ulps

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      end
!
      subroutine rand_batch_2(ri,ra)
      implicit none
      integer ri
      double precision  ra(0:100-1)
      integer i
      double precision  w(0:1009-100-1)
      double precision  tmp
      do i=0,63-1
      tmp=ra(i)+ra(i+100-63)

      w(i)=tmp-int(tmp)

      end do
      do i=63,100-1
      tmp=ra(i)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=100,1009-100-1
      tmp=w(i-100)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=1009-100,1009-100+63-1
      tmp=w(i-100)+w(i-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      do i=1009-100+63,1009-1
      tmp=w(i-100)+ra(i-1009+100-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      ri=0
      return
      end
!
      subroutine random_init_2(seed,ri,ra)
      implicit none
      integer b
      double precision  del,ulp
      parameter(b=2**14,del= 2.0d0 **(-14),ulp= 2.0d0 **(-47))
      integer a0,a1,a2,a3,a4,a5,a6,c0
      parameter(a0=15661,a1=678,a2=724,a3=5245,a4=13656,a5=11852,a6=29)
      parameter(c0=1)
      integer seed(0:8-1)
      integer ri
      double precision  ra(0:100-1)
      integer i,j,s(0:8-1)
      logical odd
      integer z(0:8-1),t
      do i=0,8-1
      s(i)=seed(i)
      end do
      odd=mod(s(7),2).ne.0
      ra(0)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      do j=1,100-1
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      odd=odd.or.(mod(s(7),2).ne.0)
      ra(j)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      end do
      ri=100
      if(odd)return
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      j=int((s(8-1)*100)/b)
      ra(j)=ra(j)+(ulp)
      return
      end
!
      subroutine decimal_to_seed_2(decimal,seed)
      implicit none
      character*(*)decimal
      integer seed(0:8-1)
      external rand_axc_2
      integer i,ten(0:8-1),c(0:8-1),ch
      data ten/10,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(decimal)
      ch=ichar(decimal(i:i))
      if(ch.ge.ichar('0').and.ch.le.ichar('9'))then
      c(0)=ch-ichar('0')
      call rand_axc_2(ten,seed,c)
      end if
      end do
      return
      end
!
      subroutine string_to_seed_2(string,seed)
      implicit none
      integer b
      parameter(b=2**14)
      character*(*)string
      integer seed(0:8-1)
      external rand_axc_2
      integer t,i,k,unity(0:8-1),c(0:8-1),ch
      data unity/1,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(string)
      ch=ichar(string(i:i))
      if(ch.gt.ichar(' ').and.ch.lt.127)then
      t=mod(seed(0),2)*(b/2)
      do k=0,8-1
      seed(k)=int(seed(k)/2)
      if(k.lt.8-1)then
      seed(k)=seed(k)+(mod(seed(k+1),2)*(b/2))
      else
      seed(k)=seed(k)+(t)
      end if
      end do
      c(0)=ch
      call rand_axc_2(unity,seed,c)
      end if
      end do
      return
      end
!
      subroutine set_random_seed_2(time,seed)
      implicit none
      integer time(8)
      integer seed(0:8-1)
      character*21 c
      external decimal_to_seed_2
      c=' '
      write(c(1:8),'(i4.4,2i2.2)')time(1),time(2),time(3)
      write(c(9:12),'(i1.1,i3.3)') (1-sign(1,time(4)))/2,abs(time(4))
      write(c(13:21),'(3i2.2,i3.3)')time(5),time(6),time(7),time(8)
      call decimal_to_seed_2(c,seed)
      return
      end
!
      subroutine seed_to_decimal_2(seed,decimal)
      implicit none
      integer pow,decbase,b
      parameter(pow=4,decbase=10**pow,b=2**14)
      character*(*)decimal
      integer seed(0:8-1)
      integer z(0:8-1),i,t,j,k
      character*36 str
      k=-1
      do i=0,8-1
      z(i)=seed(i)
      if(z(i).gt.0)k=i
      end do
      str=' '
      i=9
90000 continue
      i=i-1
      t=0
      do j=k,0,-1
      z(j)=z(j)+t*b
      t=mod(z(j),decbase)
      z(j)=int(z(j)/decbase)
      end do
      if(z(max(0,k)).eq.0)k=k-1
      j=pow*(i+1)
      if(k.ge.0)then
      str(j-(pow-1):j)='0000'
      else
      str(j-(pow-1):j)='   0'
      end if
90001 continue
      if(t.eq.0)goto 90010
      str(j:j)=char(ichar('0')+mod(t,10))
      j=j-1
      t=int(t/10)
      goto 90001
90010 continue
      if(k.ge.0)goto 90000
      if(len(decimal).gt.len(str))then
      decimal(:len(decimal)-len(str))=' '
      decimal(len(decimal)-len(str)+1:)=str
      else
      decimal=str(len(str)-len(decimal)+1:)
      end if
      return
      end
!
      subroutine rand_next_seed_2(n,ax,cx,y)
      implicit none
      integer n,ax(0:8-1),cx(0:8-1)
      integer y(0:8-1)
      external rand_axc_2
      integer a(0:8-1),c(0:8-1),z(0:8-1),t(0:8-1),m,i
      data z/8*0/
      if(n.eq.0)return
      m=n
      do i=0,8-1
      a(i)=ax(i)
      c(i)=cx(i)
      end do
90000 continue
      if(mod(m,2).gt.0)then
      call rand_axc_2(a,y,c)
      end if
      m=int(m/2)
      if(m.eq.0)return
      do i=0,8-1
      t(i)=c(i)
      end do
      call rand_axc_2(a,c,t)
      do i=0,8-1
      t(i)=a(i)
      end do
      call rand_axc_2(t,a,z)
      goto 90000
      end
!
      subroutine next_seed_23_2(n0,n1,n2,seed)
      implicit none
      integer n0,n1,n2
      integer seed(0:8-1)
      external rand_next_seed_2
      integer af0(0:8-1),cf0(0:8-1)
      integer ab0(0:8-1),cb0(0:8-1)
      integer af1(0:8-1),cf1(0:8-1)
      integer ab1(0:8-1),cb1(0:8-1)
      integer af2(0:8-1),cf2(0:8-1)
      integer ab2(0:8-1),cb2(0:8-1)
      data af0/15741,8689,9280,4732,12011,7130,6824,12302/
      data cf0/16317,10266,1198,331,10769,8310,2779,13880/
      data ab0/9173,9894,15203,15379,7981,2280,8071,429/
      data cb0/8383,3616,597,12724,15663,9639,187,4866/
      data af1/8405,4808,3603,6718,13766,9243,10375,12108/
      data cf1/13951,7170,9039,11206,8706,14101,1864,15191/
      data ab1/6269,3240,9759,7130,15320,14399,3675,1380/
      data cb1/15357,5843,6205,16275,8838,12132,2198,10330/
      data af2/445,10754,1869,6593,385,12498,14501,7383/
      data cf2/2285,8057,3864,10235,1805,10614,9615,15522/
      data ab2/405,4903,2746,1477,3263,13564,8139,2362/
      data cb2/8463,575,5876,2220,4924,1701,9060,5639/
      if(n2.gt.0)then
      call rand_next_seed_2(n2,af2,cf2,seed)
      else if(n2.lt.0)then
      call rand_next_seed_2(-n2,ab2,cb2,seed)
      end if
      if(n1.gt.0)then
      call rand_next_seed_2(n1,af1,cf1,seed)
      else if(n1.lt.0)then
      call rand_next_seed_2(-n1,ab1,cb1,seed)
      end if
      entry next_seed_2(n0,seed)
      if(n0.gt.0)then
      call rand_next_seed_2(n0,af0,cf0,seed)
      else if(n0.lt.0)then
      call rand_next_seed_2(-n0,ab0,cb0,seed)
      end if
      return
      end
!
      subroutine rand_axc_2(a,x,c)
      implicit none
      integer b
      parameter(b=2**14)
      integer a(0:8-1),c(0:8-1)
      integer x(0:8-1)
      integer z(0:8-1),i,j,t
      do i=0,8-1
      z(i)=c(i)
      end do
      do j=0,8-1
      do i=j,8-1
      z(i)=z(i)+(a(j)*x(i-j))
      end do
      end do
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      x(i)=mod(t,b)
      end do
      return
      end
!
! f90 interface to random number routine
! charles karney <karney@pppl.gov> 1999-09-30 16:58 -0400

 



 














module rngdef_3
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=kind(0.0)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(kind=double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef_3

module rng1_3
  ! the main module 
  use rngdef_3, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0_3
     subroutine rng_number_d1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_3
     subroutine rng_number_d2_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_3
     subroutine rng_number_d3_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3_3

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_number_s0_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0_3
     subroutine rng_number_s1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_3
     subroutine rng_number_s2_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_3
     subroutine rng_number_s3_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3_3

  end interface
  interface rng_gauss
     ! gaussian random numbers
     subroutine rng_gauss_d0_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0_3
     subroutine rng_gauss_d1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_3
     subroutine rng_gauss_d2_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_3
     subroutine rng_gauss_d3_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3_3

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_gauss_s0_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0_3
     subroutine rng_gauss_s1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_3
     subroutine rng_gauss_s2_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_3
     subroutine rng_gauss_s3_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3_3

  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time_3(seed,time)
       use rngdef_3
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time_3
     subroutine rng_set_seed_int_3(seed,n)
       use rngdef_3
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int_3
     subroutine rng_set_seed_string_3(seed,string)
       use rngdef_3
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string_3
  end interface

  interface
     function rng_print_seed_3(seed)
       ! return the seed in a string representation
       use rngdef_3
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed_3
     end function rng_print_seed_3
     subroutine rng_init_3(seed,state)
       ! initialize the state from the seed
       use rngdef_3
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init_3
     subroutine rng_step_seed_3(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef_3
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed_3
  end interface
end module rng1_3
 



 













module rngf77_3
  ! interface for the original f77 routines
  use rngdef_3
  implicit none
  interface
     function random1_3(ri,ra)
       use rngdef_3
       implicit none
       real(kind=double) :: random1_3
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random1_3

     function srandom1_3_3(ri,ra)
       use rngdef_3
       implicit none
       real(kind=single) :: srandom1_3_3
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom1_3_3

     subroutine random_array_3(y,n,ri,ra)
       use rngdef_3
       implicit none
       integer, intent(in) :: n
       real(kind=double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array_3

     subroutine srandom_array_3_3(y,n,ri,ra)
       use rngdef_3
       implicit none
       integer, intent(in) :: n
       real(kind=single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array_3_3

     subroutine rand_batch_3(ri,ra)
       use rngdef_3
       implicit none
       integer, intent(inout) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch_3
     subroutine random_init_3(seed,ri,ra)
       use rngdef_3
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init_3
     subroutine decimal_to_seed_3(decimal,seed)
       use rngdef_3
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed_3
     subroutine string_to_seed_3(string,seed)
       use rngdef_3
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed_3
     subroutine set_random_seed_3(time,seed)
       use rngdef_3
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed_3
     subroutine seed_to_decimal_3(seed,decimal)
       use rngdef_3
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal_3
     subroutine next_seed_33_3(n0,n1,n2,seed)
       use rngdef_3
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_33_3
     subroutine next_seed_3(n0,seed)
       use rngdef_3
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_3
  end interface
end module rngf77_3

subroutine rng_init_3(seed,state)
  use rngf77_3
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init_3(seed,state%index,state%array)
  return
end subroutine rng_init_3

subroutine rng_step_seed_3(seed,n0,n1,n2)
  use rngf77_3
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed_3(1,seed)
  else
     if (present(n0)) then
        m0=n0
     else
        m0=0
     endif
     if (present(n1)) then
        m1=n1
     else
        m1=0
     endif
     if (present(n2)) then
        m2=n2
     else
        m2=0
     endif
     call next_seed_33_3(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed_3

function rng_print_seed_3(seed)
  use rngf77_3
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed_3
  call seed_to_decimal_3(seed,rng_print_seed_3)
  return
end function rng_print_seed_3

subroutine rng_number_d0_3(state,x)
  use rngf77_3
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch_3(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0_3


subroutine rng_number_s0_3(state,x)
  use rngf77_3
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch_3(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0_3



subroutine rng_number_d1_3(state,x)
  use rngf77_3
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=state%array(i+state%index)+ulp2
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_3(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1_3


subroutine rng_number_s1_3(state,x)
  use rngf77_3
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=(int(mult*state%array(i+state%index))+0.5)*ulps
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_3(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1_3


subroutine rng_number_d2_3(state,x)
  use rngdef_3
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_3
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1_3(state,x(:,i))
  end do
end subroutine rng_number_d2_3


subroutine rng_number_s2_3(state,x)
  use rngdef_3
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_3
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1_3(state,x(:,i))
  end do
end subroutine rng_number_s2_3


subroutine rng_number_d3_3(state,x)
  use rngdef_3
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_3
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2_3(state,x(:,:,i))
  end do
end subroutine rng_number_d3_3


subroutine rng_number_s3_3(state,x)
  use rngdef_3
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_3
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2_3(state,x(:,:,i))
  end do
end subroutine rng_number_s3_3


subroutine rng_gauss_d1_3(state,x)
  use rngf77_3
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_3
  end interface
  integer :: i
  real(kind=double), parameter :: pi= 3.14159265358979323846264338328d0 
  real(kind=double) :: theta,z
  call rng_number_d1_3(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0d0 *x(i)- 1.0d0 )
     z=sqrt(- 2.0d0 *log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0d0 *x(ubound(x,1))- 1.0d0 )
  z=sqrt(- 2.0d0 *log(random1_3(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1_3

subroutine rng_gauss_d0_3(state,x)
  use rngdef_3
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_3
  end interface
  real(kind=double), dimension(1) :: y
  call rng_gauss_d1_3(state, y)
  x=y(1)
end subroutine rng_gauss_d0_3


subroutine rng_gauss_s1_3(state,x)
  use rngf77_3
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_3
  end interface
  integer :: i
  real(kind=single), parameter :: pi= 3.14159265358979323846264338328 
  real(kind=single) :: theta,z
  call rng_number_s1_3(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0*x(i)-1.0)
     z=sqrt(-2.0*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0*x(ubound(x,1))-1.0)
  z=sqrt(-2.0*log(srandom1_3_3(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1_3

subroutine rng_gauss_s0_3(state,x)
  use rngdef_3
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_3
  end interface
  real(kind=single), dimension(1) :: y
  call rng_gauss_s1_3(state, y)
  x=y(1)
end subroutine rng_gauss_s0_3


subroutine rng_gauss_d2_3(state,x)
  use rngdef_3
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_3
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1_3(state,x(:,i))
  end do
end subroutine rng_gauss_d2_3


subroutine rng_gauss_s2_3(state,x)
  use rngdef_3
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_3
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1_3(state,x(:,i))
  end do
end subroutine rng_gauss_s2_3


subroutine rng_gauss_d3_3(state,x)
  use rngdef_3
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2_3(state,x)
       use rngdef_3
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_3
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2_3(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3_3


subroutine rng_gauss_s3_3(state,x)
  use rngdef_3
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2_3(state,x)
       use rngdef_3
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_3
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2_3(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3_3


subroutine rng_set_seed_time_3(seed,time)
  use rngf77_3
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed_3(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed_3(ltime,seed)
  end if
end subroutine rng_set_seed_time_3

subroutine rng_set_seed_int_3(seed,n)
  use rngdef_3
  implicit none
  integer, dimension(0:rng_s-1), intent(out) :: seed
  integer, intent(in) :: n
  integer, parameter :: b=2**14
  integer :: m,i,sign
  if (n.lt.0) then
     seed=b-1
     m=-n-1
     sign=-1
  else
     seed=0
     m=n
     sign=1
  end if
  i=0
10 continue
  if (m.eq.0) return
  seed(i)=seed(i)+sign*mod(m,b)
  m=int(m/b)
  i=i+1
  goto 10
end subroutine rng_set_seed_int_3

subroutine rng_set_seed_string_3(seed,string)
  use rngf77_3
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed_3(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed_3(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string_3
 



 













! version 2.0 of random number routines
! author: charles karney <karney@princeton.edu>
! date: 1999-10-05 14:33 -0400
!
      function random1_3(ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      double precision  random1_3
      real srandom1_3_3
      integer ri
      double precision  ra(0:100-1)
      external rand_batch_3

      if(ri.ge.100)then
      call rand_batch_3(ri,ra(0))
      end if
      random1_3=ra(ri)+ulp2
      ri=ri+1
      return

      entry srandom1_3_3(ri,ra)
      if(ri.ge.100)then
      call rand_batch_3(ri,ra(0))
      end if

      srandom1_3_3=(int(mult*ra(ri))+0.5)*ulps

      ri=ri+1
      return
      end
!
      subroutine random_array_3(y,n,ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      integer n
      double precision  y(0:n-1)

      real ys(0:n-1)

      integer ri
      double precision  ra(0:100-1)
      integer i,k,j
      external rand_batch_3

      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      y(i)=ra(i+ri)+ulp2
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_3(ri,ra(0))
      do i=j,min(j+100,n)-1
      y(i)=ra(i-j+ri)+ulp2

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      entry srandom_array_3_3(ys,n,ri,ra)
      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      ys(i)=(int(mult*ra(i+ri))+0.5)*ulps
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_3(ri,ra(0))
      do i=j,min(j+100,n)-1
      ys(i)=(int(mult*ra(i-j+ri))+0.5)*ulps

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      end
!
      subroutine rand_batch_3(ri,ra)
      implicit none
      integer ri
      double precision  ra(0:100-1)
      integer i
      double precision  w(0:1009-100-1)
      double precision  tmp
      do i=0,63-1
      tmp=ra(i)+ra(i+100-63)

      w(i)=tmp-int(tmp)

      end do
      do i=63,100-1
      tmp=ra(i)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=100,1009-100-1
      tmp=w(i-100)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=1009-100,1009-100+63-1
      tmp=w(i-100)+w(i-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      do i=1009-100+63,1009-1
      tmp=w(i-100)+ra(i-1009+100-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      ri=0
      return
      end
!
      subroutine random_init_3(seed,ri,ra)
      implicit none
      integer b
      double precision  del,ulp
      parameter(b=2**14,del= 2.0d0 **(-14),ulp= 2.0d0 **(-47))
      integer a0,a1,a2,a3,a4,a5,a6,c0
      parameter(a0=15661,a1=678,a2=724,a3=5245,a4=13656,a5=11852,a6=29)
      parameter(c0=1)
      integer seed(0:8-1)
      integer ri
      double precision  ra(0:100-1)
      integer i,j,s(0:8-1)
      logical odd
      integer z(0:8-1),t
      do i=0,8-1
      s(i)=seed(i)
      end do
      odd=mod(s(7),2).ne.0
      ra(0)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      do j=1,100-1
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      odd=odd.or.(mod(s(7),2).ne.0)
      ra(j)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      end do
      ri=100
      if(odd)return
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      j=int((s(8-1)*100)/b)
      ra(j)=ra(j)+(ulp)
      return
      end
!
      subroutine decimal_to_seed_3(decimal,seed)
      implicit none
      character*(*)decimal
      integer seed(0:8-1)
      external rand_axc_3
      integer i,ten(0:8-1),c(0:8-1),ch
      data ten/10,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(decimal)
      ch=ichar(decimal(i:i))
      if(ch.ge.ichar('0').and.ch.le.ichar('9'))then
      c(0)=ch-ichar('0')
      call rand_axc_3(ten,seed,c)
      end if
      end do
      return
      end
!
      subroutine string_to_seed_3(string,seed)
      implicit none
      integer b
      parameter(b=2**14)
      character*(*)string
      integer seed(0:8-1)
      external rand_axc_3
      integer t,i,k,unity(0:8-1),c(0:8-1),ch
      data unity/1,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(string)
      ch=ichar(string(i:i))
      if(ch.gt.ichar(' ').and.ch.lt.127)then
      t=mod(seed(0),2)*(b/2)
      do k=0,8-1
      seed(k)=int(seed(k)/2)
      if(k.lt.8-1)then
      seed(k)=seed(k)+(mod(seed(k+1),2)*(b/2))
      else
      seed(k)=seed(k)+(t)
      end if
      end do
      c(0)=ch
      call rand_axc_3(unity,seed,c)
      end if
      end do
      return
      end
!
      subroutine set_random_seed_3(time,seed)
      implicit none
      integer time(8)
      integer seed(0:8-1)
      character*21 c
      external decimal_to_seed_3
      c=' '
      write(c(1:8),'(i4.4,2i2.2)')time(1),time(2),time(3)
      write(c(9:12),'(i1.1,i3.3)') (1-sign(1,time(4)))/2,abs(time(4))
      write(c(13:21),'(3i2.2,i3.3)')time(5),time(6),time(7),time(8)
      call decimal_to_seed_3(c,seed)
      return
      end
!
      subroutine seed_to_decimal_3(seed,decimal)
      implicit none
      integer pow,decbase,b
      parameter(pow=4,decbase=10**pow,b=2**14)
      character*(*)decimal
      integer seed(0:8-1)
      integer z(0:8-1),i,t,j,k
      character*36 str
      k=-1
      do i=0,8-1
      z(i)=seed(i)
      if(z(i).gt.0)k=i
      end do
      str=' '
      i=9
90000 continue
      i=i-1
      t=0
      do j=k,0,-1
      z(j)=z(j)+t*b
      t=mod(z(j),decbase)
      z(j)=int(z(j)/decbase)
      end do
      if(z(max(0,k)).eq.0)k=k-1
      j=pow*(i+1)
      if(k.ge.0)then
      str(j-(pow-1):j)='0000'
      else
      str(j-(pow-1):j)='   0'
      end if
90001 continue
      if(t.eq.0)goto 90010
      str(j:j)=char(ichar('0')+mod(t,10))
      j=j-1
      t=int(t/10)
      goto 90001
90010 continue
      if(k.ge.0)goto 90000
      if(len(decimal).gt.len(str))then
      decimal(:len(decimal)-len(str))=' '
      decimal(len(decimal)-len(str)+1:)=str
      else
      decimal=str(len(str)-len(decimal)+1:)
      end if
      return
      end
!
      subroutine rand_next_seed_3(n,ax,cx,y)
      implicit none
      integer n,ax(0:8-1),cx(0:8-1)
      integer y(0:8-1)
      external rand_axc_3
      integer a(0:8-1),c(0:8-1),z(0:8-1),t(0:8-1),m,i
      data z/8*0/
      if(n.eq.0)return
      m=n
      do i=0,8-1
      a(i)=ax(i)
      c(i)=cx(i)
      end do
90000 continue
      if(mod(m,2).gt.0)then
      call rand_axc_3(a,y,c)
      end if
      m=int(m/2)
      if(m.eq.0)return
      do i=0,8-1
      t(i)=c(i)
      end do
      call rand_axc_3(a,c,t)
      do i=0,8-1
      t(i)=a(i)
      end do
      call rand_axc_3(t,a,z)
      goto 90000
      end
!
      subroutine next_seed_33_3(n0,n1,n2,seed)
      implicit none
      integer n0,n1,n2
      integer seed(0:8-1)
      external rand_next_seed_3
      integer af0(0:8-1),cf0(0:8-1)
      integer ab0(0:8-1),cb0(0:8-1)
      integer af1(0:8-1),cf1(0:8-1)
      integer ab1(0:8-1),cb1(0:8-1)
      integer af2(0:8-1),cf2(0:8-1)
      integer ab2(0:8-1),cb2(0:8-1)
      data af0/15741,8689,9280,4732,12011,7130,6824,12302/
      data cf0/16317,10266,1198,331,10769,8310,2779,13880/
      data ab0/9173,9894,15203,15379,7981,2280,8071,429/
      data cb0/8383,3616,597,12724,15663,9639,187,4866/
      data af1/8405,4808,3603,6718,13766,9243,10375,12108/
      data cf1/13951,7170,9039,11206,8706,14101,1864,15191/
      data ab1/6269,3240,9759,7130,15320,14399,3675,1380/
      data cb1/15357,5843,6205,16275,8838,12132,2198,10330/
      data af2/445,10754,1869,6593,385,12498,14501,7383/
      data cf2/2285,8057,3864,10235,1805,10614,9615,15522/
      data ab2/405,4903,2746,1477,3263,13564,8139,2362/
      data cb2/8463,575,5876,2220,4924,1701,9060,5639/
      if(n2.gt.0)then
      call rand_next_seed_3(n2,af2,cf2,seed)
      else if(n2.lt.0)then
      call rand_next_seed_3(-n2,ab2,cb2,seed)
      end if
      if(n1.gt.0)then
      call rand_next_seed_3(n1,af1,cf1,seed)
      else if(n1.lt.0)then
      call rand_next_seed_3(-n1,ab1,cb1,seed)
      end if
      entry next_seed_3(n0,seed)
      if(n0.gt.0)then
      call rand_next_seed_3(n0,af0,cf0,seed)
      else if(n0.lt.0)then
      call rand_next_seed_3(-n0,ab0,cb0,seed)
      end if
      return
      end
!
      subroutine rand_axc_3(a,x,c)
      implicit none
      integer b
      parameter(b=2**14)
      integer a(0:8-1),c(0:8-1)
      integer x(0:8-1)
      integer z(0:8-1),i,j,t
      do i=0,8-1
      z(i)=c(i)
      end do
      do j=0,8-1
      do i=j,8-1
      z(i)=z(i)+(a(j)*x(i-j))
      end do
      end do
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      x(i)=mod(t,b)
      end do
      return
      end
!
! f90 interface to random number routine
! charles karney <karney@pppl.gov> 1999-09-30 16:58 -0400

 



 














module rngdef_4
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=kind(0.0)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(kind=double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef_4

module rng1_4
  ! the main module 
  use rngdef_4, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0_4
     subroutine rng_number_d1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_4
     subroutine rng_number_d2_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_4
     subroutine rng_number_d3_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3_4

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_number_s0_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0_4
     subroutine rng_number_s1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_4
     subroutine rng_number_s2_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_4
     subroutine rng_number_s3_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3_4

  end interface
  interface rng_gauss
     ! gaussian random numbers
     subroutine rng_gauss_d0_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0_4
     subroutine rng_gauss_d1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_4
     subroutine rng_gauss_d2_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_4
     subroutine rng_gauss_d3_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3_4

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_gauss_s0_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0_4
     subroutine rng_gauss_s1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_4
     subroutine rng_gauss_s2_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_4
     subroutine rng_gauss_s3_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3_4

  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time_4(seed,time)
       use rngdef_4
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time_4
     subroutine rng_set_seed_int_4(seed,n)
       use rngdef_4
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int_4
     subroutine rng_set_seed_string_4(seed,string)
       use rngdef_4
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string_4
  end interface

  interface
     function rng_print_seed_4(seed)
       ! return the seed in a string representation
       use rngdef_4
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed_4
     end function rng_print_seed_4
     subroutine rng_init_4(seed,state)
       ! initialize the state from the seed
       use rngdef_4
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init_4
     subroutine rng_step_seed_4(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef_4
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed_4
  end interface
end module rng1_4
 



 













module rngf77_4
  ! interface for the original f77 routines
  use rngdef_4
  implicit none
  interface
     function random1_4(ri,ra)
       use rngdef_4
       implicit none
       real(kind=double) :: random1_4
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random1_4

     function srandom1_4_4(ri,ra)
       use rngdef_4
       implicit none
       real(kind=single) :: srandom1_4_4
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom1_4_4

     subroutine random_array_4(y,n,ri,ra)
       use rngdef_4
       implicit none
       integer, intent(in) :: n
       real(kind=double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array_4

     subroutine srandom_array_4_4(y,n,ri,ra)
       use rngdef_4
       implicit none
       integer, intent(in) :: n
       real(kind=single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array_4_4

     subroutine rand_batch_4(ri,ra)
       use rngdef_4
       implicit none
       integer, intent(inout) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch_4
     subroutine random_init_4(seed,ri,ra)
       use rngdef_4
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init_4
     subroutine decimal_to_seed_4(decimal,seed)
       use rngdef_4
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed_4
     subroutine string_to_seed_4(string,seed)
       use rngdef_4
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed_4
     subroutine set_random_seed_4(time,seed)
       use rngdef_4
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed_4
     subroutine seed_to_decimal_4(seed,decimal)
       use rngdef_4
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal_4
     subroutine next_seed_43_4(n0,n1,n2,seed)
       use rngdef_4
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_43_4
     subroutine next_seed_4(n0,seed)
       use rngdef_4
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_4
  end interface
end module rngf77_4

subroutine rng_init_4(seed,state)
  use rngf77_4
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init_4(seed,state%index,state%array)
  return
end subroutine rng_init_4

subroutine rng_step_seed_4(seed,n0,n1,n2)
  use rngf77_4
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed_4(1,seed)
  else
     if (present(n0)) then
        m0=n0
     else
        m0=0
     endif
     if (present(n1)) then
        m1=n1
     else
        m1=0
     endif
     if (present(n2)) then
        m2=n2
     else
        m2=0
     endif
     call next_seed_43_4(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed_4

function rng_print_seed_4(seed)
  use rngf77_4
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed_4
  call seed_to_decimal_4(seed,rng_print_seed_4)
  return
end function rng_print_seed_4

subroutine rng_number_d0_4(state,x)
  use rngf77_4
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch_4(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0_4


subroutine rng_number_s0_4(state,x)
  use rngf77_4
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch_4(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0_4



subroutine rng_number_d1_4(state,x)
  use rngf77_4
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=state%array(i+state%index)+ulp2
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_4(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1_4


subroutine rng_number_s1_4(state,x)
  use rngf77_4
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=(int(mult*state%array(i+state%index))+0.5)*ulps
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_4(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1_4


subroutine rng_number_d2_4(state,x)
  use rngdef_4
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_4
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1_4(state,x(:,i))
  end do
end subroutine rng_number_d2_4


subroutine rng_number_s2_4(state,x)
  use rngdef_4
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_4
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1_4(state,x(:,i))
  end do
end subroutine rng_number_s2_4


subroutine rng_number_d3_4(state,x)
  use rngdef_4
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_4
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2_4(state,x(:,:,i))
  end do
end subroutine rng_number_d3_4


subroutine rng_number_s3_4(state,x)
  use rngdef_4
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_4
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2_4(state,x(:,:,i))
  end do
end subroutine rng_number_s3_4


subroutine rng_gauss_d1_4(state,x)
  use rngf77_4
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_4
  end interface
  integer :: i
  real(kind=double), parameter :: pi= 3.14159265358979323846264338328d0 
  real(kind=double) :: theta,z
  call rng_number_d1_4(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0d0 *x(i)- 1.0d0 )
     z=sqrt(- 2.0d0 *log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0d0 *x(ubound(x,1))- 1.0d0 )
  z=sqrt(- 2.0d0 *log(random1_4(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1_4

subroutine rng_gauss_d0_4(state,x)
  use rngdef_4
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_4
  end interface
  real(kind=double), dimension(1) :: y
  call rng_gauss_d1_4(state, y)
  x=y(1)
end subroutine rng_gauss_d0_4


subroutine rng_gauss_s1_4(state,x)
  use rngf77_4
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_4
  end interface
  integer :: i
  real(kind=single), parameter :: pi= 3.14159265358979323846264338328 
  real(kind=single) :: theta,z
  call rng_number_s1_4(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0*x(i)-1.0)
     z=sqrt(-2.0*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0*x(ubound(x,1))-1.0)
  z=sqrt(-2.0*log(srandom1_4_4(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1_4

subroutine rng_gauss_s0_4(state,x)
  use rngdef_4
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_4
  end interface
  real(kind=single), dimension(1) :: y
  call rng_gauss_s1_4(state, y)
  x=y(1)
end subroutine rng_gauss_s0_4


subroutine rng_gauss_d2_4(state,x)
  use rngdef_4
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_4
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1_4(state,x(:,i))
  end do
end subroutine rng_gauss_d2_4


subroutine rng_gauss_s2_4(state,x)
  use rngdef_4
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_4
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1_4(state,x(:,i))
  end do
end subroutine rng_gauss_s2_4


subroutine rng_gauss_d3_4(state,x)
  use rngdef_4
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2_4(state,x)
       use rngdef_4
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_4
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2_4(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3_4


subroutine rng_gauss_s3_4(state,x)
  use rngdef_4
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2_4(state,x)
       use rngdef_4
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_4
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2_4(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3_4


subroutine rng_set_seed_time_4(seed,time)
  use rngf77_4
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed_4(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed_4(ltime,seed)
  end if
end subroutine rng_set_seed_time_4

subroutine rng_set_seed_int_4(seed,n)
  use rngdef_4
  implicit none
  integer, dimension(0:rng_s-1), intent(out) :: seed
  integer, intent(in) :: n
  integer, parameter :: b=2**14
  integer :: m,i,sign
  if (n.lt.0) then
     seed=b-1
     m=-n-1
     sign=-1
  else
     seed=0
     m=n
     sign=1
  end if
  i=0
10 continue
  if (m.eq.0) return
  seed(i)=seed(i)+sign*mod(m,b)
  m=int(m/b)
  i=i+1
  goto 10
end subroutine rng_set_seed_int_4

subroutine rng_set_seed_string_4(seed,string)
  use rngf77_4
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed_4(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed_4(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string_4
 



 













! version 2.0 of random number routines
! author: charles karney <karney@princeton.edu>
! date: 1999-10-05 14:33 -0400
!
      function random1_4(ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      double precision  random1_4
      real srandom1_4_4
      integer ri
      double precision  ra(0:100-1)
      external rand_batch_4

      if(ri.ge.100)then
      call rand_batch_4(ri,ra(0))
      end if
      random1_4=ra(ri)+ulp2
      ri=ri+1
      return

      entry srandom1_4_4(ri,ra)
      if(ri.ge.100)then
      call rand_batch_4(ri,ra(0))
      end if

      srandom1_4_4=(int(mult*ra(ri))+0.5)*ulps

      ri=ri+1
      return
      end
!
      subroutine random_array_4(y,n,ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      integer n
      double precision  y(0:n-1)

      real ys(0:n-1)

      integer ri
      double precision  ra(0:100-1)
      integer i,k,j
      external rand_batch_4

      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      y(i)=ra(i+ri)+ulp2
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_4(ri,ra(0))
      do i=j,min(j+100,n)-1
      y(i)=ra(i-j+ri)+ulp2

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      entry srandom_array_4_4(ys,n,ri,ra)
      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      ys(i)=(int(mult*ra(i+ri))+0.5)*ulps
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_4(ri,ra(0))
      do i=j,min(j+100,n)-1
      ys(i)=(int(mult*ra(i-j+ri))+0.5)*ulps

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      end
!
      subroutine rand_batch_4(ri,ra)
      implicit none
      integer ri
      double precision  ra(0:100-1)
      integer i
      double precision  w(0:1009-100-1)
      double precision  tmp
      do i=0,63-1
      tmp=ra(i)+ra(i+100-63)

      w(i)=tmp-int(tmp)

      end do
      do i=63,100-1
      tmp=ra(i)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=100,1009-100-1
      tmp=w(i-100)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=1009-100,1009-100+63-1
      tmp=w(i-100)+w(i-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      do i=1009-100+63,1009-1
      tmp=w(i-100)+ra(i-1009+100-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      ri=0
      return
      end
!
      subroutine random_init_4(seed,ri,ra)
      implicit none
      integer b
      double precision  del,ulp
      parameter(b=2**14,del= 2.0d0 **(-14),ulp= 2.0d0 **(-47))
      integer a0,a1,a2,a3,a4,a5,a6,c0
      parameter(a0=15661,a1=678,a2=724,a3=5245,a4=13656,a5=11852,a6=29)
      parameter(c0=1)
      integer seed(0:8-1)
      integer ri
      double precision  ra(0:100-1)
      integer i,j,s(0:8-1)
      logical odd
      integer z(0:8-1),t
      do i=0,8-1
      s(i)=seed(i)
      end do
      odd=mod(s(7),2).ne.0
      ra(0)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      do j=1,100-1
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      odd=odd.or.(mod(s(7),2).ne.0)
      ra(j)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      end do
      ri=100
      if(odd)return
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      j=int((s(8-1)*100)/b)
      ra(j)=ra(j)+(ulp)
      return
      end
!
      subroutine decimal_to_seed_4(decimal,seed)
      implicit none
      character*(*)decimal
      integer seed(0:8-1)
      external rand_axc_4
      integer i,ten(0:8-1),c(0:8-1),ch
      data ten/10,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(decimal)
      ch=ichar(decimal(i:i))
      if(ch.ge.ichar('0').and.ch.le.ichar('9'))then
      c(0)=ch-ichar('0')
      call rand_axc_4(ten,seed,c)
      end if
      end do
      return
      end
!
      subroutine string_to_seed_4(string,seed)
      implicit none
      integer b
      parameter(b=2**14)
      character*(*)string
      integer seed(0:8-1)
      external rand_axc_4
      integer t,i,k,unity(0:8-1),c(0:8-1),ch
      data unity/1,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(string)
      ch=ichar(string(i:i))
      if(ch.gt.ichar(' ').and.ch.lt.127)then
      t=mod(seed(0),2)*(b/2)
      do k=0,8-1
      seed(k)=int(seed(k)/2)
      if(k.lt.8-1)then
      seed(k)=seed(k)+(mod(seed(k+1),2)*(b/2))
      else
      seed(k)=seed(k)+(t)
      end if
      end do
      c(0)=ch
      call rand_axc_4(unity,seed,c)
      end if
      end do
      return
      end
!
      subroutine set_random_seed_4(time,seed)
      implicit none
      integer time(8)
      integer seed(0:8-1)
      character*21 c
      external decimal_to_seed_4
      c=' '
      write(c(1:8),'(i4.4,2i2.2)')time(1),time(2),time(3)
      write(c(9:12),'(i1.1,i3.3)') (1-sign(1,time(4)))/2,abs(time(4))
      write(c(13:21),'(3i2.2,i3.3)')time(5),time(6),time(7),time(8)
      call decimal_to_seed_4(c,seed)
      return
      end
!
      subroutine seed_to_decimal_4(seed,decimal)
      implicit none
      integer pow,decbase,b
      parameter(pow=4,decbase=10**pow,b=2**14)
      character*(*)decimal
      integer seed(0:8-1)
      integer z(0:8-1),i,t,j,k
      character*36 str
      k=-1
      do i=0,8-1
      z(i)=seed(i)
      if(z(i).gt.0)k=i
      end do
      str=' '
      i=9
90000 continue
      i=i-1
      t=0
      do j=k,0,-1
      z(j)=z(j)+t*b
      t=mod(z(j),decbase)
      z(j)=int(z(j)/decbase)
      end do
      if(z(max(0,k)).eq.0)k=k-1
      j=pow*(i+1)
      if(k.ge.0)then
      str(j-(pow-1):j)='0000'
      else
      str(j-(pow-1):j)='   0'
      end if
90001 continue
      if(t.eq.0)goto 90010
      str(j:j)=char(ichar('0')+mod(t,10))
      j=j-1
      t=int(t/10)
      goto 90001
90010 continue
      if(k.ge.0)goto 90000
      if(len(decimal).gt.len(str))then
      decimal(:len(decimal)-len(str))=' '
      decimal(len(decimal)-len(str)+1:)=str
      else
      decimal=str(len(str)-len(decimal)+1:)
      end if
      return
      end
!
      subroutine rand_next_seed_4(n,ax,cx,y)
      implicit none
      integer n,ax(0:8-1),cx(0:8-1)
      integer y(0:8-1)
      external rand_axc_4
      integer a(0:8-1),c(0:8-1),z(0:8-1),t(0:8-1),m,i
      data z/8*0/
      if(n.eq.0)return
      m=n
      do i=0,8-1
      a(i)=ax(i)
      c(i)=cx(i)
      end do
90000 continue
      if(mod(m,2).gt.0)then
      call rand_axc_4(a,y,c)
      end if
      m=int(m/2)
      if(m.eq.0)return
      do i=0,8-1
      t(i)=c(i)
      end do
      call rand_axc_4(a,c,t)
      do i=0,8-1
      t(i)=a(i)
      end do
      call rand_axc_4(t,a,z)
      goto 90000
      end
!
      subroutine next_seed_43_4(n0,n1,n2,seed)
      implicit none
      integer n0,n1,n2
      integer seed(0:8-1)
      external rand_next_seed_4
      integer af0(0:8-1),cf0(0:8-1)
      integer ab0(0:8-1),cb0(0:8-1)
      integer af1(0:8-1),cf1(0:8-1)
      integer ab1(0:8-1),cb1(0:8-1)
      integer af2(0:8-1),cf2(0:8-1)
      integer ab2(0:8-1),cb2(0:8-1)
      data af0/15741,8689,9280,4732,12011,7130,6824,12302/
      data cf0/16317,10266,1198,331,10769,8310,2779,13880/
      data ab0/9173,9894,15203,15379,7981,2280,8071,429/
      data cb0/8383,3616,597,12724,15663,9639,187,4866/
      data af1/8405,4808,3603,6718,13766,9243,10375,12108/
      data cf1/13951,7170,9039,11206,8706,14101,1864,15191/
      data ab1/6269,3240,9759,7130,15320,14399,3675,1380/
      data cb1/15357,5843,6205,16275,8838,12132,2198,10330/
      data af2/445,10754,1869,6593,385,12498,14501,7383/
      data cf2/2285,8057,3864,10235,1805,10614,9615,15522/
      data ab2/405,4903,2746,1477,3263,13564,8139,2362/
      data cb2/8463,575,5876,2220,4924,1701,9060,5639/
      if(n2.gt.0)then
      call rand_next_seed_4(n2,af2,cf2,seed)
      else if(n2.lt.0)then
      call rand_next_seed_4(-n2,ab2,cb2,seed)
      end if
      if(n1.gt.0)then
      call rand_next_seed_4(n1,af1,cf1,seed)
      else if(n1.lt.0)then
      call rand_next_seed_4(-n1,ab1,cb1,seed)
      end if
      entry next_seed_4(n0,seed)
      if(n0.gt.0)then
      call rand_next_seed_4(n0,af0,cf0,seed)
      else if(n0.lt.0)then
      call rand_next_seed_4(-n0,ab0,cb0,seed)
      end if
      return
      end
!
      subroutine rand_axc_4(a,x,c)
      implicit none
      integer b
      parameter(b=2**14)
      integer a(0:8-1),c(0:8-1)
      integer x(0:8-1)
      integer z(0:8-1),i,j,t
      do i=0,8-1
      z(i)=c(i)
      end do
      do j=0,8-1
      do i=j,8-1
      z(i)=z(i)+(a(j)*x(i-j))
      end do
      end do
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      x(i)=mod(t,b)
      end do
      return
      end
!
! f90 interface to random number routine
! charles karney <karney@pppl.gov> 1999-09-30 16:58 -0400

 



 














module rngdef_5
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=kind(0.0)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(kind=double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef_5

module rng1_5
  ! the main module 
  use rngdef_5, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0_5
     subroutine rng_number_d1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_5
     subroutine rng_number_d2_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_5
     subroutine rng_number_d3_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3_5

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_number_s0_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0_5
     subroutine rng_number_s1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_5
     subroutine rng_number_s2_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_5
     subroutine rng_number_s3_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3_5

  end interface
  interface rng_gauss
     ! gaussian random numbers
     subroutine rng_gauss_d0_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0_5
     subroutine rng_gauss_d1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_5
     subroutine rng_gauss_d2_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_5
     subroutine rng_gauss_d3_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3_5

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_gauss_s0_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0_5
     subroutine rng_gauss_s1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_5
     subroutine rng_gauss_s2_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_5
     subroutine rng_gauss_s3_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3_5

  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time_5(seed,time)
       use rngdef_5
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time_5
     subroutine rng_set_seed_int_5(seed,n)
       use rngdef_5
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int_5
     subroutine rng_set_seed_string_5(seed,string)
       use rngdef_5
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string_5
  end interface

  interface
     function rng_print_seed_5(seed)
       ! return the seed in a string representation
       use rngdef_5
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed_5
     end function rng_print_seed_5
     subroutine rng_init_5(seed,state)
       ! initialize the state from the seed
       use rngdef_5
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init_5
     subroutine rng_step_seed_5(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef_5
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed_5
  end interface
end module rng1_5
 



 













module rngf77_5
  ! interface for the original f77 routines
  use rngdef_5
  implicit none
  interface
     function random1_5(ri,ra)
       use rngdef_5
       implicit none
       real(kind=double) :: random1_5
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random1_5

     function srandom1_5_5(ri,ra)
       use rngdef_5
       implicit none
       real(kind=single) :: srandom1_5_5
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom1_5_5

     subroutine random_array_5(y,n,ri,ra)
       use rngdef_5
       implicit none
       integer, intent(in) :: n
       real(kind=double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array_5

     subroutine srandom_array_5_5(y,n,ri,ra)
       use rngdef_5
       implicit none
       integer, intent(in) :: n
       real(kind=single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array_5_5

     subroutine rand_batch_5(ri,ra)
       use rngdef_5
       implicit none
       integer, intent(inout) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch_5
     subroutine random_init_5(seed,ri,ra)
       use rngdef_5
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init_5
     subroutine decimal_to_seed_5(decimal,seed)
       use rngdef_5
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed_5
     subroutine string_to_seed_5(string,seed)
       use rngdef_5
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed_5
     subroutine set_random_seed_5(time,seed)
       use rngdef_5
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed_5
     subroutine seed_to_decimal_5(seed,decimal)
       use rngdef_5
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal_5
     subroutine next_seed_53_5(n0,n1,n2,seed)
       use rngdef_5
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_53_5
     subroutine next_seed_5(n0,seed)
       use rngdef_5
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_5
  end interface
end module rngf77_5

subroutine rng_init_5(seed,state)
  use rngf77_5
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init_5(seed,state%index,state%array)
  return
end subroutine rng_init_5

subroutine rng_step_seed_5(seed,n0,n1,n2)
  use rngf77_5
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed_5(1,seed)
  else
     if (present(n0)) then
        m0=n0
     else
        m0=0
     endif
     if (present(n1)) then
        m1=n1
     else
        m1=0
     endif
     if (present(n2)) then
        m2=n2
     else
        m2=0
     endif
     call next_seed_53_5(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed_5

function rng_print_seed_5(seed)
  use rngf77_5
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed_5
  call seed_to_decimal_5(seed,rng_print_seed_5)
  return
end function rng_print_seed_5

subroutine rng_number_d0_5(state,x)
  use rngf77_5
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch_5(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0_5


subroutine rng_number_s0_5(state,x)
  use rngf77_5
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch_5(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0_5



subroutine rng_number_d1_5(state,x)
  use rngf77_5
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=state%array(i+state%index)+ulp2
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_5(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1_5


subroutine rng_number_s1_5(state,x)
  use rngf77_5
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=(int(mult*state%array(i+state%index))+0.5)*ulps
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_5(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1_5


subroutine rng_number_d2_5(state,x)
  use rngdef_5
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_5
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1_5(state,x(:,i))
  end do
end subroutine rng_number_d2_5


subroutine rng_number_s2_5(state,x)
  use rngdef_5
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_5
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1_5(state,x(:,i))
  end do
end subroutine rng_number_s2_5


subroutine rng_number_d3_5(state,x)
  use rngdef_5
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_5
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2_5(state,x(:,:,i))
  end do
end subroutine rng_number_d3_5


subroutine rng_number_s3_5(state,x)
  use rngdef_5
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_5
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2_5(state,x(:,:,i))
  end do
end subroutine rng_number_s3_5


subroutine rng_gauss_d1_5(state,x)
  use rngf77_5
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_5
  end interface
  integer :: i
  real(kind=double), parameter :: pi= 3.14159265358979323846264338328d0 
  real(kind=double) :: theta,z
  call rng_number_d1_5(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0d0 *x(i)- 1.0d0 )
     z=sqrt(- 2.0d0 *log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0d0 *x(ubound(x,1))- 1.0d0 )
  z=sqrt(- 2.0d0 *log(random1_5(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1_5

subroutine rng_gauss_d0_5(state,x)
  use rngdef_5
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_5
  end interface
  real(kind=double), dimension(1) :: y
  call rng_gauss_d1_5(state, y)
  x=y(1)
end subroutine rng_gauss_d0_5


subroutine rng_gauss_s1_5(state,x)
  use rngf77_5
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_5
  end interface
  integer :: i
  real(kind=single), parameter :: pi= 3.14159265358979323846264338328 
  real(kind=single) :: theta,z
  call rng_number_s1_5(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0*x(i)-1.0)
     z=sqrt(-2.0*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0*x(ubound(x,1))-1.0)
  z=sqrt(-2.0*log(srandom1_5_5(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1_5

subroutine rng_gauss_s0_5(state,x)
  use rngdef_5
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_5
  end interface
  real(kind=single), dimension(1) :: y
  call rng_gauss_s1_5(state, y)
  x=y(1)
end subroutine rng_gauss_s0_5


subroutine rng_gauss_d2_5(state,x)
  use rngdef_5
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_5
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1_5(state,x(:,i))
  end do
end subroutine rng_gauss_d2_5


subroutine rng_gauss_s2_5(state,x)
  use rngdef_5
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_5
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1_5(state,x(:,i))
  end do
end subroutine rng_gauss_s2_5


subroutine rng_gauss_d3_5(state,x)
  use rngdef_5
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2_5(state,x)
       use rngdef_5
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_5
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2_5(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3_5


subroutine rng_gauss_s3_5(state,x)
  use rngdef_5
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2_5(state,x)
       use rngdef_5
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_5
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2_5(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3_5


subroutine rng_set_seed_time_5(seed,time)
  use rngf77_5
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed_5(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed_5(ltime,seed)
  end if
end subroutine rng_set_seed_time_5

subroutine rng_set_seed_int_5(seed,n)
  use rngdef_5
  implicit none
  integer, dimension(0:rng_s-1), intent(out) :: seed
  integer, intent(in) :: n
  integer, parameter :: b=2**14
  integer :: m,i,sign
  if (n.lt.0) then
     seed=b-1
     m=-n-1
     sign=-1
  else
     seed=0
     m=n
     sign=1
  end if
  i=0
10 continue
  if (m.eq.0) return
  seed(i)=seed(i)+sign*mod(m,b)
  m=int(m/b)
  i=i+1
  goto 10
end subroutine rng_set_seed_int_5

subroutine rng_set_seed_string_5(seed,string)
  use rngf77_5
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed_5(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed_5(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string_5
 



 













! version 2.0 of random number routines
! author: charles karney <karney@princeton.edu>
! date: 1999-10-05 14:33 -0400
!
      function random1_5(ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      double precision  random1_5
      real srandom1_5_5
      integer ri
      double precision  ra(0:100-1)
      external rand_batch_5

      if(ri.ge.100)then
      call rand_batch_5(ri,ra(0))
      end if
      random1_5=ra(ri)+ulp2
      ri=ri+1
      return

      entry srandom1_5_5(ri,ra)
      if(ri.ge.100)then
      call rand_batch_5(ri,ra(0))
      end if

      srandom1_5_5=(int(mult*ra(ri))+0.5)*ulps

      ri=ri+1
      return
      end
!
      subroutine random_array_5(y,n,ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      integer n
      double precision  y(0:n-1)

      real ys(0:n-1)

      integer ri
      double precision  ra(0:100-1)
      integer i,k,j
      external rand_batch_5

      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      y(i)=ra(i+ri)+ulp2
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_5(ri,ra(0))
      do i=j,min(j+100,n)-1
      y(i)=ra(i-j+ri)+ulp2

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      entry srandom_array_5_5(ys,n,ri,ra)
      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      ys(i)=(int(mult*ra(i+ri))+0.5)*ulps
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_5(ri,ra(0))
      do i=j,min(j+100,n)-1
      ys(i)=(int(mult*ra(i-j+ri))+0.5)*ulps

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      end
!
      subroutine rand_batch_5(ri,ra)
      implicit none
      integer ri
      double precision  ra(0:100-1)
      integer i
      double precision  w(0:1009-100-1)
      double precision  tmp
      do i=0,63-1
      tmp=ra(i)+ra(i+100-63)

      w(i)=tmp-int(tmp)

      end do
      do i=63,100-1
      tmp=ra(i)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=100,1009-100-1
      tmp=w(i-100)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=1009-100,1009-100+63-1
      tmp=w(i-100)+w(i-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      do i=1009-100+63,1009-1
      tmp=w(i-100)+ra(i-1009+100-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      ri=0
      return
      end
!
      subroutine random_init_5(seed,ri,ra)
      implicit none
      integer b
      double precision  del,ulp
      parameter(b=2**14,del= 2.0d0 **(-14),ulp= 2.0d0 **(-47))
      integer a0,a1,a2,a3,a4,a5,a6,c0
      parameter(a0=15661,a1=678,a2=724,a3=5245,a4=13656,a5=11852,a6=29)
      parameter(c0=1)
      integer seed(0:8-1)
      integer ri
      double precision  ra(0:100-1)
      integer i,j,s(0:8-1)
      logical odd
      integer z(0:8-1),t
      do i=0,8-1
      s(i)=seed(i)
      end do
      odd=mod(s(7),2).ne.0
      ra(0)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      do j=1,100-1
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      odd=odd.or.(mod(s(7),2).ne.0)
      ra(j)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      end do
      ri=100
      if(odd)return
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      j=int((s(8-1)*100)/b)
      ra(j)=ra(j)+(ulp)
      return
      end
!
      subroutine decimal_to_seed_5(decimal,seed)
      implicit none
      character*(*)decimal
      integer seed(0:8-1)
      external rand_axc_5
      integer i,ten(0:8-1),c(0:8-1),ch
      data ten/10,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(decimal)
      ch=ichar(decimal(i:i))
      if(ch.ge.ichar('0').and.ch.le.ichar('9'))then
      c(0)=ch-ichar('0')
      call rand_axc_5(ten,seed,c)
      end if
      end do
      return
      end
!
      subroutine string_to_seed_5(string,seed)
      implicit none
      integer b
      parameter(b=2**14)
      character*(*)string
      integer seed(0:8-1)
      external rand_axc_5
      integer t,i,k,unity(0:8-1),c(0:8-1),ch
      data unity/1,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(string)
      ch=ichar(string(i:i))
      if(ch.gt.ichar(' ').and.ch.lt.127)then
      t=mod(seed(0),2)*(b/2)
      do k=0,8-1
      seed(k)=int(seed(k)/2)
      if(k.lt.8-1)then
      seed(k)=seed(k)+(mod(seed(k+1),2)*(b/2))
      else
      seed(k)=seed(k)+(t)
      end if
      end do
      c(0)=ch
      call rand_axc_5(unity,seed,c)
      end if
      end do
      return
      end
!
      subroutine set_random_seed_5(time,seed)
      implicit none
      integer time(8)
      integer seed(0:8-1)
      character*21 c
      external decimal_to_seed_5
      c=' '
      write(c(1:8),'(i4.4,2i2.2)')time(1),time(2),time(3)
      write(c(9:12),'(i1.1,i3.3)') (1-sign(1,time(4)))/2,abs(time(4))
      write(c(13:21),'(3i2.2,i3.3)')time(5),time(6),time(7),time(8)
      call decimal_to_seed_5(c,seed)
      return
      end
!
      subroutine seed_to_decimal_5(seed,decimal)
      implicit none
      integer pow,decbase,b
      parameter(pow=4,decbase=10**pow,b=2**14)
      character*(*)decimal
      integer seed(0:8-1)
      integer z(0:8-1),i,t,j,k
      character*36 str
      k=-1
      do i=0,8-1
      z(i)=seed(i)
      if(z(i).gt.0)k=i
      end do
      str=' '
      i=9
90000 continue
      i=i-1
      t=0
      do j=k,0,-1
      z(j)=z(j)+t*b
      t=mod(z(j),decbase)
      z(j)=int(z(j)/decbase)
      end do
      if(z(max(0,k)).eq.0)k=k-1
      j=pow*(i+1)
      if(k.ge.0)then
      str(j-(pow-1):j)='0000'
      else
      str(j-(pow-1):j)='   0'
      end if
90001 continue
      if(t.eq.0)goto 90010
      str(j:j)=char(ichar('0')+mod(t,10))
      j=j-1
      t=int(t/10)
      goto 90001
90010 continue
      if(k.ge.0)goto 90000
      if(len(decimal).gt.len(str))then
      decimal(:len(decimal)-len(str))=' '
      decimal(len(decimal)-len(str)+1:)=str
      else
      decimal=str(len(str)-len(decimal)+1:)
      end if
      return
      end
!
      subroutine rand_next_seed_5(n,ax,cx,y)
      implicit none
      integer n,ax(0:8-1),cx(0:8-1)
      integer y(0:8-1)
      external rand_axc_5
      integer a(0:8-1),c(0:8-1),z(0:8-1),t(0:8-1),m,i
      data z/8*0/
      if(n.eq.0)return
      m=n
      do i=0,8-1
      a(i)=ax(i)
      c(i)=cx(i)
      end do
90000 continue
      if(mod(m,2).gt.0)then
      call rand_axc_5(a,y,c)
      end if
      m=int(m/2)
      if(m.eq.0)return
      do i=0,8-1
      t(i)=c(i)
      end do
      call rand_axc_5(a,c,t)
      do i=0,8-1
      t(i)=a(i)
      end do
      call rand_axc_5(t,a,z)
      goto 90000
      end
!
      subroutine next_seed_53_5(n0,n1,n2,seed)
      implicit none
      integer n0,n1,n2
      integer seed(0:8-1)
      external rand_next_seed_5
      integer af0(0:8-1),cf0(0:8-1)
      integer ab0(0:8-1),cb0(0:8-1)
      integer af1(0:8-1),cf1(0:8-1)
      integer ab1(0:8-1),cb1(0:8-1)
      integer af2(0:8-1),cf2(0:8-1)
      integer ab2(0:8-1),cb2(0:8-1)
      data af0/15741,8689,9280,4732,12011,7130,6824,12302/
      data cf0/16317,10266,1198,331,10769,8310,2779,13880/
      data ab0/9173,9894,15203,15379,7981,2280,8071,429/
      data cb0/8383,3616,597,12724,15663,9639,187,4866/
      data af1/8405,4808,3603,6718,13766,9243,10375,12108/
      data cf1/13951,7170,9039,11206,8706,14101,1864,15191/
      data ab1/6269,3240,9759,7130,15320,14399,3675,1380/
      data cb1/15357,5843,6205,16275,8838,12132,2198,10330/
      data af2/445,10754,1869,6593,385,12498,14501,7383/
      data cf2/2285,8057,3864,10235,1805,10614,9615,15522/
      data ab2/405,4903,2746,1477,3263,13564,8139,2362/
      data cb2/8463,575,5876,2220,4924,1701,9060,5639/
      if(n2.gt.0)then
      call rand_next_seed_5(n2,af2,cf2,seed)
      else if(n2.lt.0)then
      call rand_next_seed_5(-n2,ab2,cb2,seed)
      end if
      if(n1.gt.0)then
      call rand_next_seed_5(n1,af1,cf1,seed)
      else if(n1.lt.0)then
      call rand_next_seed_5(-n1,ab1,cb1,seed)
      end if
      entry next_seed_5(n0,seed)
      if(n0.gt.0)then
      call rand_next_seed_5(n0,af0,cf0,seed)
      else if(n0.lt.0)then
      call rand_next_seed_5(-n0,ab0,cb0,seed)
      end if
      return
      end
!
      subroutine rand_axc_5(a,x,c)
      implicit none
      integer b
      parameter(b=2**14)
      integer a(0:8-1),c(0:8-1)
      integer x(0:8-1)
      integer z(0:8-1),i,j,t
      do i=0,8-1
      z(i)=c(i)
      end do
      do j=0,8-1
      do i=j,8-1
      z(i)=z(i)+(a(j)*x(i-j))
      end do
      end do
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      x(i)=mod(t,b)
      end do
      return
      end
!
! f90 interface to random number routine
! charles karney <karney@pppl.gov> 1999-09-30 16:58 -0400

 



 














module rngdef_6
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=kind(0.0)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(kind=double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef_6

module rng1_6
  ! the main module 
  use rngdef_6, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0_6
     subroutine rng_number_d1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_6
     subroutine rng_number_d2_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_6
     subroutine rng_number_d3_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3_6

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_number_s0_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0_6
     subroutine rng_number_s1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_6
     subroutine rng_number_s2_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_6
     subroutine rng_number_s3_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3_6

  end interface
  interface rng_gauss
     ! gaussian random numbers
     subroutine rng_gauss_d0_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0_6
     subroutine rng_gauss_d1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_6
     subroutine rng_gauss_d2_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_6
     subroutine rng_gauss_d3_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3_6

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_gauss_s0_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0_6
     subroutine rng_gauss_s1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_6
     subroutine rng_gauss_s2_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_6
     subroutine rng_gauss_s3_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3_6

  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time_6(seed,time)
       use rngdef_6
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time_6
     subroutine rng_set_seed_int_6(seed,n)
       use rngdef_6
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int_6
     subroutine rng_set_seed_string_6(seed,string)
       use rngdef_6
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string_6
  end interface

  interface
     function rng_print_seed_6(seed)
       ! return the seed in a string representation
       use rngdef_6
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed_6
     end function rng_print_seed_6
     subroutine rng_init_6(seed,state)
       ! initialize the state from the seed
       use rngdef_6
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init_6
     subroutine rng_step_seed_6(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef_6
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed_6
  end interface
end module rng1_6
 



 













module rngf77_6
  ! interface for the original f77 routines
  use rngdef_6
  implicit none
  interface
     function random1_6(ri,ra)
       use rngdef_6
       implicit none
       real(kind=double) :: random1_6
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random1_6

     function srandom1_6_6(ri,ra)
       use rngdef_6
       implicit none
       real(kind=single) :: srandom1_6_6
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom1_6_6

     subroutine random_array_6(y,n,ri,ra)
       use rngdef_6
       implicit none
       integer, intent(in) :: n
       real(kind=double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array_6

     subroutine srandom_array_6_6(y,n,ri,ra)
       use rngdef_6
       implicit none
       integer, intent(in) :: n
       real(kind=single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array_6_6

     subroutine rand_batch_6(ri,ra)
       use rngdef_6
       implicit none
       integer, intent(inout) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch_6
     subroutine random_init_6(seed,ri,ra)
       use rngdef_6
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init_6
     subroutine decimal_to_seed_6(decimal,seed)
       use rngdef_6
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed_6
     subroutine string_to_seed_6(string,seed)
       use rngdef_6
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed_6
     subroutine set_random_seed_6(time,seed)
       use rngdef_6
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed_6
     subroutine seed_to_decimal_6(seed,decimal)
       use rngdef_6
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal_6
     subroutine next_seed_63_6(n0,n1,n2,seed)
       use rngdef_6
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_63_6
     subroutine next_seed_6(n0,seed)
       use rngdef_6
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_6
  end interface
end module rngf77_6

subroutine rng_init_6(seed,state)
  use rngf77_6
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init_6(seed,state%index,state%array)
  return
end subroutine rng_init_6

subroutine rng_step_seed_6(seed,n0,n1,n2)
  use rngf77_6
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed_6(1,seed)
  else
     if (present(n0)) then
        m0=n0
     else
        m0=0
     endif
     if (present(n1)) then
        m1=n1
     else
        m1=0
     endif
     if (present(n2)) then
        m2=n2
     else
        m2=0
     endif
     call next_seed_63_6(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed_6

function rng_print_seed_6(seed)
  use rngf77_6
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed_6
  call seed_to_decimal_6(seed,rng_print_seed_6)
  return
end function rng_print_seed_6

subroutine rng_number_d0_6(state,x)
  use rngf77_6
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch_6(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0_6


subroutine rng_number_s0_6(state,x)
  use rngf77_6
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch_6(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0_6



subroutine rng_number_d1_6(state,x)
  use rngf77_6
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=state%array(i+state%index)+ulp2
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_6(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1_6


subroutine rng_number_s1_6(state,x)
  use rngf77_6
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=(int(mult*state%array(i+state%index))+0.5)*ulps
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_6(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1_6


subroutine rng_number_d2_6(state,x)
  use rngdef_6
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_6
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1_6(state,x(:,i))
  end do
end subroutine rng_number_d2_6


subroutine rng_number_s2_6(state,x)
  use rngdef_6
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_6
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1_6(state,x(:,i))
  end do
end subroutine rng_number_s2_6


subroutine rng_number_d3_6(state,x)
  use rngdef_6
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_6
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2_6(state,x(:,:,i))
  end do
end subroutine rng_number_d3_6


subroutine rng_number_s3_6(state,x)
  use rngdef_6
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_6
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2_6(state,x(:,:,i))
  end do
end subroutine rng_number_s3_6


subroutine rng_gauss_d1_6(state,x)
  use rngf77_6
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_6
  end interface
  integer :: i
  real(kind=double), parameter :: pi= 3.14159265358979323846264338328d0 
  real(kind=double) :: theta,z
  call rng_number_d1_6(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0d0 *x(i)- 1.0d0 )
     z=sqrt(- 2.0d0 *log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0d0 *x(ubound(x,1))- 1.0d0 )
  z=sqrt(- 2.0d0 *log(random1_6(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1_6

subroutine rng_gauss_d0_6(state,x)
  use rngdef_6
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_6
  end interface
  real(kind=double), dimension(1) :: y
  call rng_gauss_d1_6(state, y)
  x=y(1)
end subroutine rng_gauss_d0_6


subroutine rng_gauss_s1_6(state,x)
  use rngf77_6
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_6
  end interface
  integer :: i
  real(kind=single), parameter :: pi= 3.14159265358979323846264338328 
  real(kind=single) :: theta,z
  call rng_number_s1_6(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0*x(i)-1.0)
     z=sqrt(-2.0*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0*x(ubound(x,1))-1.0)
  z=sqrt(-2.0*log(srandom1_6_6(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1_6

subroutine rng_gauss_s0_6(state,x)
  use rngdef_6
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_6
  end interface
  real(kind=single), dimension(1) :: y
  call rng_gauss_s1_6(state, y)
  x=y(1)
end subroutine rng_gauss_s0_6


subroutine rng_gauss_d2_6(state,x)
  use rngdef_6
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_6
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1_6(state,x(:,i))
  end do
end subroutine rng_gauss_d2_6


subroutine rng_gauss_s2_6(state,x)
  use rngdef_6
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_6
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1_6(state,x(:,i))
  end do
end subroutine rng_gauss_s2_6


subroutine rng_gauss_d3_6(state,x)
  use rngdef_6
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2_6(state,x)
       use rngdef_6
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_6
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2_6(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3_6


subroutine rng_gauss_s3_6(state,x)
  use rngdef_6
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2_6(state,x)
       use rngdef_6
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_6
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2_6(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3_6


subroutine rng_set_seed_time_6(seed,time)
  use rngf77_6
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed_6(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed_6(ltime,seed)
  end if
end subroutine rng_set_seed_time_6

subroutine rng_set_seed_int_6(seed,n)
  use rngdef_6
  implicit none
  integer, dimension(0:rng_s-1), intent(out) :: seed
  integer, intent(in) :: n
  integer, parameter :: b=2**14
  integer :: m,i,sign
  if (n.lt.0) then
     seed=b-1
     m=-n-1
     sign=-1
  else
     seed=0
     m=n
     sign=1
  end if
  i=0
10 continue
  if (m.eq.0) return
  seed(i)=seed(i)+sign*mod(m,b)
  m=int(m/b)
  i=i+1
  goto 10
end subroutine rng_set_seed_int_6

subroutine rng_set_seed_string_6(seed,string)
  use rngf77_6
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed_6(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed_6(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string_6
 



 













! version 2.0 of random number routines
! author: charles karney <karney@princeton.edu>
! date: 1999-10-05 14:33 -0400
!
      function random1_6(ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      double precision  random1_6
      real srandom1_6_6
      integer ri
      double precision  ra(0:100-1)
      external rand_batch_6

      if(ri.ge.100)then
      call rand_batch_6(ri,ra(0))
      end if
      random1_6=ra(ri)+ulp2
      ri=ri+1
      return

      entry srandom1_6_6(ri,ra)
      if(ri.ge.100)then
      call rand_batch_6(ri,ra(0))
      end if

      srandom1_6_6=(int(mult*ra(ri))+0.5)*ulps

      ri=ri+1
      return
      end
!
      subroutine random_array_6(y,n,ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      integer n
      double precision  y(0:n-1)

      real ys(0:n-1)

      integer ri
      double precision  ra(0:100-1)
      integer i,k,j
      external rand_batch_6

      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      y(i)=ra(i+ri)+ulp2
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_6(ri,ra(0))
      do i=j,min(j+100,n)-1
      y(i)=ra(i-j+ri)+ulp2

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      entry srandom_array_6_6(ys,n,ri,ra)
      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      ys(i)=(int(mult*ra(i+ri))+0.5)*ulps
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_6(ri,ra(0))
      do i=j,min(j+100,n)-1
      ys(i)=(int(mult*ra(i-j+ri))+0.5)*ulps

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      end
!
      subroutine rand_batch_6(ri,ra)
      implicit none
      integer ri
      double precision  ra(0:100-1)
      integer i
      double precision  w(0:1009-100-1)
      double precision  tmp
      do i=0,63-1
      tmp=ra(i)+ra(i+100-63)

      w(i)=tmp-int(tmp)

      end do
      do i=63,100-1
      tmp=ra(i)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=100,1009-100-1
      tmp=w(i-100)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=1009-100,1009-100+63-1
      tmp=w(i-100)+w(i-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      do i=1009-100+63,1009-1
      tmp=w(i-100)+ra(i-1009+100-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      ri=0
      return
      end
!
      subroutine random_init_6(seed,ri,ra)
      implicit none
      integer b
      double precision  del,ulp
      parameter(b=2**14,del= 2.0d0 **(-14),ulp= 2.0d0 **(-47))
      integer a0,a1,a2,a3,a4,a5,a6,c0
      parameter(a0=15661,a1=678,a2=724,a3=5245,a4=13656,a5=11852,a6=29)
      parameter(c0=1)
      integer seed(0:8-1)
      integer ri
      double precision  ra(0:100-1)
      integer i,j,s(0:8-1)
      logical odd
      integer z(0:8-1),t
      do i=0,8-1
      s(i)=seed(i)
      end do
      odd=mod(s(7),2).ne.0
      ra(0)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      do j=1,100-1
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      odd=odd.or.(mod(s(7),2).ne.0)
      ra(j)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      end do
      ri=100
      if(odd)return
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      j=int((s(8-1)*100)/b)
      ra(j)=ra(j)+(ulp)
      return
      end
!
      subroutine decimal_to_seed_6(decimal,seed)
      implicit none
      character*(*)decimal
      integer seed(0:8-1)
      external rand_axc_6
      integer i,ten(0:8-1),c(0:8-1),ch
      data ten/10,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(decimal)
      ch=ichar(decimal(i:i))
      if(ch.ge.ichar('0').and.ch.le.ichar('9'))then
      c(0)=ch-ichar('0')
      call rand_axc_6(ten,seed,c)
      end if
      end do
      return
      end
!
      subroutine string_to_seed_6(string,seed)
      implicit none
      integer b
      parameter(b=2**14)
      character*(*)string
      integer seed(0:8-1)
      external rand_axc_6
      integer t,i,k,unity(0:8-1),c(0:8-1),ch
      data unity/1,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(string)
      ch=ichar(string(i:i))
      if(ch.gt.ichar(' ').and.ch.lt.127)then
      t=mod(seed(0),2)*(b/2)
      do k=0,8-1
      seed(k)=int(seed(k)/2)
      if(k.lt.8-1)then
      seed(k)=seed(k)+(mod(seed(k+1),2)*(b/2))
      else
      seed(k)=seed(k)+(t)
      end if
      end do
      c(0)=ch
      call rand_axc_6(unity,seed,c)
      end if
      end do
      return
      end
!
      subroutine set_random_seed_6(time,seed)
      implicit none
      integer time(8)
      integer seed(0:8-1)
      character*21 c
      external decimal_to_seed_6
      c=' '
      write(c(1:8),'(i4.4,2i2.2)')time(1),time(2),time(3)
      write(c(9:12),'(i1.1,i3.3)') (1-sign(1,time(4)))/2,abs(time(4))
      write(c(13:21),'(3i2.2,i3.3)')time(5),time(6),time(7),time(8)
      call decimal_to_seed_6(c,seed)
      return
      end
!
      subroutine seed_to_decimal_6(seed,decimal)
      implicit none
      integer pow,decbase,b
      parameter(pow=4,decbase=10**pow,b=2**14)
      character*(*)decimal
      integer seed(0:8-1)
      integer z(0:8-1),i,t,j,k
      character*36 str
      k=-1
      do i=0,8-1
      z(i)=seed(i)
      if(z(i).gt.0)k=i
      end do
      str=' '
      i=9
90000 continue
      i=i-1
      t=0
      do j=k,0,-1
      z(j)=z(j)+t*b
      t=mod(z(j),decbase)
      z(j)=int(z(j)/decbase)
      end do
      if(z(max(0,k)).eq.0)k=k-1
      j=pow*(i+1)
      if(k.ge.0)then
      str(j-(pow-1):j)='0000'
      else
      str(j-(pow-1):j)='   0'
      end if
90001 continue
      if(t.eq.0)goto 90010
      str(j:j)=char(ichar('0')+mod(t,10))
      j=j-1
      t=int(t/10)
      goto 90001
90010 continue
      if(k.ge.0)goto 90000
      if(len(decimal).gt.len(str))then
      decimal(:len(decimal)-len(str))=' '
      decimal(len(decimal)-len(str)+1:)=str
      else
      decimal=str(len(str)-len(decimal)+1:)
      end if
      return
      end
!
      subroutine rand_next_seed_6(n,ax,cx,y)
      implicit none
      integer n,ax(0:8-1),cx(0:8-1)
      integer y(0:8-1)
      external rand_axc_6
      integer a(0:8-1),c(0:8-1),z(0:8-1),t(0:8-1),m,i
      data z/8*0/
      if(n.eq.0)return
      m=n
      do i=0,8-1
      a(i)=ax(i)
      c(i)=cx(i)
      end do
90000 continue
      if(mod(m,2).gt.0)then
      call rand_axc_6(a,y,c)
      end if
      m=int(m/2)
      if(m.eq.0)return
      do i=0,8-1
      t(i)=c(i)
      end do
      call rand_axc_6(a,c,t)
      do i=0,8-1
      t(i)=a(i)
      end do
      call rand_axc_6(t,a,z)
      goto 90000
      end
!
      subroutine next_seed_63_6(n0,n1,n2,seed)
      implicit none
      integer n0,n1,n2
      integer seed(0:8-1)
      external rand_next_seed_6
      integer af0(0:8-1),cf0(0:8-1)
      integer ab0(0:8-1),cb0(0:8-1)
      integer af1(0:8-1),cf1(0:8-1)
      integer ab1(0:8-1),cb1(0:8-1)
      integer af2(0:8-1),cf2(0:8-1)
      integer ab2(0:8-1),cb2(0:8-1)
      data af0/15741,8689,9280,4732,12011,7130,6824,12302/
      data cf0/16317,10266,1198,331,10769,8310,2779,13880/
      data ab0/9173,9894,15203,15379,7981,2280,8071,429/
      data cb0/8383,3616,597,12724,15663,9639,187,4866/
      data af1/8405,4808,3603,6718,13766,9243,10375,12108/
      data cf1/13951,7170,9039,11206,8706,14101,1864,15191/
      data ab1/6269,3240,9759,7130,15320,14399,3675,1380/
      data cb1/15357,5843,6205,16275,8838,12132,2198,10330/
      data af2/445,10754,1869,6593,385,12498,14501,7383/
      data cf2/2285,8057,3864,10235,1805,10614,9615,15522/
      data ab2/405,4903,2746,1477,3263,13564,8139,2362/
      data cb2/8463,575,5876,2220,4924,1701,9060,5639/
      if(n2.gt.0)then
      call rand_next_seed_6(n2,af2,cf2,seed)
      else if(n2.lt.0)then
      call rand_next_seed_6(-n2,ab2,cb2,seed)
      end if
      if(n1.gt.0)then
      call rand_next_seed_6(n1,af1,cf1,seed)
      else if(n1.lt.0)then
      call rand_next_seed_6(-n1,ab1,cb1,seed)
      end if
      entry next_seed_6(n0,seed)
      if(n0.gt.0)then
      call rand_next_seed_6(n0,af0,cf0,seed)
      else if(n0.lt.0)then
      call rand_next_seed_6(-n0,ab0,cb0,seed)
      end if
      return
      end
!
      subroutine rand_axc_6(a,x,c)
      implicit none
      integer b
      parameter(b=2**14)
      integer a(0:8-1),c(0:8-1)
      integer x(0:8-1)
      integer z(0:8-1),i,j,t
      do i=0,8-1
      z(i)=c(i)
      end do
      do j=0,8-1
      do i=j,8-1
      z(i)=z(i)+(a(j)*x(i-j))
      end do
      end do
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      x(i)=mod(t,b)
      end do
      return
      end
!
! f90 interface to random number routine
! charles karney <karney@pppl.gov> 1999-09-30 16:58 -0400

 



 














module rngdef_7
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=kind(0.0)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(kind=double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef_7

module rng1_7
  ! the main module 
  use rngdef_7, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0_7
     subroutine rng_number_d1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_7
     subroutine rng_number_d2_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_7
     subroutine rng_number_d3_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3_7

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_number_s0_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0_7
     subroutine rng_number_s1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_7
     subroutine rng_number_s2_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_7
     subroutine rng_number_s3_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3_7

  end interface
  interface rng_gauss
     ! gaussian random numbers
     subroutine rng_gauss_d0_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0_7
     subroutine rng_gauss_d1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_7
     subroutine rng_gauss_d2_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_7
     subroutine rng_gauss_d3_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3_7

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_gauss_s0_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0_7
     subroutine rng_gauss_s1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_7
     subroutine rng_gauss_s2_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_7
     subroutine rng_gauss_s3_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3_7

  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time_7(seed,time)
       use rngdef_7
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time_7
     subroutine rng_set_seed_int_7(seed,n)
       use rngdef_7
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int_7
     subroutine rng_set_seed_string_7(seed,string)
       use rngdef_7
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string_7
  end interface

  interface
     function rng_print_seed_7(seed)
       ! return the seed in a string representation
       use rngdef_7
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed_7
     end function rng_print_seed_7
     subroutine rng_init_7(seed,state)
       ! initialize the state from the seed
       use rngdef_7
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init_7
     subroutine rng_step_seed_7(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef_7
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed_7
  end interface
end module rng1_7
 



 













module rngf77_7
  ! interface for the original f77 routines
  use rngdef_7
  implicit none
  interface
     function random1_7(ri,ra)
       use rngdef_7
       implicit none
       real(kind=double) :: random1_7
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random1_7

     function srandom1_7_7(ri,ra)
       use rngdef_7
       implicit none
       real(kind=single) :: srandom1_7_7
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom1_7_7

     subroutine random_array_7(y,n,ri,ra)
       use rngdef_7
       implicit none
       integer, intent(in) :: n
       real(kind=double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array_7

     subroutine srandom_array_7_7(y,n,ri,ra)
       use rngdef_7
       implicit none
       integer, intent(in) :: n
       real(kind=single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array_7_7

     subroutine rand_batch_7(ri,ra)
       use rngdef_7
       implicit none
       integer, intent(inout) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch_7
     subroutine random_init_7(seed,ri,ra)
       use rngdef_7
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init_7
     subroutine decimal_to_seed_7(decimal,seed)
       use rngdef_7
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed_7
     subroutine string_to_seed_7(string,seed)
       use rngdef_7
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed_7
     subroutine set_random_seed_7(time,seed)
       use rngdef_7
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed_7
     subroutine seed_to_decimal_7(seed,decimal)
       use rngdef_7
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal_7
     subroutine next_seed_73_7(n0,n1,n2,seed)
       use rngdef_7
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_73_7
     subroutine next_seed_7(n0,seed)
       use rngdef_7
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_7
  end interface
end module rngf77_7

subroutine rng_init_7(seed,state)
  use rngf77_7
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init_7(seed,state%index,state%array)
  return
end subroutine rng_init_7

subroutine rng_step_seed_7(seed,n0,n1,n2)
  use rngf77_7
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed_7(1,seed)
  else
     if (present(n0)) then
        m0=n0
     else
        m0=0
     endif
     if (present(n1)) then
        m1=n1
     else
        m1=0
     endif
     if (present(n2)) then
        m2=n2
     else
        m2=0
     endif
     call next_seed_73_7(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed_7

function rng_print_seed_7(seed)
  use rngf77_7
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed_7
  call seed_to_decimal_7(seed,rng_print_seed_7)
  return
end function rng_print_seed_7

subroutine rng_number_d0_7(state,x)
  use rngf77_7
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch_7(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0_7


subroutine rng_number_s0_7(state,x)
  use rngf77_7
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch_7(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0_7



subroutine rng_number_d1_7(state,x)
  use rngf77_7
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=state%array(i+state%index)+ulp2
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_7(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1_7


subroutine rng_number_s1_7(state,x)
  use rngf77_7
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=(int(mult*state%array(i+state%index))+0.5)*ulps
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_7(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1_7


subroutine rng_number_d2_7(state,x)
  use rngdef_7
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_7
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1_7(state,x(:,i))
  end do
end subroutine rng_number_d2_7


subroutine rng_number_s2_7(state,x)
  use rngdef_7
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_7
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1_7(state,x(:,i))
  end do
end subroutine rng_number_s2_7


subroutine rng_number_d3_7(state,x)
  use rngdef_7
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_7
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2_7(state,x(:,:,i))
  end do
end subroutine rng_number_d3_7


subroutine rng_number_s3_7(state,x)
  use rngdef_7
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_7
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2_7(state,x(:,:,i))
  end do
end subroutine rng_number_s3_7


subroutine rng_gauss_d1_7(state,x)
  use rngf77_7
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_7
  end interface
  integer :: i
  real(kind=double), parameter :: pi= 3.14159265358979323846264338328d0 
  real(kind=double) :: theta,z
  call rng_number_d1_7(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0d0 *x(i)- 1.0d0 )
     z=sqrt(- 2.0d0 *log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0d0 *x(ubound(x,1))- 1.0d0 )
  z=sqrt(- 2.0d0 *log(random1_7(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1_7

subroutine rng_gauss_d0_7(state,x)
  use rngdef_7
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_7
  end interface
  real(kind=double), dimension(1) :: y
  call rng_gauss_d1_7(state, y)
  x=y(1)
end subroutine rng_gauss_d0_7


subroutine rng_gauss_s1_7(state,x)
  use rngf77_7
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_7
  end interface
  integer :: i
  real(kind=single), parameter :: pi= 3.14159265358979323846264338328 
  real(kind=single) :: theta,z
  call rng_number_s1_7(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0*x(i)-1.0)
     z=sqrt(-2.0*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0*x(ubound(x,1))-1.0)
  z=sqrt(-2.0*log(srandom1_7_7(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1_7

subroutine rng_gauss_s0_7(state,x)
  use rngdef_7
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_7
  end interface
  real(kind=single), dimension(1) :: y
  call rng_gauss_s1_7(state, y)
  x=y(1)
end subroutine rng_gauss_s0_7


subroutine rng_gauss_d2_7(state,x)
  use rngdef_7
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_7
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1_7(state,x(:,i))
  end do
end subroutine rng_gauss_d2_7


subroutine rng_gauss_s2_7(state,x)
  use rngdef_7
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_7
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1_7(state,x(:,i))
  end do
end subroutine rng_gauss_s2_7


subroutine rng_gauss_d3_7(state,x)
  use rngdef_7
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2_7(state,x)
       use rngdef_7
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_7
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2_7(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3_7


subroutine rng_gauss_s3_7(state,x)
  use rngdef_7
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2_7(state,x)
       use rngdef_7
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_7
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2_7(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3_7


subroutine rng_set_seed_time_7(seed,time)
  use rngf77_7
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed_7(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed_7(ltime,seed)
  end if
end subroutine rng_set_seed_time_7

subroutine rng_set_seed_int_7(seed,n)
  use rngdef_7
  implicit none
  integer, dimension(0:rng_s-1), intent(out) :: seed
  integer, intent(in) :: n
  integer, parameter :: b=2**14
  integer :: m,i,sign
  if (n.lt.0) then
     seed=b-1
     m=-n-1
     sign=-1
  else
     seed=0
     m=n
     sign=1
  end if
  i=0
10 continue
  if (m.eq.0) return
  seed(i)=seed(i)+sign*mod(m,b)
  m=int(m/b)
  i=i+1
  goto 10
end subroutine rng_set_seed_int_7

subroutine rng_set_seed_string_7(seed,string)
  use rngf77_7
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed_7(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed_7(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string_7
 



 













! version 2.0 of random number routines
! author: charles karney <karney@princeton.edu>
! date: 1999-10-05 14:33 -0400
!
      function random1_7(ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      double precision  random1_7
      real srandom1_7_7
      integer ri
      double precision  ra(0:100-1)
      external rand_batch_7

      if(ri.ge.100)then
      call rand_batch_7(ri,ra(0))
      end if
      random1_7=ra(ri)+ulp2
      ri=ri+1
      return

      entry srandom1_7_7(ri,ra)
      if(ri.ge.100)then
      call rand_batch_7(ri,ra(0))
      end if

      srandom1_7_7=(int(mult*ra(ri))+0.5)*ulps

      ri=ri+1
      return
      end
!
      subroutine random_array_7(y,n,ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      integer n
      double precision  y(0:n-1)

      real ys(0:n-1)

      integer ri
      double precision  ra(0:100-1)
      integer i,k,j
      external rand_batch_7

      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      y(i)=ra(i+ri)+ulp2
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_7(ri,ra(0))
      do i=j,min(j+100,n)-1
      y(i)=ra(i-j+ri)+ulp2

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      entry srandom_array_7_7(ys,n,ri,ra)
      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      ys(i)=(int(mult*ra(i+ri))+0.5)*ulps
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_7(ri,ra(0))
      do i=j,min(j+100,n)-1
      ys(i)=(int(mult*ra(i-j+ri))+0.5)*ulps

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      end
!
      subroutine rand_batch_7(ri,ra)
      implicit none
      integer ri
      double precision  ra(0:100-1)
      integer i
      double precision  w(0:1009-100-1)
      double precision  tmp
      do i=0,63-1
      tmp=ra(i)+ra(i+100-63)

      w(i)=tmp-int(tmp)

      end do
      do i=63,100-1
      tmp=ra(i)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=100,1009-100-1
      tmp=w(i-100)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=1009-100,1009-100+63-1
      tmp=w(i-100)+w(i-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      do i=1009-100+63,1009-1
      tmp=w(i-100)+ra(i-1009+100-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      ri=0
      return
      end
!
      subroutine random_init_7(seed,ri,ra)
      implicit none
      integer b
      double precision  del,ulp
      parameter(b=2**14,del= 2.0d0 **(-14),ulp= 2.0d0 **(-47))
      integer a0,a1,a2,a3,a4,a5,a6,c0
      parameter(a0=15661,a1=678,a2=724,a3=5245,a4=13656,a5=11852,a6=29)
      parameter(c0=1)
      integer seed(0:8-1)
      integer ri
      double precision  ra(0:100-1)
      integer i,j,s(0:8-1)
      logical odd
      integer z(0:8-1),t
      do i=0,8-1
      s(i)=seed(i)
      end do
      odd=mod(s(7),2).ne.0
      ra(0)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      do j=1,100-1
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      odd=odd.or.(mod(s(7),2).ne.0)
      ra(j)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      end do
      ri=100
      if(odd)return
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      j=int((s(8-1)*100)/b)
      ra(j)=ra(j)+(ulp)
      return
      end
!
      subroutine decimal_to_seed_7(decimal,seed)
      implicit none
      character*(*)decimal
      integer seed(0:8-1)
      external rand_axc_7
      integer i,ten(0:8-1),c(0:8-1),ch
      data ten/10,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(decimal)
      ch=ichar(decimal(i:i))
      if(ch.ge.ichar('0').and.ch.le.ichar('9'))then
      c(0)=ch-ichar('0')
      call rand_axc_7(ten,seed,c)
      end if
      end do
      return
      end
!
      subroutine string_to_seed_7(string,seed)
      implicit none
      integer b
      parameter(b=2**14)
      character*(*)string
      integer seed(0:8-1)
      external rand_axc_7
      integer t,i,k,unity(0:8-1),c(0:8-1),ch
      data unity/1,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(string)
      ch=ichar(string(i:i))
      if(ch.gt.ichar(' ').and.ch.lt.127)then
      t=mod(seed(0),2)*(b/2)
      do k=0,8-1
      seed(k)=int(seed(k)/2)
      if(k.lt.8-1)then
      seed(k)=seed(k)+(mod(seed(k+1),2)*(b/2))
      else
      seed(k)=seed(k)+(t)
      end if
      end do
      c(0)=ch
      call rand_axc_7(unity,seed,c)
      end if
      end do
      return
      end
!
      subroutine set_random_seed_7(time,seed)
      implicit none
      integer time(8)
      integer seed(0:8-1)
      character*21 c
      external decimal_to_seed_7
      c=' '
      write(c(1:8),'(i4.4,2i2.2)')time(1),time(2),time(3)
      write(c(9:12),'(i1.1,i3.3)') (1-sign(1,time(4)))/2,abs(time(4))
      write(c(13:21),'(3i2.2,i3.3)')time(5),time(6),time(7),time(8)
      call decimal_to_seed_7(c,seed)
      return
      end
!
      subroutine seed_to_decimal_7(seed,decimal)
      implicit none
      integer pow,decbase,b
      parameter(pow=4,decbase=10**pow,b=2**14)
      character*(*)decimal
      integer seed(0:8-1)
      integer z(0:8-1),i,t,j,k
      character*36 str
      k=-1
      do i=0,8-1
      z(i)=seed(i)
      if(z(i).gt.0)k=i
      end do
      str=' '
      i=9
90000 continue
      i=i-1
      t=0
      do j=k,0,-1
      z(j)=z(j)+t*b
      t=mod(z(j),decbase)
      z(j)=int(z(j)/decbase)
      end do
      if(z(max(0,k)).eq.0)k=k-1
      j=pow*(i+1)
      if(k.ge.0)then
      str(j-(pow-1):j)='0000'
      else
      str(j-(pow-1):j)='   0'
      end if
90001 continue
      if(t.eq.0)goto 90010
      str(j:j)=char(ichar('0')+mod(t,10))
      j=j-1
      t=int(t/10)
      goto 90001
90010 continue
      if(k.ge.0)goto 90000
      if(len(decimal).gt.len(str))then
      decimal(:len(decimal)-len(str))=' '
      decimal(len(decimal)-len(str)+1:)=str
      else
      decimal=str(len(str)-len(decimal)+1:)
      end if
      return
      end
!
      subroutine rand_next_seed_7(n,ax,cx,y)
      implicit none
      integer n,ax(0:8-1),cx(0:8-1)
      integer y(0:8-1)
      external rand_axc_7
      integer a(0:8-1),c(0:8-1),z(0:8-1),t(0:8-1),m,i
      data z/8*0/
      if(n.eq.0)return
      m=n
      do i=0,8-1
      a(i)=ax(i)
      c(i)=cx(i)
      end do
90000 continue
      if(mod(m,2).gt.0)then
      call rand_axc_7(a,y,c)
      end if
      m=int(m/2)
      if(m.eq.0)return
      do i=0,8-1
      t(i)=c(i)
      end do
      call rand_axc_7(a,c,t)
      do i=0,8-1
      t(i)=a(i)
      end do
      call rand_axc_7(t,a,z)
      goto 90000
      end
!
      subroutine next_seed_73_7(n0,n1,n2,seed)
      implicit none
      integer n0,n1,n2
      integer seed(0:8-1)
      external rand_next_seed_7
      integer af0(0:8-1),cf0(0:8-1)
      integer ab0(0:8-1),cb0(0:8-1)
      integer af1(0:8-1),cf1(0:8-1)
      integer ab1(0:8-1),cb1(0:8-1)
      integer af2(0:8-1),cf2(0:8-1)
      integer ab2(0:8-1),cb2(0:8-1)
      data af0/15741,8689,9280,4732,12011,7130,6824,12302/
      data cf0/16317,10266,1198,331,10769,8310,2779,13880/
      data ab0/9173,9894,15203,15379,7981,2280,8071,429/
      data cb0/8383,3616,597,12724,15663,9639,187,4866/
      data af1/8405,4808,3603,6718,13766,9243,10375,12108/
      data cf1/13951,7170,9039,11206,8706,14101,1864,15191/
      data ab1/6269,3240,9759,7130,15320,14399,3675,1380/
      data cb1/15357,5843,6205,16275,8838,12132,2198,10330/
      data af2/445,10754,1869,6593,385,12498,14501,7383/
      data cf2/2285,8057,3864,10235,1805,10614,9615,15522/
      data ab2/405,4903,2746,1477,3263,13564,8139,2362/
      data cb2/8463,575,5876,2220,4924,1701,9060,5639/
      if(n2.gt.0)then
      call rand_next_seed_7(n2,af2,cf2,seed)
      else if(n2.lt.0)then
      call rand_next_seed_7(-n2,ab2,cb2,seed)
      end if
      if(n1.gt.0)then
      call rand_next_seed_7(n1,af1,cf1,seed)
      else if(n1.lt.0)then
      call rand_next_seed_7(-n1,ab1,cb1,seed)
      end if
      entry next_seed_7(n0,seed)
      if(n0.gt.0)then
      call rand_next_seed_7(n0,af0,cf0,seed)
      else if(n0.lt.0)then
      call rand_next_seed_7(-n0,ab0,cb0,seed)
      end if
      return
      end
!
      subroutine rand_axc_7(a,x,c)
      implicit none
      integer b
      parameter(b=2**14)
      integer a(0:8-1),c(0:8-1)
      integer x(0:8-1)
      integer z(0:8-1),i,j,t
      do i=0,8-1
      z(i)=c(i)
      end do
      do j=0,8-1
      do i=j,8-1
      z(i)=z(i)+(a(j)*x(i-j))
      end do
      end do
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      x(i)=mod(t,b)
      end do
      return
      end
!
! f90 interface to random number routine
! charles karney <karney@pppl.gov> 1999-09-30 16:58 -0400

 



 














module rngdef_8
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=kind(0.0)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(kind=double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef_8

module rng1_8
  ! the main module 
  use rngdef_8, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0_8
     subroutine rng_number_d1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_8
     subroutine rng_number_d2_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_8
     subroutine rng_number_d3_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3_8

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_number_s0_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0_8
     subroutine rng_number_s1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_8
     subroutine rng_number_s2_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_8
     subroutine rng_number_s3_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3_8

  end interface
  interface rng_gauss
     ! gaussian random numbers
     subroutine rng_gauss_d0_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0_8
     subroutine rng_gauss_d1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_8
     subroutine rng_gauss_d2_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_8
     subroutine rng_gauss_d3_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3_8

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_gauss_s0_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0_8
     subroutine rng_gauss_s1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_8
     subroutine rng_gauss_s2_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_8
     subroutine rng_gauss_s3_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3_8

  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time_8(seed,time)
       use rngdef_8
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time_8
     subroutine rng_set_seed_int_8(seed,n)
       use rngdef_8
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int_8
     subroutine rng_set_seed_string_8(seed,string)
       use rngdef_8
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string_8
  end interface

  interface
     function rng_print_seed_8(seed)
       ! return the seed in a string representation
       use rngdef_8
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed_8
     end function rng_print_seed_8
     subroutine rng_init_8(seed,state)
       ! initialize the state from the seed
       use rngdef_8
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init_8
     subroutine rng_step_seed_8(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef_8
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed_8
  end interface
end module rng1_8
 



 













module rngf77_8
  ! interface for the original f77 routines
  use rngdef_8
  implicit none
  interface
     function random1_8(ri,ra)
       use rngdef_8
       implicit none
       real(kind=double) :: random1_8
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random1_8

     function srandom1_8_8(ri,ra)
       use rngdef_8
       implicit none
       real(kind=single) :: srandom1_8_8
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom1_8_8

     subroutine random_array_8(y,n,ri,ra)
       use rngdef_8
       implicit none
       integer, intent(in) :: n
       real(kind=double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array_8

     subroutine srandom_array_8_8(y,n,ri,ra)
       use rngdef_8
       implicit none
       integer, intent(in) :: n
       real(kind=single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array_8_8

     subroutine rand_batch_8(ri,ra)
       use rngdef_8
       implicit none
       integer, intent(inout) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch_8
     subroutine random_init_8(seed,ri,ra)
       use rngdef_8
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init_8
     subroutine decimal_to_seed_8(decimal,seed)
       use rngdef_8
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed_8
     subroutine string_to_seed_8(string,seed)
       use rngdef_8
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed_8
     subroutine set_random_seed_8(time,seed)
       use rngdef_8
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed_8
     subroutine seed_to_decimal_8(seed,decimal)
       use rngdef_8
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal_8
     subroutine next_seed_83_8(n0,n1,n2,seed)
       use rngdef_8
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_83_8
     subroutine next_seed_8(n0,seed)
       use rngdef_8
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_8
  end interface
end module rngf77_8

subroutine rng_init_8(seed,state)
  use rngf77_8
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init_8(seed,state%index,state%array)
  return
end subroutine rng_init_8

subroutine rng_step_seed_8(seed,n0,n1,n2)
  use rngf77_8
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed_8(1,seed)
  else
     if (present(n0)) then
        m0=n0
     else
        m0=0
     endif
     if (present(n1)) then
        m1=n1
     else
        m1=0
     endif
     if (present(n2)) then
        m2=n2
     else
        m2=0
     endif
     call next_seed_83_8(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed_8

function rng_print_seed_8(seed)
  use rngf77_8
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed_8
  call seed_to_decimal_8(seed,rng_print_seed_8)
  return
end function rng_print_seed_8

subroutine rng_number_d0_8(state,x)
  use rngf77_8
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch_8(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0_8


subroutine rng_number_s0_8(state,x)
  use rngf77_8
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch_8(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0_8



subroutine rng_number_d1_8(state,x)
  use rngf77_8
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=state%array(i+state%index)+ulp2
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_8(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1_8


subroutine rng_number_s1_8(state,x)
  use rngf77_8
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=(int(mult*state%array(i+state%index))+0.5)*ulps
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_8(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1_8


subroutine rng_number_d2_8(state,x)
  use rngdef_8
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_8
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1_8(state,x(:,i))
  end do
end subroutine rng_number_d2_8


subroutine rng_number_s2_8(state,x)
  use rngdef_8
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_8
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1_8(state,x(:,i))
  end do
end subroutine rng_number_s2_8


subroutine rng_number_d3_8(state,x)
  use rngdef_8
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_8
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2_8(state,x(:,:,i))
  end do
end subroutine rng_number_d3_8


subroutine rng_number_s3_8(state,x)
  use rngdef_8
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_8
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2_8(state,x(:,:,i))
  end do
end subroutine rng_number_s3_8


subroutine rng_gauss_d1_8(state,x)
  use rngf77_8
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_8
  end interface
  integer :: i
  real(kind=double), parameter :: pi= 3.14159265358979323846264338328d0 
  real(kind=double) :: theta,z
  call rng_number_d1_8(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0d0 *x(i)- 1.0d0 )
     z=sqrt(- 2.0d0 *log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0d0 *x(ubound(x,1))- 1.0d0 )
  z=sqrt(- 2.0d0 *log(random1_8(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1_8

subroutine rng_gauss_d0_8(state,x)
  use rngdef_8
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_8
  end interface
  real(kind=double), dimension(1) :: y
  call rng_gauss_d1_8(state, y)
  x=y(1)
end subroutine rng_gauss_d0_8


subroutine rng_gauss_s1_8(state,x)
  use rngf77_8
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_8
  end interface
  integer :: i
  real(kind=single), parameter :: pi= 3.14159265358979323846264338328 
  real(kind=single) :: theta,z
  call rng_number_s1_8(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0*x(i)-1.0)
     z=sqrt(-2.0*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0*x(ubound(x,1))-1.0)
  z=sqrt(-2.0*log(srandom1_8_8(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1_8

subroutine rng_gauss_s0_8(state,x)
  use rngdef_8
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_8
  end interface
  real(kind=single), dimension(1) :: y
  call rng_gauss_s1_8(state, y)
  x=y(1)
end subroutine rng_gauss_s0_8


subroutine rng_gauss_d2_8(state,x)
  use rngdef_8
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_8
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1_8(state,x(:,i))
  end do
end subroutine rng_gauss_d2_8


subroutine rng_gauss_s2_8(state,x)
  use rngdef_8
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_8
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1_8(state,x(:,i))
  end do
end subroutine rng_gauss_s2_8


subroutine rng_gauss_d3_8(state,x)
  use rngdef_8
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2_8(state,x)
       use rngdef_8
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_8
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2_8(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3_8


subroutine rng_gauss_s3_8(state,x)
  use rngdef_8
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2_8(state,x)
       use rngdef_8
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_8
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2_8(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3_8


subroutine rng_set_seed_time_8(seed,time)
  use rngf77_8
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed_8(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed_8(ltime,seed)
  end if
end subroutine rng_set_seed_time_8

subroutine rng_set_seed_int_8(seed,n)
  use rngdef_8
  implicit none
  integer, dimension(0:rng_s-1), intent(out) :: seed
  integer, intent(in) :: n
  integer, parameter :: b=2**14
  integer :: m,i,sign
  if (n.lt.0) then
     seed=b-1
     m=-n-1
     sign=-1
  else
     seed=0
     m=n
     sign=1
  end if
  i=0
10 continue
  if (m.eq.0) return
  seed(i)=seed(i)+sign*mod(m,b)
  m=int(m/b)
  i=i+1
  goto 10
end subroutine rng_set_seed_int_8

subroutine rng_set_seed_string_8(seed,string)
  use rngf77_8
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed_8(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed_8(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string_8
 



 













! version 2.0 of random number routines
! author: charles karney <karney@princeton.edu>
! date: 1999-10-05 14:33 -0400
!
      function random1_8(ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      double precision  random1_8
      real srandom1_8_8
      integer ri
      double precision  ra(0:100-1)
      external rand_batch_8

      if(ri.ge.100)then
      call rand_batch_8(ri,ra(0))
      end if
      random1_8=ra(ri)+ulp2
      ri=ri+1
      return

      entry srandom1_8_8(ri,ra)
      if(ri.ge.100)then
      call rand_batch_8(ri,ra(0))
      end if

      srandom1_8_8=(int(mult*ra(ri))+0.5)*ulps

      ri=ri+1
      return
      end
!
      subroutine random_array_8(y,n,ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      integer n
      double precision  y(0:n-1)

      real ys(0:n-1)

      integer ri
      double precision  ra(0:100-1)
      integer i,k,j
      external rand_batch_8

      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      y(i)=ra(i+ri)+ulp2
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_8(ri,ra(0))
      do i=j,min(j+100,n)-1
      y(i)=ra(i-j+ri)+ulp2

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      entry srandom_array_8_8(ys,n,ri,ra)
      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      ys(i)=(int(mult*ra(i+ri))+0.5)*ulps
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_8(ri,ra(0))
      do i=j,min(j+100,n)-1
      ys(i)=(int(mult*ra(i-j+ri))+0.5)*ulps

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      end
!
      subroutine rand_batch_8(ri,ra)
      implicit none
      integer ri
      double precision  ra(0:100-1)
      integer i
      double precision  w(0:1009-100-1)
      double precision  tmp
      do i=0,63-1
      tmp=ra(i)+ra(i+100-63)

      w(i)=tmp-int(tmp)

      end do
      do i=63,100-1
      tmp=ra(i)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=100,1009-100-1
      tmp=w(i-100)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=1009-100,1009-100+63-1
      tmp=w(i-100)+w(i-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      do i=1009-100+63,1009-1
      tmp=w(i-100)+ra(i-1009+100-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      ri=0
      return
      end
!
      subroutine random_init_8(seed,ri,ra)
      implicit none
      integer b
      double precision  del,ulp
      parameter(b=2**14,del= 2.0d0 **(-14),ulp= 2.0d0 **(-47))
      integer a0,a1,a2,a3,a4,a5,a6,c0
      parameter(a0=15661,a1=678,a2=724,a3=5245,a4=13656,a5=11852,a6=29)
      parameter(c0=1)
      integer seed(0:8-1)
      integer ri
      double precision  ra(0:100-1)
      integer i,j,s(0:8-1)
      logical odd
      integer z(0:8-1),t
      do i=0,8-1
      s(i)=seed(i)
      end do
      odd=mod(s(7),2).ne.0
      ra(0)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      do j=1,100-1
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      odd=odd.or.(mod(s(7),2).ne.0)
      ra(j)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      end do
      ri=100
      if(odd)return
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      j=int((s(8-1)*100)/b)
      ra(j)=ra(j)+(ulp)
      return
      end
!
      subroutine decimal_to_seed_8(decimal,seed)
      implicit none
      character*(*)decimal
      integer seed(0:8-1)
      external rand_axc_8
      integer i,ten(0:8-1),c(0:8-1),ch
      data ten/10,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(decimal)
      ch=ichar(decimal(i:i))
      if(ch.ge.ichar('0').and.ch.le.ichar('9'))then
      c(0)=ch-ichar('0')
      call rand_axc_8(ten,seed,c)
      end if
      end do
      return
      end
!
      subroutine string_to_seed_8(string,seed)
      implicit none
      integer b
      parameter(b=2**14)
      character*(*)string
      integer seed(0:8-1)
      external rand_axc_8
      integer t,i,k,unity(0:8-1),c(0:8-1),ch
      data unity/1,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(string)
      ch=ichar(string(i:i))
      if(ch.gt.ichar(' ').and.ch.lt.127)then
      t=mod(seed(0),2)*(b/2)
      do k=0,8-1
      seed(k)=int(seed(k)/2)
      if(k.lt.8-1)then
      seed(k)=seed(k)+(mod(seed(k+1),2)*(b/2))
      else
      seed(k)=seed(k)+(t)
      end if
      end do
      c(0)=ch
      call rand_axc_8(unity,seed,c)
      end if
      end do
      return
      end
!
      subroutine set_random_seed_8(time,seed)
      implicit none
      integer time(8)
      integer seed(0:8-1)
      character*21 c
      external decimal_to_seed_8
      c=' '
      write(c(1:8),'(i4.4,2i2.2)')time(1),time(2),time(3)
      write(c(9:12),'(i1.1,i3.3)') (1-sign(1,time(4)))/2,abs(time(4))
      write(c(13:21),'(3i2.2,i3.3)')time(5),time(6),time(7),time(8)
      call decimal_to_seed_8(c,seed)
      return
      end
!
      subroutine seed_to_decimal_8(seed,decimal)
      implicit none
      integer pow,decbase,b
      parameter(pow=4,decbase=10**pow,b=2**14)
      character*(*)decimal
      integer seed(0:8-1)
      integer z(0:8-1),i,t,j,k
      character*36 str
      k=-1
      do i=0,8-1
      z(i)=seed(i)
      if(z(i).gt.0)k=i
      end do
      str=' '
      i=9
90000 continue
      i=i-1
      t=0
      do j=k,0,-1
      z(j)=z(j)+t*b
      t=mod(z(j),decbase)
      z(j)=int(z(j)/decbase)
      end do
      if(z(max(0,k)).eq.0)k=k-1
      j=pow*(i+1)
      if(k.ge.0)then
      str(j-(pow-1):j)='0000'
      else
      str(j-(pow-1):j)='   0'
      end if
90001 continue
      if(t.eq.0)goto 90010
      str(j:j)=char(ichar('0')+mod(t,10))
      j=j-1
      t=int(t/10)
      goto 90001
90010 continue
      if(k.ge.0)goto 90000
      if(len(decimal).gt.len(str))then
      decimal(:len(decimal)-len(str))=' '
      decimal(len(decimal)-len(str)+1:)=str
      else
      decimal=str(len(str)-len(decimal)+1:)
      end if
      return
      end
!
      subroutine rand_next_seed_8(n,ax,cx,y)
      implicit none
      integer n,ax(0:8-1),cx(0:8-1)
      integer y(0:8-1)
      external rand_axc_8
      integer a(0:8-1),c(0:8-1),z(0:8-1),t(0:8-1),m,i
      data z/8*0/
      if(n.eq.0)return
      m=n
      do i=0,8-1
      a(i)=ax(i)
      c(i)=cx(i)
      end do
90000 continue
      if(mod(m,2).gt.0)then
      call rand_axc_8(a,y,c)
      end if
      m=int(m/2)
      if(m.eq.0)return
      do i=0,8-1
      t(i)=c(i)
      end do
      call rand_axc_8(a,c,t)
      do i=0,8-1
      t(i)=a(i)
      end do
      call rand_axc_8(t,a,z)
      goto 90000
      end
!
      subroutine next_seed_83_8(n0,n1,n2,seed)
      implicit none
      integer n0,n1,n2
      integer seed(0:8-1)
      external rand_next_seed_8
      integer af0(0:8-1),cf0(0:8-1)
      integer ab0(0:8-1),cb0(0:8-1)
      integer af1(0:8-1),cf1(0:8-1)
      integer ab1(0:8-1),cb1(0:8-1)
      integer af2(0:8-1),cf2(0:8-1)
      integer ab2(0:8-1),cb2(0:8-1)
      data af0/15741,8689,9280,4732,12011,7130,6824,12302/
      data cf0/16317,10266,1198,331,10769,8310,2779,13880/
      data ab0/9173,9894,15203,15379,7981,2280,8071,429/
      data cb0/8383,3616,597,12724,15663,9639,187,4866/
      data af1/8405,4808,3603,6718,13766,9243,10375,12108/
      data cf1/13951,7170,9039,11206,8706,14101,1864,15191/
      data ab1/6269,3240,9759,7130,15320,14399,3675,1380/
      data cb1/15357,5843,6205,16275,8838,12132,2198,10330/
      data af2/445,10754,1869,6593,385,12498,14501,7383/
      data cf2/2285,8057,3864,10235,1805,10614,9615,15522/
      data ab2/405,4903,2746,1477,3263,13564,8139,2362/
      data cb2/8463,575,5876,2220,4924,1701,9060,5639/
      if(n2.gt.0)then
      call rand_next_seed_8(n2,af2,cf2,seed)
      else if(n2.lt.0)then
      call rand_next_seed_8(-n2,ab2,cb2,seed)
      end if
      if(n1.gt.0)then
      call rand_next_seed_8(n1,af1,cf1,seed)
      else if(n1.lt.0)then
      call rand_next_seed_8(-n1,ab1,cb1,seed)
      end if
      entry next_seed_8(n0,seed)
      if(n0.gt.0)then
      call rand_next_seed_8(n0,af0,cf0,seed)
      else if(n0.lt.0)then
      call rand_next_seed_8(-n0,ab0,cb0,seed)
      end if
      return
      end
!
      subroutine rand_axc_8(a,x,c)
      implicit none
      integer b
      parameter(b=2**14)
      integer a(0:8-1),c(0:8-1)
      integer x(0:8-1)
      integer z(0:8-1),i,j,t
      do i=0,8-1
      z(i)=c(i)
      end do
      do j=0,8-1
      do i=j,8-1
      z(i)=z(i)+(a(j)*x(i-j))
      end do
      end do
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      x(i)=mod(t,b)
      end do
      return
      end
!
! f90 interface to random number routine
! charles karney <karney@pppl.gov> 1999-09-30 16:58 -0400

 



 














module rngdef_9
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=kind(0.0)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(kind=double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef_9

module rng1_9
  ! the main module 
  use rngdef_9, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0_9
     subroutine rng_number_d1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_9
     subroutine rng_number_d2_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_9
     subroutine rng_number_d3_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3_9

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_number_s0_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0_9
     subroutine rng_number_s1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_9
     subroutine rng_number_s2_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_9
     subroutine rng_number_s3_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3_9

  end interface
  interface rng_gauss
     ! gaussian random numbers
     subroutine rng_gauss_d0_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0_9
     subroutine rng_gauss_d1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_9
     subroutine rng_gauss_d2_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_9
     subroutine rng_gauss_d3_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3_9

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_gauss_s0_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0_9
     subroutine rng_gauss_s1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_9
     subroutine rng_gauss_s2_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_9
     subroutine rng_gauss_s3_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3_9

  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time_9(seed,time)
       use rngdef_9
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time_9
     subroutine rng_set_seed_int_9(seed,n)
       use rngdef_9
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int_9
     subroutine rng_set_seed_string_9(seed,string)
       use rngdef_9
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string_9
  end interface

  interface
     function rng_print_seed_9(seed)
       ! return the seed in a string representation
       use rngdef_9
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed_9
     end function rng_print_seed_9
     subroutine rng_init_9(seed,state)
       ! initialize the state from the seed
       use rngdef_9
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init_9
     subroutine rng_step_seed_9(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef_9
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed_9
  end interface
end module rng1_9
 



 













module rngf77_9
  ! interface for the original f77 routines
  use rngdef_9
  implicit none
  interface
     function random1_9(ri,ra)
       use rngdef_9
       implicit none
       real(kind=double) :: random1_9
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random1_9

     function srandom1_9_9(ri,ra)
       use rngdef_9
       implicit none
       real(kind=single) :: srandom1_9_9
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom1_9_9

     subroutine random_array_9(y,n,ri,ra)
       use rngdef_9
       implicit none
       integer, intent(in) :: n
       real(kind=double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array_9

     subroutine srandom_array_9_9(y,n,ri,ra)
       use rngdef_9
       implicit none
       integer, intent(in) :: n
       real(kind=single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array_9_9

     subroutine rand_batch_9(ri,ra)
       use rngdef_9
       implicit none
       integer, intent(inout) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch_9
     subroutine random_init_9(seed,ri,ra)
       use rngdef_9
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init_9
     subroutine decimal_to_seed_9(decimal,seed)
       use rngdef_9
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed_9
     subroutine string_to_seed_9(string,seed)
       use rngdef_9
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed_9
     subroutine set_random_seed_9(time,seed)
       use rngdef_9
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed_9
     subroutine seed_to_decimal_9(seed,decimal)
       use rngdef_9
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal_9
     subroutine next_seed_93_9(n0,n1,n2,seed)
       use rngdef_9
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_93_9
     subroutine next_seed_9(n0,seed)
       use rngdef_9
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed_9
  end interface
end module rngf77_9

subroutine rng_init_9(seed,state)
  use rngf77_9
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init_9(seed,state%index,state%array)
  return
end subroutine rng_init_9

subroutine rng_step_seed_9(seed,n0,n1,n2)
  use rngf77_9
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed_9(1,seed)
  else
     if (present(n0)) then
        m0=n0
     else
        m0=0
     endif
     if (present(n1)) then
        m1=n1
     else
        m1=0
     endif
     if (present(n2)) then
        m2=n2
     else
        m2=0
     endif
     call next_seed_93_9(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed_9

function rng_print_seed_9(seed)
  use rngf77_9
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed_9
  call seed_to_decimal_9(seed,rng_print_seed_9)
  return
end function rng_print_seed_9

subroutine rng_number_d0_9(state,x)
  use rngf77_9
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch_9(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0_9


subroutine rng_number_s0_9(state,x)
  use rngf77_9
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch_9(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0_9



subroutine rng_number_d1_9(state,x)
  use rngf77_9
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=state%array(i+state%index)+ulp2
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_9(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1_9


subroutine rng_number_s1_9(state,x)
  use rngf77_9
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  integer :: i,k,j,n,n0
  n0=lbound(x,1)
  n=ubound(x,1)-n0+1
  if (n.le.0) return
  k=min(n,rng_k-state%index)
  do i=0,k-1
     x(i+n0)=(int(mult*state%array(i+state%index))+0.5)*ulps
  end do
  state%index=state%index+(k)
  do j=k,n-1,rng_k
     call rand_batch_9(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1_9


subroutine rng_number_d2_9(state,x)
  use rngdef_9
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_9
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1_9(state,x(:,i))
  end do
end subroutine rng_number_d2_9


subroutine rng_number_s2_9(state,x)
  use rngdef_9
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_9
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1_9(state,x(:,i))
  end do
end subroutine rng_number_s2_9


subroutine rng_number_d3_9(state,x)
  use rngdef_9
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2_9
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2_9(state,x(:,:,i))
  end do
end subroutine rng_number_d3_9


subroutine rng_number_s3_9(state,x)
  use rngdef_9
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2_9
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2_9(state,x(:,:,i))
  end do
end subroutine rng_number_s3_9


subroutine rng_gauss_d1_9(state,x)
  use rngf77_9
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1_9
  end interface
  integer :: i
  real(kind=double), parameter :: pi= 3.14159265358979323846264338328d0 
  real(kind=double) :: theta,z
  call rng_number_d1_9(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0d0 *x(i)- 1.0d0 )
     z=sqrt(- 2.0d0 *log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0d0 *x(ubound(x,1))- 1.0d0 )
  z=sqrt(- 2.0d0 *log(random1_9(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1_9

subroutine rng_gauss_d0_9(state,x)
  use rngdef_9
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_9
  end interface
  real(kind=double), dimension(1) :: y
  call rng_gauss_d1_9(state, y)
  x=y(1)
end subroutine rng_gauss_d0_9


subroutine rng_gauss_s1_9(state,x)
  use rngf77_9
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1_9
  end interface
  integer :: i
  real(kind=single), parameter :: pi= 3.14159265358979323846264338328 
  real(kind=single) :: theta,z
  call rng_number_s1_9(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0*x(i)-1.0)
     z=sqrt(-2.0*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0*x(ubound(x,1))-1.0)
  z=sqrt(-2.0*log(srandom1_9_9(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1_9

subroutine rng_gauss_s0_9(state,x)
  use rngdef_9
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_9
  end interface
  real(kind=single), dimension(1) :: y
  call rng_gauss_s1_9(state, y)
  x=y(1)
end subroutine rng_gauss_s0_9


subroutine rng_gauss_d2_9(state,x)
  use rngdef_9
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1_9
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1_9(state,x(:,i))
  end do
end subroutine rng_gauss_d2_9


subroutine rng_gauss_s2_9(state,x)
  use rngdef_9
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1_9
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1_9(state,x(:,i))
  end do
end subroutine rng_gauss_s2_9


subroutine rng_gauss_d3_9(state,x)
  use rngdef_9
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2_9(state,x)
       use rngdef_9
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2_9
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2_9(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3_9


subroutine rng_gauss_s3_9(state,x)
  use rngdef_9
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2_9(state,x)
       use rngdef_9
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2_9
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2_9(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3_9


subroutine rng_set_seed_time_9(seed,time)
  use rngf77_9
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed_9(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed_9(ltime,seed)
  end if
end subroutine rng_set_seed_time_9

subroutine rng_set_seed_int_9(seed,n)
  use rngdef_9
  implicit none
  integer, dimension(0:rng_s-1), intent(out) :: seed
  integer, intent(in) :: n
  integer, parameter :: b=2**14
  integer :: m,i,sign
  if (n.lt.0) then
     seed=b-1
     m=-n-1
     sign=-1
  else
     seed=0
     m=n
     sign=1
  end if
  i=0
10 continue
  if (m.eq.0) return
  seed(i)=seed(i)+sign*mod(m,b)
  m=int(m/b)
  i=i+1
  goto 10
end subroutine rng_set_seed_int_9

subroutine rng_set_seed_string_9(seed,string)
  use rngf77_9
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed_9(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed_9(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string_9
 



 













! version 2.0 of random number routines
! author: charles karney <karney@princeton.edu>
! date: 1999-10-05 14:33 -0400
!
      function random1_9(ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      double precision  random1_9
      real srandom1_9_9
      integer ri
      double precision  ra(0:100-1)
      external rand_batch_9

      if(ri.ge.100)then
      call rand_batch_9(ri,ra(0))
      end if
      random1_9=ra(ri)+ulp2
      ri=ri+1
      return

      entry srandom1_9_9(ri,ra)
      if(ri.ge.100)then
      call rand_batch_9(ri,ra(0))
      end if

      srandom1_9_9=(int(mult*ra(ri))+0.5)*ulps

      ri=ri+1
      return
      end
!
      subroutine random_array_9(y,n,ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      integer n
      double precision  y(0:n-1)

      real ys(0:n-1)

      integer ri
      double precision  ra(0:100-1)
      integer i,k,j
      external rand_batch_9

      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      y(i)=ra(i+ri)+ulp2
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_9(ri,ra(0))
      do i=j,min(j+100,n)-1
      y(i)=ra(i-j+ri)+ulp2

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      entry srandom_array_9_9(ys,n,ri,ra)
      if(n.le.0)return
      k=min(n,100-ri)
      do i=0,k-1
      ys(i)=(int(mult*ra(i+ri))+0.5)*ulps
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch_9(ri,ra(0))
      do i=j,min(j+100,n)-1
      ys(i)=(int(mult*ra(i-j+ri))+0.5)*ulps

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      end
!
      subroutine rand_batch_9(ri,ra)
      implicit none
      integer ri
      double precision  ra(0:100-1)
      integer i
      double precision  w(0:1009-100-1)
      double precision  tmp
      do i=0,63-1
      tmp=ra(i)+ra(i+100-63)

      w(i)=tmp-int(tmp)

      end do
      do i=63,100-1
      tmp=ra(i)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=100,1009-100-1
      tmp=w(i-100)+w(i-63)

      w(i)=tmp-int(tmp)

      end do
      do i=1009-100,1009-100+63-1
      tmp=w(i-100)+w(i-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      do i=1009-100+63,1009-1
      tmp=w(i-100)+ra(i-1009+100-63)

      ra(i-1009+100)=tmp-int(tmp)

      end do
      ri=0
      return
      end
!
      subroutine random_init_9(seed,ri,ra)
      implicit none
      integer b
      double precision  del,ulp
      parameter(b=2**14,del= 2.0d0 **(-14),ulp= 2.0d0 **(-47))
      integer a0,a1,a2,a3,a4,a5,a6,c0
      parameter(a0=15661,a1=678,a2=724,a3=5245,a4=13656,a5=11852,a6=29)
      parameter(c0=1)
      integer seed(0:8-1)
      integer ri
      double precision  ra(0:100-1)
      integer i,j,s(0:8-1)
      logical odd
      integer z(0:8-1),t
      do i=0,8-1
      s(i)=seed(i)
      end do
      odd=mod(s(7),2).ne.0
      ra(0)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      do j=1,100-1
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      odd=odd.or.(mod(s(7),2).ne.0)
      ra(j)=(((s(7)*del+s(6))*del+s(5))*del+int(s(4)/512))*512*del
      end do
      ri=100
      if(odd)return
      z(0)=c0+a0*s(0)
      z(1)=a0*s(1)+a1*s(0)
      z(2)=a0*s(2)+a1*s(1)+a2*s(0)
      z(3)=a0*s(3)+a1*s(2)+a2*s(1)+a3*s(0)
      z(4)=a0*s(4)+a1*s(3)+a2*s(2)+a3*s(1)+a4*s(0)
      z(5)=a0*s(5)+a1*s(4)+a2*s(3)+a3*s(2)+a4*s(1)+a5*s(0)
      z(6)=a0*s(6)+a1*s(5)+a2*s(4)+a3*s(3)+a4*s(2)+a5*s(1)+a6*s(0)
      z(7)=a0*s(7)+a1*s(6)+a2*s(5)+a3*s(4)+a4*s(3)+a5*s(2)+a6*s(1)
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      s(i)=mod(t,b)
      end do
      j=int((s(8-1)*100)/b)
      ra(j)=ra(j)+(ulp)
      return
      end
!
      subroutine decimal_to_seed_9(decimal,seed)
      implicit none
      character*(*)decimal
      integer seed(0:8-1)
      external rand_axc_9
      integer i,ten(0:8-1),c(0:8-1),ch
      data ten/10,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(decimal)
      ch=ichar(decimal(i:i))
      if(ch.ge.ichar('0').and.ch.le.ichar('9'))then
      c(0)=ch-ichar('0')
      call rand_axc_9(ten,seed,c)
      end if
      end do
      return
      end
!
      subroutine string_to_seed_9(string,seed)
      implicit none
      integer b
      parameter(b=2**14)
      character*(*)string
      integer seed(0:8-1)
      external rand_axc_9
      integer t,i,k,unity(0:8-1),c(0:8-1),ch
      data unity/1,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(string)
      ch=ichar(string(i:i))
      if(ch.gt.ichar(' ').and.ch.lt.127)then
      t=mod(seed(0),2)*(b/2)
      do k=0,8-1
      seed(k)=int(seed(k)/2)
      if(k.lt.8-1)then
      seed(k)=seed(k)+(mod(seed(k+1),2)*(b/2))
      else
      seed(k)=seed(k)+(t)
      end if
      end do
      c(0)=ch
      call rand_axc_9(unity,seed,c)
      end if
      end do
      return
      end
!
      subroutine set_random_seed_9(time,seed)
      implicit none
      integer time(8)
      integer seed(0:8-1)
      character*21 c
      external decimal_to_seed_9
      c=' '
      write(c(1:8),'(i4.4,2i2.2)')time(1),time(2),time(3)
      write(c(9:12),'(i1.1,i3.3)') (1-sign(1,time(4)))/2,abs(time(4))
      write(c(13:21),'(3i2.2,i3.3)')time(5),time(6),time(7),time(8)
      call decimal_to_seed_9(c,seed)
      return
      end
!
      subroutine seed_to_decimal_9(seed,decimal)
      implicit none
      integer pow,decbase,b
      parameter(pow=4,decbase=10**pow,b=2**14)
      character*(*)decimal
      integer seed(0:8-1)
      integer z(0:8-1),i,t,j,k
      character*36 str
      k=-1
      do i=0,8-1
      z(i)=seed(i)
      if(z(i).gt.0)k=i
      end do
      str=' '
      i=9
90000 continue
      i=i-1
      t=0
      do j=k,0,-1
      z(j)=z(j)+t*b
      t=mod(z(j),decbase)
      z(j)=int(z(j)/decbase)
      end do
      if(z(max(0,k)).eq.0)k=k-1
      j=pow*(i+1)
      if(k.ge.0)then
      str(j-(pow-1):j)='0000'
      else
      str(j-(pow-1):j)='   0'
      end if
90001 continue
      if(t.eq.0)goto 90010
      str(j:j)=char(ichar('0')+mod(t,10))
      j=j-1
      t=int(t/10)
      goto 90001
90010 continue
      if(k.ge.0)goto 90000
      if(len(decimal).gt.len(str))then
      decimal(:len(decimal)-len(str))=' '
      decimal(len(decimal)-len(str)+1:)=str
      else
      decimal=str(len(str)-len(decimal)+1:)
      end if
      return
      end
!
      subroutine rand_next_seed_9(n,ax,cx,y)
      implicit none
      integer n,ax(0:8-1),cx(0:8-1)
      integer y(0:8-1)
      external rand_axc_9
      integer a(0:8-1),c(0:8-1),z(0:8-1),t(0:8-1),m,i
      data z/8*0/
      if(n.eq.0)return
      m=n
      do i=0,8-1
      a(i)=ax(i)
      c(i)=cx(i)
      end do
90000 continue
      if(mod(m,2).gt.0)then
      call rand_axc_9(a,y,c)
      end if
      m=int(m/2)
      if(m.eq.0)return
      do i=0,8-1
      t(i)=c(i)
      end do
      call rand_axc_9(a,c,t)
      do i=0,8-1
      t(i)=a(i)
      end do
      call rand_axc_9(t,a,z)
      goto 90000
      end
!
      subroutine next_seed_93_9(n0,n1,n2,seed)
      implicit none
      integer n0,n1,n2
      integer seed(0:8-1)
      external rand_next_seed_9
      integer af0(0:8-1),cf0(0:8-1)
      integer ab0(0:8-1),cb0(0:8-1)
      integer af1(0:8-1),cf1(0:8-1)
      integer ab1(0:8-1),cb1(0:8-1)
      integer af2(0:8-1),cf2(0:8-1)
      integer ab2(0:8-1),cb2(0:8-1)
      data af0/15741,8689,9280,4732,12011,7130,6824,12302/
      data cf0/16317,10266,1198,331,10769,8310,2779,13880/
      data ab0/9173,9894,15203,15379,7981,2280,8071,429/
      data cb0/8383,3616,597,12724,15663,9639,187,4866/
      data af1/8405,4808,3603,6718,13766,9243,10375,12108/
      data cf1/13951,7170,9039,11206,8706,14101,1864,15191/
      data ab1/6269,3240,9759,7130,15320,14399,3675,1380/
      data cb1/15357,5843,6205,16275,8838,12132,2198,10330/
      data af2/445,10754,1869,6593,385,12498,14501,7383/
      data cf2/2285,8057,3864,10235,1805,10614,9615,15522/
      data ab2/405,4903,2746,1477,3263,13564,8139,2362/
      data cb2/8463,575,5876,2220,4924,1701,9060,5639/
      if(n2.gt.0)then
      call rand_next_seed_9(n2,af2,cf2,seed)
      else if(n2.lt.0)then
      call rand_next_seed_9(-n2,ab2,cb2,seed)
      end if
      if(n1.gt.0)then
      call rand_next_seed_9(n1,af1,cf1,seed)
      else if(n1.lt.0)then
      call rand_next_seed_9(-n1,ab1,cb1,seed)
      end if
      entry next_seed_9(n0,seed)
      if(n0.gt.0)then
      call rand_next_seed_9(n0,af0,cf0,seed)
      else if(n0.lt.0)then
      call rand_next_seed_9(-n0,ab0,cb0,seed)
      end if
      return
      end
!
      subroutine rand_axc_9(a,x,c)
      implicit none
      integer b
      parameter(b=2**14)
      integer a(0:8-1),c(0:8-1)
      integer x(0:8-1)
      integer z(0:8-1),i,j,t
      do i=0,8-1
      z(i)=c(i)
      end do
      do j=0,8-1
      do i=j,8-1
      z(i)=z(i)+(a(j)*x(i-j))
      end do
      end do
      t=0
      do i=0,8-1
      t=int(t/b)+z(i)
      x(i)=mod(t,b)
      end do
      return
      end
!
