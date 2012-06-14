! f90 interface to random number routine
! Charles Karney <karney@pppl.gov> 1999-09-30 16:58 -0400

 



 














module rngdef
  ! basic definitions
  implicit none
  integer, parameter :: double=selected_real_kind(12)
  integer, parameter :: single=kind(0.0)
  integer, parameter :: rng_k=100, rng_s=8, rng_c=34
  type :: rng_state
     integer :: index
     real(kind=double), dimension(0:rng_k-1) :: array
  end type rng_state
end module rngdef

module rng
  ! the main module 
  use rngdef, only: rng_state, rng_s, rng_c

  interface rng_number
     ! the basic random number routine
     subroutine rng_number_d0(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d0
     subroutine rng_number_d1(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1
     subroutine rng_number_d2(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2
     subroutine rng_number_d3(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d3

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_number_s0(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s0
     subroutine rng_number_s1(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1
     subroutine rng_number_s2(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2
     subroutine rng_number_s3(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s3

  end interface
  interface rng_gauss
     ! Gaussian random numbers
     subroutine rng_gauss_d0(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d0
     subroutine rng_gauss_d1(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1
     subroutine rng_gauss_d2(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2
     subroutine rng_gauss_d3(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d3

     ! perhaps additional interfaces are needed for single precision
     subroutine rng_gauss_s0(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s0
     subroutine rng_gauss_s1(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1
     subroutine rng_gauss_s2(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2
     subroutine rng_gauss_s3(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:,:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s3

  end interface

  interface rng_set_seed
     ! setting the random number seed in various ways
     subroutine rng_set_seed_time(seed,time)
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, dimension(8), optional, intent(in) :: time
     end subroutine rng_set_seed_time
     subroutine rng_set_seed_int(seed,n)
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       integer, intent(in) :: n
     end subroutine rng_set_seed_int
     subroutine rng_set_seed_string(seed,string)
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(out) :: seed
       character(len=*), intent(in) :: string
     end subroutine rng_set_seed_string
  end interface

  interface
     function rng_print_seed(seed)
       ! return the seed in a string representation
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       character(len=rng_c) :: rng_print_seed
     end function rng_print_seed
     subroutine rng_init(seed,state)
       ! initialize the state from the seed
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(in) :: seed
       type(rng_state), intent(out) :: state
     end subroutine rng_init
     subroutine rng_step_seed(seed,n0,n1,n2)
       ! step seed forward in 3 dimensions
       use rngdef
       implicit none
       integer, dimension(rng_s), intent(inout) :: seed
       integer, optional, intent(in) :: n0,n1,n2
     end subroutine rng_step_seed
  end interface
end module rng
 



 













module rngf77
  ! interface for the original f77 routines
  use rngdef
  implicit none
  interface
     function random(ri,ra)
       use rngdef
       implicit none
       real(kind=double) :: random
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function random

     function srandom(ri,ra)
       use rngdef
       implicit none
       real(kind=single) :: srandom
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end function srandom

     subroutine random_array(y,n,ri,ra)
       use rngdef
       implicit none
       integer, intent(in) :: n
       real(kind=double), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine random_array

     subroutine srandom_array(y,n,ri,ra)
       use rngdef
       implicit none
       integer, intent(in) :: n
       real(kind=single), dimension(0:n-1), intent(out) :: y
       integer, intent(inout) :: ri
       real(kind=double), intent(inout), dimension(0:rng_k-1) :: ra
     end subroutine srandom_array

     subroutine rand_batch(ri,ra)
       use rngdef
       implicit none
       integer, intent(inout) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(inout) :: ra
     end subroutine rand_batch
     subroutine random_init(seed,ri,ra)
       use rngdef
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       integer, intent(out) :: ri
       real(kind=double), dimension(0:rng_k-1), intent(out) :: ra
     end subroutine random_init
     subroutine decimal_to_seed(decimal,seed)
       use rngdef
       implicit none
       character(len=*), intent(in) :: decimal
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine decimal_to_seed
     subroutine string_to_seed(string,seed)
       use rngdef
       implicit none
       character(len=*), intent(in) :: string
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine string_to_seed
     subroutine set_random_seed(time,seed)
       use rngdef
       implicit none
       integer, dimension(8), intent(in) :: time(8)
       integer, dimension(0:rng_s-1), intent(out) :: seed
     end subroutine set_random_seed
     subroutine seed_to_decimal(seed,decimal)
       use rngdef
       implicit none
       integer, dimension(0:rng_s-1), intent(in) :: seed
       character(len=*), intent(out) :: decimal
     end subroutine seed_to_decimal
     subroutine next_seed3(n0,n1,n2,seed)
       use rngdef
       implicit none
       integer, intent(in) :: n0,n1,n2
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed3
     subroutine next_seed(n0,seed)
       use rngdef
       implicit none
       integer, intent(in) :: n0
       integer, dimension(0:rng_s-1), intent(inout) :: seed
     end subroutine next_seed
  end interface
end module rngf77

subroutine rng_init(seed,state)
  use rngf77
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  type(rng_state), intent(out) :: state
  call random_init(seed,state%index,state%array)
  return
end subroutine rng_init

subroutine rng_step_seed(seed,n0,n1,n2)
  use rngf77
  implicit none
  integer, dimension(rng_s), intent(inout) :: seed
  integer, optional, intent(in) :: n0,n1,n2
  integer :: m0,m1,m2
  if (.not. (present(n0) .or. present(n1) .or. present(n2))) then
     call next_seed(1,seed)
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
     call next_seed3(m0,m1,m2,seed)
  end if
  return
end subroutine rng_step_seed

function rng_print_seed(seed)
  use rngf77
  implicit none
  integer, dimension(rng_s), intent(in) :: seed
  character(len=rng_c) :: rng_print_seed
  call seed_to_decimal(seed,rng_print_seed)
  return
end function rng_print_seed

subroutine rng_number_d0(state,x)
  use rngf77
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=double), parameter :: ulp2=2.0_double**(-47-1)
  if(state%index.ge.rng_k)then
     call rand_batch(state%index,state%array)
  end if
  x=state%array(state%index)+ulp2
  state%index=state%index+1
  return
end subroutine rng_number_d0


subroutine rng_number_s0(state,x)
  use rngf77
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  real(kind=single), parameter :: ulps = 2.0**(-23)
  real(kind=double), parameter :: mult = 2.0_double**23
  if(state%index.ge.rng_k)then
     call rand_batch(state%index,state%array)
  end if
  x=(int(mult*state%array(state%index))+0.5)*ulps
  state%index=state%index+1
  return
end subroutine rng_number_s0



subroutine rng_number_d1(state,x)
  use rngf77
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
     call rand_batch(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=state%array(i-j+state%index)+ulp2
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_d1


subroutine rng_number_s1(state,x)
  use rngf77
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
     call rand_batch(state%index,state%array)
     do i=j,min(j+rng_k,n)-1
        x(i+n0)=(int(mult*state%array(i-j+state%index))+0.5)*ulps
     end do
     state%index=state%index+(min(rng_k,n-j))
  end do
  return
end subroutine rng_number_s1


subroutine rng_number_d2(state,x)
  use rngdef
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_d1(state,x(:,i))
  end do
end subroutine rng_number_d2


subroutine rng_number_s2(state,x)
  use rngdef
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_number_s1(state,x(:,i))
  end do
end subroutine rng_number_s2


subroutine rng_number_d3(state,x)
  use rngdef
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d2(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_d2(state,x(:,:,i))
  end do
end subroutine rng_number_d3


subroutine rng_number_s3(state,x)
  use rngdef
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s2(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_number_s2(state,x(:,:,i))
  end do
end subroutine rng_number_s3


subroutine rng_gauss_d1(state,x)
  use rngf77
  implicit none
  real(kind=double), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_d1(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_d1
  end interface
  integer :: i
  real(kind=double), parameter :: pi= 3.14159265358979323846264338328d0 
  real(kind=double) :: theta,z
  call rng_number_d1(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0d0 *x(i)- 1.0d0 )
     z=sqrt(- 2.0d0 *log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0d0 *x(ubound(x,1))- 1.0d0 )
  z=sqrt(- 2.0d0 *log(random(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_d1

subroutine rng_gauss_d0(state,x)
  use rngdef
  implicit none
  real(kind=double), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1
  end interface
  real(kind=double), dimension(1) :: y
  call rng_gauss_d1(state, y)
  x=y(1)
end subroutine rng_gauss_d0


subroutine rng_gauss_s1(state,x)
  use rngf77
  implicit none
  real(kind=single), intent(out), dimension(:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_number_s1(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_number_s1
  end interface
  integer :: i
  real(kind=single), parameter :: pi= 3.14159265358979323846264338328 
  real(kind=single) :: theta,z
  call rng_number_s1(state,x)
  do i=lbound(x,1),lbound(x,1)+int(size(x,1)/2)*2-1,2
     theta=pi*(2.0*x(i)-1.0)
     z=sqrt(-2.0*log(x(i+1)))
     x(i)=z*cos(theta)
     x(i+1)=z*sin(theta)
  end do

  if (mod(size(x,1),2) .eq. 0) return

  theta=pi*(2.0*x(ubound(x,1))-1.0)
  z=sqrt(-2.0*log(srandom(state%index,state%array)))
  x(ubound(x,1))=z*cos(theta)
  return
end subroutine rng_gauss_s1

subroutine rng_gauss_s0(state,x)
  use rngdef
  implicit none
  real(kind=single), intent(out) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1
  end interface
  real(kind=single), dimension(1) :: y
  call rng_gauss_s1(state, y)
  x=y(1)
end subroutine rng_gauss_s0


subroutine rng_gauss_d2(state,x)
  use rngdef
  implicit none
  real(kind=double), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d1(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_d1(state,x(:,i))
  end do
end subroutine rng_gauss_d2


subroutine rng_gauss_s2(state,x)
  use rngdef
  implicit none
  real(kind=single), intent(out), dimension(:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s1(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s1
  end interface
  integer :: i
  do i=lbound(x,2),ubound(x,2)
     call rng_gauss_s1(state,x(:,i))
  end do
end subroutine rng_gauss_s2


subroutine rng_gauss_d3(state,x)
  use rngdef
  implicit none
  real(kind=double), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_d2(state,x)
       use rngdef
       implicit none
       real(kind=double), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_d2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_d2(state,x(:,:,i))
  end do
end subroutine rng_gauss_d3


subroutine rng_gauss_s3(state,x)
  use rngdef
  implicit none
  real(kind=single), intent(out), dimension(:,:,:) :: x
  type(rng_state), intent(inout) :: state
  interface
     subroutine rng_gauss_s2(state,x)
       use rngdef
       implicit none
       real(kind=single), intent(out), dimension(:,:) :: x
       type(rng_state), intent(inout) :: state
     end subroutine rng_gauss_s2
  end interface
  integer :: i
  do i=lbound(x,3),ubound(x,3)
     call rng_gauss_s2(state,x(:,:,i))
  end do
end subroutine rng_gauss_s3


subroutine rng_set_seed_time(seed,time)
  use rngf77
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  integer, dimension(8), optional, intent(in) :: time
  integer, dimension(8) :: ltime
  if (present(time)) then
     call set_random_seed(time,seed)
  else
     call date_and_time(values=ltime)
     call set_random_seed(ltime,seed)
  end if
end subroutine rng_set_seed_time

subroutine rng_set_seed_int(seed,n)
  use rngdef
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
end subroutine rng_set_seed_int

subroutine rng_set_seed_string(seed,string)
  use rngf77
  implicit none
  integer, dimension(rng_s), intent(out) :: seed
  character(len=*), intent(in) :: string
  integer :: i,ch
  do i=1,len(string)
     ch=ichar(string(i:i))
     if (ch.ge.ichar('0').and.ch.le.ichar('9')) then
        call decimal_to_seed(string(i:),seed)
        return
     else if (ch.gt.ichar(' ').and.ch.lt.127) then
        call string_to_seed(string(i:),seed)
        return
     end if
  end do
  seed=0
  return
end subroutine rng_set_seed_string
 



 













! Version 2.0 of random number routines
! Author: Charles Karney <karney@princeton.edu>
! Date: 1999-10-05 14:33 -0400
!
      function random(ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      Real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      double precision  random
      Real srandom
      integer ri
      double precision  ra(0:100-1)
      external rand_batch

      if(ri.GE.100)then
      call rand_batch(ri,ra(0))
      end if
      random=ra(ri)+ulp2
      ri=ri+1
      return

      entry srandom(ri,ra)
      if(ri.GE.100)then
      call rand_batch(ri,ra(0))
      end if

      srandom=(int(mult*ra(ri))+0.5)*ulps

      ri=ri+1
      return
      end
!
      subroutine random_array(y,n,ri,ra)
      implicit none
      double precision  ulp2
      parameter(ulp2= 2.0d0 **(-47-1))

      Real ulps
      double precision  mult
      parameter(ulps=2.0**(-23),mult= 2.0d0 **23)

      integer n
      double precision  y(0:n-1)

      Real ys(0:n-1)

      integer ri
      double precision  ra(0:100-1)
      integer i,k,j
      external rand_batch

      if(n.LE.0)return
      k=min(n,100-ri)
      do i=0,k-1
      y(i)=ra(i+ri)+ulp2
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch(ri,ra(0))
      do i=j,min(j+100,n)-1
      y(i)=ra(i-j+ri)+ulp2

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      entry srandom_array(ys,n,ri,ra)
      if(n.LE.0)return
      k=min(n,100-ri)
      do i=0,k-1
      ys(i)=(int(mult*ra(i+ri))+0.5)*ulps
      end do
      ri=ri+(k)
      do j=k,n-1,100
      call rand_batch(ri,ra(0))
      do i=j,min(j+100,n)-1
      ys(i)=(int(mult*ra(i-j+ri))+0.5)*ulps

      end do
      ri=ri+(min(100,n-j))
      end do
      return
      end
!
      subroutine rand_batch(ri,ra)
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
      subroutine random_init(seed,ri,ra)
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
      odd=mod(s(7),2).NE.0
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
      odd=odd.OR.(mod(s(7),2).NE.0)
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
      subroutine decimal_to_seed(decimal,seed)
      implicit none
      character*(*)decimal
      integer seed(0:8-1)
      external rand_axc
      integer i,ten(0:8-1),c(0:8-1),ch
      data ten/10,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(decimal)
      ch=ichar(decimal(i:i))
      if(ch.GE.ichar('0').AND.ch.LE.ichar('9'))then
      c(0)=ch-ichar('0')
      call rand_axc(ten,seed,c)
      end if
      end do
      return
      end
!
      subroutine string_to_seed(string,seed)
      implicit none
      integer b
      parameter(b=2**14)
      character*(*)string
      integer seed(0:8-1)
      external rand_axc
      integer t,i,k,unity(0:8-1),c(0:8-1),ch
      data unity/1,7*0/
      do i=0,8-1
      seed(i)=0
      c(i)=0
      end do
      do i=1,len(string)
      ch=ichar(string(i:i))
      if(ch.GT.ichar(' ').AND.ch.LT.127)then
      t=mod(seed(0),2)*(b/2)
      do k=0,8-1
      seed(k)=int(seed(k)/2)
      if(k.LT.8-1)then
      seed(k)=seed(k)+(mod(seed(k+1),2)*(b/2))
      else
      seed(k)=seed(k)+(t)
      end if
      end do
      c(0)=ch
      call rand_axc(unity,seed,c)
      end if
      end do
      return
      end
!
      subroutine set_random_seed(time,seed)
      implicit none
      integer time(8)
      integer seed(0:8-1)
      character*21 c
      external decimal_to_seed
      c=' '
      write(c(1:8),'(i4.4,2i2.2)')time(1),time(2),time(3)
      write(c(9:12),'(i1.1,i3.3)') (1-sign(1,time(4)))/2,abs(time(4))
      write(c(13:21),'(3i2.2,i3.3)')time(5),time(6),time(7),time(8)
      call decimal_to_seed(c,seed)
      return
      end
!
      subroutine seed_to_decimal(seed,decimal)
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
      if(z(i).GT.0)k=i
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
      if(z(max(0,k)).EQ.0)k=k-1
      j=pow*(i+1)
      if(k.GE.0)then
      str(j-(pow-1):j)='0000'
      else
      str(j-(pow-1):j)='   0'
      end if
90001 continue
      if(t.EQ.0)goto 90010
      str(j:j)=char(ichar('0')+mod(t,10))
      j=j-1
      t=int(t/10)
      goto 90001
90010 continue
      if(k.GE.0)goto 90000
      if(len(decimal).GT.len(str))then
      decimal(:len(decimal)-len(str))=' '
      decimal(len(decimal)-len(str)+1:)=str
      else
      decimal=str(len(str)-len(decimal)+1:)
      end if
      return
      end
!
      subroutine rand_next_seed(n,ax,cx,y)
      implicit none
      integer n,ax(0:8-1),cx(0:8-1)
      integer y(0:8-1)
      external rand_axc
      integer a(0:8-1),c(0:8-1),z(0:8-1),t(0:8-1),m,i
      data z/8*0/
      if(n.EQ.0)return
      m=n
      do i=0,8-1
      a(i)=ax(i)
      c(i)=cx(i)
      end do
90000 continue
      if(mod(m,2).GT.0)then
      call rand_axc(a,y,c)
      end if
      m=int(m/2)
      if(m.EQ.0)return
      do i=0,8-1
      t(i)=c(i)
      end do
      call rand_axc(a,c,t)
      do i=0,8-1
      t(i)=a(i)
      end do
      call rand_axc(t,a,z)
      goto 90000
      end
!
      subroutine next_seed3(n0,n1,n2,seed)
      implicit none
      integer n0,n1,n2
      integer seed(0:8-1)
      external rand_next_seed
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
      if(n2.GT.0)then
      call rand_next_seed(n2,af2,cf2,seed)
      else if(n2.LT.0)then
      call rand_next_seed(-n2,ab2,cb2,seed)
      end if
      if(n1.GT.0)then
      call rand_next_seed(n1,af1,cf1,seed)
      else if(n1.LT.0)then
      call rand_next_seed(-n1,ab1,cb1,seed)
      end if
      entry next_seed(n0,seed)
      if(n0.GT.0)then
      call rand_next_seed(n0,af0,cf0,seed)
      else if(n0.LT.0)then
      call rand_next_seed(-n0,ab0,cb0,seed)
      end if
      return
      end
!
      subroutine rand_axc(a,x,c)
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
