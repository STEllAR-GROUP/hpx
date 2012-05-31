subroutine gtc()
  !use global_parameters
  !use particle_array
  !use particle_tracking
  !use field_array
  !use diagnosis_array

  implicit none
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
  integer i,ierror
  real(doubleprec) time(9),timewc(9),t0,dt,t0wc,dtwc,loop_time
  real(doubleprec) tracktcpu,tracktwc,tr0,tr0wc
  character(len=10) ic(8)

  !tr0=0.0
  !tr0wc=0.0
  !tracktcpu=0.0
  !tracktwc=0.0
  print*,' hello world TEST'
!
!  time=0.0
!  t0=0.0
!  timewc=0.0
!  t0wc=0.0
!  call timer(t0,dt,t0wc,dtwc)
!  time(8)=t0
!  timewc(8)=t0wc
!  istep=0

end subroutine
