module precision
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
#ifdef DOUBLE_PRECISION
  integer, parameter :: wp=doubleprec

  real(wp), parameter :: ONE=1.0d+00
  real(wp), parameter :: TEN=10.0d+00
  real(wp), parameter :: ZERO=0.0d+00
  real(wp), parameter :: PT_ONE=0.1d+00
  real(wp), parameter :: ONE_TEN_THOUSANDTH=1.0d-04
  real(wp), parameter :: SMALL=1.0d-20
  real(wp), parameter :: BIG=1.0d+20

#else
  integer, parameter :: wp=singleprec

  real(wp), parameter :: ONE=1.0e+00
  real(wp), parameter :: TEN=10.0e+00
  real(wp), parameter :: ZERO=0.0e+00
  real(wp), parameter :: PT_ONE=0.1e+00
  real(wp), parameter :: ONE_TEN_THOUSANDTH=1.0e-04
  real(wp), parameter :: SMALL=1.0e-20
  real(wp), parameter :: BIG=1.0e+20

#endif





end module precision

module global_parameters
  use precision
  integer,parameter :: ihistory=22,snapout=33,maxmpsi=192
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

#ifdef _SX
! SX-6 trick to minimize bank conflict in chargei
  integer,dimension(0:maxmpsi) :: mmtheta
!cdir duplicate(mmtheta,1024)
#endif

end module global_parameters

module particle_array
  use precision
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,zelectron0,zelectron1
end module particle_array

module particle_tracking
  use precision
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp
end module particle_tracking

module field_array
  use precision
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: Total_field_energy(3)

#ifdef _SX
! SX-6 trick to minimize bank conflicts in chargei
!cdir duplicate(iigrid,1024)
  integer,dimension(0:mmpsi) :: iigrid
!cdir duplicate(qqtinv,1024)
  real(wp),dimension(0:mmpsi) :: qqtinv
#endif

end module field_array

module diagnosis_array
  use precision
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)
end module diagnosis_array

