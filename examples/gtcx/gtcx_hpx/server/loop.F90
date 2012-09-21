!========================================================================

    Subroutine loop(hpx4_bti,hpx4_mype,hpx4_numberpe)

!========================================================================

  use, intrinsic :: iso_c_binding, only : c_ptr
  use precision
  implicit none
! interface {{{
  interface
    Subroutine setup(hpx4_bti,&
             t_gids, p_gids,&
! global parameters
                       ihistory,snapout,maxmpsi,&
          mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile,&
           nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron,&
                                       mtheta, &
                                       deltat, &
           do_collision, &
! particle decomp
              ntoroidal,npartdom,&
              partd_comm,nproc_partd,myrank_partd,&
              toroidal_comm,nproc_toroidal,myrank_toroidal,&
              left_pe,right_pe,&
              toroidal_domain_location,particle_domain_location,&
! particle array
                                      kzion,kzelectron,jtelectron0,jtelectron1,&
                                        jtion0,jtion1,&
                                       wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1, &
                                        wpion,wtion0,wtion1,&
                                         zion,zion0,zelectron,&
        zelectron0,zelectron1, &
! field array
                      mmpsi, &
                                      itran,igrid, &
                                          jtp1,jtp2,&
                                       phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt,&
                                         phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit,&
                                           evector,wtp1,wtp2,phisave,&
              Total_field_energy, &
! diagnosis array
                       mflux,num_mode,m_poloidal,&
          nmode,mmode,&
           efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy,eflux,&
       rmarker,rdtemi,rdteme,pfluxpsi,&
       amp_mode, &
                                       hfluxpsi,&
                                           eigenmode,&
           etracer,ptracer, &
! particle_tracking
          track_particles,nptrack,isnap,&
                                           ptracked,&
                                      ntrackp)
      
      use, intrinsic :: iso_c_binding, only : c_ptr
      use precision
      integer,dimension(:),allocatable :: t_gids
      integer,dimension(:),allocatable :: p_gids
!  global parameters
      integer :: ihistory,snapout,maxmpsi
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

!     particle decomp
      integer  :: ntoroidal,npartdom
      integer  :: partd_comm,nproc_partd,myrank_partd
      integer  :: toroidal_comm,nproc_toroidal,myrank_toroidal
      integer  :: left_pe,right_pe
      integer  :: toroidal_domain_location,particle_domain_location

!     particle array
      integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
      integer,dimension(:,:),allocatable :: jtion0,jtion1
      real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
           wtelectron0,wtelectron1
      real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
      real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,&
            zelectron0,zelectron1

!     field array
      integer :: mmpsi
      integer,dimension(:),allocatable :: itran,igrid
      integer,dimension(:,:,:),allocatable :: jtp1,jtp2
      real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
           pmarki,pmarke,zonali,zonale,gradt
      real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
           markere,pgyro,tgyro,dtemper,heatflux,phit
      real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
      real(wp) :: Total_field_energy(3)

!     diagnosis array
      integer :: mflux,num_mode,m_poloidal
      integer nmode(num_mode),mmode(num_mode)
      real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
           entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
           rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
           amp_mode(2,num_mode,2)
      real(wp),dimension(:),allocatable :: hfluxpsi
      real(wp),dimension(:,:,:),allocatable :: eigenmode
      real(wp) etracer,ptracer(4)

!     particle_tracking
      integer track_particles,nptrack,isnap
      real(wp),dimension(:,:,:),allocatable :: ptracked
      integer,dimension(:),allocatable :: ntrackp

      TYPE(C_PTR), INTENT(IN), VALUE :: hpx4_bti
    end subroutine setup
    subroutine load(&
! global parameters
                       ihistory,snapout,maxmpsi,&
          mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile,&
           nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron,&
                                       mtheta, &
                                       deltat, &
           do_collision, &
! particle array
                                      kzion,kzelectron,jtelectron0,jtelectron1,&
                                        jtion0,jtion1,&
                                       wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1, &
                                        wpion,wtion0,wtion1,&
                                         zion,zion0,zelectron,&
        zelectron0,zelectron1, &
! field array
                      mmpsi, &
                                      itran,igrid, &
                                          jtp1,jtp2,&
                                       phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt,&
                                         phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit,&
                                           evector,wtp1,wtp2,phisave,&
              Total_field_energy, &
! particle_tracking
          track_particles,nptrack,isnap,&
                                           ptracked,&
                                      ntrackp,&
! particle decomp
              ntoroidal,npartdom,&
              partd_comm,nproc_partd,myrank_partd,&
              toroidal_comm,nproc_toroidal,myrank_toroidal,&
              left_pe,right_pe,&
              toroidal_domain_location,particle_domain_location,&
! diagnosis array
                       mflux,num_mode,m_poloidal,&
          nmode,mmode,&
           efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy,eflux,&
       rmarker,rdtemi,rdteme,pfluxpsi,&
       amp_mode, &
                                       hfluxpsi,&
                                           eigenmode,&
           etracer,ptracer &
               )
    use precision
    implicit none

!    global parameters
     integer :: ihistory,snapout,maxmpsi
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

!   particle array
    integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
    integer,dimension(:,:),allocatable :: jtion0,jtion1
    real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
         wtelectron0,wtelectron1
    real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
    real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,&
          zelectron0,zelectron1

!   field array
    integer :: mmpsi
    integer,dimension(:),allocatable :: itran,igrid
    integer,dimension(:,:,:),allocatable :: jtp1,jtp2
    real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
         pmarki,pmarke,zonali,zonale,gradt
    real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
         markere,pgyro,tgyro,dtemper,heatflux,phit
    real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
    real(wp) :: Total_field_energy(3)

!   particle_tracking
    integer track_particles,nptrack,isnap
    real(wp),dimension(:,:,:),allocatable :: ptracked
    integer,dimension(:),allocatable :: ntrackp

!   particle decomp
    integer  :: ntoroidal,npartdom
    integer  :: partd_comm,nproc_partd,myrank_partd
    integer  :: toroidal_comm,nproc_toroidal,myrank_toroidal
    integer  :: left_pe,right_pe
    integer  :: toroidal_domain_location,particle_domain_location

!   diagnosis array
    integer :: mflux,num_mode,m_poloidal
    integer nmode(num_mode),mmode(num_mode)
    real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
          entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
          rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
          amp_mode(2,num_mode,2)
    real(wp),dimension(:),allocatable :: hfluxpsi
    real(wp),dimension(:,:,:),allocatable :: eigenmode
    real(wp) etracer,ptracer(4)
    end subroutine load
  end interface
! }}}

  integer hpx4_mype,hpx4_numberpe
  TYPE(C_PTR), INTENT(IN), VALUE :: hpx4_bti
! hjw
  integer,dimension(:),allocatable :: t_gids
  integer,dimension(:),allocatable :: p_gids

!  global parameters
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

! particle decomp
  integer  :: ntoroidal,npartdom
  integer  :: partd_comm,nproc_partd,myrank_partd
  integer  :: toroidal_comm,nproc_toroidal,myrank_toroidal
  integer  :: left_pe,right_pe
  integer  :: toroidal_domain_location,particle_domain_location

! particle array
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,&
        zelectron0,zelectron1

! field array
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: Total_field_energy(3)

! diagnosis array
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)

! particle_tracking
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp

! local variables
  integer i

  mype = hpx4_mype
  numberpe = hpx4_numberpe

  istep = 0

  ! input parameters, setup equilibrium, allocate memory 
  call setup(hpx4_bti,&  ! {{{
             t_gids, p_gids,&
! global parameters
                       ihistory,snapout,maxmpsi,&
          mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile,&
           nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron,&
                                       mtheta, &
                                       deltat, &
           do_collision, &
! particle decomp
              ntoroidal,npartdom,&
              partd_comm,nproc_partd,myrank_partd,&
              toroidal_comm,nproc_toroidal,myrank_toroidal,&
              left_pe,right_pe,&
              toroidal_domain_location,particle_domain_location,&
! particle array
                                      kzion,kzelectron,jtelectron0,jtelectron1,&
                                        jtion0,jtion1,&
                                       wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1, &
                                        wpion,wtion0,wtion1,&
                                         zion,zion0,zelectron,&
        zelectron0,zelectron1, &
! field array
                      mmpsi, &
                                      itran,igrid, &
                                          jtp1,jtp2,&
                                       phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt,&
                                         phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit,&
                                           evector,wtp1,wtp2,phisave,&
              Total_field_energy, &
! diagnosis array
                       mflux,num_mode,m_poloidal,&
          nmode,mmode,&
           efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy,eflux,&
       rmarker,rdtemi,rdteme,pfluxpsi,&
       amp_mode, &
                                       hfluxpsi,&
                                           eigenmode,&
           etracer,ptracer, &
! particle_tracking
          track_particles,nptrack,isnap,&
                                           ptracked,&
                                      ntrackp) ! }}}

  ! initialize particle position and velocity
  call load(& ! {{{
! global parameters
                       ihistory,snapout,maxmpsi,&
          mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile,&
           nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron,&
                                       mtheta, &
                                       deltat, &
           do_collision, &
! particle array
                                      kzion,kzelectron,jtelectron0,jtelectron1,&
                                        jtion0,jtion1,&
                                       wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1, &
                                        wpion,wtion0,wtion1,&
                                         zion,zion0,zelectron,&
        zelectron0,zelectron1, &
! field array
                      mmpsi, &
                                      itran,igrid, &
                                          jtp1,jtp2,&
                                       phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt,&
                                         phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit,&
                                           evector,wtp1,wtp2,phisave,&
              Total_field_energy, &
! particle_tracking
          track_particles,nptrack,isnap,&
                                           ptracked,&
                                      ntrackp,&
! particle decomp
              ntoroidal,npartdom,&
              partd_comm,nproc_partd,myrank_partd,&
              toroidal_comm,nproc_toroidal,myrank_toroidal,&
              left_pe,right_pe,&
              toroidal_domain_location,particle_domain_location,&
! diagnosis array
                       mflux,num_mode,m_poloidal,&
          nmode,mmode,&
           efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy,eflux,&
       rmarker,rdtemi,rdteme,pfluxpsi,&
       amp_mode, &
                                       hfluxpsi,&
                                           eigenmode,&
           etracer,ptracer &
              ) ! }}}

end subroutine loop
