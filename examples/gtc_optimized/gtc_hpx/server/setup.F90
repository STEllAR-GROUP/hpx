!========================================================================

    module particle_decomp

!========================================================================
  integer  :: ntoroidal,npartdom
  integer  :: partd_comm,nproc_partd,myrank_partd
  integer  :: toroidal_comm,nproc_toroidal,myrank_toroidal
  integer  :: left_pe,right_pe
  integer  :: toroidal_domain_location,particle_domain_location
end module particle_decomp


!========================================================================

    Subroutine setup(hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal, &
                     hpx_left_pe, hpx_right_pe)

!========================================================================

  use global_parameters
  use particle_decomp
  use particle_array
  use field_array
  use diagnosis_array
  use particle_tracking
  implicit none

  integer i,j,k,ierror,ij,mid_theta,ip,jt,indp,indt,mtest,micell,mecell
  integer mi_local,me_local,hpx_left_pe, hpx_right_pe
  integer hpx_numberpe,hpx_mype,hpx_npartdom,hpx_ntoroidal
  real(wp) r0,b0,temperature,tdum,r,q,sint,dtheta_dx,rhoi,b,zdum,&
       edensity0,delr,delt,rmax,rmin,wt,tau_vth,zeff
  CHARACTER(LEN=10) date, time
  namelist /run_parameters/ numberpe,mi,mgrid,mid_theta,mtdiag,delr,delt,&
       ulength,utime,gyroradius,tstep
! hjw
  integer nmem
! hjw

#ifdef __AIX
#define FLUSH flush_
#else
#define FLUSH flush
#endif

! total # of PE and rank of PE
  numberpe = hpx_numberpe
  mype = hpx_mype
!  call mpi_comm_size(mpi_comm_world,numberpe,ierror)
!  call mpi_comm_rank(mpi_comm_world,mype,ierror)

! Read the input file that contains the run parameters
  call read_input_params(micell,mecell,r0,b0,temperature,edensity0)

! numerical constant
  pi=4.0_wp*atan(1.0_wp)
  mstep=max(2,mstep)
  msnap=min(msnap,mstep/ndiag)
  isnap=mstep/msnap
  idiag1=mpsi/2
  idiag2=mpsi/2
  if(nonlinear < 0.5_wp)then
     paranl=0.0_wp
     mode00=0
     idiag1=1
     idiag2=mpsi
  endif
  rc=rc*(a0+a1)
  rw=1.0_wp/(rw*(a1-a0))

! Set up the particle decomposition within each toroidal domain
  call set_particle_decomp(hpx_npartdom,hpx_ntoroidal,hpx_left_pe,hpx_right_pe)

! equilibrium unit: length (unit=cm) and time (unit=second) unit
  ulength=r0
  utime=1.0_wp/(9580._wp*b0) ! time unit = inverse gyrofrequency of proton 
! primary ion thermal gyroradius in equilibrium unit, vthermal=sqrt(T/m)
  gyroradius=102.0_wp*sqrt(aion*temperature)/(abs(qion)*b0)/ulength
  tstep=tstep*aion/(abs(qion)*gyroradius*kappati)
  
! basic ion-ion collision time, Braginskii definition
  if(tauii>0.0_wp)then
     !tau_vth=24.0_wp-log(sqrt(edensity0)/temperature) ! Coulomb logarithm
     !tau_vth=2.09e7_wp*(temperature)**1.5_wp/(edensity0*tau_vth*2.31_wp)*sqrt(2.0_wp)/utime
     r=0.5*(a0+a1)
     q=q0+q1*r/a+q2*r*r/(a*a)
     zeff=qion  ! Set zeff to main ion species
     tau_vth=23.0-log(sqrt(zeff*zeff*edensity0)/temperature**1.5)
     tau_vth=2.09e7*(temperature)**1.5_wp*sqrt(aion)/(edensity0*tau_vth*utime*zeff)
     tauii=tau_vth*tauii  ! Multiply v_thermal tau by scaling factor tauii
     do_collision=.TRUE.
     if(mype==0)write(stdout,*)'Collision time tauii=',tauii,'  nu_star=',q/(tauii*gyroradius*r**1.5),'  q=',q
  else
     do_collision=.FALSE.
  endif

! allocate memory
     nmem = (16 * (mpsi+1)) + (m_poloidal*num_mode*mpsi)
  allocate (qtinv(0:mpsi),itran(0:mpsi),mtheta(0:mpsi),&
     deltat(0:mpsi),rtemi(0:mpsi),rteme(0:mpsi),&
     rden(0:mpsi),igrid(0:mpsi),pmarki(0:mpsi),&
     pmarke(0:mpsi),phi00(0:mpsi),phip00(0:mpsi),&
     hfluxpsi(0:mpsi),zonali(0:mpsi),zonale(0:mpsi),gradt(mpsi),&
     eigenmode(m_poloidal,num_mode,mpsi),STAT=mtest)
  if (mtest /= 0 .and. mtest /= 5014) then
! hjw
!    write(0,*)mype,'*** Cannot allocate qtinv: mtest=',mtest
     write(0,*)mype,'*** qtinv: Allocate Error: ',nmem, ' words mtest= ',mtest
! hjw
!     call MPI_ABORT(MPI_COMM_WORLD,1,ierror)
  endif

! --- Define poloidal grid ---
! grid spacing
  deltar=(a1-a0)/real(mpsi)

! grid shift associated with fieldline following coordinates
  tdum=2.0_wp*pi*a1/real(mthetamax)
  do i=0,mpsi
     r=a0+deltar*real(i)
     mtheta(i)=max(2,min(mthetamax,2*int(pi*r/tdum+0.5_wp))) !even # poloidal grid
     deltat(i)=2.0_wp*pi/real(mtheta(i))
     q=q0+q1*r/a+q2*r*r/(a*a)
     itran(i)=int(real(mtheta(i))/q+0.5_wp)
     qtinv(i)=real(mtheta(i))/real(itran(i)) !q value for coordinate transformation
     qtinv(i)=1.0_wp/qtinv(i) !inverse q to avoid divide operation
     itran(i)=itran(i)-mtheta(i)*(itran(i)/mtheta(i))
  enddo
! un-comment the next two lines to use magnetic coordinate
!  qtinv=0.0
!  itran=0

! When doing mode diagnostics, we need to switch from the field-line following
! coordinates alpha-zeta to a normal geometric grid in theta-zeta. This
! translates to a greater number of grid points in the zeta direction, which
! is mtdiag. Precisely, mtdiag should be mtheta/q but since mtheta changes
! from one flux surface to another, we use a formula that gives us enough
! grid points for all the flux surfaces considered.
  mtdiag=(mthetamax/mzetamax)*mzetamax
! In the case of a non-linear run, we need only half the points.
  if(nonlinear > 0.5_wp)mtdiag=mtdiag/2

! starting point for a poloidal grid
  igrid(0)=1
  do i=1,mpsi
     igrid(i)=igrid(i-1)+mtheta(i-1)+1
  enddo

! number of grids on a poloidal plane
  mgrid=sum(mtheta+1)
  mi_local=micell*(mgrid-mpsi)*mzeta          !# of ions in toroidal domain
  mi=micell*(mgrid-mpsi)*mzeta/npartdom       !# of ions per processor
  if(mi<mod(mi_local,npartdom))mi=mi+1
  me_local=mecell*(mgrid-mpsi)*mzeta          !# of electrons in toroidal domain
  me=mecell*(mgrid-mpsi)*mzeta/npartdom       !# of electrons per processor
  if(me<mod(me_local,npartdom))me=me+1
  mimax=mi+100*ceiling(sqrt(real(mi))) !ions array upper bound
  memax=me+100*ceiling(sqrt(real(me))) !electrons array upper bound

   if (mype==0) then
    write(0,*)' #ions per proc= ',mi,' #electrons per proc= ',me
    write(0,*)' #ions per grid cell= ',micell,'   #electrons per grid cell= ',mecell
    write(0,*)' #ions per toroidal domain=',mi_local,'   #electrons per toroidal domain= ',me_local
   endif


  delr=deltar/gyroradius
  delt=deltat(mpsi/2)*(a0+deltar*real(mpsi/2))/gyroradius
  mid_theta=mtheta(mpsi/2)
  if(mype == 0)write(stdout,run_parameters)

! allocate memory
  allocate(pgyro(4,mgrid),tgyro(4,mgrid),markeri(mzeta,mgrid),&
     densityi(0:mzeta,mgrid),phi(0:mzeta,mgrid),evector(3,0:mzeta,mgrid),&
     jtp1(2,mgrid,mzeta),jtp2(2,mgrid,mzeta),wtp1(2,mgrid,mzeta),&
     wtp2(2,mgrid,mzeta),dtemper(mgrid,mzeta),heatflux(mgrid,mzeta),&
     STAT=mtest)
  if (mtest /= 0 .and. mtest /= 5014) then
! hjw
     nmem= &
     ((4*mgrid)+(4*mgrid)+(mzeta*mgrid)+&
     (mzeta*mgrid)+(mzeta*mgrid)+(3*mzeta*mgrid)+&
     (2*mgrid*mzeta)+(2*mgrid*mzeta)+(2*mgrid*mzeta)+&
     (2*mgrid*mzeta)+(mgrid*mzeta)+(mgrid*mzeta))

!   write(0,*)mype,'*** setup: Cannot allocate pgyro: mtest=',mtest
    write(0,*)mype,'*** pgyro: Allocate Error: ',nmem, ' words mtest= ',mtest
! hjw
!     call MPI_ABORT(MPI_COMM_WORLD,1,ierror)
  endif

! temperature and density on the grid, T_i=n_0=1 at mid-radius
  rtemi=1.0_wp
  rteme=1.0_wp
  rden=1.0_wp
  phi=0.0_wp
  phip00=0.0_wp
  pfluxpsi=0.0_wp
  rdtemi=0.0_wp
  rdteme=0.0_wp
  zonali=0.0_wp
  zonale=0.0_wp
 
! # of marker per grid, Jacobian=(1.0+r*cos(theta+r*sin(theta)))*(1.0+r*cos(theta))
  pmarki=0.0_wp
!$omp parallel do private(i,j,k,r,ij,zdum,tdum,rmax,rmin)
  do i=0,mpsi
     r=a0+deltar*real(i)
     do j=1,mtheta(i)
        ij=igrid(i)+j
        do k=1,mzeta
           zdum=zetamin+real(k)*deltaz
           tdum=real(j)*deltat(i)+zdum*qtinv(i)
           markeri(k,ij)=(1.0_wp+r*cos(tdum))**2
           pmarki(i)=pmarki(i)+markeri(k,ij)
        enddo
     enddo
     rmax=min(a1,r+0.5_wp*deltar)
     rmin=max(a0,r-0.5_wp*deltar)
     !!tdum=real(mi)*(rmax*rmax-rmin*rmin)/(a1*a1-a0*a0)
     tdum=real(mi*npartdom)*(rmax*rmax-rmin*rmin)/(a1*a1-a0*a0)
     do j=1,mtheta(i)
        ij=igrid(i)+j
        do k=1,mzeta
           markeri(k,ij)=tdum*markeri(k,ij)/pmarki(i)
           markeri(k,ij)=1.0_wp/markeri(k,ij) !to avoid divide operation
        enddo
     enddo
     !!pmarki(i)=1.0_wp/(real(numberpe)*tdum)
     pmarki(i)=1.0_wp/(real(ntoroidal)*tdum)
     markeri(:,igrid(i))=markeri(:,igrid(i)+mtheta(i))
  enddo

  if(track_particles == 1)then
    ! We keep track of the particles by tagging them with a number
    ! We add an extra element to the particle array, which will hold
    ! the particle tag, i.e. just a number
    ! Each processor has its own "ptracked" array to accumulate the tracked
    ! particles that may or may not reside in its subdomain.
    ! The vector "ntrackp" keeps contains the number of tracked particles
    ! currently residing on the processor at each time step. We write out
    ! the data to file every (mstep/msnap) steps.
     nparam=7
     allocate(ptracked(nparam,max(nptrack,1),isnap))
     allocate(ntrackp(isnap))
  else
    ! No tagging of the particles
     nparam=6
  endif

! allocate memory
  allocate(zion(nparam,mimax),zion0(nparam,mimax),jtion0(4,mimax),&
     jtion1(4,mimax),kzion(mimax),wzion(mimax),wpion(4,mimax),&
     wtion0(4,mimax),wtion1(4,mimax),STAT=mtest)
  if (mtest /= 0 .and. mtest /= 5014) then
! hjw
  nmem=((nparam*mimax)+(nparam*mimax)+(4*mimax)+&
     (4*mimax)+(mimax)+(mimax)+(4*mimax)+&
     (4*mimax)+(4*mimax))
!   write(0,*)mype,'*** Cannot allocate zion: mtest=',mtest
    write(0,*)mype,'*** zion: Allocate Error: ',nmem, ' words mtest= ',mtest
! hjw
!     call MPI_ABORT(MPI_COMM_WORLD,1,ierror)
  endif
  if(nhybrid>0)then
     allocate(zelectron(6,memax),zelectron0(6,memax),jtelectron0(memax),&
        jtelectron1(memax),kzelectron(memax),wzelectron(memax),&
        wpelectron(memax),wtelectron0(memax),wtelectron1(memax),&
        markere(mzeta,mgrid),densitye(0:mzeta,mgrid),zelectron1(6,memax),&
        phisave(0:mzeta,mgrid,2*nhybrid),phit(0:mzeta,mgrid),STAT=mtest)
     if(mtest /= 0) then
! hjw
     nmem=((6*memax)+(6*memax)+(memax)+&
        (memax)+(memax)+(memax)+&
        (memax)+(memax)+(memax)+&
        (mzeta*mgrid)+(mzeta*mgrid)+(6*memax)+&
        (mzeta*mgrid*2*nhybrid)+(mzeta*mgrid))
!   write(0,*)mype,'*** Cannot allocate zelectron: mtest=',mtest
    write(0,*)mype,'*** zelectron: Allocate Error: ',nmem, ' words mtest= ',mtest
! hjw
!        call MPI_ABORT(MPI_COMM_WORLD,1,ierror)
     endif
     markere=markeri*real(mi,wp)/real(me,wp)
     pmarke=pmarki*real(mi,wp)/real(me,wp)

! initial potential
     phisave=0.0_wp
  endif

! 4-point gyro-averaging for sqrt(mu)=gyroradius on grid of magnetic coordinates
! rho=gyroradius*sqrt(2/(b/b_0))*sqrt(mu/mu_0), mu_0*b_0=m*v_th^2
! dtheta/delta_x=1/(r*(1+r*cos(theta))), delta_x=poloidal length increase
!$omp parallel do private(i,j,r,ij,tdum,q,b,dtheta_dx,rhoi)
  do i=0,mpsi
     r=a0+deltar*real(i,wp)
     do j=0,mtheta(i)
        ij=igrid(i)+j
        tdum=deltat(i)*real(j,wp)
        q=q0+q1*r/a+q2*r*r/(a*a)
        b=1.0_wp/(1.0_wp+r*cos(tdum))
        dtheta_dx=1.0_wp/r
! first two points perpendicular to field line on poloidal surface            
        rhoi=sqrt(2.0_wp/b)*gyroradius
        pgyro(1,ij)=-rhoi
        pgyro(2,ij)=rhoi
! non-orthorgonality between psi and theta: tgyro=-rhoi*dtheta_dx*r*sin(tdum)
        tgyro(1,ij)=0.0_wp
        tgyro(2,ij)=0.0_wp

! the other two points tangential to field line
        tgyro(3,ij)=-rhoi*dtheta_dx
        tgyro(4,ij)=rhoi*dtheta_dx
        pgyro(3:4,ij)=rhoi*0.5_wp*rhoi/r
     enddo
  enddo

! initiate radial interpolation for grid
  do k=1,mzeta
     zdum=zetamin+deltaz*real(k)
!$omp parallel do private(i,ip,j,indp,indt,ij,tdum,jt,wt)
     do i=1,mpsi-1
        do ip=1,2
           indp=min(mpsi,i+ip)
           indt=max(0,i-ip)
           do j=1,mtheta(i)
              ij=igrid(i)+j
! upward
              tdum=(real(j,wp)*deltat(i)+zdum*(qtinv(i)-qtinv(indp)))/deltat(indp)
              jt=floor(tdum)
              wt=tdum-real(jt,wp)
              jt=mod(jt+mtheta(indp),mtheta(indp))
              if(ip==1)then
                 wtp1(1,ij,k)=wt
                 jtp1(1,ij,k)=igrid(indp)+jt
              else
                 wtp2(1,ij,k)=wt
                 jtp2(1,ij,k)=igrid(indp)+jt
              endif
! downward
               
              tdum=(real(j)*deltat(i)+zdum*(qtinv(i)-qtinv(indt)))/deltat(indt)
              jt=floor(tdum)
              wt=tdum-real(jt,wp)
              jt=mod(jt+mtheta(indt),mtheta(indt))
              if(ip==1)then
                 wtp1(2,ij,k)=wt
                 jtp1(2,ij,k)=igrid(indt)+jt
              else
                 wtp2(2,ij,k)=wt
                 jtp2(2,ij,k)=igrid(indt)+jt
              endif
           enddo
        enddo
     enddo
  enddo

end subroutine setup


!=============================================================================

  Subroutine read_input_params(micell,mecell,r0,b0,temperature,edensity0)

!=============================================================================

  use global_parameters
  use particle_decomp
  use particle_tracking
  use diagnosis_array
  implicit none

  logical file_exist
  integer ierror,micell,mecell
  real(wp),intent(INOUT) :: r0,b0,temperature,edensity0
  CHARACTER(LEN=10) date, time

#ifdef _OPENMP
  integer nthreads,omp_get_num_threads
#endif

  namelist /input_parameters/ irun,mstep,msnap,ndiag,nhybrid,nonlinear,paranl,&
       mode00,tstep,micell,mecell,mpsi,mthetamax,mzetamax,npartdom,&
       ncycle,a,a0,a1,q0,q1,q2,rc,rw,aion,qion,aelectron,qelectron,kappati,&
       kappate,kappan,tite,fixed_Tprofile,flow0,&
       flow1,flow2,r0,b0,temperature,edensity0,stdout,nbound,umax,iload,&
       tauii,track_particles,nptrack,rng_control,nmode,mmode
!
! Since it is preferable to have only one MPI process reading the input file,
! we choose the master process to set the default run parameters and to read
! the input file. The parameters will then be broadcast to the other processes.
!

  if(mype==0) then
! Default control parameters
    irun=0                 ! 0 for initial run, any non-zero value for restart
    mstep=1500             ! # of time steps
    msnap=1                ! # of snapshots
    ndiag=4                ! do diag when mod(istep,ndiag)=0
    nonlinear=1.0          ! 1.0 nonlinear run, 0.0 linear run
    nhybrid=0              ! 0: adiabatic electron, >1: kinetic electron
    paranl=0.0             ! 1: keep parallel nonlinearity
    mode00=1               ! 1 include (0,0) mode, 0 exclude (0,0) mode

! run size (both mtheta and mzetamax should be multiples of # of PEs)
    tstep=0.2              ! time step (unit=L_T/v_th), tstep*\omega_transit<0.1 
    micell=2               ! # of ions per grid cell
    mecell=2               ! # of electrons per grid cell
    mpsi=90                ! total # of radial grid points
    mthetamax=640          ! poloidal grid, even and factors of 2,3,5 for FFT
    mzetamax=64            ! total # of toroidal grid points, domain decomp.
    npartdom=1             ! number of particle domain partitions per tor dom.
    ncycle=5               ! subcycle electron
     
! run geometry
    a=0.358                ! minor radius, unit=R_0
    a0=0.1                 ! inner boundary, unit=a
    a1=0.9                 ! outer boundary, unit=a
    q0=0.854               ! q_profile, q=q0 + q1*r/a + q2 (r/a)^2
    q1=0.0
    q2=2.184
    rc=0.5                 ! kappa=exp{-[(r-rc)/rw]**6}
    rw=0.35                ! rc in unit of (a1+a0) and rw in unit of (a1-a0)

! species information
    aion=1.0               ! species isotope #
    qion=1.0               ! charge state
    aelectron=1.0/1837.0
    qelectron=-1.0

! equilibrium unit: R_0=1, Omega_c=1, B_0=1, m=1, e=1
    kappati=6.9            ! grad_T/T
    kappate=6.9
    kappan=kappati*0.319   ! inverse of eta_i, grad_n/grad_T
    fixed_Tprofile=1       ! Maintain Temperature profile (0=no, >0 =yes)
    tite=1.0               ! T_i/T_e
    flow0=0.0              ! d phi/dpsi=gyroradius*[flow0+flow1*r/a+
    flow1=0.0              !                              flow2*(r/a)**2]
    flow2=0.0

! physical unit
    r0=93.4                ! major radius (unit=cm)
    b0=19100.0             ! on-axis vacuum field (unit=gauss)
    temperature=2500.0     ! electron temperature (unit=ev)
    edensity0=0.46e14      ! electron number density (1/cm^3)

! standard output: use 0 or 6 to terminal and 11 to file 'stdout.out'
    stdout=6  
    nbound=4               ! 0 for periodic, >0 for zero boundary 
    umax=4.0               ! unit=v_th, maximum velocity in each direction
    iload=0                ! 0: uniform, 1: non-uniform
    tauii=-1.0             ! -1.0: no collisions, 1.0: collisions
    track_particles=0      ! 1: keep track of some particles
    nptrack=0              ! track nptrack particles every time step
    rng_control=1          ! controls seed and algorithm for random num. gen.
                           ! rng_control>0 uses the portable random num. gen.

! mode diagnostic: 8 modes.
    nmode=(/5, 7, 9,11,13,15,18,20/)    ! n: toroidal mode number
    mmode=(/7,10,13,15,18,21,25,28/)    ! m: poloidal mode number

! Test if the input file gtc.input exists
    inquire(file='gtc.input',exist=file_exist)
    if (file_exist) then
       open(55,file='gtc.input',status='old')
       read(55,nml=input_parameters)
       close(55)
    else
       write(0,*)'******************************************'
       write(0,*)'*** NOTE!!! Cannot find file gtc.input !!!'
       write(0,*)'*** Using default run parameters...'
       write(0,*)'******************************************'
    endif

! Changing the units of a0 and a1 from units of "a" to units of "R_0"
    a0=a0*a
    a1=a1*a

! open file for standard output, record program starting time
    if(stdout /= 6 .and. stdout /= 0)open(stdout,file='stdout.out',status='replace')
    call date_and_time(date,time)
    write(stdout,*) 'Program starts at DATE=', date, 'TIME=', time
    write(stdout,input_parameters)

#ifdef _OPENMP
!$omp parallel private(nthreads)
    nthreads=omp_get_num_threads()  !Get the number of threads if using OMP
!$omp single
    write(stdout,'(/,"===================================")')
    write(stdout,*)' Number of OpenMP threads = ',nthreads
    write(stdout,'("===================================",/)')
!$omp end single nowait
!$omp end parallel
#else
    write(stdout,'(/,"===================================")')
    write(stdout,*)' Run without OpenMP threads'
    write(stdout,'("===================================",/)')
#endif
  endif

! Now send the parameter values to all the other MPI processes
!  call broadcast_input_params(micell,mecell,r0,b0,temperature,edensity0)

end Subroutine read_input_params


!=============================================================================

  Subroutine broadcast_input_params(micell,mecell,r0,b0,temperature,edensity0)

!=============================================================================

  use global_parameters
  use particle_tracking
  use particle_decomp
  use diagnosis_array

  integer,parameter :: n_integers=20+2*num_mode,n_reals=28
  integer  :: integer_params(n_integers)
  real(wp) :: real_params(n_reals)
  integer ierror,micell,mecell
  real(wp),intent(INOUT) :: r0,b0,temperature,edensity0

! The master process, mype=0, holds all the input parameters. We need
! to broadcast their values to the other processes. Instead of issuing
! an expensive MPI_BCAST() for each parameter, it is better to pack
! everything in a single vector, broadcast it, and unpack it.

  if(mype==0)then
!   Pack all the integer parameters in integer_params() array
    integer_params(1)=irun
    integer_params(2)=mstep
    integer_params(3)=msnap
    integer_params(4)=ndiag
    integer_params(5)=nhybrid
    integer_params(6)=mode00
    integer_params(7)=micell
    integer_params(8)=mecell
    integer_params(9)=mpsi
    integer_params(10)=mthetamax
    integer_params(11)=mzetamax
    integer_params(12)=npartdom
    integer_params(13)=ncycle
    integer_params(14)=stdout
    integer_params(15)=nbound
    integer_params(16)=iload
    integer_params(17)=track_particles
    integer_params(18)=nptrack
    integer_params(19)=rng_control
    integer_params(20)=fixed_Tprofile
    integer_params(21:21+num_mode-1)=nmode(1:num_mode)
    integer_params(21+num_mode:21+2*num_mode-1)=mmode(1:num_mode)

!   Pack all the real parameters in real_params() array
    real_params(1)=nonlinear
    real_params(2)=paranl
    real_params(3)=tstep
    real_params(4)=a
    real_params(5)=a0
    real_params(6)=a1
    real_params(7)=q0
    real_params(8)=q1
    real_params(9)=q2
    real_params(10)=rc
    real_params(11)=rw
    real_params(12)=aion
    real_params(13)=qion
    real_params(14)=aelectron
    real_params(15)=qelectron
    real_params(16)=kappati
    real_params(17)=kappate
    real_params(18)=kappan
    real_params(19)=tite
    real_params(20)=flow0
    real_params(21)=flow1
    real_params(22)=flow2
    real_params(23)=r0
    real_params(24)=b0
    real_params(25)=temperature
    real_params(26)=edensity0
    real_params(27)=umax
    real_params(28)=tauii
  endif

! Send input parameters to all processes
!  call MPI_BCAST(integer_params,n_integers,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
!  call MPI_BCAST(real_params,n_reals,mpi_Rsize,0,MPI_COMM_WORLD,ierror)

  if(mype/=0)then
!   Unpack integer parameters
    irun=integer_params(1)
    mstep=integer_params(2)
    msnap=integer_params(3)
    ndiag=integer_params(4)
    nhybrid=integer_params(5)
    mode00=integer_params(6)
    micell=integer_params(7)
    mecell=integer_params(8)
    mpsi=integer_params(9)
    mthetamax=integer_params(10)
    mzetamax=integer_params(11)
    npartdom=integer_params(12)
    ncycle=integer_params(13)
    stdout=integer_params(14)
    nbound=integer_params(15)
    iload=integer_params(16)
    track_particles=integer_params(17)
    nptrack=integer_params(18)
    rng_control=integer_params(19)
    fixed_Tprofile=integer_params(20)
    nmode(1:num_mode)=integer_params(21:21+num_mode-1)
    mmode(1:num_mode)=integer_params(21+num_mode:21+2*num_mode-1)

!   Unpack real parameters
    nonlinear=real_params(1)
    paranl=real_params(2)
    tstep=real_params(3)
    a=real_params(4)
    a0=real_params(5)
    a1=real_params(6)
    q0=real_params(7)
    q1=real_params(8)
    q2=real_params(9)
    rc=real_params(10)
    rw=real_params(11)
    aion=real_params(12)
    qion=real_params(13)
    aelectron=real_params(14)
    qelectron=real_params(15)
    kappati=real_params(16)
    kappate=real_params(17)
    kappan=real_params(18)
    tite=real_params(19)
    flow0=real_params(20)
    flow1=real_params(21)
    flow2=real_params(22)
    r0=real_params(23)
    b0=real_params(24)
    temperature=real_params(25)
    edensity0=real_params(26)
    umax=real_params(27)
    tauii=real_params(28)
  endif

#ifdef DEBUG_BCAST
!    write(mype+10,*)irun,mstep,msnap,ndiag,nhybrid,mode00,micell,mecell,&
!       mpsi,mthetamax,mzetamax,npartdom,ncycle,stdout,nbound,iload,&
!       track_particles,nptrack,rng_control,nmode(1:num_mode),mmode(1:num_mode)
!
!    write(mype+10,*)nonlinear,paranl,tstep,a,a0,a1,q0,q1,q2,rc,rw,aion,qion,&
!       aelectron,qelectron,kappati,kappate,kappan,tite,flow0,flow1,flow2,&
!       r0,b0,temperature,edensity0,umax,tauii
!    close(mype+10)
#endif

end subroutine broadcast_input_params


!=============================================================================

    Subroutine set_particle_decomp(hpx_npartdom,hpx_ntoroidal, &
                              hpx_left_pe,hpx_right_pe)

!=============================================================================

  use global_parameters
  use particle_decomp
!  use particle_array
!  use field_array
!  use diagnosis_array
!  use particle_tracking
  implicit none

  integer  :: i,j,k,pe_number,mtest,ierror
  integer  :: hpx_npartdom, hpx_ntoroidal
  integer  :: hpx_left_pe, hpx_right_pe

! ----- First we verify the consistency of ntoroidal and npartdom -------
! The number of toroidal domains (ntoroidal) times the number of particle
! "domains" (npartdom) needs to be equal to the number of processor "numberpe".
! numberpe cannot be changed since it is given on the command line.

! numberpe must be a multiple of npartdom so change npartdom accordingly
  do while (mod(numberpe,npartdom) /= 0)
     npartdom=npartdom-1
     if(npartdom==1)exit
  enddo
  ntoroidal=numberpe/npartdom
  if(mype==0)then
    write(stdout,*)'*******************************************************'
    write(stdout,*)'  Using npartdom=',npartdom,' and ntoroidal=',ntoroidal
    write(stdout,*)'*******************************************************'
    write(stdout,*)
  endif
  hpx_npartdom = npartdom
  hpx_ntoroidal = ntoroidal

! make sure that mzetamax is a multiple of ntoroidal
  mzetamax=ntoroidal*max(1,int(real(mzetamax)/real(ntoroidal)+0.5_wp))

! Make sure that "mpsi", the total number of flux surfaces, is an even
! number since this quantity will be used in Fast Fourier Transforms
  mpsi=2*(mpsi/2)

! We now give each PE (task) a unique domain identified by 2 numbers: the
! particle and toroidal domain numbers.
!    particle_domain_location = rank of the particle domain holding mype
!    toroidal_domain_location = rank of the toroidal domain holding mype
! 
! On the IBM SP, the MPI tasks are distributed in an orderly fashion to each
! node unless the LoadLeveler instruction "#@ blocking = unlimited" is used.
! On Seaborg for example, the first 16 tasks (mype=0-15) will be assigned to
! the first node that has been allocated to the job, then the next 16
! (mype=16-31) will be assigned to the second node, etc. When not using the
! OpenMP, we want the particle domains to sit on the same node because
! communication is more intensive. To achieve this, successive PE numbers are
! assigned to the particle domains first.
! It is easy to achieve this ordering by simply using mype/npartdom for
! the toroidal domain and mod(mype,npartdom) for the particle domain.
!
!  pe_number=0
!  do j=0,ntoroidal-1
!     do i=0,npartdom-1
!        pe_grid(i,j)=pe_number
!        particle_domain_location(pe_number)=i
!        toroidal_domain_location(pe_number)=j
!        pe_number=pe_number+1
!     enddo
!  enddo

  particle_domain_location=mod(mype,npartdom)
  toroidal_domain_location=mype/npartdom

!  write(0,*)'mype=',mype,"  particle_domain_location =",&
!            particle_domain_location,' toroidal_domain_location =',&
!            toroidal_domain_location,' pi=',pi

! Domain decomposition in toroidal direction.
  mzeta=mzetamax/ntoroidal
  zetamin=2.0_wp*pi*real(toroidal_domain_location)/real(ntoroidal)
  zetamax=2.0_wp*pi*real(toroidal_domain_location+1)/real(ntoroidal)

!  write(0,*)mype,' in set_particle_decomp: mzeta=',mzeta,'  zetamin=',&
!            zetamin,'  zetamax=',zetamax

! grid spacing in the toroidal direction
  deltaz=(zetamax-zetamin)/real(mzeta)

! ---- Create particle domain communicator and toroidal communicator -----
! We now need to create a new communicator which will include only the
! processes located in the same toroidal domain. The particles inside
! each toroidal domain are divided equally between "npartdom" processes.
! Each one of these processes will do a charge deposition on a copy of
! the same grid, requiring a toroidal-domain-wide reduction after the
! deposition. The new communicator will allow the reduction to be done
! only between those processes located in the same toroidal domain.
!
! We also need to create a purely toroidal communicator so that the
! particles with the same particle domain id can exchange with their
! toroidal neighbors.
!
! Form 2 subcommunicators: one that includes all the processes located in
! the same toroidal domain (partd_comm), and one that includes all the
! processes part of the same particle domain (toroidal_comm).
! Here is how to create a new communicator from an old one by using
! the MPI call "MPI_COMM_SPLIT()".
! All the processes passing the same value of "color" will be placed in
! the same communicator. The "rank_in_new_comm" value will be used to
! set the rank of that process on the communicator.
!  call MPI_COMM_SPLIT(old_comm,color,rank_in_new_comm,new_comm,ierror)

! particle domain communicator (for communications between the particle
! domains WITHIN the same toroidal domain)
  !print*,' TEST toroidal location ', toroidal_domain_location,mype
  !print*,' TEST particle location ', particle_domain_location
!*  call MPI_COMM_SPLIT(MPI_COMM_WORLD,toroidal_domain_location,&
!*                      particle_domain_location,partd_comm,ierror)

! toroidal communicator (for communications BETWEEN toroidal domains of same
! particle domain number)
!*  call MPI_COMM_SPLIT(MPI_COMM_WORLD,particle_domain_location,&
!*                      toroidal_domain_location,toroidal_comm,ierror)

!*  call mpi_comm_size(partd_comm,nproc_partd,ierror)
!*  call mpi_comm_rank(partd_comm,myrank_partd,ierror)

!*  call mpi_comm_size(toroidal_comm,nproc_toroidal,ierror)
!*  call mpi_comm_rank(toroidal_comm,myrank_toroidal,ierror)

!  write(0,*)'mype=',mype,'  nproc_toroidal=',nproc_toroidal,&
!       ' myrank_toroidal=',myrank_toroidal,'  nproc_partd=',nproc_partd,&
!       ' myrank_partd=',myrank_partd

  ! component ids for communications between the particle
  ! domains WITHIN the same toroidal domain

  ! component ids for for communications BETWEEN toroidal domains of same
  ! particle domain number

  myrank_toroidal = toroidal_domain_location
  myrank_partd = particle_domain_location

  if ( myrank_toroidal .gt. particle_domain_location ) then
    myrank_toroidal = particle_domain_location
  end if

!  if(nproc_partd/=npartdom)then
!    write(0,*)'*** nproc_partd=',nproc_partd,' NOT EQUAL to npartdom=',npartdom
!    call MPI_ABORT(MPI_COMM_WORLD,1,ierror)
!  endif

!  if(nproc_toroidal/=ntoroidal)then
!    write(0,*)'*** nproc_toroidal=',nproc_toroidal,' NOT EQUAL to ntoroidal=',&
!              ntoroidal
!    call MPI_ABORT(MPI_COMM_WORLD,1,ierror)
!  endif

! We now find the toroidal neighbors of the current toroidal domain and
! store that information in 2 easily accessible variables. This information
! is needed several times inside the code, such as when particles cross
! the domain boundaries. We will use the toroidal communicator to do these
! transfers so we don't need to worry about the value of myrank_partd.
! We have periodic boundary conditions in the toroidal direction so the
! neighbor to the left of myrank_toroidal=0 is (ntoroidal-1).

  left_pe=mod(myrank_toroidal-1+ntoroidal,ntoroidal)
  right_pe=mod(myrank_toroidal+1,ntoroidal)
  hpx_left_pe = left_pe
  hpx_right_pe = right_pe
  !print*,' TEST left_pe ', left_pe
  !print*,' TEST right_pe ',right_pe

end subroutine set_particle_decomp
