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
  use rngdef
  implicit none
  interface ! {{{
    subroutine restart_write(&
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

!      global parameters
      integer :: ihistory,snapout,maxmpsi
      integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
           istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
           mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
           ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile
      real(kind=wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
           flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
           zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
      integer,dimension(:),allocatable :: mtheta
      real(kind=wp),dimension(:),allocatable :: deltat
      logical  do_collision
!     particle array
      integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
      integer,dimension(:,:),allocatable :: jtion0,jtion1
      real(kind=wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
           wtelectron0,wtelectron1
      real(kind=wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
      real(kind=wp),dimension(:,:),allocatable :: zion,zion0,zelectron,&
            zelectron0,zelectron1
!     field array
      integer :: mmpsi
      integer,dimension(:),allocatable :: itran,igrid
      integer,dimension(:,:,:),allocatable :: jtp1,jtp2
      real(kind=wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
           pmarki,pmarke,zonali,zonale,gradt
      real(kind=wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
           markere,pgyro,tgyro,dtemper,heatflux,phit
      real(kind=wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
      real(kind=wp) :: Total_field_energy(3)

!     diagnosis array
      integer :: mflux,num_mode,m_poloidal
      integer nmode(num_mode),mmode(num_mode)
      real(kind=wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
           entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
           rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
           amp_mode(2,num_mode,2)
      real(kind=wp),dimension(:),allocatable :: hfluxpsi
      real(kind=wp),dimension(:,:,:),allocatable :: eigenmode
      real(kind=wp) etracer,ptracer(4)
    end subroutine restart_write 
    subroutine tag_particles(& 
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
! particle_tracking
          track_particles,nptrack,isnap,&
                                           ptracked,&
                                      ntrackp)
      use precision
      implicit none
      integer :: m,np
!      global parameters
        integer :: ihistory,snapout,maxmpsi
        integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
           istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
           mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
           ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile
        real(kind=wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
           flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
           zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
        integer,dimension(:),allocatable :: mtheta
        real(kind=wp),dimension(:),allocatable :: deltat
        logical  do_collision
!     particle array
      integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
      integer,dimension(:,:),allocatable :: jtion0,jtion1
      real(kind=wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
           wtelectron0,wtelectron1
      real(kind=wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
      real(kind=wp),dimension(:,:),allocatable :: zion,zion0,zelectron,&
            zelectron0,zelectron1
!     particle_tracking
      integer track_particles,nptrack,isnap
      real(kind=wp),dimension(:,:,:),allocatable :: ptracked
      integer,dimension(:),allocatable :: ntrackp
    end subroutine tag_particles
    subroutine set_random_zion(mi,rng_control,zion,state)
      use precision
      use rng
      implicit none
      integer mi,rng_control
      type(rng_state) :: state
      real(kind=wp),dimension(:,:),allocatable :: zion
    end subroutine set_random_zion
    subroutine rand_num_gen_init(rng_control,mype,irun,stdout,state)
      use precision
      use rng
      implicit none
      integer mype,rng_control,irun,stdout
      type(rng_state) :: state
    end subroutine rand_num_gen_init
  end interface ! }}}

!  global parameters
   integer :: ihistory,snapout,maxmpsi
   integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
        istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
        mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
        ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile
   real(kind=wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
        flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
        zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
   integer,dimension(:),allocatable :: mtheta
   real(kind=wp),dimension(:),allocatable :: deltat
   logical  do_collision

! particle array
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(kind=wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(kind=wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(kind=wp),dimension(:,:),allocatable :: zion,zion0,zelectron,&
        zelectron0,zelectron1

! field array
  integer :: mmpsi
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(kind=wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(kind=wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(kind=wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(kind=wp) :: Total_field_energy(3)

! particle_tracking
  integer track_particles,nptrack,isnap
  real(kind=wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp

! particle decomp
  integer  :: ntoroidal,npartdom
  integer  :: partd_comm,nproc_partd,myrank_partd
  integer  :: toroidal_comm,nproc_toroidal,myrank_toroidal
  integer  :: left_pe,right_pe
  integer  :: toroidal_domain_location,particle_domain_location

! diagnosis array
   integer :: mflux,num_mode,m_poloidal
   integer nmode(num_mode),mmode(num_mode)
   real(kind=wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
        entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
        rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
        amp_mode(2,num_mode,2)
   real(kind=wp),dimension(:),allocatable :: hfluxpsi
   real(kind=wp),dimension(:,:,:),allocatable :: eigenmode
   real(kind=wp) etracer,ptracer(4)

! local variables
  type(rng_state) :: state
  integer i,m,ierr
  real(kind=wp) :: c0=2.515517,c1=0.802853,c2=0.010328,&
              d1=1.432788,d2=0.189269,d3=0.001308
  real(kind=wp) :: w_initial,energy,momentum,r,rmi,pi2_inv,delr,ainv,vthe,tsqrt,&
              vthi,cost,b,upara,eperp,cmratio,z4tmp

! Initialize random number generator
  call rand_num_gen_init(rng_control,mype,irun,stdout,state)

! restart from previous runs
  if(irun /= 0)then
     call restart_read(&
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
     return
  endif

  !!rmi=1.0/real(mi)
  rmi=1.0/real(mi*npartdom)
  pi2_inv=0.5/pi
  delr=1.0/deltar
  ainv=1.0/a
  w_initial=1.0e-3
  if(nonlinear<0.5)w_initial=1.0e-12
  if(track_particles < -10 .and. track_particles > -20)w_initial=0.1_wp*real(-10-track_particles,wp)
  if(mype==0)write(0,*)'w_initial =',w_initial
  ntracer=0
  !if(mype==0)ntracer=mi
  if(mype==0)ntracer=1
      
! radial: uniformly distributed in r^2, later transform to psi
  do m=1,mi
     !!zion(1,m)=sqrt(a0*a0+(real(m)-0.5)*(a1*a1-a0*a0)*rmi)
     zion(1,m)=sqrt(a0*a0+(real(m+myrank_partd*mi)-0.5)*(a1*a1-a0*a0)*rmi)
  enddo

! If particle tracking is "on", tag each particle with a unique number
  if(track_particles == 1)call tag_particles(&
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
! particle_tracking
          track_particles,nptrack,isnap,&
                                           ptracked,&
                                      ntrackp)

! Set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
  call set_random_zion(mi,rng_control,zion,state)

! poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
  do m=1,mi                
     zion(2,m)=2.0*pi*(zion(2,m)-0.5)
     zion0(2,m)=zion(2,m) !zion0(2,:) for temporary storage
  enddo
  do i=1,10
     do m=1,mi
        zion(2,m)=zion0(2,m)-2.0*zion(1,m)*sin(zion(2,m))
     enddo
  enddo
  do m=1,mi                
     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
  enddo

! Maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
  do m=1,mi                
     z4tmp=zion(4,m)
     zion(4,m)=zion(4,m)-0.5
     !zion0(4,m)=sign(1.0d+00,zion(4,m))
     zion0(4,m)=sign(ONE,zion(4,m))
     !zion(4,m)=sqrt(max(1.0d-20,log(1.0/max(1.0d-20,zion(4,m)**2))))
     zion(4,m)=sqrt(max(SMALL,log(1.0/max(SMALL,zion(4,m)**2))))
     zion(4,m)=zion(4,m)-(c0+c1*zion(4,m)+c2*zion(4,m)**2)/&
          (1.0+d1*zion(4,m)+d2*zion(4,m)**2+d3*zion(4,m)**3)
     if(zion(4,m)>umax)zion(4,m)=z4tmp
  enddo

  do m=1,mi

! toroidal:  uniformly distributed in zeta
     zion(3,m)=zetamin+(zetamax-zetamin)*zion(3,m)

     zion(4,m)=zion0(4,m)*min(umax,zion(4,m))

! initial random weight
     zion(5,m)=2.0*w_initial*(zion(5,m)-0.5)*(1.0+cos(zion(2,m)))

! Maxwellian distribution in v_perp, <v_perp^2>=1.0
     !zion(6,m)=max(1.0d-20,min(umax*umax,-log(max(1.0d-20,zion(6,m)))))
     zion(6,m)=max(SMALL,min(umax*umax,-log(max(SMALL,zion(6,m)))))
  enddo

! transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
  vthi=gyroradius*abs(qion)/aion
  do m=1,mi
     zion0(1,m)=1.0/(1.0+zion(1,m)*cos(zion(2,m))) !B-field
     zion(1,m)=0.5*zion(1,m)*zion(1,m)
     zion(4,m)=vthi*zion(4,m)*aion/(qion*zion0(1,m))
     zion(6,m)=sqrt(aion*vthi*vthi*zion(6,m)/zion0(1,m))
  enddo

! tracer particle initially resides on PE=0
  if(mype == 0)then
     zion(1,ntracer)=0.5*(0.5*(a0+a1))**2
     zion(2,ntracer)=0.0
     zion(3,ntracer)=0.5*(zetamin+zetamax)
     zion(4,ntracer)=0.5*vthi*aion/qion
     zion(5,ntracer)=0.0
     zion(6,ntracer)=sqrt(aion*vthi*vthi)
!     zion0(1:5,ntracer)=zion(1:5,ntracer)
  endif

  if(iload==0)then
! uniform loading
     do m=1,mi
        zion0(6,m)=1.0
     enddo
  else
! true nonuniform for temperature profile, density profile by weight
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        i=max(0,min(mpsi,int((r-a0)*delr+0.5)))
        zion(4,m)=zion(4,m)*sqrt(rtemi(i))
        zion(6,m)=zion(6,m)*sqrt(rtemi(i))
!        zion0(6,m)=max(0.1d+00,min(10.0d+00,rden(i)))
        zion0(6,m)=max(PT_ONE,min(TEN,rden(i)))
     enddo
  endif
  
! load electron on top of ion if mi=me
  if(nhybrid>0)then
     vthe=sqrt(aion/(aelectron*tite))*(qion/aion)*(aelectron/qelectron)
     tsqrt=sqrt(1.0/tite)

! keep trapped electrons
     cmratio=qion/aion
     me=0
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        cost=cos(zion(2,m))
        b=1.0/(1.0+r*cost)
        upara=zion(4,m)*b*cmratio
        energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
        eperp=zion(6,m)*zion(6,m)/(1.0-r)
        if(eperp>energy)then
           me=me+1
           zelectron(1,me)=zion(1,m)
           zelectron(2,me)=zion(2,m)
           zelectron(3,me)=zion(3,m)
           zelectron(4,me)=zion(4,m)*vthe
           zelectron(5,me)=zion(5,m)
           zelectron(6,me)=zion(6,m)*tsqrt
           zelectron0(6,me)=zion0(6,m)
        endif
     enddo
     if(mype == 0)then
        ntracer=me
!        zelectron0(1:5,ntracer)=zelectron(1:5,ntracer)
     endif
  endif

!  write(mype+40,'(6e15.6)')zion(1:6,1:mi)
!  do m=1,mi
!     write(mype+40,*)zion(:,m)
!  enddo
!  close(mype+40)

end subroutine load


