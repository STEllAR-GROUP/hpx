module precision_0
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
#ifdef double_precision_0
  integer, parameter :: wp=doubleprec

  real(wp), parameter :: one=1.0d+00
  real(wp), parameter :: ten=10.0d+00
  real(wp), parameter :: zero=0.0d+00
  real(wp), parameter :: pt_one=0.1d+00
  real(wp), parameter :: one_ten_thousandth=1.0d-04
  real(wp), parameter :: small=1.0d-20
  real(wp), parameter :: big=1.0d+20

#else
  integer, parameter :: wp=singleprec

  real(wp), parameter :: one=1.0e+00
  real(wp), parameter :: ten=10.0e+00
  real(wp), parameter :: zero=0.0e+00
  real(wp), parameter :: pt_one=0.1e+00
  real(wp), parameter :: one_ten_thousandth=1.0e-04
  real(wp), parameter :: small=1.0e-20
  real(wp), parameter :: big=1.0e+20

#endif





end module precision_0

module global_parameters_0
  use precision_0
  integer,parameter :: ihistory=22,snapout=33,maxmpsi=192
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload_0,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

#ifdef _sx
! sx-6 trick to minimize bank conflict in chargei_0
  integer,dimension(0:maxmpsi) :: mmtheta
!cdir duplicate(mmtheta,1024)
#endif

end module global_parameters_0

module particle_array_0
  use precision_0
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,zelectron0,zelectron1
end module particle_array_0

module particle_tracking_0
  use precision_0
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp
end module particle_tracking_0

module field_array_0
  use precision_0
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: total_field_energy(3)

#ifdef _sx
! sx-6 trick to minimize bank conflicts in chargei_0
!cdir duplicate(iigrid,1024)
  integer,dimension(0:mmpsi) :: iigrid
!cdir duplicate(qqtinv,1024)
  real(wp),dimension(0:mmpsi) :: qqtinv
#endif

end module field_array_0

module diagnosis_array_0
  use precision_0
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)
end module diagnosis_array_0

module precision_1
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
#ifdef double_precision_1
  integer, parameter :: wp=doubleprec

  real(wp), parameter :: one=1.0d+00
  real(wp), parameter :: ten=10.0d+00
  real(wp), parameter :: zero=0.0d+00
  real(wp), parameter :: pt_one=0.1d+00
  real(wp), parameter :: one_ten_thousandth=1.0d-04
  real(wp), parameter :: small=1.0d-20
  real(wp), parameter :: big=1.0d+20

#else
  integer, parameter :: wp=singleprec

  real(wp), parameter :: one=1.0e+00
  real(wp), parameter :: ten=10.0e+00
  real(wp), parameter :: zero=0.0e+00
  real(wp), parameter :: pt_one=0.1e+00
  real(wp), parameter :: one_ten_thousandth=1.0e-04
  real(wp), parameter :: small=1.0e-20
  real(wp), parameter :: big=1.0e+20

#endif





end module precision_1

module global_parameters_1
  use precision_1
  integer,parameter :: ihistory=22,snapout=33,maxmpsi=192
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload_1,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

#ifdef _sx
! sx-6 trick to minimize bank conflict in chargei_1
  integer,dimension(0:maxmpsi) :: mmtheta
!cdir duplicate(mmtheta,1024)
#endif

end module global_parameters_1

module particle_array_1
  use precision_1
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,zelectron0,zelectron1
end module particle_array_1

module particle_tracking_1
  use precision_1
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp
end module particle_tracking_1

module field_array_1
  use precision_1
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: total_field_energy(3)

#ifdef _sx
! sx-6 trick to minimize bank conflicts in chargei_1
!cdir duplicate(iigrid,1024)
  integer,dimension(0:mmpsi) :: iigrid
!cdir duplicate(qqtinv,1024)
  real(wp),dimension(0:mmpsi) :: qqtinv
#endif

end module field_array_1

module diagnosis_array_1
  use precision_1
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)
end module diagnosis_array_1

module precision_2
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
#ifdef double_precision_2
  integer, parameter :: wp=doubleprec

  real(wp), parameter :: one=1.0d+00
  real(wp), parameter :: ten=10.0d+00
  real(wp), parameter :: zero=0.0d+00
  real(wp), parameter :: pt_one=0.1d+00
  real(wp), parameter :: one_ten_thousandth=1.0d-04
  real(wp), parameter :: small=1.0d-20
  real(wp), parameter :: big=1.0d+20

#else
  integer, parameter :: wp=singleprec

  real(wp), parameter :: one=1.0e+00
  real(wp), parameter :: ten=10.0e+00
  real(wp), parameter :: zero=0.0e+00
  real(wp), parameter :: pt_one=0.1e+00
  real(wp), parameter :: one_ten_thousandth=1.0e-04
  real(wp), parameter :: small=1.0e-20
  real(wp), parameter :: big=1.0e+20

#endif





end module precision_2

module global_parameters_2
  use precision_2
  integer,parameter :: ihistory=22,snapout=33,maxmpsi=192
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload_2,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

#ifdef _sx
! sx-6 trick to minimize bank conflict in chargei_2
  integer,dimension(0:maxmpsi) :: mmtheta
!cdir duplicate(mmtheta,1024)
#endif

end module global_parameters_2

module particle_array_2
  use precision_2
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,zelectron0,zelectron1
end module particle_array_2

module particle_tracking_2
  use precision_2
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp
end module particle_tracking_2

module field_array_2
  use precision_2
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: total_field_energy(3)

#ifdef _sx
! sx-6 trick to minimize bank conflicts in chargei_2
!cdir duplicate(iigrid,1024)
  integer,dimension(0:mmpsi) :: iigrid
!cdir duplicate(qqtinv,1024)
  real(wp),dimension(0:mmpsi) :: qqtinv
#endif

end module field_array_2

module diagnosis_array_2
  use precision_2
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)
end module diagnosis_array_2

module precision_3
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
#ifdef double_precision_3
  integer, parameter :: wp=doubleprec

  real(wp), parameter :: one=1.0d+00
  real(wp), parameter :: ten=10.0d+00
  real(wp), parameter :: zero=0.0d+00
  real(wp), parameter :: pt_one=0.1d+00
  real(wp), parameter :: one_ten_thousandth=1.0d-04
  real(wp), parameter :: small=1.0d-20
  real(wp), parameter :: big=1.0d+20

#else
  integer, parameter :: wp=singleprec

  real(wp), parameter :: one=1.0e+00
  real(wp), parameter :: ten=10.0e+00
  real(wp), parameter :: zero=0.0e+00
  real(wp), parameter :: pt_one=0.1e+00
  real(wp), parameter :: one_ten_thousandth=1.0e-04
  real(wp), parameter :: small=1.0e-20
  real(wp), parameter :: big=1.0e+20

#endif





end module precision_3

module global_parameters_3
  use precision_3
  integer,parameter :: ihistory=22,snapout=33,maxmpsi=192
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload_3,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

#ifdef _sx
! sx-6 trick to minimize bank conflict in chargei_3
  integer,dimension(0:maxmpsi) :: mmtheta
!cdir duplicate(mmtheta,1024)
#endif

end module global_parameters_3

module particle_array_3
  use precision_3
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,zelectron0,zelectron1
end module particle_array_3

module particle_tracking_3
  use precision_3
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp
end module particle_tracking_3

module field_array_3
  use precision_3
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: total_field_energy(3)

#ifdef _sx
! sx-6 trick to minimize bank conflicts in chargei_3
!cdir duplicate(iigrid,1024)
  integer,dimension(0:mmpsi) :: iigrid
!cdir duplicate(qqtinv,1024)
  real(wp),dimension(0:mmpsi) :: qqtinv
#endif

end module field_array_3

module diagnosis_array_3
  use precision_3
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)
end module diagnosis_array_3

module precision_4
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
#ifdef double_precision_4
  integer, parameter :: wp=doubleprec

  real(wp), parameter :: one=1.0d+00
  real(wp), parameter :: ten=10.0d+00
  real(wp), parameter :: zero=0.0d+00
  real(wp), parameter :: pt_one=0.1d+00
  real(wp), parameter :: one_ten_thousandth=1.0d-04
  real(wp), parameter :: small=1.0d-20
  real(wp), parameter :: big=1.0d+20

#else
  integer, parameter :: wp=singleprec

  real(wp), parameter :: one=1.0e+00
  real(wp), parameter :: ten=10.0e+00
  real(wp), parameter :: zero=0.0e+00
  real(wp), parameter :: pt_one=0.1e+00
  real(wp), parameter :: one_ten_thousandth=1.0e-04
  real(wp), parameter :: small=1.0e-20
  real(wp), parameter :: big=1.0e+20

#endif





end module precision_4

module global_parameters_4
  use precision_4
  integer,parameter :: ihistory=22,snapout=33,maxmpsi=192
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload_4,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

#ifdef _sx
! sx-6 trick to minimize bank conflict in chargei_4
  integer,dimension(0:maxmpsi) :: mmtheta
!cdir duplicate(mmtheta,1024)
#endif

end module global_parameters_4

module particle_array_4
  use precision_4
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,zelectron0,zelectron1
end module particle_array_4

module particle_tracking_4
  use precision_4
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp
end module particle_tracking_4

module field_array_4
  use precision_4
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: total_field_energy(3)

#ifdef _sx
! sx-6 trick to minimize bank conflicts in chargei_4
!cdir duplicate(iigrid,1024)
  integer,dimension(0:mmpsi) :: iigrid
!cdir duplicate(qqtinv,1024)
  real(wp),dimension(0:mmpsi) :: qqtinv
#endif

end module field_array_4

module diagnosis_array_4
  use precision_4
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)
end module diagnosis_array_4

module precision_5
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
#ifdef double_precision_5
  integer, parameter :: wp=doubleprec

  real(wp), parameter :: one=1.0d+00
  real(wp), parameter :: ten=10.0d+00
  real(wp), parameter :: zero=0.0d+00
  real(wp), parameter :: pt_one=0.1d+00
  real(wp), parameter :: one_ten_thousandth=1.0d-04
  real(wp), parameter :: small=1.0d-20
  real(wp), parameter :: big=1.0d+20

#else
  integer, parameter :: wp=singleprec

  real(wp), parameter :: one=1.0e+00
  real(wp), parameter :: ten=10.0e+00
  real(wp), parameter :: zero=0.0e+00
  real(wp), parameter :: pt_one=0.1e+00
  real(wp), parameter :: one_ten_thousandth=1.0e-04
  real(wp), parameter :: small=1.0e-20
  real(wp), parameter :: big=1.0e+20

#endif





end module precision_5

module global_parameters_5
  use precision_5
  integer,parameter :: ihistory=22,snapout=33,maxmpsi=192
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload_5,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

#ifdef _sx
! sx-6 trick to minimize bank conflict in chargei_5
  integer,dimension(0:maxmpsi) :: mmtheta
!cdir duplicate(mmtheta,1024)
#endif

end module global_parameters_5

module particle_array_5
  use precision_5
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,zelectron0,zelectron1
end module particle_array_5

module particle_tracking_5
  use precision_5
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp
end module particle_tracking_5

module field_array_5
  use precision_5
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: total_field_energy(3)

#ifdef _sx
! sx-6 trick to minimize bank conflicts in chargei_5
!cdir duplicate(iigrid,1024)
  integer,dimension(0:mmpsi) :: iigrid
!cdir duplicate(qqtinv,1024)
  real(wp),dimension(0:mmpsi) :: qqtinv
#endif

end module field_array_5

module diagnosis_array_5
  use precision_5
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)
end module diagnosis_array_5

module precision_6
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
#ifdef double_precision_6
  integer, parameter :: wp=doubleprec

  real(wp), parameter :: one=1.0d+00
  real(wp), parameter :: ten=10.0d+00
  real(wp), parameter :: zero=0.0d+00
  real(wp), parameter :: pt_one=0.1d+00
  real(wp), parameter :: one_ten_thousandth=1.0d-04
  real(wp), parameter :: small=1.0d-20
  real(wp), parameter :: big=1.0d+20

#else
  integer, parameter :: wp=singleprec

  real(wp), parameter :: one=1.0e+00
  real(wp), parameter :: ten=10.0e+00
  real(wp), parameter :: zero=0.0e+00
  real(wp), parameter :: pt_one=0.1e+00
  real(wp), parameter :: one_ten_thousandth=1.0e-04
  real(wp), parameter :: small=1.0e-20
  real(wp), parameter :: big=1.0e+20

#endif





end module precision_6

module global_parameters_6
  use precision_6
  integer,parameter :: ihistory=22,snapout=33,maxmpsi=192
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload_6,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

#ifdef _sx
! sx-6 trick to minimize bank conflict in chargei_6
  integer,dimension(0:maxmpsi) :: mmtheta
!cdir duplicate(mmtheta,1024)
#endif

end module global_parameters_6

module particle_array_6
  use precision_6
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,zelectron0,zelectron1
end module particle_array_6

module particle_tracking_6
  use precision_6
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp
end module particle_tracking_6

module field_array_6
  use precision_6
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: total_field_energy(3)

#ifdef _sx
! sx-6 trick to minimize bank conflicts in chargei_6
!cdir duplicate(iigrid,1024)
  integer,dimension(0:mmpsi) :: iigrid
!cdir duplicate(qqtinv,1024)
  real(wp),dimension(0:mmpsi) :: qqtinv
#endif

end module field_array_6

module diagnosis_array_6
  use precision_6
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)
end module diagnosis_array_6

module precision_7
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
#ifdef double_precision_7
  integer, parameter :: wp=doubleprec

  real(wp), parameter :: one=1.0d+00
  real(wp), parameter :: ten=10.0d+00
  real(wp), parameter :: zero=0.0d+00
  real(wp), parameter :: pt_one=0.1d+00
  real(wp), parameter :: one_ten_thousandth=1.0d-04
  real(wp), parameter :: small=1.0d-20
  real(wp), parameter :: big=1.0d+20

#else
  integer, parameter :: wp=singleprec

  real(wp), parameter :: one=1.0e+00
  real(wp), parameter :: ten=10.0e+00
  real(wp), parameter :: zero=0.0e+00
  real(wp), parameter :: pt_one=0.1e+00
  real(wp), parameter :: one_ten_thousandth=1.0e-04
  real(wp), parameter :: small=1.0e-20
  real(wp), parameter :: big=1.0e+20

#endif





end module precision_7

module global_parameters_7
  use precision_7
  integer,parameter :: ihistory=22,snapout=33,maxmpsi=192
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload_7,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

#ifdef _sx
! sx-6 trick to minimize bank conflict in chargei_7
  integer,dimension(0:maxmpsi) :: mmtheta
!cdir duplicate(mmtheta,1024)
#endif

end module global_parameters_7

module particle_array_7
  use precision_7
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,zelectron0,zelectron1
end module particle_array_7

module particle_tracking_7
  use precision_7
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp
end module particle_tracking_7

module field_array_7
  use precision_7
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: total_field_energy(3)

#ifdef _sx
! sx-6 trick to minimize bank conflicts in chargei_7
!cdir duplicate(iigrid,1024)
  integer,dimension(0:mmpsi) :: iigrid
!cdir duplicate(qqtinv,1024)
  real(wp),dimension(0:mmpsi) :: qqtinv
#endif

end module field_array_7

module diagnosis_array_7
  use precision_7
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)
end module diagnosis_array_7

module precision_8
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
#ifdef double_precision_8
  integer, parameter :: wp=doubleprec

  real(wp), parameter :: one=1.0d+00
  real(wp), parameter :: ten=10.0d+00
  real(wp), parameter :: zero=0.0d+00
  real(wp), parameter :: pt_one=0.1d+00
  real(wp), parameter :: one_ten_thousandth=1.0d-04
  real(wp), parameter :: small=1.0d-20
  real(wp), parameter :: big=1.0d+20

#else
  integer, parameter :: wp=singleprec

  real(wp), parameter :: one=1.0e+00
  real(wp), parameter :: ten=10.0e+00
  real(wp), parameter :: zero=0.0e+00
  real(wp), parameter :: pt_one=0.1e+00
  real(wp), parameter :: one_ten_thousandth=1.0e-04
  real(wp), parameter :: small=1.0e-20
  real(wp), parameter :: big=1.0e+20

#endif





end module precision_8

module global_parameters_8
  use precision_8
  integer,parameter :: ihistory=22,snapout=33,maxmpsi=192
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload_8,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

#ifdef _sx
! sx-6 trick to minimize bank conflict in chargei_8
  integer,dimension(0:maxmpsi) :: mmtheta
!cdir duplicate(mmtheta,1024)
#endif

end module global_parameters_8

module particle_array_8
  use precision_8
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,zelectron0,zelectron1
end module particle_array_8

module particle_tracking_8
  use precision_8
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp
end module particle_tracking_8

module field_array_8
  use precision_8
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: total_field_energy(3)

#ifdef _sx
! sx-6 trick to minimize bank conflicts in chargei_8
!cdir duplicate(iigrid,1024)
  integer,dimension(0:mmpsi) :: iigrid
!cdir duplicate(qqtinv,1024)
  real(wp),dimension(0:mmpsi) :: qqtinv
#endif

end module field_array_8

module diagnosis_array_8
  use precision_8
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)
end module diagnosis_array_8

module precision_9
  integer, parameter :: doubleprec=selected_real_kind(12),&
       singleprec=selected_real_kind(6),&
       defaultprec=kind(0.0)
#ifdef double_precision_9
  integer, parameter :: wp=doubleprec

  real(wp), parameter :: one=1.0d+00
  real(wp), parameter :: ten=10.0d+00
  real(wp), parameter :: zero=0.0d+00
  real(wp), parameter :: pt_one=0.1d+00
  real(wp), parameter :: one_ten_thousandth=1.0d-04
  real(wp), parameter :: small=1.0d-20
  real(wp), parameter :: big=1.0d+20

#else
  integer, parameter :: wp=singleprec

  real(wp), parameter :: one=1.0e+00
  real(wp), parameter :: ten=10.0e+00
  real(wp), parameter :: zero=0.0e+00
  real(wp), parameter :: pt_one=0.1e+00
  real(wp), parameter :: one_ten_thousandth=1.0e-04
  real(wp), parameter :: small=1.0e-20
  real(wp), parameter :: big=1.0e+20

#endif





end module precision_9

module global_parameters_9
  use precision_9
  integer,parameter :: ihistory=22,snapout=33,maxmpsi=192
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload_9,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_tprofile
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

#ifdef _sx
! sx-6 trick to minimize bank conflict in chargei_9
  integer,dimension(0:maxmpsi) :: mmtheta
!cdir duplicate(mmtheta,1024)
#endif

end module global_parameters_9

module particle_array_9
  use precision_9
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,zelectron0,zelectron1
end module particle_array_9

module particle_tracking_9
  use precision_9
  integer track_particles,nptrack,isnap
  real(wp),dimension(:,:,:),allocatable :: ptracked
  integer,dimension(:),allocatable :: ntrackp
end module particle_tracking_9

module field_array_9
  use precision_9
  integer,parameter :: mmpsi=192
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(wp) :: total_field_energy(3)

#ifdef _sx
! sx-6 trick to minimize bank conflicts in chargei_9
!cdir duplicate(iigrid,1024)
  integer,dimension(0:mmpsi) :: iigrid
!cdir duplicate(qqtinv,1024)
  real(wp),dimension(0:mmpsi) :: qqtinv
#endif

end module field_array_9

module diagnosis_array_9
  use precision_9
  integer,parameter :: mflux=5,num_mode=8,m_poloidal=9
  integer nmode(num_mode),mmode(num_mode)
  real(wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(wp),dimension(:),allocatable :: hfluxpsi
  real(wp),dimension(:,:,:),allocatable :: eigenmode
  real(wp) etracer,ptracer(4)
end module diagnosis_array_9

