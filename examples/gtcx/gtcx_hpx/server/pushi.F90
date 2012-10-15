subroutine pushi(hpx4_bti,&
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
! particle array
                                      kzion,kzelectron,jtelectron0,jtelectron1,&
                                        jtion0,jtion1,&
                                       wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1, &
                                        wpion,wtion0,wtion1,&
                                         zion,zion0,zelectron,&
        zelectron0,zelectron1, &
! particle decomp
              ntoroidal,npartdom,&
              partd_comm,nproc_partd,myrank_partd,&
              toroidal_comm,nproc_toroidal,myrank_toroidal,&
              left_pe,right_pe,&
              toroidal_domain_location,particle_domain_location,&
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
  !use global_parameters
  !use particle_array
  !use particle_decomp
  !use field_array
  !use diagnosis_array
  use, intrinsic :: iso_c_binding, only : c_ptr
  use precision
  implicit none

  TYPE(C_PTR), INTENT(IN), VALUE :: hpx4_bti
  integer,dimension(:),allocatable :: t_gids
  integer,dimension(:),allocatable :: p_gids
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

! particle decomp
  integer  :: ntoroidal,npartdom
  integer  :: partd_comm,nproc_partd,myrank_partd
  integer  :: toroidal_comm,nproc_toroidal,myrank_toroidal
  integer  :: left_pe,right_pe
  integer  :: toroidal_domain_location,particle_domain_location

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

  integer i,im,ip,j,jt,k,kz,kk,m,ii,j11,j10,j01,j00,jtheta,num,ierror,&
       larmor,ij,ipjt
  real(kind=wp) cost0,sint0,cost,sint,rdum,tdum,dtime,q,r,b,g,gp,ri,rip,dbdp,dbdt,&
       dedb,deni,upara,energy,kappa,dptdp,dptdt,dptdz,vdr,pdot,tdot,zdot,rdot,&
       wdot,wdrift,vdrenergy,delr,pi2,pi2_inv,uleft,uright,adum(0:mpsi),&
       dmark(mflux),pstat,xdum,ydum,xdot,ydot,tem_inv,wz1,wz0,wp1,wp0,wt11,&
       wt10,wt01,wt00,temp(0:mpsi),dtemp(0:mpsi),denp(0:mpsi),g2,tem,rfac,&
       sbound,qdum,tflr,d_inv,epara,wpara,psimax,psimin,paxis,pdum,wdrive,&
       ainv,rinv,qinv,psitmp,thetatmp,zetatmp,rhoi,e1,e2,e3,cmratio,cinv,&
       wpi(3,mi),rdel,vthi,dtem(mflux),dden(mflux),ddentmp(mflux),&
       dtemtmp(mflux),vdrtmp(mflux),dmarktmp(mflux),hfluxpsitmp(0:mpsi)
  real(kind=wp) Epsi,Etheta,Ezeta,En_field(3),zdum,drdp,grad_zeta,jacobian,dphi_square(3)
  real(kind=wp) sum_of_weights,total_sum_of_weights,residual_weight
  integer  mi_total,conserve_particles

  limit_vpara=0   ! limit_vpara=1 :  parallel velocity kept <= abs(umax)
  conserve_particles=0
  delr=1.0/deltar
  pi2=2.0*pi
  !pi2_inv=0.5/pi
  sbound=1.0
  if(nbound==0)sbound=0.0
  psimax=0.5*a1*a1
  psimin=0.5*a0*a0
  paxis=0.5*(8.0*gyroradius)**2
  cmratio=qion/aion
  cinv=1.0/qion
  vthi=gyroradius*abs(qion)/aion
  tem_inv=1.0/(aion*vthi*vthi)
  d_inv=real(mflux)/(a1-a0)
  uright=umax*vthi
  uleft=-uright

  if(irk==1)then
! 1st step of Runge-Kutta method
     dtime=0.5*tstep
     do m=1,mi
        zion0(1:5,m)=zion(1:5,m)
     enddo

     vdrtmp=0.0

! 2nd step of Runge-Kutta method
  else
     dtime=tstep

     if(nonlinear<0.5)vdrtmp=0.0
     if(nonlinear>0.5)vdrtmp=pfluxpsi
  endif

! gather e_field using 4-point gyro-averaging, sorting in poloidal angle
! The following line is an OpenMP directive for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
  do m=1,mi
     e1=0.0
     e2=0.0
     e3=0.0
     kk=kzion(m)
     wz1=wzion(m)
     wz0=1.0-wz1

     do larmor=1,4

        ij=jtion0(larmor,m)
        wp0=1.0-wpion(larmor,m)
        wt00=1.0-wtion0(larmor,m)
        e1=e1+wp0*wt00*(wz0*evector(1,kk,ij)+wz1*evector(1,kk+1,ij))
        e2=e2+wp0*wt00*(wz0*evector(2,kk,ij)+wz1*evector(2,kk+1,ij))
        e3=e3+wp0*wt00*(wz0*evector(3,kk,ij)+wz1*evector(3,kk+1,ij))
        
        ij=ij+1
        wt10=1.0-wt00
        e1=e1+wp0*wt10*(wz0*evector(1,kk,ij)+wz1*evector(1,kk+1,ij))
        e2=e2+wp0*wt10*(wz0*evector(2,kk,ij)+wz1*evector(2,kk+1,ij))
        e3=e3+wp0*wt10*(wz0*evector(3,kk,ij)+wz1*evector(3,kk+1,ij))
        
        ij=jtion1(larmor,m)
        wp1=1.0-wp0
        wt01=1.0-wtion1(larmor,m)
        e1=e1+wp1*wt01*(wz0*evector(1,kk,ij)+wz1*evector(1,kk+1,ij))
        e2=e2+wp1*wt01*(wz0*evector(2,kk,ij)+wz1*evector(2,kk+1,ij))
        e3=e3+wp1*wt01*(wz0*evector(3,kk,ij)+wz1*evector(3,kk+1,ij))
        
        ij=ij+1
        wt11=1.0-wt01
        e1=e1+wp1*wt11*(wz0*evector(1,kk,ij)+wz1*evector(1,kk+1,ij))
        e2=e2+wp1*wt11*(wz0*evector(2,kk,ij)+wz1*evector(2,kk+1,ij))
        e3=e3+wp1*wt11*(wz0*evector(3,kk,ij)+wz1*evector(3,kk+1,ij))

     enddo

     wpi(1,m)=0.25*e1
     wpi(2,m)=0.25*e2
     wpi(3,m)=0.25*e3

  enddo

! primary ion marker temperature and parallel flow velocity
  temp=1.0
  dtemp=0.0
  temp=1.0/(temp*rtemi*aion*vthi*vthi) !inverse local temperature
  ainv=1.0/a

!********** test gradual kappan **********
!     kappati=min(6.9_wp,(kappan+(6.9_wp-kappan)*real(istep,wp)/4000._wp))
!     if(mype==0.and.irk==2)write(36,*)istep,kappan,kappati
! ****************************************

! update GC position
! Another OpenMP parallel loop...
  do m=1,mi
     r=sqrt(2.0_wp*zion(1,m))
     rinv=1.0_wp/r
     ii=max(0,min(mpsi-1,int((r-a0)*delr)))
     ip=max(1,min(mflux,1+int((r-a0)*d_inv)))
     wp0=real(ii+1)-(r-a0)*delr
     wp1=1.0-wp0
     tem=wp0*temp(ii)+wp1*temp(ii+1)
     q=q0+q1*r*ainv+q2*r*r*ainv*ainv
     qinv=1.0/q
     cost=cos(zion(2,m))
     sint=sin(zion(2,m))
!     cost0=cos(zion(2,m)+r*sint)
!     sint0=sin(zion(2,m)+r*sint)
     b=1.0/(1.0+r*cost)
     g=1.0
     gp=0.0
!     ri=r*r*qinv
!     rip=(2.0*q0+q1*r*ainv)*qinv*qinv
     ri=0.0
     rip=0.0
     dbdp=-b*b*cost*rinv
     dbdt=b*b*r*sint
     dedb=cinv*(zion(4,m)*zion(4,m)*qion*b*cmratio+zion(6,m)*zion(6,m))
     deni=1.0/(g*q + ri + zion(4,m)*(g*rip-ri*gp))
     upara=zion(4,m)*b*cmratio
     energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
     rfac=rw*(r-rc)
     rfac=rfac*rfac
     rfac=rfac*rfac*rfac
     rfac=exp(-rfac)
     kappa=1.0-sbound+sbound*rfac
!     kappa=((min(umax*umax,energy*tem)-1.5)*kappati+kappan)*kappa*rinv
     kappa=((energy*tem-1.5)*kappati+kappan)*kappa*rinv

! perturbed quantities
     dptdp=wpi(1,m)
     dptdt=wpi(2,m)
     dptdz=wpi(3,m)-wpi(2,m)*qinv
     epara=-wpi(3,m)*b*q*deni

! subtract net particle flow
     dptdt=dptdt+vdrtmp(ip)

! ExB drift in radial direction for w-dot and flux diagnostics
     vdr=q*(ri*dptdz-g*dptdt)*deni
     wdrive=vdr*kappa
     wpara=epara*(upara-dtemp(ii))*qion*tem
     wdrift=q*(g*dbdt*dptdp-g*dbdp*dptdt+ri*dbdp*dptdz)*deni*dedb*qion*tem
     wdot=(zion0(6,m)-paranl*zion(5,m))*(wdrive+wpara+wdrift)

! self-consistent and external electric field for marker orbits
     dptdp=dptdp*nonlinear+gyroradius*(flow0+flow1*r*ainv+flow2*r*r*ainv*ainv)
     dptdt=dptdt*nonlinear
     dptdz=dptdz*nonlinear

! particle velocity
     pdot = q*(-g*dedb*dbdt - g*dptdt + ri*dptdz)*deni
     tdot = (upara*b*(1.0-q*gp*zion(4,m)) + q*g*(dedb*dbdp + dptdp))*deni
     zdot = (upara*b*q*(1.0+rip*zion(4,m)) - q*ri*(dedb*dbdp + dptdp))*deni
     rdot = ((gp*zion(4,m)-1.0)*(dedb*dbdt + paranl*dptdt)-&
          paranl*q*(1.0+rip*zion(4,m))*dptdz)*deni 
         
! update particle position
!     if(zion0(1,m) < paxis)then
! particles close to axis use (x,y) coordinates
!        pdum=sqrt(zion(1,m))
!        xdum   = pdum*cost  
!        ydum   = pdum*sint
!        pdum=1.0/zion(1,m)
!        xdot   = 0.5*pdot*xdum*pdum-ydum*tdot
!        ydot   = 0.5*pdot*ydum*pdum+xdum*tdot
!        pdum=sqrt(zion0(1,m))
!        xdum   = pdum*cos(zion0(2,m)) + dtime*xdot
!        ydum   = pdum*sin(zion0(2,m)) + dtime*ydot
!        zion(1,m) = max(1.0e-8_wp*psimax,xdum*xdum+ydum*ydum)
!        zion(2,m) = sign(1.0_wp,ydum)*acos(max(-1.0_wp,min(1.0_wp,xdum/sqrt(zion(1,m)))))
!     else
     zion(1,m) = max(1.0e-8_wp*psimax,zion0(1,m)+dtime*pdot)
     zion(2,m) = zion0(2,m)+dtime*tdot
!     endif

     zion(3,m) = zion0(3,m)+dtime*zdot
     zion(4,m) = zion0(4,m)+dtime*rdot
     zion(5,m) = zion0(5,m)+dtime*wdot
     
! theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
! procedure on Seaborg. However, modulo works better and is preferable.
!!!     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
!!!     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
!!!     zion(3,m)=zion(3,m)*pi2_inv+10.0
!!!     zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m)))

     zion(2,m)=modulo(zion(2,m),pi2)
     zion(3,m)=modulo(zion(3,m),pi2)

! UPDATE: 02/10/2006  The modulo function now streams on the X1E
! 02/20/2004  The modulo function seems to prevent streaming on the X1
! 02/20/2004 mod() does the same thing as modulo but it streams.
! 02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
!  to catch the negative values of zion.
!     zion(2,m)=mod((pi2+zion(2,m)),pi2)
!     zion(3,m)=mod((pi2+zion(3,m)),pi2)

! store GC information for flux measurements
     wpi(1,m)=vdr*rinv
     wpi(2,m)=energy
     wpi(3,m)=b
 enddo 

 if(irk==2)then

  ! Avoid runaway particles by limiting absolute value of parallel velocity
  ! if parameter limit_vpara=1
    if(limit_vpara==1)then
       do m=1,mi
          r=sqrt(2.0_wp*zion(1,m))
          b=1.0_wp/(1.0_wp+r*cos(zion(2,m)))
          upara=zion(4,m)*b*cmratio
          if(abs(upara) > uright)then
             zion(4,m)=sign(1._wp,upara)*uright/(b*cmratio)
             !wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          endif
       enddo
    endif

! out of boundary particle
    do m=1,mi
       if(zion(1,m) > psimax)then
          zion(1,m)=zion0(1,m)
          zion(2,m)=2.0*pi-zion0(2,m)
          zion(3,m)=zion0(3,m)
          zion(4,m)=zion0(4,m)
          zion(5,m)=zion0(5,m)
          
       elseif(zion(1,m) < psimin)then
          zion(1,m)=zion0(1,m)
          zion(2,m)=2.0*pi-zion0(2,m)
          zion(3,m)=zion0(3,m)
          zion(4,m)=zion0(4,m)
          zion(5,m)=zion0(5,m)
       endif
    enddo

    if(conserve_particles==1)then
       do m=1,mi
          sum_of_weights=sum_of_weights+zion(5,m)
       enddo
       !call MPI_ALLREDUCE(sum_of_weights,total_sum_of_weights,1,mpi_Rsize,&
       !     MPI_SUM,MPI_COMM_WORLD,ierror)
       call comm_allreduce_cmm(hpx4_bti,sum_of_weights,total_sum_of_weights,1)
       !call MPI_ALLREDUCE(mi,mi_total,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierror)
       call int_comm_allreduce_cmm(hpx4_bti,mi,mi_total,1)
       residual_weight=total_sum_of_weights/real(mi_total,wp)
       if(mype==0)write(49,*)istep,mi_total,total_sum_of_weights,residual_weight
       do m=1,mi
          zion(5,m)=zion(5,m)-residual_weight
       enddo
    endif

! Restore temperature profile when running a nonlinear calculation
! (nonlinear=1.0) and parameter fixed_Tprofile > 0.
    if(nonlinear > 0.5 .and. fixed_Tprofile > 0)then
       if(mod(istep,ndiag)==0)then
          dtem=0.0
          dden=0.0
          do m=1,mi
             wpi(1,m)=sqrt(2.0*zion(1,m))
             cost=cos(zion(2,m))
             b=1.0/(1.0+wpi(1,m)*cost)
             upara=zion(4,m)*b*cmratio
             wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          enddo
          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             dtem(ip)=dtem(ip)+wpi(2,m)*zion(5,m)
             dden(ip)=dden(ip)+1.0
          enddo
          dtemtmp=0.
          ddentmp=0.
! S.Ethier 06/04/03  According to the MPI standard, the send and receive
! buffers cannot be the same for MPI_Reduce or MPI_Allreduce. It generates
! an error on the Linux platform. We thus make sure that the 2 buffers
! are different.
          !call MPI_ALLREDUCE(dtem,dtemtmp,mflux,mpi_Rsize,MPI_SUM,MPI_COMM_WORLD,ierror)
          call comm_allreduce_cmm(hpx4_bti,dtem,dtemtmp,mflux) 
          !call MPI_ALLREDUCE(dden,ddentmp,mflux,mpi_Rsize,MPI_SUM,MPI_COMM_WORLD,ierror)
          call comm_allreduce_cmm(hpx4_bti,dden,ddentmp,mflux)
          dtem=dtemtmp*tem_inv/max(1.0_wp,ddentmp) !perturbed temperature
          tdum=0.01*real(ndiag)
          rdtemi=(1.0-tdum)*rdtemi+tdum*dtem

          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             zion(5,m)=zion(5,m)-(wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
         ! Correction to include paranl effect according to Weixing
         ! 08/27/04 The correction may not be necessary according to Jerome
         !          because the restoration uses delta_f/f0
         !!!    zion(5,m)=zion(5,m)-(1.0-paranl*zion(5,m))*&
         !!!                 (wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
          enddo
       endif
    endif
 endif

 if(idiag==0)then
! fluxes diagnose at irk=1
    rmarker=0.0
    eflux=0.0
    efluxi=0.0
    pfluxi=0.0
    dflowi=0.0
    entropyi=0.0
    hfluxpsi=0.0
    dmark=0.0
    dden=0.0
    particles_energy=0.0
    do m=1,mi
       r=sqrt(2.0*zion0(1,m))

! radial location in diagnostic bin
       ip=max(1,min(mflux,1+int((r-a0)*d_inv)))
       ii=max(0,min(mpsi,int((r-a0)*delr+0.5)))
       vdrenergy=wpi(1,m)*(wpi(2,m)-1.5*aion*vthi*vthi*rtemi(ii))*zion0(5,m)

! radial profile of heat flux
       hfluxpsi(ii)=hfluxpsi(ii)+vdrenergy ! energy flux profile       

! marker,energy,particle,momentum fluxes,parallel flows,entropy and kinetic energy
       rmarker(ip)=rmarker(ip)+zion0(6,m)
       eflux(ip)=eflux(ip)+vdrenergy
       efluxi=efluxi+vdrenergy
       pfluxi=pfluxi+wpi(1,m)*zion0(5,m)
       dflowi=dflowi+wpi(3,m)*zion0(4,m)*zion0(5,m)
       entropyi=entropyi+zion0(5,m)*zion0(5,m)
     ! S.Ethier 9/21/04  We want to keep track of the total energy in the
     ! particles, so we add up the contributions of energy*weight for each
     ! particle. This kinetic energy was saved in wpi(2,m) in the irk=1
     ! section above.
       particles_energy(1)=particles_energy(1)+wpi(2,m)*zion0(5,m)
       particles_energy(2)=particles_energy(2)+wpi(2,m)

       dmark(ip)=dmark(ip)+wpi(1,m)*r
       dden(ip)=dden(ip)+1.0
    enddo
    hfluxpsitmp=0.
    !call MPI_ALLREDUCE(hfluxpsi,hfluxpsitmp,mpsi+1,mpi_Rsize,MPI_SUM,MPI_COMM_WORLD,ierror)
    call comm_allreduce_cmm(hpx4_bti,hfluxpsi,hfluxpsitmp,mpsi+1)
    hfluxpsi=hfluxpsitmp*pmarki

    dmarktmp=0.
    !call MPI_ALLREDUCE(dmark,dmarktmp,mflux,mpi_Rsize,MPI_SUM,MPI_COMM_WORLD,ierror)
    call comm_allreduce_cmm(hpx4_bti,dmark,dmarktmp,mflux)
    ddentmp=0.
    !call MPI_ALLREDUCE(dden,ddentmp,mflux,mpi_Rsize,MPI_SUM,MPI_COMM_WORLD,ierror)
    call comm_allreduce_cmm(hpx4_bti,dden,ddentmp,mflux)
    dmark=dmarktmp/max(1.0_wp,ddentmp)
    tdum=0.01*real(ndiag)
    pfluxpsi=(1.0-tdum)*pfluxpsi+tdum*dmark

  ! Field energy: 1/(4*pi)sum_ijk{|Grad_phi|^2*Jacobian*r*deltar*deltat*deltaz}
    En_field=0._wp
    Total_field_energy=0._wp
    do k=1,mzeta
       zdum=zetamin+real(k)*deltaz
       do i=1,mpsi-1
          r=a0+deltar*real(i)
          !q=q0+q1*r*ainv+q2*r*r*ainv*ainv
          drdp=1.0_wp/r
          do j=1,mtheta(i)
             ij=igrid(i)+j
             tdum=real(j)*deltat(i)+zdum*qtinv(i)
             grad_zeta=1.0_wp+r*cos(tdum)
             jacobian=grad_zeta*grad_zeta
             Epsi=evector(1,k,ij)-phip00(i)*drdp
             Etheta=evector(2,k,ij)
             Ezeta=(evector(3,k,ij)-evector(2,k,ij)*qtinv(i))
           ! Turbulence energy: field energy without the zonal flow
             dphi_square(1)=r*r*Epsi*Epsi + &    ! dphi_dpsi square
                         drdp*drdp*Etheta*Etheta + &  !dphi_dtheta sq
                         Ezeta*Ezeta/(grad_zeta*grad_zeta)
           ! Total field energy: turbulence + zonal flow energies
             dphi_square(2)=r*r*evector(1,k,ij)*evector(1,k,ij) + &
                            drdp*drdp*Etheta*Etheta + &
                            Ezeta*Ezeta/(grad_zeta*grad_zeta)
           ! Zonal flow energy only
             dphi_square(3)=phip00(i)*phip00(i)
             En_field=En_field+dphi_square*jacobian*r*deltat(i)
          enddo
       enddo
    enddo
    En_field=En_field*deltar*deltaz/(4.0_wp*pi)

  ! Sum up the values from all the toroidal domains. We need the value of
  ! only one processor per domain.
    !call MPI_REDUCE(En_field,Total_field_energy,3,mpi_Rsize,MPI_SUM,0,&
    !                toroidal_comm,ierror)
    call toroidal_reduce_cmm(hpx4_bti,En_field,Total_field_energy,3,0)

 endif
 
end subroutine pushi
