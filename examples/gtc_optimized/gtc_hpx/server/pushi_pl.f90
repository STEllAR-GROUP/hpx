subroutine pushi_0(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_0
  use particle_array_0
  use particle_decomp_0
  use field_array_0
  use diagnosis_array_0
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,im,ip,j,jt,k,kz,kk,m,ii,j11,j10,j01,j00,jtheta,num,ierror,&
       larmor,ij,ipjt
  real(8) cost0,sint0,cost,sint,rdum,tdum,dtime,q,r,b,g,gp,ri,rip,dbdp,dbdt,&
       dedb,deni,upara,energy,kappa,dptdp,dptdt,dptdz,vdr,pdot,tdot,zdot,rdot,&
       wdot,wdrift,vdrenergy,delr,pi2,pi2_inv,uleft,uright,adum(0:mpsi),&
       dmark(mflux),pstat,xdum,ydum,xdot,ydot,tem_inv,wz1,wz0,wp1,wp0,wt11,&
       wt10,wt01,wt00,temp(0:mpsi),dtemp(0:mpsi),denp(0:mpsi),g2,tem,rfac,&
       sbound,qdum,tflr,d_inv,epara,wpara,psimax,psimin,paxis,pdum,wdrive,&
       ainv,rinv,qinv,psitmp,thetatmp,zetatmp,rhoi,e1,e2,e3,cmratio,cinv,&
       wpi(3,mi),rdel,vthi,dtem(mflux),dden(mflux),ddentmp(mflux),&
       dtemtmp(mflux),vdrtmp(mflux),dmarktmp(mflux),hfluxpsitmp(0:mpsi)
  real(8) epsi,etheta,ezeta,en_field(3),zdum,drdp,grad_zeta,jacobian,dphi_square(3)
  real(8) sum_of_weights,total_sum_of_weights,residual_weight
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
! 1st step of runge-kutta method
     dtime=0.5*tstep
!$omp parallel do private(m)
     do m=1,mi
        zion0(1:5,m)=zion(1:5,m)
     enddo

     vdrtmp=0.0

! 2nd step of runge-kutta method
  else
     dtime=tstep

     if(nonlinear<0.5)vdrtmp=0.0
     if(nonlinear>0.5)vdrtmp=pfluxpsi
  endif

! gather e_field using 4-point gyro-averaging, sorting in poloidal angle
! the following line is an openmp directive for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
!$omp parallel do private(m,e1,e2,e3,kk,wz1,wz0,larmor,ij,wp0,wt00,&
!$omp& wt10,wp1,wt01,wt11)
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

! update gc position
! another openmp parallel loop...
!$omp parallel do private(m,r,rinv,ii,ip,wp0,wp1,tem,q,qinv,cost,sint,cost0,&
!$omp& sint0,b,g,gp,ri,rip,dbdp,dbdt,dedb,deni,upara,energy,rfac,kappa,dptdp,&
!$omp& dptdt,dptdz,epara,vdr,wdrive,wpara,wdrift,wdot,pdot,tdot,zdot,rdot)
  do m=1,mi
     r=sqrt(2.0d0*zion(1,m))
     rinv=1.0d0/r
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

! exb drift in radial direction for w-dot and flux diagnostics
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
     zion(1,m) = max(1.0e-8*psimax,zion0(1,m)+dtime*pdot)
     zion(2,m) = zion0(2,m)+dtime*tdot
!     endif

     zion(3,m) = zion0(3,m)+dtime*zdot
     zion(4,m) = zion0(4,m)+dtime*rdot
     zion(5,m) = zion0(5,m)+dtime*wdot
     
! theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
! procedure on seaborg. however, modulo works better and is preferable.
!!!     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
!!!     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
!!!     zion(3,m)=zion(3,m)*pi2_inv+10.0
!!!     zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m)))

     zion(2,m)=modulo(zion(2,m),pi2)
     zion(3,m)=modulo(zion(3,m),pi2)

! update: 02/10/2006  the modulo function now streams on the x1e
! 02/20/2004  the modulo function seems to prevent streaming on the x1
! 02/20/2004 mod() does the same thing as modulo but it streams.
! 02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
!  to catch the negative values of zion.
!     zion(2,m)=mod((pi2+zion(2,m)),pi2)
!     zion(3,m)=mod((pi2+zion(3,m)),pi2)

! store gc information for flux measurements
     wpi(1,m)=vdr*rinv
     wpi(2,m)=energy
     wpi(3,m)=b
 enddo 

 if(irk==2)then
  ! avoid runaway particles by limiting absolute value of parallel velocity
  ! if parameter limit_vpara=1
    if(limit_vpara==1)then
       do m=1,mi
          r=sqrt(2.0d0*zion(1,m))
          b=1.0d0/(1.0d0+r*cos(zion(2,m)))
          upara=zion(4,m)*b*cmratio
          if(abs(upara) > uright)then
             zion(4,m)=sign(1.0d0,upara)*uright/(b*cmratio)
             !wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          endif
       enddo
    endif

! out of boundary particle
!$omp parallel do private(m)
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
       !call mpi_allreduce(sum_of_weights,total_sum_of_weights,1,mpi_rsize,&
       !     mpi_sum,mpi_comm_world,ierror)
       call comm_allreduce_cmm(ptr,sum_of_weights,total_sum_of_weights,1);
       !call mpi_allreduce(mi,mi_total,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
       !residual_weight=total_sum_of_weights/real(mi_total,wp)
       call int_comm_allreduce_cmm(ptr,mi,mi_total,one);
       if(mype==0)write(49,*)istep,mi_total,total_sum_of_weights,residual_weight
       do m=1,mi
          zion(5,m)=zion(5,m)-residual_weight
       enddo
    endif

! restore temperature profile when running a nonlinear calculation
! (nonlinear=1.0) and parameter fixed_tprofile > 0.
    if(nonlinear > 0.5 .and. fixed_tprofile > 0)then
       if(mod(istep,ndiag)==0)then
          dtem=0.0
          dden=0.0
!$omp parallel do private(m,cost,b,upara)
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
! s.ethier 06/04/03  according to the mpi standard, the send and receive
! buffers cannot be the same for mpi_reduce or mpi_allreduce. it generates
! an error on the linux platform. we thus make sure that the 2 buffers
! are different.
          !call mpi_allreduce(dtem,dtemtmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dtem,dtemtmp,mflux);
          !call mpi_allreduce(dden,ddentmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dden,ddentmp,mflux);
          dtem=dtemtmp*tem_inv/max(1.0d0,ddentmp) !perturbed temperature
          tdum=0.01*real(ndiag)
          rdtemi=(1.0-tdum)*rdtemi+tdum*dtem

!$omp parallel do private(m,ip)
          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             zion(5,m)=zion(5,m)-(wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
         ! correction to include paranl effect according to weixing
         ! 08/27/04 the correction may not be necessary according to jerome
         !          because the restoration uses delta_f/f0
         !!!    zion(5,m)=zion(5,m)-(1.0-paranl*zion(5,m))*&
         !!!                 (wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
          enddo
       endif
    endif
 endif
 
end subroutine pushi_0
subroutine pushi_1(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_1
  use particle_array_1
  use particle_decomp_1
  use field_array_1
  use diagnosis_array_1
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,im,ip,j,jt,k,kz,kk,m,ii,j11,j10,j01,j00,jtheta,num,ierror,&
       larmor,ij,ipjt
  real(8) cost0,sint0,cost,sint,rdum,tdum,dtime,q,r,b,g,gp,ri,rip,dbdp,dbdt,&
       dedb,deni,upara,energy,kappa,dptdp,dptdt,dptdz,vdr,pdot,tdot,zdot,rdot,&
       wdot,wdrift,vdrenergy,delr,pi2,pi2_inv,uleft,uright,adum(0:mpsi),&
       dmark(mflux),pstat,xdum,ydum,xdot,ydot,tem_inv,wz1,wz0,wp1,wp0,wt11,&
       wt10,wt01,wt00,temp(0:mpsi),dtemp(0:mpsi),denp(0:mpsi),g2,tem,rfac,&
       sbound,qdum,tflr,d_inv,epara,wpara,psimax,psimin,paxis,pdum,wdrive,&
       ainv,rinv,qinv,psitmp,thetatmp,zetatmp,rhoi,e1,e2,e3,cmratio,cinv,&
       wpi(3,mi),rdel,vthi,dtem(mflux),dden(mflux),ddentmp(mflux),&
       dtemtmp(mflux),vdrtmp(mflux),dmarktmp(mflux),hfluxpsitmp(0:mpsi)
  real(8) epsi,etheta,ezeta,en_field(3),zdum,drdp,grad_zeta,jacobian,dphi_square(3)
  real(8) sum_of_weights,total_sum_of_weights,residual_weight
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
! 1st step of runge-kutta method
     dtime=0.5*tstep
!$omp parallel do private(m)
     do m=1,mi
        zion0(1:5,m)=zion(1:5,m)
     enddo

     vdrtmp=0.0

! 2nd step of runge-kutta method
  else
     dtime=tstep

     if(nonlinear<0.5)vdrtmp=0.0
     if(nonlinear>0.5)vdrtmp=pfluxpsi
  endif

! gather e_field using 4-point gyro-averaging, sorting in poloidal angle
! the following line is an openmp directive for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
!$omp parallel do private(m,e1,e2,e3,kk,wz1,wz0,larmor,ij,wp0,wt00,&
!$omp& wt10,wp1,wt01,wt11)
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

! update gc position
! another openmp parallel loop...
!$omp parallel do private(m,r,rinv,ii,ip,wp0,wp1,tem,q,qinv,cost,sint,cost0,&
!$omp& sint0,b,g,gp,ri,rip,dbdp,dbdt,dedb,deni,upara,energy,rfac,kappa,dptdp,&
!$omp& dptdt,dptdz,epara,vdr,wdrive,wpara,wdrift,wdot,pdot,tdot,zdot,rdot)
  do m=1,mi
     r=sqrt(2.0d0*zion(1,m))
     rinv=1.0d0/r
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

! exb drift in radial direction for w-dot and flux diagnostics
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
     zion(1,m) = max(1.0e-8*psimax,zion0(1,m)+dtime*pdot)
     zion(2,m) = zion0(2,m)+dtime*tdot
!     endif

     zion(3,m) = zion0(3,m)+dtime*zdot
     zion(4,m) = zion0(4,m)+dtime*rdot
     zion(5,m) = zion0(5,m)+dtime*wdot
     
! theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
! procedure on seaborg. however, modulo works better and is preferable.
!!!     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
!!!     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
!!!     zion(3,m)=zion(3,m)*pi2_inv+10.0
!!!     zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m)))

     zion(2,m)=modulo(zion(2,m),pi2)
     zion(3,m)=modulo(zion(3,m),pi2)

! update: 02/10/2006  the modulo function now streams on the x1e
! 02/20/2004  the modulo function seems to prevent streaming on the x1
! 02/20/2004 mod() does the same thing as modulo but it streams.
! 02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
!  to catch the negative values of zion.
!     zion(2,m)=mod((pi2+zion(2,m)),pi2)
!     zion(3,m)=mod((pi2+zion(3,m)),pi2)

! store gc information for flux measurements
     wpi(1,m)=vdr*rinv
     wpi(2,m)=energy
     wpi(3,m)=b
 enddo 

 if(irk==2)then
  ! avoid runaway particles by limiting absolute value of parallel velocity
  ! if parameter limit_vpara=1
    if(limit_vpara==1)then
       do m=1,mi
          r=sqrt(2.0d0*zion(1,m))
          b=1.0d0/(1.0d0+r*cos(zion(2,m)))
          upara=zion(4,m)*b*cmratio
          if(abs(upara) > uright)then
             zion(4,m)=sign(1.0d0,upara)*uright/(b*cmratio)
             !wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          endif
       enddo
    endif

! out of boundary particle
!$omp parallel do private(m)
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
       !call mpi_allreduce(sum_of_weights,total_sum_of_weights,1,mpi_rsize,&
       !     mpi_sum,mpi_comm_world,ierror)
       call comm_allreduce_cmm(ptr,sum_of_weights,total_sum_of_weights,1);
       !call mpi_allreduce(mi,mi_total,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
       !residual_weight=total_sum_of_weights/real(mi_total,wp)
       call int_comm_allreduce_cmm(ptr,mi,mi_total,one);
       if(mype==0)write(49,*)istep,mi_total,total_sum_of_weights,residual_weight
       do m=1,mi
          zion(5,m)=zion(5,m)-residual_weight
       enddo
    endif

! restore temperature profile when running a nonlinear calculation
! (nonlinear=1.0) and parameter fixed_tprofile > 0.
    if(nonlinear > 0.5 .and. fixed_tprofile > 0)then
       if(mod(istep,ndiag)==0)then
          dtem=0.0
          dden=0.0
!$omp parallel do private(m,cost,b,upara)
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
! s.ethier 06/04/03  according to the mpi standard, the send and receive
! buffers cannot be the same for mpi_reduce or mpi_allreduce. it generates
! an error on the linux platform. we thus make sure that the 2 buffers
! are different.
          !call mpi_allreduce(dtem,dtemtmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dtem,dtemtmp,mflux);
          !call mpi_allreduce(dden,ddentmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dden,ddentmp,mflux);
          dtem=dtemtmp*tem_inv/max(1.0d0,ddentmp) !perturbed temperature
          tdum=0.01*real(ndiag)
          rdtemi=(1.0-tdum)*rdtemi+tdum*dtem

!$omp parallel do private(m,ip)
          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             zion(5,m)=zion(5,m)-(wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
         ! correction to include paranl effect according to weixing
         ! 08/27/04 the correction may not be necessary according to jerome
         !          because the restoration uses delta_f/f0
         !!!    zion(5,m)=zion(5,m)-(1.0-paranl*zion(5,m))*&
         !!!                 (wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
          enddo
       endif
    endif
 endif
 
end subroutine pushi_1
subroutine pushi_2(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_2
  use particle_array_2
  use particle_decomp_2
  use field_array_2
  use diagnosis_array_2
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,im,ip,j,jt,k,kz,kk,m,ii,j11,j10,j01,j00,jtheta,num,ierror,&
       larmor,ij,ipjt
  real(8) cost0,sint0,cost,sint,rdum,tdum,dtime,q,r,b,g,gp,ri,rip,dbdp,dbdt,&
       dedb,deni,upara,energy,kappa,dptdp,dptdt,dptdz,vdr,pdot,tdot,zdot,rdot,&
       wdot,wdrift,vdrenergy,delr,pi2,pi2_inv,uleft,uright,adum(0:mpsi),&
       dmark(mflux),pstat,xdum,ydum,xdot,ydot,tem_inv,wz1,wz0,wp1,wp0,wt11,&
       wt10,wt01,wt00,temp(0:mpsi),dtemp(0:mpsi),denp(0:mpsi),g2,tem,rfac,&
       sbound,qdum,tflr,d_inv,epara,wpara,psimax,psimin,paxis,pdum,wdrive,&
       ainv,rinv,qinv,psitmp,thetatmp,zetatmp,rhoi,e1,e2,e3,cmratio,cinv,&
       wpi(3,mi),rdel,vthi,dtem(mflux),dden(mflux),ddentmp(mflux),&
       dtemtmp(mflux),vdrtmp(mflux),dmarktmp(mflux),hfluxpsitmp(0:mpsi)
  real(8) epsi,etheta,ezeta,en_field(3),zdum,drdp,grad_zeta,jacobian,dphi_square(3)
  real(8) sum_of_weights,total_sum_of_weights,residual_weight
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
! 1st step of runge-kutta method
     dtime=0.5*tstep
!$omp parallel do private(m)
     do m=1,mi
        zion0(1:5,m)=zion(1:5,m)
     enddo

     vdrtmp=0.0

! 2nd step of runge-kutta method
  else
     dtime=tstep

     if(nonlinear<0.5)vdrtmp=0.0
     if(nonlinear>0.5)vdrtmp=pfluxpsi
  endif

! gather e_field using 4-point gyro-averaging, sorting in poloidal angle
! the following line is an openmp directive for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
!$omp parallel do private(m,e1,e2,e3,kk,wz1,wz0,larmor,ij,wp0,wt00,&
!$omp& wt10,wp1,wt01,wt11)
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

! update gc position
! another openmp parallel loop...
!$omp parallel do private(m,r,rinv,ii,ip,wp0,wp1,tem,q,qinv,cost,sint,cost0,&
!$omp& sint0,b,g,gp,ri,rip,dbdp,dbdt,dedb,deni,upara,energy,rfac,kappa,dptdp,&
!$omp& dptdt,dptdz,epara,vdr,wdrive,wpara,wdrift,wdot,pdot,tdot,zdot,rdot)
  do m=1,mi
     r=sqrt(2.0d0*zion(1,m))
     rinv=1.0d0/r
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

! exb drift in radial direction for w-dot and flux diagnostics
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
     zion(1,m) = max(1.0e-8*psimax,zion0(1,m)+dtime*pdot)
     zion(2,m) = zion0(2,m)+dtime*tdot
!     endif

     zion(3,m) = zion0(3,m)+dtime*zdot
     zion(4,m) = zion0(4,m)+dtime*rdot
     zion(5,m) = zion0(5,m)+dtime*wdot
     
! theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
! procedure on seaborg. however, modulo works better and is preferable.
!!!     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
!!!     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
!!!     zion(3,m)=zion(3,m)*pi2_inv+10.0
!!!     zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m)))

     zion(2,m)=modulo(zion(2,m),pi2)
     zion(3,m)=modulo(zion(3,m),pi2)

! update: 02/10/2006  the modulo function now streams on the x1e
! 02/20/2004  the modulo function seems to prevent streaming on the x1
! 02/20/2004 mod() does the same thing as modulo but it streams.
! 02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
!  to catch the negative values of zion.
!     zion(2,m)=mod((pi2+zion(2,m)),pi2)
!     zion(3,m)=mod((pi2+zion(3,m)),pi2)

! store gc information for flux measurements
     wpi(1,m)=vdr*rinv
     wpi(2,m)=energy
     wpi(3,m)=b
 enddo 

 if(irk==2)then
  ! avoid runaway particles by limiting absolute value of parallel velocity
  ! if parameter limit_vpara=1
    if(limit_vpara==1)then
       do m=1,mi
          r=sqrt(2.0d0*zion(1,m))
          b=1.0d0/(1.0d0+r*cos(zion(2,m)))
          upara=zion(4,m)*b*cmratio
          if(abs(upara) > uright)then
             zion(4,m)=sign(1.0d0,upara)*uright/(b*cmratio)
             !wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          endif
       enddo
    endif

! out of boundary particle
!$omp parallel do private(m)
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
       !call mpi_allreduce(sum_of_weights,total_sum_of_weights,1,mpi_rsize,&
       !     mpi_sum,mpi_comm_world,ierror)
       call comm_allreduce_cmm(ptr,sum_of_weights,total_sum_of_weights,1);
       !call mpi_allreduce(mi,mi_total,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
       !residual_weight=total_sum_of_weights/real(mi_total,wp)
       call int_comm_allreduce_cmm(ptr,mi,mi_total,one);
       if(mype==0)write(49,*)istep,mi_total,total_sum_of_weights,residual_weight
       do m=1,mi
          zion(5,m)=zion(5,m)-residual_weight
       enddo
    endif

! restore temperature profile when running a nonlinear calculation
! (nonlinear=1.0) and parameter fixed_tprofile > 0.
    if(nonlinear > 0.5 .and. fixed_tprofile > 0)then
       if(mod(istep,ndiag)==0)then
          dtem=0.0
          dden=0.0
!$omp parallel do private(m,cost,b,upara)
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
! s.ethier 06/04/03  according to the mpi standard, the send and receive
! buffers cannot be the same for mpi_reduce or mpi_allreduce. it generates
! an error on the linux platform. we thus make sure that the 2 buffers
! are different.
          !call mpi_allreduce(dtem,dtemtmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dtem,dtemtmp,mflux);
          !call mpi_allreduce(dden,ddentmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dden,ddentmp,mflux);
          dtem=dtemtmp*tem_inv/max(1.0d0,ddentmp) !perturbed temperature
          tdum=0.01*real(ndiag)
          rdtemi=(1.0-tdum)*rdtemi+tdum*dtem

!$omp parallel do private(m,ip)
          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             zion(5,m)=zion(5,m)-(wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
         ! correction to include paranl effect according to weixing
         ! 08/27/04 the correction may not be necessary according to jerome
         !          because the restoration uses delta_f/f0
         !!!    zion(5,m)=zion(5,m)-(1.0-paranl*zion(5,m))*&
         !!!                 (wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
          enddo
       endif
    endif
 endif
 
end subroutine pushi_2
subroutine pushi_3(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_3
  use particle_array_3
  use particle_decomp_3
  use field_array_3
  use diagnosis_array_3
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,im,ip,j,jt,k,kz,kk,m,ii,j11,j10,j01,j00,jtheta,num,ierror,&
       larmor,ij,ipjt
  real(8) cost0,sint0,cost,sint,rdum,tdum,dtime,q,r,b,g,gp,ri,rip,dbdp,dbdt,&
       dedb,deni,upara,energy,kappa,dptdp,dptdt,dptdz,vdr,pdot,tdot,zdot,rdot,&
       wdot,wdrift,vdrenergy,delr,pi2,pi2_inv,uleft,uright,adum(0:mpsi),&
       dmark(mflux),pstat,xdum,ydum,xdot,ydot,tem_inv,wz1,wz0,wp1,wp0,wt11,&
       wt10,wt01,wt00,temp(0:mpsi),dtemp(0:mpsi),denp(0:mpsi),g2,tem,rfac,&
       sbound,qdum,tflr,d_inv,epara,wpara,psimax,psimin,paxis,pdum,wdrive,&
       ainv,rinv,qinv,psitmp,thetatmp,zetatmp,rhoi,e1,e2,e3,cmratio,cinv,&
       wpi(3,mi),rdel,vthi,dtem(mflux),dden(mflux),ddentmp(mflux),&
       dtemtmp(mflux),vdrtmp(mflux),dmarktmp(mflux),hfluxpsitmp(0:mpsi)
  real(8) epsi,etheta,ezeta,en_field(3),zdum,drdp,grad_zeta,jacobian,dphi_square(3)
  real(8) sum_of_weights,total_sum_of_weights,residual_weight
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
! 1st step of runge-kutta method
     dtime=0.5*tstep
!$omp parallel do private(m)
     do m=1,mi
        zion0(1:5,m)=zion(1:5,m)
     enddo

     vdrtmp=0.0

! 2nd step of runge-kutta method
  else
     dtime=tstep

     if(nonlinear<0.5)vdrtmp=0.0
     if(nonlinear>0.5)vdrtmp=pfluxpsi
  endif

! gather e_field using 4-point gyro-averaging, sorting in poloidal angle
! the following line is an openmp directive for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
!$omp parallel do private(m,e1,e2,e3,kk,wz1,wz0,larmor,ij,wp0,wt00,&
!$omp& wt10,wp1,wt01,wt11)
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

! update gc position
! another openmp parallel loop...
!$omp parallel do private(m,r,rinv,ii,ip,wp0,wp1,tem,q,qinv,cost,sint,cost0,&
!$omp& sint0,b,g,gp,ri,rip,dbdp,dbdt,dedb,deni,upara,energy,rfac,kappa,dptdp,&
!$omp& dptdt,dptdz,epara,vdr,wdrive,wpara,wdrift,wdot,pdot,tdot,zdot,rdot)
  do m=1,mi
     r=sqrt(2.0d0*zion(1,m))
     rinv=1.0d0/r
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

! exb drift in radial direction for w-dot and flux diagnostics
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
     zion(1,m) = max(1.0e-8*psimax,zion0(1,m)+dtime*pdot)
     zion(2,m) = zion0(2,m)+dtime*tdot
!     endif

     zion(3,m) = zion0(3,m)+dtime*zdot
     zion(4,m) = zion0(4,m)+dtime*rdot
     zion(5,m) = zion0(5,m)+dtime*wdot
     
! theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
! procedure on seaborg. however, modulo works better and is preferable.
!!!     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
!!!     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
!!!     zion(3,m)=zion(3,m)*pi2_inv+10.0
!!!     zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m)))

     zion(2,m)=modulo(zion(2,m),pi2)
     zion(3,m)=modulo(zion(3,m),pi2)

! update: 02/10/2006  the modulo function now streams on the x1e
! 02/20/2004  the modulo function seems to prevent streaming on the x1
! 02/20/2004 mod() does the same thing as modulo but it streams.
! 02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
!  to catch the negative values of zion.
!     zion(2,m)=mod((pi2+zion(2,m)),pi2)
!     zion(3,m)=mod((pi2+zion(3,m)),pi2)

! store gc information for flux measurements
     wpi(1,m)=vdr*rinv
     wpi(2,m)=energy
     wpi(3,m)=b
 enddo 

 if(irk==2)then
  ! avoid runaway particles by limiting absolute value of parallel velocity
  ! if parameter limit_vpara=1
    if(limit_vpara==1)then
       do m=1,mi
          r=sqrt(2.0d0*zion(1,m))
          b=1.0d0/(1.0d0+r*cos(zion(2,m)))
          upara=zion(4,m)*b*cmratio
          if(abs(upara) > uright)then
             zion(4,m)=sign(1.0d0,upara)*uright/(b*cmratio)
             !wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          endif
       enddo
    endif

! out of boundary particle
!$omp parallel do private(m)
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
       !call mpi_allreduce(sum_of_weights,total_sum_of_weights,1,mpi_rsize,&
       !     mpi_sum,mpi_comm_world,ierror)
       call comm_allreduce_cmm(ptr,sum_of_weights,total_sum_of_weights,1);
       !call mpi_allreduce(mi,mi_total,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
       !residual_weight=total_sum_of_weights/real(mi_total,wp)
       call int_comm_allreduce_cmm(ptr,mi,mi_total,one);
       if(mype==0)write(49,*)istep,mi_total,total_sum_of_weights,residual_weight
       do m=1,mi
          zion(5,m)=zion(5,m)-residual_weight
       enddo
    endif

! restore temperature profile when running a nonlinear calculation
! (nonlinear=1.0) and parameter fixed_tprofile > 0.
    if(nonlinear > 0.5 .and. fixed_tprofile > 0)then
       if(mod(istep,ndiag)==0)then
          dtem=0.0
          dden=0.0
!$omp parallel do private(m,cost,b,upara)
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
! s.ethier 06/04/03  according to the mpi standard, the send and receive
! buffers cannot be the same for mpi_reduce or mpi_allreduce. it generates
! an error on the linux platform. we thus make sure that the 2 buffers
! are different.
          !call mpi_allreduce(dtem,dtemtmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dtem,dtemtmp,mflux);
          !call mpi_allreduce(dden,ddentmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dden,ddentmp,mflux);
          dtem=dtemtmp*tem_inv/max(1.0d0,ddentmp) !perturbed temperature
          tdum=0.01*real(ndiag)
          rdtemi=(1.0-tdum)*rdtemi+tdum*dtem

!$omp parallel do private(m,ip)
          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             zion(5,m)=zion(5,m)-(wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
         ! correction to include paranl effect according to weixing
         ! 08/27/04 the correction may not be necessary according to jerome
         !          because the restoration uses delta_f/f0
         !!!    zion(5,m)=zion(5,m)-(1.0-paranl*zion(5,m))*&
         !!!                 (wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
          enddo
       endif
    endif
 endif
 
end subroutine pushi_3
subroutine pushi_4(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_4
  use particle_array_4
  use particle_decomp_4
  use field_array_4
  use diagnosis_array_4
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,im,ip,j,jt,k,kz,kk,m,ii,j11,j10,j01,j00,jtheta,num,ierror,&
       larmor,ij,ipjt
  real(8) cost0,sint0,cost,sint,rdum,tdum,dtime,q,r,b,g,gp,ri,rip,dbdp,dbdt,&
       dedb,deni,upara,energy,kappa,dptdp,dptdt,dptdz,vdr,pdot,tdot,zdot,rdot,&
       wdot,wdrift,vdrenergy,delr,pi2,pi2_inv,uleft,uright,adum(0:mpsi),&
       dmark(mflux),pstat,xdum,ydum,xdot,ydot,tem_inv,wz1,wz0,wp1,wp0,wt11,&
       wt10,wt01,wt00,temp(0:mpsi),dtemp(0:mpsi),denp(0:mpsi),g2,tem,rfac,&
       sbound,qdum,tflr,d_inv,epara,wpara,psimax,psimin,paxis,pdum,wdrive,&
       ainv,rinv,qinv,psitmp,thetatmp,zetatmp,rhoi,e1,e2,e3,cmratio,cinv,&
       wpi(3,mi),rdel,vthi,dtem(mflux),dden(mflux),ddentmp(mflux),&
       dtemtmp(mflux),vdrtmp(mflux),dmarktmp(mflux),hfluxpsitmp(0:mpsi)
  real(8) epsi,etheta,ezeta,en_field(3),zdum,drdp,grad_zeta,jacobian,dphi_square(3)
  real(8) sum_of_weights,total_sum_of_weights,residual_weight
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
! 1st step of runge-kutta method
     dtime=0.5*tstep
!$omp parallel do private(m)
     do m=1,mi
        zion0(1:5,m)=zion(1:5,m)
     enddo

     vdrtmp=0.0

! 2nd step of runge-kutta method
  else
     dtime=tstep

     if(nonlinear<0.5)vdrtmp=0.0
     if(nonlinear>0.5)vdrtmp=pfluxpsi
  endif

! gather e_field using 4-point gyro-averaging, sorting in poloidal angle
! the following line is an openmp directive for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
!$omp parallel do private(m,e1,e2,e3,kk,wz1,wz0,larmor,ij,wp0,wt00,&
!$omp& wt10,wp1,wt01,wt11)
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

! update gc position
! another openmp parallel loop...
!$omp parallel do private(m,r,rinv,ii,ip,wp0,wp1,tem,q,qinv,cost,sint,cost0,&
!$omp& sint0,b,g,gp,ri,rip,dbdp,dbdt,dedb,deni,upara,energy,rfac,kappa,dptdp,&
!$omp& dptdt,dptdz,epara,vdr,wdrive,wpara,wdrift,wdot,pdot,tdot,zdot,rdot)
  do m=1,mi
     r=sqrt(2.0d0*zion(1,m))
     rinv=1.0d0/r
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

! exb drift in radial direction for w-dot and flux diagnostics
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
     zion(1,m) = max(1.0e-8*psimax,zion0(1,m)+dtime*pdot)
     zion(2,m) = zion0(2,m)+dtime*tdot
!     endif

     zion(3,m) = zion0(3,m)+dtime*zdot
     zion(4,m) = zion0(4,m)+dtime*rdot
     zion(5,m) = zion0(5,m)+dtime*wdot
     
! theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
! procedure on seaborg. however, modulo works better and is preferable.
!!!     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
!!!     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
!!!     zion(3,m)=zion(3,m)*pi2_inv+10.0
!!!     zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m)))

     zion(2,m)=modulo(zion(2,m),pi2)
     zion(3,m)=modulo(zion(3,m),pi2)

! update: 02/10/2006  the modulo function now streams on the x1e
! 02/20/2004  the modulo function seems to prevent streaming on the x1
! 02/20/2004 mod() does the same thing as modulo but it streams.
! 02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
!  to catch the negative values of zion.
!     zion(2,m)=mod((pi2+zion(2,m)),pi2)
!     zion(3,m)=mod((pi2+zion(3,m)),pi2)

! store gc information for flux measurements
     wpi(1,m)=vdr*rinv
     wpi(2,m)=energy
     wpi(3,m)=b
 enddo 

 if(irk==2)then
  ! avoid runaway particles by limiting absolute value of parallel velocity
  ! if parameter limit_vpara=1
    if(limit_vpara==1)then
       do m=1,mi
          r=sqrt(2.0d0*zion(1,m))
          b=1.0d0/(1.0d0+r*cos(zion(2,m)))
          upara=zion(4,m)*b*cmratio
          if(abs(upara) > uright)then
             zion(4,m)=sign(1.0d0,upara)*uright/(b*cmratio)
             !wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          endif
       enddo
    endif

! out of boundary particle
!$omp parallel do private(m)
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
       !call mpi_allreduce(sum_of_weights,total_sum_of_weights,1,mpi_rsize,&
       !     mpi_sum,mpi_comm_world,ierror)
       call comm_allreduce_cmm(ptr,sum_of_weights,total_sum_of_weights,1);
       !call mpi_allreduce(mi,mi_total,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
       !residual_weight=total_sum_of_weights/real(mi_total,wp)
       call int_comm_allreduce_cmm(ptr,mi,mi_total,one);
       if(mype==0)write(49,*)istep,mi_total,total_sum_of_weights,residual_weight
       do m=1,mi
          zion(5,m)=zion(5,m)-residual_weight
       enddo
    endif

! restore temperature profile when running a nonlinear calculation
! (nonlinear=1.0) and parameter fixed_tprofile > 0.
    if(nonlinear > 0.5 .and. fixed_tprofile > 0)then
       if(mod(istep,ndiag)==0)then
          dtem=0.0
          dden=0.0
!$omp parallel do private(m,cost,b,upara)
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
! s.ethier 06/04/03  according to the mpi standard, the send and receive
! buffers cannot be the same for mpi_reduce or mpi_allreduce. it generates
! an error on the linux platform. we thus make sure that the 2 buffers
! are different.
          !call mpi_allreduce(dtem,dtemtmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dtem,dtemtmp,mflux);
          !call mpi_allreduce(dden,ddentmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dden,ddentmp,mflux);
          dtem=dtemtmp*tem_inv/max(1.0d0,ddentmp) !perturbed temperature
          tdum=0.01*real(ndiag)
          rdtemi=(1.0-tdum)*rdtemi+tdum*dtem

!$omp parallel do private(m,ip)
          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             zion(5,m)=zion(5,m)-(wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
         ! correction to include paranl effect according to weixing
         ! 08/27/04 the correction may not be necessary according to jerome
         !          because the restoration uses delta_f/f0
         !!!    zion(5,m)=zion(5,m)-(1.0-paranl*zion(5,m))*&
         !!!                 (wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
          enddo
       endif
    endif
 endif
 
end subroutine pushi_4
subroutine pushi_5(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_5
  use particle_array_5
  use particle_decomp_5
  use field_array_5
  use diagnosis_array_5
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,im,ip,j,jt,k,kz,kk,m,ii,j11,j10,j01,j00,jtheta,num,ierror,&
       larmor,ij,ipjt
  real(8) cost0,sint0,cost,sint,rdum,tdum,dtime,q,r,b,g,gp,ri,rip,dbdp,dbdt,&
       dedb,deni,upara,energy,kappa,dptdp,dptdt,dptdz,vdr,pdot,tdot,zdot,rdot,&
       wdot,wdrift,vdrenergy,delr,pi2,pi2_inv,uleft,uright,adum(0:mpsi),&
       dmark(mflux),pstat,xdum,ydum,xdot,ydot,tem_inv,wz1,wz0,wp1,wp0,wt11,&
       wt10,wt01,wt00,temp(0:mpsi),dtemp(0:mpsi),denp(0:mpsi),g2,tem,rfac,&
       sbound,qdum,tflr,d_inv,epara,wpara,psimax,psimin,paxis,pdum,wdrive,&
       ainv,rinv,qinv,psitmp,thetatmp,zetatmp,rhoi,e1,e2,e3,cmratio,cinv,&
       wpi(3,mi),rdel,vthi,dtem(mflux),dden(mflux),ddentmp(mflux),&
       dtemtmp(mflux),vdrtmp(mflux),dmarktmp(mflux),hfluxpsitmp(0:mpsi)
  real(8) epsi,etheta,ezeta,en_field(3),zdum,drdp,grad_zeta,jacobian,dphi_square(3)
  real(8) sum_of_weights,total_sum_of_weights,residual_weight
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
! 1st step of runge-kutta method
     dtime=0.5*tstep
!$omp parallel do private(m)
     do m=1,mi
        zion0(1:5,m)=zion(1:5,m)
     enddo

     vdrtmp=0.0

! 2nd step of runge-kutta method
  else
     dtime=tstep

     if(nonlinear<0.5)vdrtmp=0.0
     if(nonlinear>0.5)vdrtmp=pfluxpsi
  endif

! gather e_field using 4-point gyro-averaging, sorting in poloidal angle
! the following line is an openmp directive for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
!$omp parallel do private(m,e1,e2,e3,kk,wz1,wz0,larmor,ij,wp0,wt00,&
!$omp& wt10,wp1,wt01,wt11)
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

! update gc position
! another openmp parallel loop...
!$omp parallel do private(m,r,rinv,ii,ip,wp0,wp1,tem,q,qinv,cost,sint,cost0,&
!$omp& sint0,b,g,gp,ri,rip,dbdp,dbdt,dedb,deni,upara,energy,rfac,kappa,dptdp,&
!$omp& dptdt,dptdz,epara,vdr,wdrive,wpara,wdrift,wdot,pdot,tdot,zdot,rdot)
  do m=1,mi
     r=sqrt(2.0d0*zion(1,m))
     rinv=1.0d0/r
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

! exb drift in radial direction for w-dot and flux diagnostics
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
     zion(1,m) = max(1.0e-8*psimax,zion0(1,m)+dtime*pdot)
     zion(2,m) = zion0(2,m)+dtime*tdot
!     endif

     zion(3,m) = zion0(3,m)+dtime*zdot
     zion(4,m) = zion0(4,m)+dtime*rdot
     zion(5,m) = zion0(5,m)+dtime*wdot
     
! theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
! procedure on seaborg. however, modulo works better and is preferable.
!!!     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
!!!     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
!!!     zion(3,m)=zion(3,m)*pi2_inv+10.0
!!!     zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m)))

     zion(2,m)=modulo(zion(2,m),pi2)
     zion(3,m)=modulo(zion(3,m),pi2)

! update: 02/10/2006  the modulo function now streams on the x1e
! 02/20/2004  the modulo function seems to prevent streaming on the x1
! 02/20/2004 mod() does the same thing as modulo but it streams.
! 02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
!  to catch the negative values of zion.
!     zion(2,m)=mod((pi2+zion(2,m)),pi2)
!     zion(3,m)=mod((pi2+zion(3,m)),pi2)

! store gc information for flux measurements
     wpi(1,m)=vdr*rinv
     wpi(2,m)=energy
     wpi(3,m)=b
 enddo 

 if(irk==2)then
  ! avoid runaway particles by limiting absolute value of parallel velocity
  ! if parameter limit_vpara=1
    if(limit_vpara==1)then
       do m=1,mi
          r=sqrt(2.0d0*zion(1,m))
          b=1.0d0/(1.0d0+r*cos(zion(2,m)))
          upara=zion(4,m)*b*cmratio
          if(abs(upara) > uright)then
             zion(4,m)=sign(1.0d0,upara)*uright/(b*cmratio)
             !wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          endif
       enddo
    endif

! out of boundary particle
!$omp parallel do private(m)
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
       !call mpi_allreduce(sum_of_weights,total_sum_of_weights,1,mpi_rsize,&
       !     mpi_sum,mpi_comm_world,ierror)
       call comm_allreduce_cmm(ptr,sum_of_weights,total_sum_of_weights,1);
       !call mpi_allreduce(mi,mi_total,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
       !residual_weight=total_sum_of_weights/real(mi_total,wp)
       call int_comm_allreduce_cmm(ptr,mi,mi_total,one);
       if(mype==0)write(49,*)istep,mi_total,total_sum_of_weights,residual_weight
       do m=1,mi
          zion(5,m)=zion(5,m)-residual_weight
       enddo
    endif

! restore temperature profile when running a nonlinear calculation
! (nonlinear=1.0) and parameter fixed_tprofile > 0.
    if(nonlinear > 0.5 .and. fixed_tprofile > 0)then
       if(mod(istep,ndiag)==0)then
          dtem=0.0
          dden=0.0
!$omp parallel do private(m,cost,b,upara)
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
! s.ethier 06/04/03  according to the mpi standard, the send and receive
! buffers cannot be the same for mpi_reduce or mpi_allreduce. it generates
! an error on the linux platform. we thus make sure that the 2 buffers
! are different.
          !call mpi_allreduce(dtem,dtemtmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dtem,dtemtmp,mflux);
          !call mpi_allreduce(dden,ddentmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dden,ddentmp,mflux);
          dtem=dtemtmp*tem_inv/max(1.0d0,ddentmp) !perturbed temperature
          tdum=0.01*real(ndiag)
          rdtemi=(1.0-tdum)*rdtemi+tdum*dtem

!$omp parallel do private(m,ip)
          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             zion(5,m)=zion(5,m)-(wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
         ! correction to include paranl effect according to weixing
         ! 08/27/04 the correction may not be necessary according to jerome
         !          because the restoration uses delta_f/f0
         !!!    zion(5,m)=zion(5,m)-(1.0-paranl*zion(5,m))*&
         !!!                 (wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
          enddo
       endif
    endif
 endif
 
end subroutine pushi_5
subroutine pushi_6(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_6
  use particle_array_6
  use particle_decomp_6
  use field_array_6
  use diagnosis_array_6
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,im,ip,j,jt,k,kz,kk,m,ii,j11,j10,j01,j00,jtheta,num,ierror,&
       larmor,ij,ipjt
  real(8) cost0,sint0,cost,sint,rdum,tdum,dtime,q,r,b,g,gp,ri,rip,dbdp,dbdt,&
       dedb,deni,upara,energy,kappa,dptdp,dptdt,dptdz,vdr,pdot,tdot,zdot,rdot,&
       wdot,wdrift,vdrenergy,delr,pi2,pi2_inv,uleft,uright,adum(0:mpsi),&
       dmark(mflux),pstat,xdum,ydum,xdot,ydot,tem_inv,wz1,wz0,wp1,wp0,wt11,&
       wt10,wt01,wt00,temp(0:mpsi),dtemp(0:mpsi),denp(0:mpsi),g2,tem,rfac,&
       sbound,qdum,tflr,d_inv,epara,wpara,psimax,psimin,paxis,pdum,wdrive,&
       ainv,rinv,qinv,psitmp,thetatmp,zetatmp,rhoi,e1,e2,e3,cmratio,cinv,&
       wpi(3,mi),rdel,vthi,dtem(mflux),dden(mflux),ddentmp(mflux),&
       dtemtmp(mflux),vdrtmp(mflux),dmarktmp(mflux),hfluxpsitmp(0:mpsi)
  real(8) epsi,etheta,ezeta,en_field(3),zdum,drdp,grad_zeta,jacobian,dphi_square(3)
  real(8) sum_of_weights,total_sum_of_weights,residual_weight
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
! 1st step of runge-kutta method
     dtime=0.5*tstep
!$omp parallel do private(m)
     do m=1,mi
        zion0(1:5,m)=zion(1:5,m)
     enddo

     vdrtmp=0.0

! 2nd step of runge-kutta method
  else
     dtime=tstep

     if(nonlinear<0.5)vdrtmp=0.0
     if(nonlinear>0.5)vdrtmp=pfluxpsi
  endif

! gather e_field using 4-point gyro-averaging, sorting in poloidal angle
! the following line is an openmp directive for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
!$omp parallel do private(m,e1,e2,e3,kk,wz1,wz0,larmor,ij,wp0,wt00,&
!$omp& wt10,wp1,wt01,wt11)
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

! update gc position
! another openmp parallel loop...
!$omp parallel do private(m,r,rinv,ii,ip,wp0,wp1,tem,q,qinv,cost,sint,cost0,&
!$omp& sint0,b,g,gp,ri,rip,dbdp,dbdt,dedb,deni,upara,energy,rfac,kappa,dptdp,&
!$omp& dptdt,dptdz,epara,vdr,wdrive,wpara,wdrift,wdot,pdot,tdot,zdot,rdot)
  do m=1,mi
     r=sqrt(2.0d0*zion(1,m))
     rinv=1.0d0/r
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

! exb drift in radial direction for w-dot and flux diagnostics
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
     zion(1,m) = max(1.0e-8*psimax,zion0(1,m)+dtime*pdot)
     zion(2,m) = zion0(2,m)+dtime*tdot
!     endif

     zion(3,m) = zion0(3,m)+dtime*zdot
     zion(4,m) = zion0(4,m)+dtime*rdot
     zion(5,m) = zion0(5,m)+dtime*wdot
     
! theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
! procedure on seaborg. however, modulo works better and is preferable.
!!!     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
!!!     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
!!!     zion(3,m)=zion(3,m)*pi2_inv+10.0
!!!     zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m)))

     zion(2,m)=modulo(zion(2,m),pi2)
     zion(3,m)=modulo(zion(3,m),pi2)

! update: 02/10/2006  the modulo function now streams on the x1e
! 02/20/2004  the modulo function seems to prevent streaming on the x1
! 02/20/2004 mod() does the same thing as modulo but it streams.
! 02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
!  to catch the negative values of zion.
!     zion(2,m)=mod((pi2+zion(2,m)),pi2)
!     zion(3,m)=mod((pi2+zion(3,m)),pi2)

! store gc information for flux measurements
     wpi(1,m)=vdr*rinv
     wpi(2,m)=energy
     wpi(3,m)=b
 enddo 

 if(irk==2)then
  ! avoid runaway particles by limiting absolute value of parallel velocity
  ! if parameter limit_vpara=1
    if(limit_vpara==1)then
       do m=1,mi
          r=sqrt(2.0d0*zion(1,m))
          b=1.0d0/(1.0d0+r*cos(zion(2,m)))
          upara=zion(4,m)*b*cmratio
          if(abs(upara) > uright)then
             zion(4,m)=sign(1.0d0,upara)*uright/(b*cmratio)
             !wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          endif
       enddo
    endif

! out of boundary particle
!$omp parallel do private(m)
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
       !call mpi_allreduce(sum_of_weights,total_sum_of_weights,1,mpi_rsize,&
       !     mpi_sum,mpi_comm_world,ierror)
       call comm_allreduce_cmm(ptr,sum_of_weights,total_sum_of_weights,1);
       !call mpi_allreduce(mi,mi_total,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
       !residual_weight=total_sum_of_weights/real(mi_total,wp)
       call int_comm_allreduce_cmm(ptr,mi,mi_total,one);
       if(mype==0)write(49,*)istep,mi_total,total_sum_of_weights,residual_weight
       do m=1,mi
          zion(5,m)=zion(5,m)-residual_weight
       enddo
    endif

! restore temperature profile when running a nonlinear calculation
! (nonlinear=1.0) and parameter fixed_tprofile > 0.
    if(nonlinear > 0.5 .and. fixed_tprofile > 0)then
       if(mod(istep,ndiag)==0)then
          dtem=0.0
          dden=0.0
!$omp parallel do private(m,cost,b,upara)
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
! s.ethier 06/04/03  according to the mpi standard, the send and receive
! buffers cannot be the same for mpi_reduce or mpi_allreduce. it generates
! an error on the linux platform. we thus make sure that the 2 buffers
! are different.
          !call mpi_allreduce(dtem,dtemtmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dtem,dtemtmp,mflux);
          !call mpi_allreduce(dden,ddentmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dden,ddentmp,mflux);
          dtem=dtemtmp*tem_inv/max(1.0d0,ddentmp) !perturbed temperature
          tdum=0.01*real(ndiag)
          rdtemi=(1.0-tdum)*rdtemi+tdum*dtem

!$omp parallel do private(m,ip)
          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             zion(5,m)=zion(5,m)-(wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
         ! correction to include paranl effect according to weixing
         ! 08/27/04 the correction may not be necessary according to jerome
         !          because the restoration uses delta_f/f0
         !!!    zion(5,m)=zion(5,m)-(1.0-paranl*zion(5,m))*&
         !!!                 (wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
          enddo
       endif
    endif
 endif
 
end subroutine pushi_6
subroutine pushi_7(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_7
  use particle_array_7
  use particle_decomp_7
  use field_array_7
  use diagnosis_array_7
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,im,ip,j,jt,k,kz,kk,m,ii,j11,j10,j01,j00,jtheta,num,ierror,&
       larmor,ij,ipjt
  real(8) cost0,sint0,cost,sint,rdum,tdum,dtime,q,r,b,g,gp,ri,rip,dbdp,dbdt,&
       dedb,deni,upara,energy,kappa,dptdp,dptdt,dptdz,vdr,pdot,tdot,zdot,rdot,&
       wdot,wdrift,vdrenergy,delr,pi2,pi2_inv,uleft,uright,adum(0:mpsi),&
       dmark(mflux),pstat,xdum,ydum,xdot,ydot,tem_inv,wz1,wz0,wp1,wp0,wt11,&
       wt10,wt01,wt00,temp(0:mpsi),dtemp(0:mpsi),denp(0:mpsi),g2,tem,rfac,&
       sbound,qdum,tflr,d_inv,epara,wpara,psimax,psimin,paxis,pdum,wdrive,&
       ainv,rinv,qinv,psitmp,thetatmp,zetatmp,rhoi,e1,e2,e3,cmratio,cinv,&
       wpi(3,mi),rdel,vthi,dtem(mflux),dden(mflux),ddentmp(mflux),&
       dtemtmp(mflux),vdrtmp(mflux),dmarktmp(mflux),hfluxpsitmp(0:mpsi)
  real(8) epsi,etheta,ezeta,en_field(3),zdum,drdp,grad_zeta,jacobian,dphi_square(3)
  real(8) sum_of_weights,total_sum_of_weights,residual_weight
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
! 1st step of runge-kutta method
     dtime=0.5*tstep
!$omp parallel do private(m)
     do m=1,mi
        zion0(1:5,m)=zion(1:5,m)
     enddo

     vdrtmp=0.0

! 2nd step of runge-kutta method
  else
     dtime=tstep

     if(nonlinear<0.5)vdrtmp=0.0
     if(nonlinear>0.5)vdrtmp=pfluxpsi
  endif

! gather e_field using 4-point gyro-averaging, sorting in poloidal angle
! the following line is an openmp directive for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
!$omp parallel do private(m,e1,e2,e3,kk,wz1,wz0,larmor,ij,wp0,wt00,&
!$omp& wt10,wp1,wt01,wt11)
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

! update gc position
! another openmp parallel loop...
!$omp parallel do private(m,r,rinv,ii,ip,wp0,wp1,tem,q,qinv,cost,sint,cost0,&
!$omp& sint0,b,g,gp,ri,rip,dbdp,dbdt,dedb,deni,upara,energy,rfac,kappa,dptdp,&
!$omp& dptdt,dptdz,epara,vdr,wdrive,wpara,wdrift,wdot,pdot,tdot,zdot,rdot)
  do m=1,mi
     r=sqrt(2.0d0*zion(1,m))
     rinv=1.0d0/r
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

! exb drift in radial direction for w-dot and flux diagnostics
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
     zion(1,m) = max(1.0e-8*psimax,zion0(1,m)+dtime*pdot)
     zion(2,m) = zion0(2,m)+dtime*tdot
!     endif

     zion(3,m) = zion0(3,m)+dtime*zdot
     zion(4,m) = zion0(4,m)+dtime*rdot
     zion(5,m) = zion0(5,m)+dtime*wdot
     
! theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
! procedure on seaborg. however, modulo works better and is preferable.
!!!     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
!!!     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
!!!     zion(3,m)=zion(3,m)*pi2_inv+10.0
!!!     zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m)))

     zion(2,m)=modulo(zion(2,m),pi2)
     zion(3,m)=modulo(zion(3,m),pi2)

! update: 02/10/2006  the modulo function now streams on the x1e
! 02/20/2004  the modulo function seems to prevent streaming on the x1
! 02/20/2004 mod() does the same thing as modulo but it streams.
! 02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
!  to catch the negative values of zion.
!     zion(2,m)=mod((pi2+zion(2,m)),pi2)
!     zion(3,m)=mod((pi2+zion(3,m)),pi2)

! store gc information for flux measurements
     wpi(1,m)=vdr*rinv
     wpi(2,m)=energy
     wpi(3,m)=b
 enddo 

 if(irk==2)then
  ! avoid runaway particles by limiting absolute value of parallel velocity
  ! if parameter limit_vpara=1
    if(limit_vpara==1)then
       do m=1,mi
          r=sqrt(2.0d0*zion(1,m))
          b=1.0d0/(1.0d0+r*cos(zion(2,m)))
          upara=zion(4,m)*b*cmratio
          if(abs(upara) > uright)then
             zion(4,m)=sign(1.0d0,upara)*uright/(b*cmratio)
             !wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          endif
       enddo
    endif

! out of boundary particle
!$omp parallel do private(m)
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
       !call mpi_allreduce(sum_of_weights,total_sum_of_weights,1,mpi_rsize,&
       !     mpi_sum,mpi_comm_world,ierror)
       call comm_allreduce_cmm(ptr,sum_of_weights,total_sum_of_weights,1);
       !call mpi_allreduce(mi,mi_total,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
       !residual_weight=total_sum_of_weights/real(mi_total,wp)
       call int_comm_allreduce_cmm(ptr,mi,mi_total,one);
       if(mype==0)write(49,*)istep,mi_total,total_sum_of_weights,residual_weight
       do m=1,mi
          zion(5,m)=zion(5,m)-residual_weight
       enddo
    endif

! restore temperature profile when running a nonlinear calculation
! (nonlinear=1.0) and parameter fixed_tprofile > 0.
    if(nonlinear > 0.5 .and. fixed_tprofile > 0)then
       if(mod(istep,ndiag)==0)then
          dtem=0.0
          dden=0.0
!$omp parallel do private(m,cost,b,upara)
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
! s.ethier 06/04/03  according to the mpi standard, the send and receive
! buffers cannot be the same for mpi_reduce or mpi_allreduce. it generates
! an error on the linux platform. we thus make sure that the 2 buffers
! are different.
          !call mpi_allreduce(dtem,dtemtmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dtem,dtemtmp,mflux);
          !call mpi_allreduce(dden,ddentmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dden,ddentmp,mflux);
          dtem=dtemtmp*tem_inv/max(1.0d0,ddentmp) !perturbed temperature
          tdum=0.01*real(ndiag)
          rdtemi=(1.0-tdum)*rdtemi+tdum*dtem

!$omp parallel do private(m,ip)
          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             zion(5,m)=zion(5,m)-(wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
         ! correction to include paranl effect according to weixing
         ! 08/27/04 the correction may not be necessary according to jerome
         !          because the restoration uses delta_f/f0
         !!!    zion(5,m)=zion(5,m)-(1.0-paranl*zion(5,m))*&
         !!!                 (wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
          enddo
       endif
    endif
 endif
 
end subroutine pushi_7
subroutine pushi_8(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_8
  use particle_array_8
  use particle_decomp_8
  use field_array_8
  use diagnosis_array_8
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,im,ip,j,jt,k,kz,kk,m,ii,j11,j10,j01,j00,jtheta,num,ierror,&
       larmor,ij,ipjt
  real(8) cost0,sint0,cost,sint,rdum,tdum,dtime,q,r,b,g,gp,ri,rip,dbdp,dbdt,&
       dedb,deni,upara,energy,kappa,dptdp,dptdt,dptdz,vdr,pdot,tdot,zdot,rdot,&
       wdot,wdrift,vdrenergy,delr,pi2,pi2_inv,uleft,uright,adum(0:mpsi),&
       dmark(mflux),pstat,xdum,ydum,xdot,ydot,tem_inv,wz1,wz0,wp1,wp0,wt11,&
       wt10,wt01,wt00,temp(0:mpsi),dtemp(0:mpsi),denp(0:mpsi),g2,tem,rfac,&
       sbound,qdum,tflr,d_inv,epara,wpara,psimax,psimin,paxis,pdum,wdrive,&
       ainv,rinv,qinv,psitmp,thetatmp,zetatmp,rhoi,e1,e2,e3,cmratio,cinv,&
       wpi(3,mi),rdel,vthi,dtem(mflux),dden(mflux),ddentmp(mflux),&
       dtemtmp(mflux),vdrtmp(mflux),dmarktmp(mflux),hfluxpsitmp(0:mpsi)
  real(8) epsi,etheta,ezeta,en_field(3),zdum,drdp,grad_zeta,jacobian,dphi_square(3)
  real(8) sum_of_weights,total_sum_of_weights,residual_weight
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
! 1st step of runge-kutta method
     dtime=0.5*tstep
!$omp parallel do private(m)
     do m=1,mi
        zion0(1:5,m)=zion(1:5,m)
     enddo

     vdrtmp=0.0

! 2nd step of runge-kutta method
  else
     dtime=tstep

     if(nonlinear<0.5)vdrtmp=0.0
     if(nonlinear>0.5)vdrtmp=pfluxpsi
  endif

! gather e_field using 4-point gyro-averaging, sorting in poloidal angle
! the following line is an openmp directive for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
!$omp parallel do private(m,e1,e2,e3,kk,wz1,wz0,larmor,ij,wp0,wt00,&
!$omp& wt10,wp1,wt01,wt11)
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

! update gc position
! another openmp parallel loop...
!$omp parallel do private(m,r,rinv,ii,ip,wp0,wp1,tem,q,qinv,cost,sint,cost0,&
!$omp& sint0,b,g,gp,ri,rip,dbdp,dbdt,dedb,deni,upara,energy,rfac,kappa,dptdp,&
!$omp& dptdt,dptdz,epara,vdr,wdrive,wpara,wdrift,wdot,pdot,tdot,zdot,rdot)
  do m=1,mi
     r=sqrt(2.0d0*zion(1,m))
     rinv=1.0d0/r
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

! exb drift in radial direction for w-dot and flux diagnostics
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
     zion(1,m) = max(1.0e-8*psimax,zion0(1,m)+dtime*pdot)
     zion(2,m) = zion0(2,m)+dtime*tdot
!     endif

     zion(3,m) = zion0(3,m)+dtime*zdot
     zion(4,m) = zion0(4,m)+dtime*rdot
     zion(5,m) = zion0(5,m)+dtime*wdot
     
! theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
! procedure on seaborg. however, modulo works better and is preferable.
!!!     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
!!!     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
!!!     zion(3,m)=zion(3,m)*pi2_inv+10.0
!!!     zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m)))

     zion(2,m)=modulo(zion(2,m),pi2)
     zion(3,m)=modulo(zion(3,m),pi2)

! update: 02/10/2006  the modulo function now streams on the x1e
! 02/20/2004  the modulo function seems to prevent streaming on the x1
! 02/20/2004 mod() does the same thing as modulo but it streams.
! 02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
!  to catch the negative values of zion.
!     zion(2,m)=mod((pi2+zion(2,m)),pi2)
!     zion(3,m)=mod((pi2+zion(3,m)),pi2)

! store gc information for flux measurements
     wpi(1,m)=vdr*rinv
     wpi(2,m)=energy
     wpi(3,m)=b
 enddo 

 if(irk==2)then
  ! avoid runaway particles by limiting absolute value of parallel velocity
  ! if parameter limit_vpara=1
    if(limit_vpara==1)then
       do m=1,mi
          r=sqrt(2.0d0*zion(1,m))
          b=1.0d0/(1.0d0+r*cos(zion(2,m)))
          upara=zion(4,m)*b*cmratio
          if(abs(upara) > uright)then
             zion(4,m)=sign(1.0d0,upara)*uright/(b*cmratio)
             !wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          endif
       enddo
    endif

! out of boundary particle
!$omp parallel do private(m)
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
       !call mpi_allreduce(sum_of_weights,total_sum_of_weights,1,mpi_rsize,&
       !     mpi_sum,mpi_comm_world,ierror)
       call comm_allreduce_cmm(ptr,sum_of_weights,total_sum_of_weights,1);
       !call mpi_allreduce(mi,mi_total,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
       !residual_weight=total_sum_of_weights/real(mi_total,wp)
       call int_comm_allreduce_cmm(ptr,mi,mi_total,one);
       if(mype==0)write(49,*)istep,mi_total,total_sum_of_weights,residual_weight
       do m=1,mi
          zion(5,m)=zion(5,m)-residual_weight
       enddo
    endif

! restore temperature profile when running a nonlinear calculation
! (nonlinear=1.0) and parameter fixed_tprofile > 0.
    if(nonlinear > 0.5 .and. fixed_tprofile > 0)then
       if(mod(istep,ndiag)==0)then
          dtem=0.0
          dden=0.0
!$omp parallel do private(m,cost,b,upara)
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
! s.ethier 06/04/03  according to the mpi standard, the send and receive
! buffers cannot be the same for mpi_reduce or mpi_allreduce. it generates
! an error on the linux platform. we thus make sure that the 2 buffers
! are different.
          !call mpi_allreduce(dtem,dtemtmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dtem,dtemtmp,mflux);
          !call mpi_allreduce(dden,ddentmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dden,ddentmp,mflux);
          dtem=dtemtmp*tem_inv/max(1.0d0,ddentmp) !perturbed temperature
          tdum=0.01*real(ndiag)
          rdtemi=(1.0-tdum)*rdtemi+tdum*dtem

!$omp parallel do private(m,ip)
          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             zion(5,m)=zion(5,m)-(wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
         ! correction to include paranl effect according to weixing
         ! 08/27/04 the correction may not be necessary according to jerome
         !          because the restoration uses delta_f/f0
         !!!    zion(5,m)=zion(5,m)-(1.0-paranl*zion(5,m))*&
         !!!                 (wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
          enddo
       endif
    endif
 endif
 
end subroutine pushi_8
subroutine pushi_9(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_9
  use particle_array_9
  use particle_decomp_9
  use field_array_9
  use diagnosis_array_9
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,im,ip,j,jt,k,kz,kk,m,ii,j11,j10,j01,j00,jtheta,num,ierror,&
       larmor,ij,ipjt
  real(8) cost0,sint0,cost,sint,rdum,tdum,dtime,q,r,b,g,gp,ri,rip,dbdp,dbdt,&
       dedb,deni,upara,energy,kappa,dptdp,dptdt,dptdz,vdr,pdot,tdot,zdot,rdot,&
       wdot,wdrift,vdrenergy,delr,pi2,pi2_inv,uleft,uright,adum(0:mpsi),&
       dmark(mflux),pstat,xdum,ydum,xdot,ydot,tem_inv,wz1,wz0,wp1,wp0,wt11,&
       wt10,wt01,wt00,temp(0:mpsi),dtemp(0:mpsi),denp(0:mpsi),g2,tem,rfac,&
       sbound,qdum,tflr,d_inv,epara,wpara,psimax,psimin,paxis,pdum,wdrive,&
       ainv,rinv,qinv,psitmp,thetatmp,zetatmp,rhoi,e1,e2,e3,cmratio,cinv,&
       wpi(3,mi),rdel,vthi,dtem(mflux),dden(mflux),ddentmp(mflux),&
       dtemtmp(mflux),vdrtmp(mflux),dmarktmp(mflux),hfluxpsitmp(0:mpsi)
  real(8) epsi,etheta,ezeta,en_field(3),zdum,drdp,grad_zeta,jacobian,dphi_square(3)
  real(8) sum_of_weights,total_sum_of_weights,residual_weight
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
! 1st step of runge-kutta method
     dtime=0.5*tstep
!$omp parallel do private(m)
     do m=1,mi
        zion0(1:5,m)=zion(1:5,m)
     enddo

     vdrtmp=0.0

! 2nd step of runge-kutta method
  else
     dtime=tstep

     if(nonlinear<0.5)vdrtmp=0.0
     if(nonlinear>0.5)vdrtmp=pfluxpsi
  endif

! gather e_field using 4-point gyro-averaging, sorting in poloidal angle
! the following line is an openmp directive for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
!$omp parallel do private(m,e1,e2,e3,kk,wz1,wz0,larmor,ij,wp0,wt00,&
!$omp& wt10,wp1,wt01,wt11)
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

! update gc position
! another openmp parallel loop...
!$omp parallel do private(m,r,rinv,ii,ip,wp0,wp1,tem,q,qinv,cost,sint,cost0,&
!$omp& sint0,b,g,gp,ri,rip,dbdp,dbdt,dedb,deni,upara,energy,rfac,kappa,dptdp,&
!$omp& dptdt,dptdz,epara,vdr,wdrive,wpara,wdrift,wdot,pdot,tdot,zdot,rdot)
  do m=1,mi
     r=sqrt(2.0d0*zion(1,m))
     rinv=1.0d0/r
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

! exb drift in radial direction for w-dot and flux diagnostics
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
     zion(1,m) = max(1.0e-8*psimax,zion0(1,m)+dtime*pdot)
     zion(2,m) = zion0(2,m)+dtime*tdot
!     endif

     zion(3,m) = zion0(3,m)+dtime*zdot
     zion(4,m) = zion0(4,m)+dtime*rdot
     zion(5,m) = zion0(5,m)+dtime*wdot
     
! theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
! procedure on seaborg. however, modulo works better and is preferable.
!!!     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
!!!     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
!!!     zion(3,m)=zion(3,m)*pi2_inv+10.0
!!!     zion(3,m)=2.0*pi*(zion(3,m)-aint(zion(3,m)))

     zion(2,m)=modulo(zion(2,m),pi2)
     zion(3,m)=modulo(zion(3,m),pi2)

! update: 02/10/2006  the modulo function now streams on the x1e
! 02/20/2004  the modulo function seems to prevent streaming on the x1
! 02/20/2004 mod() does the same thing as modulo but it streams.
! 02/23/2004 need to do mod((pi2+zion),pi2) instead of mod(zion,pi2) in order
!  to catch the negative values of zion.
!     zion(2,m)=mod((pi2+zion(2,m)),pi2)
!     zion(3,m)=mod((pi2+zion(3,m)),pi2)

! store gc information for flux measurements
     wpi(1,m)=vdr*rinv
     wpi(2,m)=energy
     wpi(3,m)=b
 enddo 

 if(irk==2)then
  ! avoid runaway particles by limiting absolute value of parallel velocity
  ! if parameter limit_vpara=1
    if(limit_vpara==1)then
       do m=1,mi
          r=sqrt(2.0d0*zion(1,m))
          b=1.0d0/(1.0d0+r*cos(zion(2,m)))
          upara=zion(4,m)*b*cmratio
          if(abs(upara) > uright)then
             zion(4,m)=sign(1.0d0,upara)*uright/(b*cmratio)
             !wpi(2,m)=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
          endif
       enddo
    endif

! out of boundary particle
!$omp parallel do private(m)
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
       !call mpi_allreduce(sum_of_weights,total_sum_of_weights,1,mpi_rsize,&
       !     mpi_sum,mpi_comm_world,ierror)
       call comm_allreduce_cmm(ptr,sum_of_weights,total_sum_of_weights,1);
       !call mpi_allreduce(mi,mi_total,1,mpi_integer,mpi_sum,mpi_comm_world,ierror)
       !residual_weight=total_sum_of_weights/real(mi_total,wp)
       call int_comm_allreduce_cmm(ptr,mi,mi_total,one);
       if(mype==0)write(49,*)istep,mi_total,total_sum_of_weights,residual_weight
       do m=1,mi
          zion(5,m)=zion(5,m)-residual_weight
       enddo
    endif

! restore temperature profile when running a nonlinear calculation
! (nonlinear=1.0) and parameter fixed_tprofile > 0.
    if(nonlinear > 0.5 .and. fixed_tprofile > 0)then
       if(mod(istep,ndiag)==0)then
          dtem=0.0
          dden=0.0
!$omp parallel do private(m,cost,b,upara)
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
! s.ethier 06/04/03  according to the mpi standard, the send and receive
! buffers cannot be the same for mpi_reduce or mpi_allreduce. it generates
! an error on the linux platform. we thus make sure that the 2 buffers
! are different.
          !call mpi_allreduce(dtem,dtemtmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dtem,dtemtmp,mflux);
          !call mpi_allreduce(dden,ddentmp,mflux,mpi_rsize,mpi_sum,mpi_comm_world,ierror)
          call comm_allreduce_cmm(ptr,dden,ddentmp,mflux);
          dtem=dtemtmp*tem_inv/max(1.0d0,ddentmp) !perturbed temperature
          tdum=0.01*real(ndiag)
          rdtemi=(1.0-tdum)*rdtemi+tdum*dtem

!$omp parallel do private(m,ip)
          do m=1,mi
             ip=max(1,min(mflux,1+int((wpi(1,m)-a0)*d_inv)))
             zion(5,m)=zion(5,m)-(wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
         ! correction to include paranl effect according to weixing
         ! 08/27/04 the correction may not be necessary according to jerome
         !          because the restoration uses delta_f/f0
         !!!    zion(5,m)=zion(5,m)-(1.0-paranl*zion(5,m))*&
         !!!                 (wpi(2,m)*tem_inv-1.5)*rdtemi(ip)
          enddo
       endif
    endif
 endif
 
end subroutine pushi_9
