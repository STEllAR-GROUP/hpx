subroutine collision(hpx4_bti,&
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
        zelectron0,zelectron1)

  !use global_parameters
  !use particle_array
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
  real(wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(wp),dimension(:),allocatable :: deltat
  logical  do_collision

! particle array
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(wp),dimension(:,:),allocatable :: zion,zion0,zelectron,&
        zelectron0,zelectron1

  integer,parameter :: neop=32,neot=1,neoz=1

  integer m,mcell(mi),ip,jt,kz,ierror,k,icount
  real(wp) delp,delt,delz,b,cost0,q,r,sint,v,xv,zv,vths,vth,vmin,freq,&
           phix,dphi,f,g,h,sp,sn,dp,dn,dpn,delu,delv2,delm(neop*neot*neoz),&
           dele(neop*neot*neoz),marker(neop*neot*neoz),v3,v5,zeff,&
           ddum(neop*neot*neoz),den_electron,tem_ion,aspecie,psimin,psimax
  real(wp) maxwell(100001)
  real(wp) :: r2psi
  
  save maxwell

! read maxwell.dat  
  if(istep==ndiag)then
! basic collision time, Braginskii
!     tauii=24.0-log(sqrt(0.46e14)/2555.0)
!     tauii=2.09e7*(2555.0)**1.5/(0.46e14*tauii*2.31)*sqrt(2.0)/utime
     !zeff=2.31
     zeff=1.00
     den_electron=0.46e14
     tem_ion=2500.0
     aspecie=1.0
     q=1.4
     r=0.5*(a0+a1)
! assume carbon impurity
  !!   tauii=23.0-log(sqrt(6.0*zeff*den_electron)/tem_ion**1.5)
  !!   tauii=2.09e7*(tem_ion)**1.5*sqrt(aspecie)/(den_electron*tauii*utime*zeff)

     if(mype==0)then
        !!write(stdout,*)'tau_Braginskii=',tauii,' mu_star=',q/(tauii*gyroradius*r**1.5)
        open(99,file='maxwell.dat',status='old')
        read(99,*)(maxwell(m),m=1,100001)
        close(99)
     endif
     !call MPI_BCAST(maxwell,100001,mpi_Rsize,0,MPI_COMM_WORLD,ierror)
     call broadcast_real_cmm(maxwell,100001)
  endif

  psimin=r2psi(a0)
  psimax=r2psi(a1)
  delp=real(neop)/(psimax-psimin)
  delt=real(neot)/(2.0*pi)
  delz=real(neoz)/(2.0*pi)
  do m=1,mi
     q=q0*exp(2.0*q1*zion(1,m)/(a*a))
!     r=a*sqrt(max(1.0d-4,(q-q0)/q1))
     r=a*sqrt(max(ONE_TEN_THOUSANDTH,(q-q0)/q1))
     sint=sin(zion(2,m))
     cost0=cos(zion(2,m)+r*sint)
     b=1.0/(1.0+r*cost0)

! use zion0(1,:) for v_para, zion0(2,:) for v_perp^2, zion0(3,:) for B-field
     zion0(3,m)=b
     zion0(1,m)=zion(4,m)*zion0(3,m)
     zion0(2,m)=2.0*zion(6,m)*zion(6,m)*zion0(3,m)

! use cell in psi for near equal volume
     ip=max(1,min(neop,ceiling((zion(1,m)-psimin)*delp)))
! use cell in theta0 for near equal volume
     jt=max(1,min(neot,ceiling(zion(2,m)*delt)))
! use cell in global zeta address
     kz=max(1,min(neoz,ceiling(zion(3,m)*delz)))
     mcell(m)=(kz-1)*neop*neot+(jt-1)*neop+ip
  enddo

  vth=2.0*gyroradius*gyroradius
  vths=1.0/sqrt(vth)
  vmin=1.0e-20*vth
  v3=1.0e-10*vth*gyroradius
  v5=vth*v3
  marker=0.0
  dele=0.0
  delm=0.0

! use zion0(4,1:mi), zion0(5,1:mi) to store random number
  call random_number(zion0(4,1:mi))
  call random_number(zion0(5,1:mi))

  do m=1,mi
     v=sqrt(max(vmin,zion0(1,m)*zion0(1,m)+zion0(2,m)))
!     zv=max(0.1d+00,min(10.0d+00,v*vths))
     zv=max(PT_ONE,min(TEN,v*vths))
     xv=zv*zv
     freq=1.88*real(ndiag)*tstep/(tauii*zv*xv)

! Maxwellian integral by table look-up for range of (0.0001,10.0)
     k = min(100000,int(xv*10000.0) + 1)
     phix = (real(k)-10000.0*xv)*maxwell(k) + (10000.0*xv-real(k-1))*maxwell(k+1)
     if(xv .lt. 0.025)phix=4.0/3.0*sqrt(xv/pi)*xv*(1.0-0.6*xv+3.0/14.0*xv*xv)
     if(xv .gt. 10.0)phix=1.0-2.0/exp(xv)*sqrt(xv/pi)*(1.0+1.0/(2.0*xv)-1.0/(4.0*xv*xv))
     dphi = 2.0*sqrt(xv/pi)/exp(xv)

! coefficients for like-species collisions
     f=2.0*phix
     g=(phix-0.5*phix/xv+dphi)
     h=phix/xv

     sp=zion0(1,m)*f
     sn=zion0(2,m)*(2.0*f-h-g)-2.0*zion0(1,m)*zion0(1,m)*g
     dp=max(v3,zion0(1,m)*zion0(1,m)*h+zion0(2,m)*g)
     dn=max(v5,4.0*zion0(2,m)*v*v*v*v*g*h/dp)
     dpn=2.0*zion0(2,m)*zion0(1,m)*(h-g)

! parallel and perpendicular drag and diffusion
     delu=(zion0(4,m)-0.5)*sqrt(12.0*dp*freq)-sp*freq
     delv2=dpn*(zion0(4,m)-0.5)*sqrt(12.0/dp*freq)-sn*freq+&
          (zion0(5,m)-0.5)*sqrt(12.0*dn*freq)

     if(delu*delu+abs(delv2) .gt. vth)then
        delu=0.0
        delv2=0.0
     endif

! momentum and energy changes due to collisions
     delm(mcell(m))=delm(mcell(m))+zion(5,m)*delu
     dele(mcell(m))=dele(mcell(m))+zion(5,m)*(delv2+delu*(2.0*zion0(1,m)+delu))
     zion0(1,m)=zion0(1,m)+delu
     zion0(2,m)=max(vmin,zion0(2,m)+delv2)

     marker(mcell(m))=marker(mcell(m))+1.0
  enddo

! global sum
  icount=neop*neot*neoz
  ddum=0._wp
  !call MPI_ALLREDUCE(marker,ddum,icount,mpi_Rsize,MPI_SUM,MPI_COMM_WORLD,ierror)
  call comm_allreduce_cmm(hpx4_bti,marker,ddum,icount)
! marker=max(1.0d+00,ddum)
  marker=max(ONE,ddum)

  ddum=0._wp
  !call MPI_ALLREDUCE(delm,ddum,icount,mpi_Rsize,MPI_SUM,MPI_COMM_WORLD,ierror)
  call comm_allreduce_cmm(hpx4_bti,delm,ddum,icount)
  delm=sqrt(4.5*pi)*2.0*delm/(vth*marker)

  ddum=0._wp
  !call MPI_ALLREDUCE(dele,ddum,icount,mpi_Rsize,MPI_SUM,MPI_COMM_WORLD,ierror)
  call comm_allreduce_cmm(hpx4_bti,dele,ddum,icount)
  dele=sqrt(4.5*pi)*dele/(1.5*vth*marker)
  dele=0.0

! local conservation
  do m=1,mi
     v=sqrt(max(vmin,zion0(1,m)*zion0(1,m)+zion0(2,m)))
!     zv=max(0.1d+00,min(10.0d+00,v*vths))
     zv=max(PT_ONE,min(TEN,v*vths))
     xv=zv*zv


     k = min(100000,int(xv*10000.0) + 1)
     phix = (k-10000.0*xv)*maxwell(k) + (10000.0*xv-k+1)*maxwell(k+1)
     if(xv .lt. 0.025)phix=4.0/3.0*sqrt(xv/pi)*xv*(1.0-0.6*xv+3.0/14.0*xv*xv)
     if(xv .gt. 10.0)phix=1.0-2.0/exp(xv)*sqrt(xv/pi)*(1.0+1.0/(2.0*xv)-1.0/(4.0*xv*xv))
     dphi = 2.0*sqrt(xv/pi)/exp(xv)

     zion(5,m)=zion(5,m)-phix/(xv*zv)*zion0(1,m)*delm(mcell(m))&
          -(phix-dphi)/zv*dele(mcell(m))
  enddo

! update velocity variables
  zion(4,1:mi)=zion0(1,1:mi)/zion0(3,1:mi)
  zion(6,1:mi)=sqrt(0.5*zion0(2,1:mi)/zion0(3,1:mi))

end subroutine collision
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
