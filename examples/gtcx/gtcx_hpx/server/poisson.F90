subroutine poisson(iflag,hpx4_bti,&
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
! particle decomp
              ntoroidal,npartdom,&
              partd_comm,nproc_partd,myrank_partd,&
              toroidal_comm,nproc_toroidal,myrank_toroidal,&
              left_pe,right_pe,&
              toroidal_domain_location,particle_domain_location&
                  )
  !use global_parameters
  !use field_array
  !use particle_decomp
  use, intrinsic :: iso_c_binding, only : c_ptr
  use precision
  implicit none
  interface ! {{{
    subroutine poisson_initial(mring,mindex,nindex,indexp,ring,&
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
! field array
                      mmpsi, &
                                      itran,igrid, &
                                          jtp1,jtp2,&
                                       phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt,&
                                         phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit,&
                                           evector,wtp1,wtp2,phisave,&
              Total_field_energy)
  !use global_parameters
  !use field_array
  use, intrinsic :: iso_c_binding, only : c_ptr
  use precision
  implicit none

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

  integer mring,mindex,ierr
  integer,dimension(mgrid,mzeta) :: nindex
  integer,dimension(mindex,mgrid,mzeta) :: indexp
  real(kind=wp),dimension(mindex,mgrid,mzeta) :: ring        
  end subroutine poisson_initial
  end interface ! }}}

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

! particle decomp
  integer  :: ntoroidal,npartdom
  integer  :: partd_comm,nproc_partd,myrank_partd
  integer  :: toroidal_comm,nproc_toroidal,myrank_toroidal
  integer  :: left_pe,right_pe
  integer  :: toroidal_domain_location,particle_domain_location

  integer iflag,i,it,ij,j,k,n,iteration,mring,mindex,mtest,ierr
  integer,dimension(:,:),allocatable :: nindex
  integer,dimension(:,:,:),allocatable :: indexp
  real(kind=wp),dimension(:,:,:),allocatable :: ring
  real(kind=wp) gamma,tmp,prms,perr(mgrid)
  real(kind=wp) ptilde(mgrid),phitmp(mgrid),dentmp(mgrid)

  integer :: ipartd,nzeta,izeta1,izeta2
  real(kind=wp),dimension(:),allocatable :: sendbuf,recvbuf
! hjw
  integer :: nmem
! hjw

  save nindex,indexp,ring

! number of gyro-ring
  mring=2

! number of summation: maximum is 32*mring+1
  mindex=32*mring+1
!  mindex=53

! gamma=0.75: max. resolution for k=0.577
  gamma=0.75
  iteration=5

! initialize poisson solver
  if(istep==1 .and. irk==1 .and. iflag==0)then
     allocate(indexp(mindex,mgrid,mzeta),ring(mindex,mgrid,mzeta),&
          nindex(mgrid,mzeta),STAT=mtest)
     if (mtest /= 0) then
! hjw
     nmem = (2*(mindex*mgrid*mzeta))+(mgrid*mzeta)
!     write(0,*)mype,'*** Cannot allocate indexp: mtest=',mtest
      write(0,*)mype,'*** indexp: Allocate Error: ',nmem, ' words mtest= ',mtest
! hjw
        !call MPI_ABORT(MPI_COMM_WORLD,1,ierr)
     endif


! initialize
     call poisson_initial(mring,mindex,nindex,indexp,ring,&
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
! field array
                      mmpsi, &
                                      itran,igrid, &
                                          jtp1,jtp2,&
                                       phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt,&
                                         phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit,&
                                           evector,wtp1,wtp2,phisave,&
              Total_field_energy)
  endif

  tmp=1.0/(tite+1.0-gamma)

  if(npartdom>1 .and. mod(mzeta,npartdom)==0)then
  ! mzeta is a multiple of npartdom so we can split the work of the
  ! k=1,mzeta loop between the processors having the same toroidal
  ! domain but a different particle domain.
  ! First we split mzeta into npartdom and determine the loop indices
  ! for each process part of the particle domain.
    ipartd=1
    nzeta=mzeta/npartdom
    izeta1=myrank_partd*nzeta+1
    izeta2=(myrank_partd+1)*nzeta
  else
    ipartd=0
    izeta1=1
    izeta2=mzeta
  endif

  !!!do k=1,mzeta
  do k=izeta1,izeta2

! first iteration, first guess of phi. (1+T_i/T_e) phi - phi_title = n_i
     if(iflag==0)then
!$omp parallel do private(i)
        do i=1,mgrid
           dentmp(i)=qion*densityi(k,i)
        enddo
     else
!$omp parallel do private(i)
        do i=1,mgrid
           dentmp(i)=qion*densityi(k,i)+qelectron*densitye(k,i)
        enddo
     endif

!$omp parallel do private(i)
     do i=1,mgrid
        phitmp(i)=dentmp(i)*tmp
     enddo

     do it=2,iteration

!$omp parallel do private(i,j)
        do i=1,mgrid
           ptilde(i)=0.0
           do j=1,nindex(i,k)
              ptilde(i)=ptilde(i)+ring(j,i,k)*phitmp(indexp(j,i,k))
           enddo
        enddo

!$omp parallel do private(i)
        do i=1,mgrid
           perr(i)=ptilde(i)-gamma*phitmp(i)
           phitmp(i)=(dentmp(i)+perr(i))*tmp
        enddo
!        prms=sum(perr*perr)/sum(phitmp*phitmp)
!        if(mype==0)write(*,*)istep,prms

! radial boundary
        phitmp(igrid(0):igrid(0)+mtheta(0))=0.0
        phitmp(igrid(mpsi):igrid(mpsi)+mtheta(mpsi))=0.0
     enddo

! store final results
!$omp parallel do private(i)
     do i=1,mgrid
        phi(k,i)=phitmp(i)
     enddo
  enddo

  if(ipartd==1)then
  ! Since the work was split between the processors of different particle
  ! domains, we need to gather the full array phi on these processors.
    allocate(sendbuf(nzeta*mgrid),recvbuf(mzeta*mgrid))
    do k=izeta1,izeta2
       do i=1,mgrid
          sendbuf((k-izeta1)*mgrid+i)=phi(k,i)
       enddo
    enddo
    !call MPI_ALLGATHER(sendbuf,nzeta*mgrid,mpi_Rsize,recvbuf,nzeta*mgrid,&
    !                   mpi_Rsize,partd_comm,ierr)
     call partd_allgather_cmm(hpx4_bti,sendbuf,recvbuf,nzeta*mgrid)
    do k=1,mzeta
       do i=1,mgrid
          phi(k,i)=recvbuf((k-1)*mgrid+i)
       enddo
    enddo
    deallocate(sendbuf,recvbuf)
  endif

! in equilibrium unit
!$omp parallel do private(i)
  do i=0,mpsi
     phi(1:mzeta,igrid(i)+1:igrid(i)+mtheta(i))=phi(1:mzeta,igrid(i)+1:igrid(i)+&
          mtheta(i))*rtemi(i)*(qion*gyroradius)**2/aion
! poloidal BC
     phi(1:mzeta,igrid(i))=phi(1:mzeta,igrid(i)+mtheta(i))
  enddo

end subroutine poisson
 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine poisson_initial(mring,mindex,nindex,indexp,ring,&
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
! field array
                      mmpsi, &
                                      itran,igrid, &
                                          jtp1,jtp2,&
                                       phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt,&
                                         phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit,&
                                           evector,wtp1,wtp2,phisave,&
              Total_field_energy)
  !use global_parameters
  !use field_array
  use, intrinsic :: iso_c_binding, only : c_ptr
  use precision
  implicit none

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

  integer mring,mindex,ierr
  integer,dimension(mgrid,mzeta) :: nindex
  integer,dimension(mindex,mgrid,mzeta) :: indexp
  real(kind=wp),dimension(mindex,mgrid,mzeta) :: ring        
  integer i,ii,ij,ij0,ipjt,j,jt,j1,j0,jm,k,kr,kp,nt,np,minn,maxn
  real(kind=wp) vring(3),fring(3),rgrid,tgrid,ddelr,ddelt,wght,r,rr,t,rdum,wr,&
       wt1,wt0,tdum,zdum,b,pi2_inv,delr,delt(0:mpsi)
  
  if(mring==1)then
! one ring, velocity in unit of gyroradius
     vring(1)=sqrt(2.0)
     fring(1)=1.0
     
  elseif(mring==2)then
! two rings good for up to k_perp=1.5
     vring(1)=0.9129713024553
     vring(2)=2.233935334042
     fring(1)=0.7193896325719
     fring(2)=0.2806103674281

  else      
! three rings: exact(<0.8%) for up to k_perp=1.5
     vring(1)=0.388479356715
     vring(2)=1.414213562373
     vring(3)=2.647840808818
     fring(1)=0.3043424333839
     fring(2)=0.5833550690524
     fring(3)=0.1123024975637
  endif

  pi2_inv=0.5/pi
  delr=1.0/deltar
  delt=2.0*pi/deltat
  nindex=0
  ring=0.0
  indexp=1
  do k=1,mzeta
     zdum=zetamin+deltaz*real(k)
!$omp parallel do private(i,j,kr,kp,np,nt,ij0,rgrid,tgrid,jt,b,ipjt,ddelr,&
!$omp& ddelt,wght,r,t,rdum,ii,wr,tdum,j1,wt1,j0,wt0,ij,rr)
     do i=0,mpsi
        do j=1,mtheta(i)
           ij0=igrid(i)+j

! 1st point is the original grid point
           nindex(ij0,k)=1
           indexp(1,ij0,k)=ij0
           ring(1,ij0,k)=0.25

! position of grid points
           rgrid=a0+deltar*real(i)
           tgrid=deltat(i)*real(j)+zdum*qtinv(i)
           tgrid=tgrid*pi2_inv
           tgrid=2.0*pi*(tgrid-aint(tgrid))
           jt=max(0,min(mtheta(i),int(pi2_inv*delt(i)*tgrid+0.5)))

! B-field
           b=1.0/(1.0+rgrid*cos(tgrid))
           ipjt=igrid(i)+jt

           do kr=1,mring

! FLR from grid point and weight of 8-point for each ring
              do kp=1,8
                 if(kp<5)then
                    ddelr=pgyro(kp,ipjt)
                    ddelt=tgyro(kp,ipjt)
                    wght=0.0625*fring(kr)
                 
                 elseif(kp==5)then
                    ddelr=0.5*(pgyro(1,ipjt)+pgyro(3,ipjt))
                    ddelt=0.5*(tgyro(1,ipjt)+tgyro(3,ipjt))
                    wght=0.125*fring(kr)
                 elseif(kp==6)then
                    ddelr=0.5*(pgyro(2,ipjt)+pgyro(3,ipjt))
                    ddelt=0.5*(tgyro(2,ipjt)+tgyro(3,ipjt))
                 elseif(kp==7)then
                    ddelr=0.5*(pgyro(2,ipjt)+pgyro(4,ipjt))
                    ddelt=0.5*(tgyro(2,ipjt)+tgyro(4,ipjt))
                 elseif(kp==8)then
                    ddelr=0.5*(pgyro(1,ipjt)+pgyro(4,ipjt))
                    ddelt=0.5*(tgyro(1,ipjt)+tgyro(4,ipjt))
                 endif

! position for each point with rho_i=2.0*vring
                 r=rgrid+ddelr*2.0*vring(kr)*sqrt(0.5/b)
                 t=tgrid+ddelt*2.0*vring(kr)*sqrt(0.5/b)

! linear interpolation
                 rdum=delr*max(0.0,min(a1-a0,r-a0))
                 ii=max(0,min(mpsi-1,int(rdum)))
                 wr=rdum-real(ii)
                 if(wr>0.95)wr=1.0
                 if(wr<0.05)wr=0.0

! outer flux surface
                 tdum=t-zdum*qtinv(ii+1)
                 tdum=tdum*pi2_inv+10.0
                 tdum=delt(ii+1)*(tdum-aint(tdum))
                 j1=max(0,min(mtheta(ii+1)-1,int(tdum)))
                 wt1=tdum-real(j1)
                 if(wt1>0.95)wt1=1.0
                 if(wt1<0.05)wt1=0.0

! inner flux surface
                 tdum=t-zdum*qtinv(ii)
                 tdum=tdum*pi2_inv+10.0
                 tdum=delt(ii)*(tdum-aint(tdum))
                 j0=max(0,min(mtheta(ii)-1,int(tdum)))
                 wt0=tdum-real(j0)
                 if(wt0>0.95)wt0=1.0
                 if(wt0<0.05)wt0=0.0

! index and weight of each point
                 do np=1,4
                    if(np==1)then
                       ij=igrid(ii+1)+j1+1
                       rr=wght*wr*wt1
                    elseif(np==2)then
                       if(j1==0)j1=mtheta(ii+1)
                       ij=igrid(ii+1)+j1
                       rr=wght*wr*(1.0-wt1)
                    elseif(np==3)then
                       ij=igrid(ii)+j0+1
                       rr=wght*(1.0-wr)*wt0
                    else
                       if(j0==0)j0=mtheta(ii)
                       ij=igrid(ii)+j0
                       rr=wght*(1.0-wr)*(1.0-wt0)
                    endif

! insignificant point replaced by the original grid point
                    if(rr<0.001)then
                       ring(1,ij0,k)=ring(1,ij0,k)+rr
                       goto 100
                    endif

                    do nt=1,nindex(ij0,k)
! redundant point
                       if(ij==indexp(nt,ij0,k))then
                          ring(nt,ij0,k)=ring(nt,ij0,k)+rr
                          goto 100
                       endif
                    enddo
! new point
                    nindex(ij0,k)=nindex(ij0,k)+1
                    nt=nindex(ij0,k)
                    indexp(nt,ij0,k)=ij
                    ring(  nt,ij0,k)=rr
                 
100                 continue
                 enddo  !end of 4-point interpolation loop
              enddo     !end of 8-point-per-ring loop
           enddo        !end of ring loop
        enddo           !end of poloidal loop
     enddo              !end of radial loop     
  enddo                 !end of toroidal loop

! check array size
  if(maxval(nindex)>mindex)then
     write(0,*)'Poisson error',mype,maxval(nindex),' > ',mindex
     !call MPI_ABORT(MPI_COMM_WORLD,1,ierr)
  endif

  rdum=0.0
  tdum=0.0
  zdum=1.0
  do k=1,mzeta
     do i=1,mgrid
        rdum=sum(ring(1:nindex(i,k),i,k))
        tdum=max(tdum,rdum)
        zdum=min(zdum,rdum)
     enddo
  enddo
  if(mype==0)write(stdout,*)'poisson solver=',maxval(nindex),minval(nindex),&
       tdum,zdum,mgrid,sum(nindex)
  
end subroutine poisson_initial

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



