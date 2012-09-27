subroutine chargei_0(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_0
  use particle_array_0
  use field_array_0
  use diagnosis_array_0
  use particle_decomp_0
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer m,i,im,j,k,ip,jt,kk,ii,j11,j10,j01,j00,ierror,larmor,ij,ipjt,&
       icount,idest,isource,isendtag,irecvtag !,istatus(mpi_status_size)
  real(8) weight,wemark,rdum,tdum,delr,delt(0:mpsi),delz,smu_inv,r,wz1,wz0,&
       wp1,wp0,wt11,wt10,wt01,wt00,adum(0:mpsi),tflr,damping,pi2_inv,&
       psitmp,thetatmp,zetatmp,rhoi,deltheta
  real(8) sendl(mgrid),recvr(mgrid)
  real(8) dnitmp(0:mzeta,mgrid)

#ifdef _openmp
! we need the following temporary array only when using openmp
!!!  real,dimension(:,:),allocatable :: dnitmp
#endif

  delr=1.0/deltar 
  delt=2.0*pi/deltat
  delz=1.0/deltaz
  smu_inv=sqrt(aion)/(abs(qion)*gyroradius)
  pi2_inv=0.5/pi
  densityi=0.0

!$omp parallel do private(m,larmor,psitmp,thetatmp,zetatmp,rhoi,r,ip,jt,ipjt,&
!$omp& wz1,kk,rdum,ii,wp1,tflr,im,tdum,j00,j01)
  do m=1,mi
     psitmp=zion(1,m)
     thetatmp=zion(2,m)
     zetatmp=zion(3,m)
     rhoi=zion(6,m)*smu_inv

     r=sqrt(2.0*psitmp)
     ip=max(0,min(mpsi,int((r-a0)*delr+0.5)))
     jt=max(0,min(mtheta(ip),int(thetatmp*pi2_inv*delt(ip)+0.5)))
     ipjt=igrid(ip)+jt
     wz1=(zetatmp-zetamin)*delz
     kk=max(0,min(mzeta-1,int(wz1)))
     kzion(m)=kk
     wzion(m)=wz1-real(kk)

     do larmor=1,4
!        rdum=delr*max(0.0d+00,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        rdum=delr*max(zero,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        ii=max(0,min(mpsi-1,int(rdum)))
        wp1=rdum-real(ii)
        wpion(larmor,m)=wp1

! particle position in theta
        tflr=thetatmp+rhoi*tgyro(larmor,ipjt)

! inner flux surface
        im=ii
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j00=max(0,min(mtheta(im)-1,int(tdum)))
        jtion0(larmor,m)=igrid(im)+j00
        wtion0(larmor,m)=tdum-real(j00)

! outer flux surface
        im=ii+1
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j01=max(0,min(mtheta(im)-1,int(tdum)))
        jtion1(larmor,m)=igrid(im)+j01
        wtion1(larmor,m)=tdum-real(j01)
     enddo
  enddo

  if(istep==0)return

#ifdef _openmp
! the following lines are openmp directives for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
! when compiling with the openmp option (-qsmp=omp for the xlf compiler on
! aix and -mp for the mipspro compiler on irix), the preprocessor variable
! _openmp is automatically defined and we use it here to add pieces of codes
! needed to avoid contentions between threads. in the following parallel
! loop each thread has its own private array "dnitmp()" which is used to
! prevent the thread from writing into the shared array "densityi()"
!
!$omp parallel private(dnitmp)
!!!!  allocate(dnitmp(0:mzeta,mgrid))
  dnitmp=0.   ! set array elements to zero
#endif
!$omp do private(m,larmor,weight,kk,wz1,wz0,wp1,wp0,wt10,wt00,wt11,wt01,ij)
  do m=1,mi
     weight=zion(5,m)

     kk=kzion(m)
     wz1=weight*wzion(m)
     wz0=weight-wz1     

     do larmor=1,4
        wp1=wpion(larmor,m)
        wp0=1.0-wp1

        wt10=wp0*wtion0(larmor,m)
        wt00=wp0-wt10

        wt11=wp1*wtion1(larmor,m)
        wt01=wp1-wt11

#ifdef _openmp
! use thread-private temporary array dnitmp to store the results
        ij=jtion0(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt00
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt00
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt10
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt01
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt01
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt11
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt11
#else
! if no loop-level parallelism, use original algorithm (write directly
! into array "densityi()".
        ij=jtion0(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt00
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt00

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt10
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt01
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt01

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt11
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt11
#endif

     enddo
  enddo

#ifdef _openmp
! for openmp, we now write the results accumulated in each thread-private
! array dnitmp() back into the shared array densityi(). the loop is enclosed
! in a critical section so that one thread at a time updates densityi(). 
!$omp critical
  do ij=1,mgrid
     do kk=0,mzeta
        densityi(kk,ij) = densityi(kk,ij) + dnitmp(kk,ij)
     enddo
  enddo
!$omp end critical
!!!!  deallocate(dnitmp)
#endif
!$omp end parallel

! if we have a particle decomposition on the toroidal domains, do a reduce
! operation to add up all the contributions to charge density on the grid
  if(npartdom>1)then
   !$omp parallel do private(ij,kk)
    do ij=1,mgrid
       do kk=0,mzeta
          dnitmp(kk,ij)=densityi(kk,ij)
          densityi(kk,ij)=0.
       enddo
    enddo
    call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);
    !call mpi_allreduce(dnitmp,densityi,(mgrid*(mzeta+1)),mpi_rsize,&
    !                   mpi_sum,partd_comm,ierror)
  endif

! poloidal end cell, discard ghost cell j=0
  do i=0,mpsi
     densityi(:,igrid(i)+mtheta(i))=densityi(:,igrid(i)+mtheta(i))+densityi(:,igrid(i))
  enddo

! toroidal end cell
  sendl=densityi(0,:)
  recvr=0.0
  icount=mgrid
  !!!idest=mod(myrank_toroidal-1+ntoroidal,ntoroidal)
  idest=left_pe
  !!!isource=mod(myrank_toroidal+1,ntoroidal)
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource

! send densityi to left and receive from right
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)
     
  if(myrank_toroidal == ntoroidal-1)then
! b.c. at zeta=2*pi is shifted
     do i=0,mpsi
        ii=igrid(i)
        jt=mtheta(i)
        densityi(mzeta,ii+1:ii+jt)=densityi(mzeta,ii+1:ii+jt)+&
             cshift(recvr(ii+1:ii+jt),itran(i))
     enddo
  else
! b.c. at zeta<2*pi is continuous
     densityi(mzeta,:)=densityi(mzeta,:)+recvr
  endif
  
! zero out charge in radial boundary cell
  do i=0,nbound-1
     densityi(:,igrid(i):igrid(i)+mtheta(i))=&
	densityi(:,igrid(i):igrid(i)+mtheta(i))*real(i)/real(nbound)
     densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))=&
	densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))*real(i)/real(nbound)
  enddo

! flux surface average and normalization  
  zonali=0.0
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           zonali(i)=zonali(i)+0.25*densityi(k,ij)
           densityi(k,ij)=0.25*densityi(k,ij)*markeri(k,ij)
        enddo        
     enddo
  enddo

! global sum of phi00, broadcast to every toroidal pe
  !call mpi_allreduce(zonali,adum,mpsi+1,mpi_rsize,mpi_sum,toroidal_comm,ierror)
  call toroidal_allreduce_cmm(ptr,zonali,adum,mpsi+1)
  zonali=adum*pmarki

! densityi subtracted (0,0) mode
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           densityi(k,ij)=densityi(k,ij)-zonali(i)
        enddo
     enddo
! poloidal bc condition
     densityi(1:mzeta,igrid(i))=densityi(1:mzeta,igrid(i)+mtheta(i))
  enddo
  
! enforce charge conservation for zonal flow mode
  rdum=0.0
  tdum=0.0
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     rdum=rdum+r
     tdum=tdum+r*zonali(i)
  enddo
  tdum=tdum/rdum
  ddeni=tdum !for diagnostic
  do i=1,mpsi-1
     zonali(i)=zonali(i)-tdum
  enddo

end subroutine chargei_0

subroutine chargei_1(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_1
  use particle_array_1
  use field_array_1
  use diagnosis_array_1
  use particle_decomp_1
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer m,i,im,j,k,ip,jt,kk,ii,j11,j10,j01,j00,ierror,larmor,ij,ipjt,&
       icount,idest,isource,isendtag,irecvtag !,istatus(mpi_status_size)
  real(8) weight,wemark,rdum,tdum,delr,delt(0:mpsi),delz,smu_inv,r,wz1,wz0,&
       wp1,wp0,wt11,wt10,wt01,wt00,adum(0:mpsi),tflr,damping,pi2_inv,&
       psitmp,thetatmp,zetatmp,rhoi,deltheta
  real(8) sendl(mgrid),recvr(mgrid)
  real(8) dnitmp(0:mzeta,mgrid)

#ifdef _openmp
! we need the following temporary array only when using openmp
!!!  real,dimension(:,:),allocatable :: dnitmp
#endif

  delr=1.0/deltar 
  delt=2.0*pi/deltat
  delz=1.0/deltaz
  smu_inv=sqrt(aion)/(abs(qion)*gyroradius)
  pi2_inv=0.5/pi
  densityi=0.0

!$omp parallel do private(m,larmor,psitmp,thetatmp,zetatmp,rhoi,r,ip,jt,ipjt,&
!$omp& wz1,kk,rdum,ii,wp1,tflr,im,tdum,j00,j01)
  do m=1,mi
     psitmp=zion(1,m)
     thetatmp=zion(2,m)
     zetatmp=zion(3,m)
     rhoi=zion(6,m)*smu_inv

     r=sqrt(2.0*psitmp)
     ip=max(0,min(mpsi,int((r-a0)*delr+0.5)))
     jt=max(0,min(mtheta(ip),int(thetatmp*pi2_inv*delt(ip)+0.5)))
     ipjt=igrid(ip)+jt
     wz1=(zetatmp-zetamin)*delz
     kk=max(0,min(mzeta-1,int(wz1)))
     kzion(m)=kk
     wzion(m)=wz1-real(kk)

     do larmor=1,4
!        rdum=delr*max(0.0d+00,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        rdum=delr*max(zero,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        ii=max(0,min(mpsi-1,int(rdum)))
        wp1=rdum-real(ii)
        wpion(larmor,m)=wp1

! particle position in theta
        tflr=thetatmp+rhoi*tgyro(larmor,ipjt)

! inner flux surface
        im=ii
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j00=max(0,min(mtheta(im)-1,int(tdum)))
        jtion0(larmor,m)=igrid(im)+j00
        wtion0(larmor,m)=tdum-real(j00)

! outer flux surface
        im=ii+1
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j01=max(0,min(mtheta(im)-1,int(tdum)))
        jtion1(larmor,m)=igrid(im)+j01
        wtion1(larmor,m)=tdum-real(j01)
     enddo
  enddo

  if(istep==0)return

#ifdef _openmp
! the following lines are openmp directives for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
! when compiling with the openmp option (-qsmp=omp for the xlf compiler on
! aix and -mp for the mipspro compiler on irix), the preprocessor variable
! _openmp is automatically defined and we use it here to add pieces of codes
! needed to avoid contentions between threads. in the following parallel
! loop each thread has its own private array "dnitmp()" which is used to
! prevent the thread from writing into the shared array "densityi()"
!
!$omp parallel private(dnitmp)
!!!!  allocate(dnitmp(0:mzeta,mgrid))
  dnitmp=0.   ! set array elements to zero
#endif
!$omp do private(m,larmor,weight,kk,wz1,wz0,wp1,wp0,wt10,wt00,wt11,wt01,ij)
  do m=1,mi
     weight=zion(5,m)

     kk=kzion(m)
     wz1=weight*wzion(m)
     wz0=weight-wz1     

     do larmor=1,4
        wp1=wpion(larmor,m)
        wp0=1.0-wp1

        wt10=wp0*wtion0(larmor,m)
        wt00=wp0-wt10

        wt11=wp1*wtion1(larmor,m)
        wt01=wp1-wt11

#ifdef _openmp
! use thread-private temporary array dnitmp to store the results
        ij=jtion0(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt00
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt00
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt10
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt01
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt01
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt11
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt11
#else
! if no loop-level parallelism, use original algorithm (write directly
! into array "densityi()".
        ij=jtion0(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt00
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt00

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt10
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt01
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt01

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt11
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt11
#endif

     enddo
  enddo

#ifdef _openmp
! for openmp, we now write the results accumulated in each thread-private
! array dnitmp() back into the shared array densityi(). the loop is enclosed
! in a critical section so that one thread at a time updates densityi(). 
!$omp critical
  do ij=1,mgrid
     do kk=0,mzeta
        densityi(kk,ij) = densityi(kk,ij) + dnitmp(kk,ij)
     enddo
  enddo
!$omp end critical
!!!!  deallocate(dnitmp)
#endif
!$omp end parallel

! if we have a particle decomposition on the toroidal domains, do a reduce
! operation to add up all the contributions to charge density on the grid
  if(npartdom>1)then
   !$omp parallel do private(ij,kk)
    do ij=1,mgrid
       do kk=0,mzeta
          dnitmp(kk,ij)=densityi(kk,ij)
          densityi(kk,ij)=0.
       enddo
    enddo
    call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);
    !call mpi_allreduce(dnitmp,densityi,(mgrid*(mzeta+1)),mpi_rsize,&
    !                   mpi_sum,partd_comm,ierror)
  endif

! poloidal end cell, discard ghost cell j=0
  do i=0,mpsi
     densityi(:,igrid(i)+mtheta(i))=densityi(:,igrid(i)+mtheta(i))+densityi(:,igrid(i))
  enddo

! toroidal end cell
  sendl=densityi(0,:)
  recvr=0.0
  icount=mgrid
  !!!idest=mod(myrank_toroidal-1+ntoroidal,ntoroidal)
  idest=left_pe
  !!!isource=mod(myrank_toroidal+1,ntoroidal)
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource

! send densityi to left and receive from right
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)
     
  if(myrank_toroidal == ntoroidal-1)then
! b.c. at zeta=2*pi is shifted
     do i=0,mpsi
        ii=igrid(i)
        jt=mtheta(i)
        densityi(mzeta,ii+1:ii+jt)=densityi(mzeta,ii+1:ii+jt)+&
             cshift(recvr(ii+1:ii+jt),itran(i))
     enddo
  else
! b.c. at zeta<2*pi is continuous
     densityi(mzeta,:)=densityi(mzeta,:)+recvr
  endif
  
! zero out charge in radial boundary cell
  do i=0,nbound-1
     densityi(:,igrid(i):igrid(i)+mtheta(i))=&
	densityi(:,igrid(i):igrid(i)+mtheta(i))*real(i)/real(nbound)
     densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))=&
	densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))*real(i)/real(nbound)
  enddo

! flux surface average and normalization  
  zonali=0.0
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           zonali(i)=zonali(i)+0.25*densityi(k,ij)
           densityi(k,ij)=0.25*densityi(k,ij)*markeri(k,ij)
        enddo        
     enddo
  enddo

! global sum of phi00, broadcast to every toroidal pe
  !call mpi_allreduce(zonali,adum,mpsi+1,mpi_rsize,mpi_sum,toroidal_comm,ierror)
  call toroidal_allreduce_cmm(ptr,zonali,adum,mpsi+1)
  zonali=adum*pmarki

! densityi subtracted (0,0) mode
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           densityi(k,ij)=densityi(k,ij)-zonali(i)
        enddo
     enddo
! poloidal bc condition
     densityi(1:mzeta,igrid(i))=densityi(1:mzeta,igrid(i)+mtheta(i))
  enddo
  
! enforce charge conservation for zonal flow mode
  rdum=0.0
  tdum=0.0
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     rdum=rdum+r
     tdum=tdum+r*zonali(i)
  enddo
  tdum=tdum/rdum
  ddeni=tdum !for diagnostic
  do i=1,mpsi-1
     zonali(i)=zonali(i)-tdum
  enddo

end subroutine chargei_1

subroutine chargei_2(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_2
  use particle_array_2
  use field_array_2
  use diagnosis_array_2
  use particle_decomp_2
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer m,i,im,j,k,ip,jt,kk,ii,j11,j10,j01,j00,ierror,larmor,ij,ipjt,&
       icount,idest,isource,isendtag,irecvtag !,istatus(mpi_status_size)
  real(8) weight,wemark,rdum,tdum,delr,delt(0:mpsi),delz,smu_inv,r,wz1,wz0,&
       wp1,wp0,wt11,wt10,wt01,wt00,adum(0:mpsi),tflr,damping,pi2_inv,&
       psitmp,thetatmp,zetatmp,rhoi,deltheta
  real(8) sendl(mgrid),recvr(mgrid)
  real(8) dnitmp(0:mzeta,mgrid)

#ifdef _openmp
! we need the following temporary array only when using openmp
!!!  real,dimension(:,:),allocatable :: dnitmp
#endif

  delr=1.0/deltar 
  delt=2.0*pi/deltat
  delz=1.0/deltaz
  smu_inv=sqrt(aion)/(abs(qion)*gyroradius)
  pi2_inv=0.5/pi
  densityi=0.0

!$omp parallel do private(m,larmor,psitmp,thetatmp,zetatmp,rhoi,r,ip,jt,ipjt,&
!$omp& wz1,kk,rdum,ii,wp1,tflr,im,tdum,j00,j01)
  do m=1,mi
     psitmp=zion(1,m)
     thetatmp=zion(2,m)
     zetatmp=zion(3,m)
     rhoi=zion(6,m)*smu_inv

     r=sqrt(2.0*psitmp)
     ip=max(0,min(mpsi,int((r-a0)*delr+0.5)))
     jt=max(0,min(mtheta(ip),int(thetatmp*pi2_inv*delt(ip)+0.5)))
     ipjt=igrid(ip)+jt
     wz1=(zetatmp-zetamin)*delz
     kk=max(0,min(mzeta-1,int(wz1)))
     kzion(m)=kk
     wzion(m)=wz1-real(kk)

     do larmor=1,4
!        rdum=delr*max(0.0d+00,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        rdum=delr*max(zero,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        ii=max(0,min(mpsi-1,int(rdum)))
        wp1=rdum-real(ii)
        wpion(larmor,m)=wp1

! particle position in theta
        tflr=thetatmp+rhoi*tgyro(larmor,ipjt)

! inner flux surface
        im=ii
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j00=max(0,min(mtheta(im)-1,int(tdum)))
        jtion0(larmor,m)=igrid(im)+j00
        wtion0(larmor,m)=tdum-real(j00)

! outer flux surface
        im=ii+1
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j01=max(0,min(mtheta(im)-1,int(tdum)))
        jtion1(larmor,m)=igrid(im)+j01
        wtion1(larmor,m)=tdum-real(j01)
     enddo
  enddo

  if(istep==0)return

#ifdef _openmp
! the following lines are openmp directives for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
! when compiling with the openmp option (-qsmp=omp for the xlf compiler on
! aix and -mp for the mipspro compiler on irix), the preprocessor variable
! _openmp is automatically defined and we use it here to add pieces of codes
! needed to avoid contentions between threads. in the following parallel
! loop each thread has its own private array "dnitmp()" which is used to
! prevent the thread from writing into the shared array "densityi()"
!
!$omp parallel private(dnitmp)
!!!!  allocate(dnitmp(0:mzeta,mgrid))
  dnitmp=0.   ! set array elements to zero
#endif
!$omp do private(m,larmor,weight,kk,wz1,wz0,wp1,wp0,wt10,wt00,wt11,wt01,ij)
  do m=1,mi
     weight=zion(5,m)

     kk=kzion(m)
     wz1=weight*wzion(m)
     wz0=weight-wz1     

     do larmor=1,4
        wp1=wpion(larmor,m)
        wp0=1.0-wp1

        wt10=wp0*wtion0(larmor,m)
        wt00=wp0-wt10

        wt11=wp1*wtion1(larmor,m)
        wt01=wp1-wt11

#ifdef _openmp
! use thread-private temporary array dnitmp to store the results
        ij=jtion0(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt00
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt00
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt10
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt01
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt01
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt11
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt11
#else
! if no loop-level parallelism, use original algorithm (write directly
! into array "densityi()".
        ij=jtion0(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt00
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt00

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt10
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt01
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt01

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt11
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt11
#endif

     enddo
  enddo

#ifdef _openmp
! for openmp, we now write the results accumulated in each thread-private
! array dnitmp() back into the shared array densityi(). the loop is enclosed
! in a critical section so that one thread at a time updates densityi(). 
!$omp critical
  do ij=1,mgrid
     do kk=0,mzeta
        densityi(kk,ij) = densityi(kk,ij) + dnitmp(kk,ij)
     enddo
  enddo
!$omp end critical
!!!!  deallocate(dnitmp)
#endif
!$omp end parallel

! if we have a particle decomposition on the toroidal domains, do a reduce
! operation to add up all the contributions to charge density on the grid
  if(npartdom>1)then
   !$omp parallel do private(ij,kk)
    do ij=1,mgrid
       do kk=0,mzeta
          dnitmp(kk,ij)=densityi(kk,ij)
          densityi(kk,ij)=0.
       enddo
    enddo
    call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);
    !call mpi_allreduce(dnitmp,densityi,(mgrid*(mzeta+1)),mpi_rsize,&
    !                   mpi_sum,partd_comm,ierror)
  endif

! poloidal end cell, discard ghost cell j=0
  do i=0,mpsi
     densityi(:,igrid(i)+mtheta(i))=densityi(:,igrid(i)+mtheta(i))+densityi(:,igrid(i))
  enddo

! toroidal end cell
  sendl=densityi(0,:)
  recvr=0.0
  icount=mgrid
  !!!idest=mod(myrank_toroidal-1+ntoroidal,ntoroidal)
  idest=left_pe
  !!!isource=mod(myrank_toroidal+1,ntoroidal)
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource

! send densityi to left and receive from right
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)
     
  if(myrank_toroidal == ntoroidal-1)then
! b.c. at zeta=2*pi is shifted
     do i=0,mpsi
        ii=igrid(i)
        jt=mtheta(i)
        densityi(mzeta,ii+1:ii+jt)=densityi(mzeta,ii+1:ii+jt)+&
             cshift(recvr(ii+1:ii+jt),itran(i))
     enddo
  else
! b.c. at zeta<2*pi is continuous
     densityi(mzeta,:)=densityi(mzeta,:)+recvr
  endif
  
! zero out charge in radial boundary cell
  do i=0,nbound-1
     densityi(:,igrid(i):igrid(i)+mtheta(i))=&
	densityi(:,igrid(i):igrid(i)+mtheta(i))*real(i)/real(nbound)
     densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))=&
	densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))*real(i)/real(nbound)
  enddo

! flux surface average and normalization  
  zonali=0.0
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           zonali(i)=zonali(i)+0.25*densityi(k,ij)
           densityi(k,ij)=0.25*densityi(k,ij)*markeri(k,ij)
        enddo        
     enddo
  enddo

! global sum of phi00, broadcast to every toroidal pe
  !call mpi_allreduce(zonali,adum,mpsi+1,mpi_rsize,mpi_sum,toroidal_comm,ierror)
  call toroidal_allreduce_cmm(ptr,zonali,adum,mpsi+1)
  zonali=adum*pmarki

! densityi subtracted (0,0) mode
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           densityi(k,ij)=densityi(k,ij)-zonali(i)
        enddo
     enddo
! poloidal bc condition
     densityi(1:mzeta,igrid(i))=densityi(1:mzeta,igrid(i)+mtheta(i))
  enddo
  
! enforce charge conservation for zonal flow mode
  rdum=0.0
  tdum=0.0
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     rdum=rdum+r
     tdum=tdum+r*zonali(i)
  enddo
  tdum=tdum/rdum
  ddeni=tdum !for diagnostic
  do i=1,mpsi-1
     zonali(i)=zonali(i)-tdum
  enddo

end subroutine chargei_2

subroutine chargei_3(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_3
  use particle_array_3
  use field_array_3
  use diagnosis_array_3
  use particle_decomp_3
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer m,i,im,j,k,ip,jt,kk,ii,j11,j10,j01,j00,ierror,larmor,ij,ipjt,&
       icount,idest,isource,isendtag,irecvtag !,istatus(mpi_status_size)
  real(8) weight,wemark,rdum,tdum,delr,delt(0:mpsi),delz,smu_inv,r,wz1,wz0,&
       wp1,wp0,wt11,wt10,wt01,wt00,adum(0:mpsi),tflr,damping,pi2_inv,&
       psitmp,thetatmp,zetatmp,rhoi,deltheta
  real(8) sendl(mgrid),recvr(mgrid)
  real(8) dnitmp(0:mzeta,mgrid)

#ifdef _openmp
! we need the following temporary array only when using openmp
!!!  real,dimension(:,:),allocatable :: dnitmp
#endif

  delr=1.0/deltar 
  delt=2.0*pi/deltat
  delz=1.0/deltaz
  smu_inv=sqrt(aion)/(abs(qion)*gyroradius)
  pi2_inv=0.5/pi
  densityi=0.0

!$omp parallel do private(m,larmor,psitmp,thetatmp,zetatmp,rhoi,r,ip,jt,ipjt,&
!$omp& wz1,kk,rdum,ii,wp1,tflr,im,tdum,j00,j01)
  do m=1,mi
     psitmp=zion(1,m)
     thetatmp=zion(2,m)
     zetatmp=zion(3,m)
     rhoi=zion(6,m)*smu_inv

     r=sqrt(2.0*psitmp)
     ip=max(0,min(mpsi,int((r-a0)*delr+0.5)))
     jt=max(0,min(mtheta(ip),int(thetatmp*pi2_inv*delt(ip)+0.5)))
     ipjt=igrid(ip)+jt
     wz1=(zetatmp-zetamin)*delz
     kk=max(0,min(mzeta-1,int(wz1)))
     kzion(m)=kk
     wzion(m)=wz1-real(kk)

     do larmor=1,4
!        rdum=delr*max(0.0d+00,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        rdum=delr*max(zero,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        ii=max(0,min(mpsi-1,int(rdum)))
        wp1=rdum-real(ii)
        wpion(larmor,m)=wp1

! particle position in theta
        tflr=thetatmp+rhoi*tgyro(larmor,ipjt)

! inner flux surface
        im=ii
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j00=max(0,min(mtheta(im)-1,int(tdum)))
        jtion0(larmor,m)=igrid(im)+j00
        wtion0(larmor,m)=tdum-real(j00)

! outer flux surface
        im=ii+1
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j01=max(0,min(mtheta(im)-1,int(tdum)))
        jtion1(larmor,m)=igrid(im)+j01
        wtion1(larmor,m)=tdum-real(j01)
     enddo
  enddo

  if(istep==0)return

#ifdef _openmp
! the following lines are openmp directives for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
! when compiling with the openmp option (-qsmp=omp for the xlf compiler on
! aix and -mp for the mipspro compiler on irix), the preprocessor variable
! _openmp is automatically defined and we use it here to add pieces of codes
! needed to avoid contentions between threads. in the following parallel
! loop each thread has its own private array "dnitmp()" which is used to
! prevent the thread from writing into the shared array "densityi()"
!
!$omp parallel private(dnitmp)
!!!!  allocate(dnitmp(0:mzeta,mgrid))
  dnitmp=0.   ! set array elements to zero
#endif
!$omp do private(m,larmor,weight,kk,wz1,wz0,wp1,wp0,wt10,wt00,wt11,wt01,ij)
  do m=1,mi
     weight=zion(5,m)

     kk=kzion(m)
     wz1=weight*wzion(m)
     wz0=weight-wz1     

     do larmor=1,4
        wp1=wpion(larmor,m)
        wp0=1.0-wp1

        wt10=wp0*wtion0(larmor,m)
        wt00=wp0-wt10

        wt11=wp1*wtion1(larmor,m)
        wt01=wp1-wt11

#ifdef _openmp
! use thread-private temporary array dnitmp to store the results
        ij=jtion0(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt00
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt00
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt10
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt01
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt01
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt11
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt11
#else
! if no loop-level parallelism, use original algorithm (write directly
! into array "densityi()".
        ij=jtion0(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt00
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt00

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt10
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt01
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt01

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt11
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt11
#endif

     enddo
  enddo

#ifdef _openmp
! for openmp, we now write the results accumulated in each thread-private
! array dnitmp() back into the shared array densityi(). the loop is enclosed
! in a critical section so that one thread at a time updates densityi(). 
!$omp critical
  do ij=1,mgrid
     do kk=0,mzeta
        densityi(kk,ij) = densityi(kk,ij) + dnitmp(kk,ij)
     enddo
  enddo
!$omp end critical
!!!!  deallocate(dnitmp)
#endif
!$omp end parallel

! if we have a particle decomposition on the toroidal domains, do a reduce
! operation to add up all the contributions to charge density on the grid
  if(npartdom>1)then
   !$omp parallel do private(ij,kk)
    do ij=1,mgrid
       do kk=0,mzeta
          dnitmp(kk,ij)=densityi(kk,ij)
          densityi(kk,ij)=0.
       enddo
    enddo
    call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);
    !call mpi_allreduce(dnitmp,densityi,(mgrid*(mzeta+1)),mpi_rsize,&
    !                   mpi_sum,partd_comm,ierror)
  endif

! poloidal end cell, discard ghost cell j=0
  do i=0,mpsi
     densityi(:,igrid(i)+mtheta(i))=densityi(:,igrid(i)+mtheta(i))+densityi(:,igrid(i))
  enddo

! toroidal end cell
  sendl=densityi(0,:)
  recvr=0.0
  icount=mgrid
  !!!idest=mod(myrank_toroidal-1+ntoroidal,ntoroidal)
  idest=left_pe
  !!!isource=mod(myrank_toroidal+1,ntoroidal)
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource

! send densityi to left and receive from right
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)
     
  if(myrank_toroidal == ntoroidal-1)then
! b.c. at zeta=2*pi is shifted
     do i=0,mpsi
        ii=igrid(i)
        jt=mtheta(i)
        densityi(mzeta,ii+1:ii+jt)=densityi(mzeta,ii+1:ii+jt)+&
             cshift(recvr(ii+1:ii+jt),itran(i))
     enddo
  else
! b.c. at zeta<2*pi is continuous
     densityi(mzeta,:)=densityi(mzeta,:)+recvr
  endif
  
! zero out charge in radial boundary cell
  do i=0,nbound-1
     densityi(:,igrid(i):igrid(i)+mtheta(i))=&
	densityi(:,igrid(i):igrid(i)+mtheta(i))*real(i)/real(nbound)
     densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))=&
	densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))*real(i)/real(nbound)
  enddo

! flux surface average and normalization  
  zonali=0.0
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           zonali(i)=zonali(i)+0.25*densityi(k,ij)
           densityi(k,ij)=0.25*densityi(k,ij)*markeri(k,ij)
        enddo        
     enddo
  enddo

! global sum of phi00, broadcast to every toroidal pe
  !call mpi_allreduce(zonali,adum,mpsi+1,mpi_rsize,mpi_sum,toroidal_comm,ierror)
  call toroidal_allreduce_cmm(ptr,zonali,adum,mpsi+1)
  zonali=adum*pmarki

! densityi subtracted (0,0) mode
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           densityi(k,ij)=densityi(k,ij)-zonali(i)
        enddo
     enddo
! poloidal bc condition
     densityi(1:mzeta,igrid(i))=densityi(1:mzeta,igrid(i)+mtheta(i))
  enddo
  
! enforce charge conservation for zonal flow mode
  rdum=0.0
  tdum=0.0
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     rdum=rdum+r
     tdum=tdum+r*zonali(i)
  enddo
  tdum=tdum/rdum
  ddeni=tdum !for diagnostic
  do i=1,mpsi-1
     zonali(i)=zonali(i)-tdum
  enddo

end subroutine chargei_3

subroutine chargei_4(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_4
  use particle_array_4
  use field_array_4
  use diagnosis_array_4
  use particle_decomp_4
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer m,i,im,j,k,ip,jt,kk,ii,j11,j10,j01,j00,ierror,larmor,ij,ipjt,&
       icount,idest,isource,isendtag,irecvtag !,istatus(mpi_status_size)
  real(8) weight,wemark,rdum,tdum,delr,delt(0:mpsi),delz,smu_inv,r,wz1,wz0,&
       wp1,wp0,wt11,wt10,wt01,wt00,adum(0:mpsi),tflr,damping,pi2_inv,&
       psitmp,thetatmp,zetatmp,rhoi,deltheta
  real(8) sendl(mgrid),recvr(mgrid)
  real(8) dnitmp(0:mzeta,mgrid)

#ifdef _openmp
! we need the following temporary array only when using openmp
!!!  real,dimension(:,:),allocatable :: dnitmp
#endif

  delr=1.0/deltar 
  delt=2.0*pi/deltat
  delz=1.0/deltaz
  smu_inv=sqrt(aion)/(abs(qion)*gyroradius)
  pi2_inv=0.5/pi
  densityi=0.0

!$omp parallel do private(m,larmor,psitmp,thetatmp,zetatmp,rhoi,r,ip,jt,ipjt,&
!$omp& wz1,kk,rdum,ii,wp1,tflr,im,tdum,j00,j01)
  do m=1,mi
     psitmp=zion(1,m)
     thetatmp=zion(2,m)
     zetatmp=zion(3,m)
     rhoi=zion(6,m)*smu_inv

     r=sqrt(2.0*psitmp)
     ip=max(0,min(mpsi,int((r-a0)*delr+0.5)))
     jt=max(0,min(mtheta(ip),int(thetatmp*pi2_inv*delt(ip)+0.5)))
     ipjt=igrid(ip)+jt
     wz1=(zetatmp-zetamin)*delz
     kk=max(0,min(mzeta-1,int(wz1)))
     kzion(m)=kk
     wzion(m)=wz1-real(kk)

     do larmor=1,4
!        rdum=delr*max(0.0d+00,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        rdum=delr*max(zero,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        ii=max(0,min(mpsi-1,int(rdum)))
        wp1=rdum-real(ii)
        wpion(larmor,m)=wp1

! particle position in theta
        tflr=thetatmp+rhoi*tgyro(larmor,ipjt)

! inner flux surface
        im=ii
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j00=max(0,min(mtheta(im)-1,int(tdum)))
        jtion0(larmor,m)=igrid(im)+j00
        wtion0(larmor,m)=tdum-real(j00)

! outer flux surface
        im=ii+1
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j01=max(0,min(mtheta(im)-1,int(tdum)))
        jtion1(larmor,m)=igrid(im)+j01
        wtion1(larmor,m)=tdum-real(j01)
     enddo
  enddo

  if(istep==0)return

#ifdef _openmp
! the following lines are openmp directives for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
! when compiling with the openmp option (-qsmp=omp for the xlf compiler on
! aix and -mp for the mipspro compiler on irix), the preprocessor variable
! _openmp is automatically defined and we use it here to add pieces of codes
! needed to avoid contentions between threads. in the following parallel
! loop each thread has its own private array "dnitmp()" which is used to
! prevent the thread from writing into the shared array "densityi()"
!
!$omp parallel private(dnitmp)
!!!!  allocate(dnitmp(0:mzeta,mgrid))
  dnitmp=0.   ! set array elements to zero
#endif
!$omp do private(m,larmor,weight,kk,wz1,wz0,wp1,wp0,wt10,wt00,wt11,wt01,ij)
  do m=1,mi
     weight=zion(5,m)

     kk=kzion(m)
     wz1=weight*wzion(m)
     wz0=weight-wz1     

     do larmor=1,4
        wp1=wpion(larmor,m)
        wp0=1.0-wp1

        wt10=wp0*wtion0(larmor,m)
        wt00=wp0-wt10

        wt11=wp1*wtion1(larmor,m)
        wt01=wp1-wt11

#ifdef _openmp
! use thread-private temporary array dnitmp to store the results
        ij=jtion0(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt00
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt00
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt10
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt01
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt01
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt11
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt11
#else
! if no loop-level parallelism, use original algorithm (write directly
! into array "densityi()".
        ij=jtion0(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt00
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt00

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt10
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt01
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt01

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt11
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt11
#endif

     enddo
  enddo

#ifdef _openmp
! for openmp, we now write the results accumulated in each thread-private
! array dnitmp() back into the shared array densityi(). the loop is enclosed
! in a critical section so that one thread at a time updates densityi(). 
!$omp critical
  do ij=1,mgrid
     do kk=0,mzeta
        densityi(kk,ij) = densityi(kk,ij) + dnitmp(kk,ij)
     enddo
  enddo
!$omp end critical
!!!!  deallocate(dnitmp)
#endif
!$omp end parallel

! if we have a particle decomposition on the toroidal domains, do a reduce
! operation to add up all the contributions to charge density on the grid
  if(npartdom>1)then
   !$omp parallel do private(ij,kk)
    do ij=1,mgrid
       do kk=0,mzeta
          dnitmp(kk,ij)=densityi(kk,ij)
          densityi(kk,ij)=0.
       enddo
    enddo
    call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);
    !call mpi_allreduce(dnitmp,densityi,(mgrid*(mzeta+1)),mpi_rsize,&
    !                   mpi_sum,partd_comm,ierror)
  endif

! poloidal end cell, discard ghost cell j=0
  do i=0,mpsi
     densityi(:,igrid(i)+mtheta(i))=densityi(:,igrid(i)+mtheta(i))+densityi(:,igrid(i))
  enddo

! toroidal end cell
  sendl=densityi(0,:)
  recvr=0.0
  icount=mgrid
  !!!idest=mod(myrank_toroidal-1+ntoroidal,ntoroidal)
  idest=left_pe
  !!!isource=mod(myrank_toroidal+1,ntoroidal)
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource

! send densityi to left and receive from right
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)
     
  if(myrank_toroidal == ntoroidal-1)then
! b.c. at zeta=2*pi is shifted
     do i=0,mpsi
        ii=igrid(i)
        jt=mtheta(i)
        densityi(mzeta,ii+1:ii+jt)=densityi(mzeta,ii+1:ii+jt)+&
             cshift(recvr(ii+1:ii+jt),itran(i))
     enddo
  else
! b.c. at zeta<2*pi is continuous
     densityi(mzeta,:)=densityi(mzeta,:)+recvr
  endif
  
! zero out charge in radial boundary cell
  do i=0,nbound-1
     densityi(:,igrid(i):igrid(i)+mtheta(i))=&
	densityi(:,igrid(i):igrid(i)+mtheta(i))*real(i)/real(nbound)
     densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))=&
	densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))*real(i)/real(nbound)
  enddo

! flux surface average and normalization  
  zonali=0.0
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           zonali(i)=zonali(i)+0.25*densityi(k,ij)
           densityi(k,ij)=0.25*densityi(k,ij)*markeri(k,ij)
        enddo        
     enddo
  enddo

! global sum of phi00, broadcast to every toroidal pe
  !call mpi_allreduce(zonali,adum,mpsi+1,mpi_rsize,mpi_sum,toroidal_comm,ierror)
  call toroidal_allreduce_cmm(ptr,zonali,adum,mpsi+1)
  zonali=adum*pmarki

! densityi subtracted (0,0) mode
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           densityi(k,ij)=densityi(k,ij)-zonali(i)
        enddo
     enddo
! poloidal bc condition
     densityi(1:mzeta,igrid(i))=densityi(1:mzeta,igrid(i)+mtheta(i))
  enddo
  
! enforce charge conservation for zonal flow mode
  rdum=0.0
  tdum=0.0
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     rdum=rdum+r
     tdum=tdum+r*zonali(i)
  enddo
  tdum=tdum/rdum
  ddeni=tdum !for diagnostic
  do i=1,mpsi-1
     zonali(i)=zonali(i)-tdum
  enddo

end subroutine chargei_4

subroutine chargei_5(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_5
  use particle_array_5
  use field_array_5
  use diagnosis_array_5
  use particle_decomp_5
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer m,i,im,j,k,ip,jt,kk,ii,j11,j10,j01,j00,ierror,larmor,ij,ipjt,&
       icount,idest,isource,isendtag,irecvtag !,istatus(mpi_status_size)
  real(8) weight,wemark,rdum,tdum,delr,delt(0:mpsi),delz,smu_inv,r,wz1,wz0,&
       wp1,wp0,wt11,wt10,wt01,wt00,adum(0:mpsi),tflr,damping,pi2_inv,&
       psitmp,thetatmp,zetatmp,rhoi,deltheta
  real(8) sendl(mgrid),recvr(mgrid)
  real(8) dnitmp(0:mzeta,mgrid)

#ifdef _openmp
! we need the following temporary array only when using openmp
!!!  real,dimension(:,:),allocatable :: dnitmp
#endif

  delr=1.0/deltar 
  delt=2.0*pi/deltat
  delz=1.0/deltaz
  smu_inv=sqrt(aion)/(abs(qion)*gyroradius)
  pi2_inv=0.5/pi
  densityi=0.0

!$omp parallel do private(m,larmor,psitmp,thetatmp,zetatmp,rhoi,r,ip,jt,ipjt,&
!$omp& wz1,kk,rdum,ii,wp1,tflr,im,tdum,j00,j01)
  do m=1,mi
     psitmp=zion(1,m)
     thetatmp=zion(2,m)
     zetatmp=zion(3,m)
     rhoi=zion(6,m)*smu_inv

     r=sqrt(2.0*psitmp)
     ip=max(0,min(mpsi,int((r-a0)*delr+0.5)))
     jt=max(0,min(mtheta(ip),int(thetatmp*pi2_inv*delt(ip)+0.5)))
     ipjt=igrid(ip)+jt
     wz1=(zetatmp-zetamin)*delz
     kk=max(0,min(mzeta-1,int(wz1)))
     kzion(m)=kk
     wzion(m)=wz1-real(kk)

     do larmor=1,4
!        rdum=delr*max(0.0d+00,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        rdum=delr*max(zero,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        ii=max(0,min(mpsi-1,int(rdum)))
        wp1=rdum-real(ii)
        wpion(larmor,m)=wp1

! particle position in theta
        tflr=thetatmp+rhoi*tgyro(larmor,ipjt)

! inner flux surface
        im=ii
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j00=max(0,min(mtheta(im)-1,int(tdum)))
        jtion0(larmor,m)=igrid(im)+j00
        wtion0(larmor,m)=tdum-real(j00)

! outer flux surface
        im=ii+1
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j01=max(0,min(mtheta(im)-1,int(tdum)))
        jtion1(larmor,m)=igrid(im)+j01
        wtion1(larmor,m)=tdum-real(j01)
     enddo
  enddo

  if(istep==0)return

#ifdef _openmp
! the following lines are openmp directives for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
! when compiling with the openmp option (-qsmp=omp for the xlf compiler on
! aix and -mp for the mipspro compiler on irix), the preprocessor variable
! _openmp is automatically defined and we use it here to add pieces of codes
! needed to avoid contentions between threads. in the following parallel
! loop each thread has its own private array "dnitmp()" which is used to
! prevent the thread from writing into the shared array "densityi()"
!
!$omp parallel private(dnitmp)
!!!!  allocate(dnitmp(0:mzeta,mgrid))
  dnitmp=0.   ! set array elements to zero
#endif
!$omp do private(m,larmor,weight,kk,wz1,wz0,wp1,wp0,wt10,wt00,wt11,wt01,ij)
  do m=1,mi
     weight=zion(5,m)

     kk=kzion(m)
     wz1=weight*wzion(m)
     wz0=weight-wz1     

     do larmor=1,4
        wp1=wpion(larmor,m)
        wp0=1.0-wp1

        wt10=wp0*wtion0(larmor,m)
        wt00=wp0-wt10

        wt11=wp1*wtion1(larmor,m)
        wt01=wp1-wt11

#ifdef _openmp
! use thread-private temporary array dnitmp to store the results
        ij=jtion0(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt00
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt00
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt10
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt01
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt01
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt11
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt11
#else
! if no loop-level parallelism, use original algorithm (write directly
! into array "densityi()".
        ij=jtion0(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt00
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt00

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt10
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt01
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt01

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt11
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt11
#endif

     enddo
  enddo

#ifdef _openmp
! for openmp, we now write the results accumulated in each thread-private
! array dnitmp() back into the shared array densityi(). the loop is enclosed
! in a critical section so that one thread at a time updates densityi(). 
!$omp critical
  do ij=1,mgrid
     do kk=0,mzeta
        densityi(kk,ij) = densityi(kk,ij) + dnitmp(kk,ij)
     enddo
  enddo
!$omp end critical
!!!!  deallocate(dnitmp)
#endif
!$omp end parallel

! if we have a particle decomposition on the toroidal domains, do a reduce
! operation to add up all the contributions to charge density on the grid
  if(npartdom>1)then
   !$omp parallel do private(ij,kk)
    do ij=1,mgrid
       do kk=0,mzeta
          dnitmp(kk,ij)=densityi(kk,ij)
          densityi(kk,ij)=0.
       enddo
    enddo
    call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);
    !call mpi_allreduce(dnitmp,densityi,(mgrid*(mzeta+1)),mpi_rsize,&
    !                   mpi_sum,partd_comm,ierror)
  endif

! poloidal end cell, discard ghost cell j=0
  do i=0,mpsi
     densityi(:,igrid(i)+mtheta(i))=densityi(:,igrid(i)+mtheta(i))+densityi(:,igrid(i))
  enddo

! toroidal end cell
  sendl=densityi(0,:)
  recvr=0.0
  icount=mgrid
  !!!idest=mod(myrank_toroidal-1+ntoroidal,ntoroidal)
  idest=left_pe
  !!!isource=mod(myrank_toroidal+1,ntoroidal)
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource

! send densityi to left and receive from right
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)
     
  if(myrank_toroidal == ntoroidal-1)then
! b.c. at zeta=2*pi is shifted
     do i=0,mpsi
        ii=igrid(i)
        jt=mtheta(i)
        densityi(mzeta,ii+1:ii+jt)=densityi(mzeta,ii+1:ii+jt)+&
             cshift(recvr(ii+1:ii+jt),itran(i))
     enddo
  else
! b.c. at zeta<2*pi is continuous
     densityi(mzeta,:)=densityi(mzeta,:)+recvr
  endif
  
! zero out charge in radial boundary cell
  do i=0,nbound-1
     densityi(:,igrid(i):igrid(i)+mtheta(i))=&
	densityi(:,igrid(i):igrid(i)+mtheta(i))*real(i)/real(nbound)
     densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))=&
	densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))*real(i)/real(nbound)
  enddo

! flux surface average and normalization  
  zonali=0.0
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           zonali(i)=zonali(i)+0.25*densityi(k,ij)
           densityi(k,ij)=0.25*densityi(k,ij)*markeri(k,ij)
        enddo        
     enddo
  enddo

! global sum of phi00, broadcast to every toroidal pe
  !call mpi_allreduce(zonali,adum,mpsi+1,mpi_rsize,mpi_sum,toroidal_comm,ierror)
  call toroidal_allreduce_cmm(ptr,zonali,adum,mpsi+1)
  zonali=adum*pmarki

! densityi subtracted (0,0) mode
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           densityi(k,ij)=densityi(k,ij)-zonali(i)
        enddo
     enddo
! poloidal bc condition
     densityi(1:mzeta,igrid(i))=densityi(1:mzeta,igrid(i)+mtheta(i))
  enddo
  
! enforce charge conservation for zonal flow mode
  rdum=0.0
  tdum=0.0
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     rdum=rdum+r
     tdum=tdum+r*zonali(i)
  enddo
  tdum=tdum/rdum
  ddeni=tdum !for diagnostic
  do i=1,mpsi-1
     zonali(i)=zonali(i)-tdum
  enddo

end subroutine chargei_5

subroutine chargei_6(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_6
  use particle_array_6
  use field_array_6
  use diagnosis_array_6
  use particle_decomp_6
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer m,i,im,j,k,ip,jt,kk,ii,j11,j10,j01,j00,ierror,larmor,ij,ipjt,&
       icount,idest,isource,isendtag,irecvtag !,istatus(mpi_status_size)
  real(8) weight,wemark,rdum,tdum,delr,delt(0:mpsi),delz,smu_inv,r,wz1,wz0,&
       wp1,wp0,wt11,wt10,wt01,wt00,adum(0:mpsi),tflr,damping,pi2_inv,&
       psitmp,thetatmp,zetatmp,rhoi,deltheta
  real(8) sendl(mgrid),recvr(mgrid)
  real(8) dnitmp(0:mzeta,mgrid)

#ifdef _openmp
! we need the following temporary array only when using openmp
!!!  real,dimension(:,:),allocatable :: dnitmp
#endif

  delr=1.0/deltar 
  delt=2.0*pi/deltat
  delz=1.0/deltaz
  smu_inv=sqrt(aion)/(abs(qion)*gyroradius)
  pi2_inv=0.5/pi
  densityi=0.0

!$omp parallel do private(m,larmor,psitmp,thetatmp,zetatmp,rhoi,r,ip,jt,ipjt,&
!$omp& wz1,kk,rdum,ii,wp1,tflr,im,tdum,j00,j01)
  do m=1,mi
     psitmp=zion(1,m)
     thetatmp=zion(2,m)
     zetatmp=zion(3,m)
     rhoi=zion(6,m)*smu_inv

     r=sqrt(2.0*psitmp)
     ip=max(0,min(mpsi,int((r-a0)*delr+0.5)))
     jt=max(0,min(mtheta(ip),int(thetatmp*pi2_inv*delt(ip)+0.5)))
     ipjt=igrid(ip)+jt
     wz1=(zetatmp-zetamin)*delz
     kk=max(0,min(mzeta-1,int(wz1)))
     kzion(m)=kk
     wzion(m)=wz1-real(kk)

     do larmor=1,4
!        rdum=delr*max(0.0d+00,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        rdum=delr*max(zero,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        ii=max(0,min(mpsi-1,int(rdum)))
        wp1=rdum-real(ii)
        wpion(larmor,m)=wp1

! particle position in theta
        tflr=thetatmp+rhoi*tgyro(larmor,ipjt)

! inner flux surface
        im=ii
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j00=max(0,min(mtheta(im)-1,int(tdum)))
        jtion0(larmor,m)=igrid(im)+j00
        wtion0(larmor,m)=tdum-real(j00)

! outer flux surface
        im=ii+1
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j01=max(0,min(mtheta(im)-1,int(tdum)))
        jtion1(larmor,m)=igrid(im)+j01
        wtion1(larmor,m)=tdum-real(j01)
     enddo
  enddo

  if(istep==0)return

#ifdef _openmp
! the following lines are openmp directives for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
! when compiling with the openmp option (-qsmp=omp for the xlf compiler on
! aix and -mp for the mipspro compiler on irix), the preprocessor variable
! _openmp is automatically defined and we use it here to add pieces of codes
! needed to avoid contentions between threads. in the following parallel
! loop each thread has its own private array "dnitmp()" which is used to
! prevent the thread from writing into the shared array "densityi()"
!
!$omp parallel private(dnitmp)
!!!!  allocate(dnitmp(0:mzeta,mgrid))
  dnitmp=0.   ! set array elements to zero
#endif
!$omp do private(m,larmor,weight,kk,wz1,wz0,wp1,wp0,wt10,wt00,wt11,wt01,ij)
  do m=1,mi
     weight=zion(5,m)

     kk=kzion(m)
     wz1=weight*wzion(m)
     wz0=weight-wz1     

     do larmor=1,4
        wp1=wpion(larmor,m)
        wp0=1.0-wp1

        wt10=wp0*wtion0(larmor,m)
        wt00=wp0-wt10

        wt11=wp1*wtion1(larmor,m)
        wt01=wp1-wt11

#ifdef _openmp
! use thread-private temporary array dnitmp to store the results
        ij=jtion0(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt00
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt00
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt10
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt01
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt01
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt11
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt11
#else
! if no loop-level parallelism, use original algorithm (write directly
! into array "densityi()".
        ij=jtion0(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt00
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt00

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt10
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt01
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt01

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt11
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt11
#endif

     enddo
  enddo

#ifdef _openmp
! for openmp, we now write the results accumulated in each thread-private
! array dnitmp() back into the shared array densityi(). the loop is enclosed
! in a critical section so that one thread at a time updates densityi(). 
!$omp critical
  do ij=1,mgrid
     do kk=0,mzeta
        densityi(kk,ij) = densityi(kk,ij) + dnitmp(kk,ij)
     enddo
  enddo
!$omp end critical
!!!!  deallocate(dnitmp)
#endif
!$omp end parallel

! if we have a particle decomposition on the toroidal domains, do a reduce
! operation to add up all the contributions to charge density on the grid
  if(npartdom>1)then
   !$omp parallel do private(ij,kk)
    do ij=1,mgrid
       do kk=0,mzeta
          dnitmp(kk,ij)=densityi(kk,ij)
          densityi(kk,ij)=0.
       enddo
    enddo
    call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);
    !call mpi_allreduce(dnitmp,densityi,(mgrid*(mzeta+1)),mpi_rsize,&
    !                   mpi_sum,partd_comm,ierror)
  endif

! poloidal end cell, discard ghost cell j=0
  do i=0,mpsi
     densityi(:,igrid(i)+mtheta(i))=densityi(:,igrid(i)+mtheta(i))+densityi(:,igrid(i))
  enddo

! toroidal end cell
  sendl=densityi(0,:)
  recvr=0.0
  icount=mgrid
  !!!idest=mod(myrank_toroidal-1+ntoroidal,ntoroidal)
  idest=left_pe
  !!!isource=mod(myrank_toroidal+1,ntoroidal)
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource

! send densityi to left and receive from right
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)
     
  if(myrank_toroidal == ntoroidal-1)then
! b.c. at zeta=2*pi is shifted
     do i=0,mpsi
        ii=igrid(i)
        jt=mtheta(i)
        densityi(mzeta,ii+1:ii+jt)=densityi(mzeta,ii+1:ii+jt)+&
             cshift(recvr(ii+1:ii+jt),itran(i))
     enddo
  else
! b.c. at zeta<2*pi is continuous
     densityi(mzeta,:)=densityi(mzeta,:)+recvr
  endif
  
! zero out charge in radial boundary cell
  do i=0,nbound-1
     densityi(:,igrid(i):igrid(i)+mtheta(i))=&
	densityi(:,igrid(i):igrid(i)+mtheta(i))*real(i)/real(nbound)
     densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))=&
	densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))*real(i)/real(nbound)
  enddo

! flux surface average and normalization  
  zonali=0.0
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           zonali(i)=zonali(i)+0.25*densityi(k,ij)
           densityi(k,ij)=0.25*densityi(k,ij)*markeri(k,ij)
        enddo        
     enddo
  enddo

! global sum of phi00, broadcast to every toroidal pe
  !call mpi_allreduce(zonali,adum,mpsi+1,mpi_rsize,mpi_sum,toroidal_comm,ierror)
  call toroidal_allreduce_cmm(ptr,zonali,adum,mpsi+1)
  zonali=adum*pmarki

! densityi subtracted (0,0) mode
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           densityi(k,ij)=densityi(k,ij)-zonali(i)
        enddo
     enddo
! poloidal bc condition
     densityi(1:mzeta,igrid(i))=densityi(1:mzeta,igrid(i)+mtheta(i))
  enddo
  
! enforce charge conservation for zonal flow mode
  rdum=0.0
  tdum=0.0
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     rdum=rdum+r
     tdum=tdum+r*zonali(i)
  enddo
  tdum=tdum/rdum
  ddeni=tdum !for diagnostic
  do i=1,mpsi-1
     zonali(i)=zonali(i)-tdum
  enddo

end subroutine chargei_6

subroutine chargei_7(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_7
  use particle_array_7
  use field_array_7
  use diagnosis_array_7
  use particle_decomp_7
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer m,i,im,j,k,ip,jt,kk,ii,j11,j10,j01,j00,ierror,larmor,ij,ipjt,&
       icount,idest,isource,isendtag,irecvtag !,istatus(mpi_status_size)
  real(8) weight,wemark,rdum,tdum,delr,delt(0:mpsi),delz,smu_inv,r,wz1,wz0,&
       wp1,wp0,wt11,wt10,wt01,wt00,adum(0:mpsi),tflr,damping,pi2_inv,&
       psitmp,thetatmp,zetatmp,rhoi,deltheta
  real(8) sendl(mgrid),recvr(mgrid)
  real(8) dnitmp(0:mzeta,mgrid)

#ifdef _openmp
! we need the following temporary array only when using openmp
!!!  real,dimension(:,:),allocatable :: dnitmp
#endif

  delr=1.0/deltar 
  delt=2.0*pi/deltat
  delz=1.0/deltaz
  smu_inv=sqrt(aion)/(abs(qion)*gyroradius)
  pi2_inv=0.5/pi
  densityi=0.0

!$omp parallel do private(m,larmor,psitmp,thetatmp,zetatmp,rhoi,r,ip,jt,ipjt,&
!$omp& wz1,kk,rdum,ii,wp1,tflr,im,tdum,j00,j01)
  do m=1,mi
     psitmp=zion(1,m)
     thetatmp=zion(2,m)
     zetatmp=zion(3,m)
     rhoi=zion(6,m)*smu_inv

     r=sqrt(2.0*psitmp)
     ip=max(0,min(mpsi,int((r-a0)*delr+0.5)))
     jt=max(0,min(mtheta(ip),int(thetatmp*pi2_inv*delt(ip)+0.5)))
     ipjt=igrid(ip)+jt
     wz1=(zetatmp-zetamin)*delz
     kk=max(0,min(mzeta-1,int(wz1)))
     kzion(m)=kk
     wzion(m)=wz1-real(kk)

     do larmor=1,4
!        rdum=delr*max(0.0d+00,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        rdum=delr*max(zero,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        ii=max(0,min(mpsi-1,int(rdum)))
        wp1=rdum-real(ii)
        wpion(larmor,m)=wp1

! particle position in theta
        tflr=thetatmp+rhoi*tgyro(larmor,ipjt)

! inner flux surface
        im=ii
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j00=max(0,min(mtheta(im)-1,int(tdum)))
        jtion0(larmor,m)=igrid(im)+j00
        wtion0(larmor,m)=tdum-real(j00)

! outer flux surface
        im=ii+1
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j01=max(0,min(mtheta(im)-1,int(tdum)))
        jtion1(larmor,m)=igrid(im)+j01
        wtion1(larmor,m)=tdum-real(j01)
     enddo
  enddo

  if(istep==0)return

#ifdef _openmp
! the following lines are openmp directives for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
! when compiling with the openmp option (-qsmp=omp for the xlf compiler on
! aix and -mp for the mipspro compiler on irix), the preprocessor variable
! _openmp is automatically defined and we use it here to add pieces of codes
! needed to avoid contentions between threads. in the following parallel
! loop each thread has its own private array "dnitmp()" which is used to
! prevent the thread from writing into the shared array "densityi()"
!
!$omp parallel private(dnitmp)
!!!!  allocate(dnitmp(0:mzeta,mgrid))
  dnitmp=0.   ! set array elements to zero
#endif
!$omp do private(m,larmor,weight,kk,wz1,wz0,wp1,wp0,wt10,wt00,wt11,wt01,ij)
  do m=1,mi
     weight=zion(5,m)

     kk=kzion(m)
     wz1=weight*wzion(m)
     wz0=weight-wz1     

     do larmor=1,4
        wp1=wpion(larmor,m)
        wp0=1.0-wp1

        wt10=wp0*wtion0(larmor,m)
        wt00=wp0-wt10

        wt11=wp1*wtion1(larmor,m)
        wt01=wp1-wt11

#ifdef _openmp
! use thread-private temporary array dnitmp to store the results
        ij=jtion0(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt00
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt00
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt10
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt01
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt01
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt11
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt11
#else
! if no loop-level parallelism, use original algorithm (write directly
! into array "densityi()".
        ij=jtion0(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt00
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt00

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt10
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt01
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt01

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt11
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt11
#endif

     enddo
  enddo

#ifdef _openmp
! for openmp, we now write the results accumulated in each thread-private
! array dnitmp() back into the shared array densityi(). the loop is enclosed
! in a critical section so that one thread at a time updates densityi(). 
!$omp critical
  do ij=1,mgrid
     do kk=0,mzeta
        densityi(kk,ij) = densityi(kk,ij) + dnitmp(kk,ij)
     enddo
  enddo
!$omp end critical
!!!!  deallocate(dnitmp)
#endif
!$omp end parallel

! if we have a particle decomposition on the toroidal domains, do a reduce
! operation to add up all the contributions to charge density on the grid
  if(npartdom>1)then
   !$omp parallel do private(ij,kk)
    do ij=1,mgrid
       do kk=0,mzeta
          dnitmp(kk,ij)=densityi(kk,ij)
          densityi(kk,ij)=0.
       enddo
    enddo
    call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);
    !call mpi_allreduce(dnitmp,densityi,(mgrid*(mzeta+1)),mpi_rsize,&
    !                   mpi_sum,partd_comm,ierror)
  endif

! poloidal end cell, discard ghost cell j=0
  do i=0,mpsi
     densityi(:,igrid(i)+mtheta(i))=densityi(:,igrid(i)+mtheta(i))+densityi(:,igrid(i))
  enddo

! toroidal end cell
  sendl=densityi(0,:)
  recvr=0.0
  icount=mgrid
  !!!idest=mod(myrank_toroidal-1+ntoroidal,ntoroidal)
  idest=left_pe
  !!!isource=mod(myrank_toroidal+1,ntoroidal)
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource

! send densityi to left and receive from right
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)
     
  if(myrank_toroidal == ntoroidal-1)then
! b.c. at zeta=2*pi is shifted
     do i=0,mpsi
        ii=igrid(i)
        jt=mtheta(i)
        densityi(mzeta,ii+1:ii+jt)=densityi(mzeta,ii+1:ii+jt)+&
             cshift(recvr(ii+1:ii+jt),itran(i))
     enddo
  else
! b.c. at zeta<2*pi is continuous
     densityi(mzeta,:)=densityi(mzeta,:)+recvr
  endif
  
! zero out charge in radial boundary cell
  do i=0,nbound-1
     densityi(:,igrid(i):igrid(i)+mtheta(i))=&
	densityi(:,igrid(i):igrid(i)+mtheta(i))*real(i)/real(nbound)
     densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))=&
	densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))*real(i)/real(nbound)
  enddo

! flux surface average and normalization  
  zonali=0.0
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           zonali(i)=zonali(i)+0.25*densityi(k,ij)
           densityi(k,ij)=0.25*densityi(k,ij)*markeri(k,ij)
        enddo        
     enddo
  enddo

! global sum of phi00, broadcast to every toroidal pe
  !call mpi_allreduce(zonali,adum,mpsi+1,mpi_rsize,mpi_sum,toroidal_comm,ierror)
  call toroidal_allreduce_cmm(ptr,zonali,adum,mpsi+1)
  zonali=adum*pmarki

! densityi subtracted (0,0) mode
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           densityi(k,ij)=densityi(k,ij)-zonali(i)
        enddo
     enddo
! poloidal bc condition
     densityi(1:mzeta,igrid(i))=densityi(1:mzeta,igrid(i)+mtheta(i))
  enddo
  
! enforce charge conservation for zonal flow mode
  rdum=0.0
  tdum=0.0
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     rdum=rdum+r
     tdum=tdum+r*zonali(i)
  enddo
  tdum=tdum/rdum
  ddeni=tdum !for diagnostic
  do i=1,mpsi-1
     zonali(i)=zonali(i)-tdum
  enddo

end subroutine chargei_7

subroutine chargei_8(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_8
  use particle_array_8
  use field_array_8
  use diagnosis_array_8
  use particle_decomp_8
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer m,i,im,j,k,ip,jt,kk,ii,j11,j10,j01,j00,ierror,larmor,ij,ipjt,&
       icount,idest,isource,isendtag,irecvtag !,istatus(mpi_status_size)
  real(8) weight,wemark,rdum,tdum,delr,delt(0:mpsi),delz,smu_inv,r,wz1,wz0,&
       wp1,wp0,wt11,wt10,wt01,wt00,adum(0:mpsi),tflr,damping,pi2_inv,&
       psitmp,thetatmp,zetatmp,rhoi,deltheta
  real(8) sendl(mgrid),recvr(mgrid)
  real(8) dnitmp(0:mzeta,mgrid)

#ifdef _openmp
! we need the following temporary array only when using openmp
!!!  real,dimension(:,:),allocatable :: dnitmp
#endif

  delr=1.0/deltar 
  delt=2.0*pi/deltat
  delz=1.0/deltaz
  smu_inv=sqrt(aion)/(abs(qion)*gyroradius)
  pi2_inv=0.5/pi
  densityi=0.0

!$omp parallel do private(m,larmor,psitmp,thetatmp,zetatmp,rhoi,r,ip,jt,ipjt,&
!$omp& wz1,kk,rdum,ii,wp1,tflr,im,tdum,j00,j01)
  do m=1,mi
     psitmp=zion(1,m)
     thetatmp=zion(2,m)
     zetatmp=zion(3,m)
     rhoi=zion(6,m)*smu_inv

     r=sqrt(2.0*psitmp)
     ip=max(0,min(mpsi,int((r-a0)*delr+0.5)))
     jt=max(0,min(mtheta(ip),int(thetatmp*pi2_inv*delt(ip)+0.5)))
     ipjt=igrid(ip)+jt
     wz1=(zetatmp-zetamin)*delz
     kk=max(0,min(mzeta-1,int(wz1)))
     kzion(m)=kk
     wzion(m)=wz1-real(kk)

     do larmor=1,4
!        rdum=delr*max(0.0d+00,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        rdum=delr*max(zero,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        ii=max(0,min(mpsi-1,int(rdum)))
        wp1=rdum-real(ii)
        wpion(larmor,m)=wp1

! particle position in theta
        tflr=thetatmp+rhoi*tgyro(larmor,ipjt)

! inner flux surface
        im=ii
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j00=max(0,min(mtheta(im)-1,int(tdum)))
        jtion0(larmor,m)=igrid(im)+j00
        wtion0(larmor,m)=tdum-real(j00)

! outer flux surface
        im=ii+1
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j01=max(0,min(mtheta(im)-1,int(tdum)))
        jtion1(larmor,m)=igrid(im)+j01
        wtion1(larmor,m)=tdum-real(j01)
     enddo
  enddo

  if(istep==0)return

#ifdef _openmp
! the following lines are openmp directives for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
! when compiling with the openmp option (-qsmp=omp for the xlf compiler on
! aix and -mp for the mipspro compiler on irix), the preprocessor variable
! _openmp is automatically defined and we use it here to add pieces of codes
! needed to avoid contentions between threads. in the following parallel
! loop each thread has its own private array "dnitmp()" which is used to
! prevent the thread from writing into the shared array "densityi()"
!
!$omp parallel private(dnitmp)
!!!!  allocate(dnitmp(0:mzeta,mgrid))
  dnitmp=0.   ! set array elements to zero
#endif
!$omp do private(m,larmor,weight,kk,wz1,wz0,wp1,wp0,wt10,wt00,wt11,wt01,ij)
  do m=1,mi
     weight=zion(5,m)

     kk=kzion(m)
     wz1=weight*wzion(m)
     wz0=weight-wz1     

     do larmor=1,4
        wp1=wpion(larmor,m)
        wp0=1.0-wp1

        wt10=wp0*wtion0(larmor,m)
        wt00=wp0-wt10

        wt11=wp1*wtion1(larmor,m)
        wt01=wp1-wt11

#ifdef _openmp
! use thread-private temporary array dnitmp to store the results
        ij=jtion0(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt00
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt00
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt10
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt01
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt01
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt11
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt11
#else
! if no loop-level parallelism, use original algorithm (write directly
! into array "densityi()".
        ij=jtion0(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt00
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt00

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt10
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt01
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt01

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt11
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt11
#endif

     enddo
  enddo

#ifdef _openmp
! for openmp, we now write the results accumulated in each thread-private
! array dnitmp() back into the shared array densityi(). the loop is enclosed
! in a critical section so that one thread at a time updates densityi(). 
!$omp critical
  do ij=1,mgrid
     do kk=0,mzeta
        densityi(kk,ij) = densityi(kk,ij) + dnitmp(kk,ij)
     enddo
  enddo
!$omp end critical
!!!!  deallocate(dnitmp)
#endif
!$omp end parallel

! if we have a particle decomposition on the toroidal domains, do a reduce
! operation to add up all the contributions to charge density on the grid
  if(npartdom>1)then
   !$omp parallel do private(ij,kk)
    do ij=1,mgrid
       do kk=0,mzeta
          dnitmp(kk,ij)=densityi(kk,ij)
          densityi(kk,ij)=0.
       enddo
    enddo
    call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);
    !call mpi_allreduce(dnitmp,densityi,(mgrid*(mzeta+1)),mpi_rsize,&
    !                   mpi_sum,partd_comm,ierror)
  endif

! poloidal end cell, discard ghost cell j=0
  do i=0,mpsi
     densityi(:,igrid(i)+mtheta(i))=densityi(:,igrid(i)+mtheta(i))+densityi(:,igrid(i))
  enddo

! toroidal end cell
  sendl=densityi(0,:)
  recvr=0.0
  icount=mgrid
  !!!idest=mod(myrank_toroidal-1+ntoroidal,ntoroidal)
  idest=left_pe
  !!!isource=mod(myrank_toroidal+1,ntoroidal)
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource

! send densityi to left and receive from right
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)
     
  if(myrank_toroidal == ntoroidal-1)then
! b.c. at zeta=2*pi is shifted
     do i=0,mpsi
        ii=igrid(i)
        jt=mtheta(i)
        densityi(mzeta,ii+1:ii+jt)=densityi(mzeta,ii+1:ii+jt)+&
             cshift(recvr(ii+1:ii+jt),itran(i))
     enddo
  else
! b.c. at zeta<2*pi is continuous
     densityi(mzeta,:)=densityi(mzeta,:)+recvr
  endif
  
! zero out charge in radial boundary cell
  do i=0,nbound-1
     densityi(:,igrid(i):igrid(i)+mtheta(i))=&
	densityi(:,igrid(i):igrid(i)+mtheta(i))*real(i)/real(nbound)
     densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))=&
	densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))*real(i)/real(nbound)
  enddo

! flux surface average and normalization  
  zonali=0.0
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           zonali(i)=zonali(i)+0.25*densityi(k,ij)
           densityi(k,ij)=0.25*densityi(k,ij)*markeri(k,ij)
        enddo        
     enddo
  enddo

! global sum of phi00, broadcast to every toroidal pe
  !call mpi_allreduce(zonali,adum,mpsi+1,mpi_rsize,mpi_sum,toroidal_comm,ierror)
  call toroidal_allreduce_cmm(ptr,zonali,adum,mpsi+1)
  zonali=adum*pmarki

! densityi subtracted (0,0) mode
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           densityi(k,ij)=densityi(k,ij)-zonali(i)
        enddo
     enddo
! poloidal bc condition
     densityi(1:mzeta,igrid(i))=densityi(1:mzeta,igrid(i)+mtheta(i))
  enddo
  
! enforce charge conservation for zonal flow mode
  rdum=0.0
  tdum=0.0
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     rdum=rdum+r
     tdum=tdum+r*zonali(i)
  enddo
  tdum=tdum/rdum
  ddeni=tdum !for diagnostic
  do i=1,mpsi-1
     zonali(i)=zonali(i)-tdum
  enddo

end subroutine chargei_8

subroutine chargei_9(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_9
  use particle_array_9
  use field_array_9
  use diagnosis_array_9
  use particle_decomp_9
  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer m,i,im,j,k,ip,jt,kk,ii,j11,j10,j01,j00,ierror,larmor,ij,ipjt,&
       icount,idest,isource,isendtag,irecvtag !,istatus(mpi_status_size)
  real(8) weight,wemark,rdum,tdum,delr,delt(0:mpsi),delz,smu_inv,r,wz1,wz0,&
       wp1,wp0,wt11,wt10,wt01,wt00,adum(0:mpsi),tflr,damping,pi2_inv,&
       psitmp,thetatmp,zetatmp,rhoi,deltheta
  real(8) sendl(mgrid),recvr(mgrid)
  real(8) dnitmp(0:mzeta,mgrid)

#ifdef _openmp
! we need the following temporary array only when using openmp
!!!  real,dimension(:,:),allocatable :: dnitmp
#endif

  delr=1.0/deltar 
  delt=2.0*pi/deltat
  delz=1.0/deltaz
  smu_inv=sqrt(aion)/(abs(qion)*gyroradius)
  pi2_inv=0.5/pi
  densityi=0.0

!$omp parallel do private(m,larmor,psitmp,thetatmp,zetatmp,rhoi,r,ip,jt,ipjt,&
!$omp& wz1,kk,rdum,ii,wp1,tflr,im,tdum,j00,j01)
  do m=1,mi
     psitmp=zion(1,m)
     thetatmp=zion(2,m)
     zetatmp=zion(3,m)
     rhoi=zion(6,m)*smu_inv

     r=sqrt(2.0*psitmp)
     ip=max(0,min(mpsi,int((r-a0)*delr+0.5)))
     jt=max(0,min(mtheta(ip),int(thetatmp*pi2_inv*delt(ip)+0.5)))
     ipjt=igrid(ip)+jt
     wz1=(zetatmp-zetamin)*delz
     kk=max(0,min(mzeta-1,int(wz1)))
     kzion(m)=kk
     wzion(m)=wz1-real(kk)

     do larmor=1,4
!        rdum=delr*max(0.0d+00,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        rdum=delr*max(zero,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
        ii=max(0,min(mpsi-1,int(rdum)))
        wp1=rdum-real(ii)
        wpion(larmor,m)=wp1

! particle position in theta
        tflr=thetatmp+rhoi*tgyro(larmor,ipjt)

! inner flux surface
        im=ii
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j00=max(0,min(mtheta(im)-1,int(tdum)))
        jtion0(larmor,m)=igrid(im)+j00
        wtion0(larmor,m)=tdum-real(j00)

! outer flux surface
        im=ii+1
        tdum=pi2_inv*(tflr-zetatmp*qtinv(im))+10.0
        tdum=(tdum-aint(tdum))*delt(im)
        j01=max(0,min(mtheta(im)-1,int(tdum)))
        jtion1(larmor,m)=igrid(im)+j01
        wtion1(larmor,m)=tdum-real(j01)
     enddo
  enddo

  if(istep==0)return

#ifdef _openmp
! the following lines are openmp directives for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
! when compiling with the openmp option (-qsmp=omp for the xlf compiler on
! aix and -mp for the mipspro compiler on irix), the preprocessor variable
! _openmp is automatically defined and we use it here to add pieces of codes
! needed to avoid contentions between threads. in the following parallel
! loop each thread has its own private array "dnitmp()" which is used to
! prevent the thread from writing into the shared array "densityi()"
!
!$omp parallel private(dnitmp)
!!!!  allocate(dnitmp(0:mzeta,mgrid))
  dnitmp=0.   ! set array elements to zero
#endif
!$omp do private(m,larmor,weight,kk,wz1,wz0,wp1,wp0,wt10,wt00,wt11,wt01,ij)
  do m=1,mi
     weight=zion(5,m)

     kk=kzion(m)
     wz1=weight*wzion(m)
     wz0=weight-wz1     

     do larmor=1,4
        wp1=wpion(larmor,m)
        wp0=1.0-wp1

        wt10=wp0*wtion0(larmor,m)
        wt00=wp0-wt10

        wt11=wp1*wtion1(larmor,m)
        wt01=wp1-wt11

#ifdef _openmp
! use thread-private temporary array dnitmp to store the results
        ij=jtion0(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt00
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt00
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt10
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt01
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt01
        
        ij=ij+1
        dnitmp(kk,ij) = dnitmp(kk,ij) + wz0*wt11
        dnitmp(kk+1,ij)   = dnitmp(kk+1,ij)   + wz1*wt11
#else
! if no loop-level parallelism, use original algorithm (write directly
! into array "densityi()".
        ij=jtion0(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt00
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt00

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt10
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt10

        ij=jtion1(larmor,m)
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt01
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt01

        ij=ij+1
        densityi(kk,ij) = densityi(kk,ij) + wz0*wt11
        densityi(kk+1,ij)   = densityi(kk+1,ij)   + wz1*wt11
#endif

     enddo
  enddo

#ifdef _openmp
! for openmp, we now write the results accumulated in each thread-private
! array dnitmp() back into the shared array densityi(). the loop is enclosed
! in a critical section so that one thread at a time updates densityi(). 
!$omp critical
  do ij=1,mgrid
     do kk=0,mzeta
        densityi(kk,ij) = densityi(kk,ij) + dnitmp(kk,ij)
     enddo
  enddo
!$omp end critical
!!!!  deallocate(dnitmp)
#endif
!$omp end parallel

! if we have a particle decomposition on the toroidal domains, do a reduce
! operation to add up all the contributions to charge density on the grid
  if(npartdom>1)then
   !$omp parallel do private(ij,kk)
    do ij=1,mgrid
       do kk=0,mzeta
          dnitmp(kk,ij)=densityi(kk,ij)
          densityi(kk,ij)=0.
       enddo
    enddo
    call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);
    !call mpi_allreduce(dnitmp,densityi,(mgrid*(mzeta+1)),mpi_rsize,&
    !                   mpi_sum,partd_comm,ierror)
  endif

! poloidal end cell, discard ghost cell j=0
  do i=0,mpsi
     densityi(:,igrid(i)+mtheta(i))=densityi(:,igrid(i)+mtheta(i))+densityi(:,igrid(i))
  enddo

! toroidal end cell
  sendl=densityi(0,:)
  recvr=0.0
  icount=mgrid
  !!!idest=mod(myrank_toroidal-1+ntoroidal,ntoroidal)
  idest=left_pe
  !!!isource=mod(myrank_toroidal+1,ntoroidal)
  isource=right_pe
  isendtag=myrank_toroidal
  irecvtag=isource

! send densityi to left and receive from right
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)
     
  if(myrank_toroidal == ntoroidal-1)then
! b.c. at zeta=2*pi is shifted
     do i=0,mpsi
        ii=igrid(i)
        jt=mtheta(i)
        densityi(mzeta,ii+1:ii+jt)=densityi(mzeta,ii+1:ii+jt)+&
             cshift(recvr(ii+1:ii+jt),itran(i))
     enddo
  else
! b.c. at zeta<2*pi is continuous
     densityi(mzeta,:)=densityi(mzeta,:)+recvr
  endif
  
! zero out charge in radial boundary cell
  do i=0,nbound-1
     densityi(:,igrid(i):igrid(i)+mtheta(i))=&
	densityi(:,igrid(i):igrid(i)+mtheta(i))*real(i)/real(nbound)
     densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))=&
	densityi(:,igrid(mpsi-i):igrid(mpsi-i)+mtheta(mpsi-i))*real(i)/real(nbound)
  enddo

! flux surface average and normalization  
  zonali=0.0
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           zonali(i)=zonali(i)+0.25*densityi(k,ij)
           densityi(k,ij)=0.25*densityi(k,ij)*markeri(k,ij)
        enddo        
     enddo
  enddo

! global sum of phi00, broadcast to every toroidal pe
  !call mpi_allreduce(zonali,adum,mpsi+1,mpi_rsize,mpi_sum,toroidal_comm,ierror)
  call toroidal_allreduce_cmm(ptr,zonali,adum,mpsi+1)
  zonali=adum*pmarki

! densityi subtracted (0,0) mode
!$omp parallel do private(i,j,k,ij)
  do i=0,mpsi
     do j=1,mtheta(i)
        do k=1,mzeta
           ij=igrid(i)+j
           densityi(k,ij)=densityi(k,ij)-zonali(i)
        enddo
     enddo
! poloidal bc condition
     densityi(1:mzeta,igrid(i))=densityi(1:mzeta,igrid(i)+mtheta(i))
  enddo
  
! enforce charge conservation for zonal flow mode
  rdum=0.0
  tdum=0.0
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     rdum=rdum+r
     tdum=tdum+r*zonali(i)
  enddo
  tdum=tdum/rdum
  ddeni=tdum !for diagnostic
  do i=1,mpsi-1
     zonali(i)=zonali(i)-tdum
  enddo

end subroutine chargei_9

