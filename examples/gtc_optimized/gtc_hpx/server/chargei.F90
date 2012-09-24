subroutine chargei(ptr)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters
  use particle_array
  use field_array
  use diagnosis_array
  use particle_decomp
  implicit none
  TYPE(C_PTR), INTENT(IN), VALUE :: ptr
  integer m,i,im,j,k,ip,jt,kk,ii,j11,j10,j01,j00,ierror,larmor,ij,ipjt,&
       icount,idest,isource,isendtag,irecvtag !,istatus(MPI_STATUS_SIZE)
  real(wp) weight,wemark,rdum,tdum,delr,delt(0:mpsi),delz,smu_inv,r,wz1,wz0,&
       wp1,wp0,wt11,wt10,wt01,wt00,adum(0:mpsi),tflr,damping,pi2_inv,&
       psitmp,thetatmp,zetatmp,rhoi,deltheta
  real(wp) sendl(mgrid),recvr(mgrid)
  real(wp) dnitmp(0:mzeta,mgrid)

#ifdef _OPENMP
! We need the following temporary array only when using OpenMP
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
        rdum=delr*max(ZERO,min(a1-a0,r+rhoi*pgyro(larmor,ipjt)-a0))
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

#ifdef _OPENMP
! The following lines are OpenMP directives for loop-level parallelism
! on shared memory machines (see http://www.openmp.org).
! When compiling with the OpenMP option (-qsmp=omp for the XLF compiler on
! AIX and -mp for the MIPSpro compiler on IRIX), the preprocessor variable
! _OPENMP is automatically defined and we use it here to add pieces of codes
! needed to avoid contentions between threads. In the following parallel
! loop each thread has its own private array "dnitmp()" which is used to
! prevent the thread from writing into the shared array "densityi()"
!
!$omp parallel private(dnitmp)
!!!!  allocate(dnitmp(0:mzeta,mgrid))
  dnitmp=0.   ! Set array elements to zero
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

#ifdef _OPENMP
! Use thread-private temporary array dnitmp to store the results
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
! If no loop-level parallelism, use original algorithm (write directly
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

#ifdef _OPENMP
! For OpenMP, we now write the results accumulated in each thread-private
! array dnitmp() back into the shared array densityi(). The loop is enclosed
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

! If we have a particle decomposition on the toroidal domains, do a reduce
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
    !call MPI_ALLREDUCE(dnitmp,densityi,(mgrid*(mzeta+1)),mpi_Rsize,&
    !                   MPI_SUM,partd_comm,ierror)
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
  !call MPI_SENDRECV(sendl,icount,mpi_Rsize,idest,isendtag,&
  !     recvr,icount,mpi_Rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)
     
  if(myrank_toroidal == ntoroidal-1)then
! B.C. at zeta=2*pi is shifted
     do i=0,mpsi
        ii=igrid(i)
        jt=mtheta(i)
        densityi(mzeta,ii+1:ii+jt)=densityi(mzeta,ii+1:ii+jt)+&
             cshift(recvr(ii+1:ii+jt),itran(i))
     enddo
  else
! B.C. at zeta<2*pi is continuous
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

! global sum of phi00, broadcast to every toroidal PE
  !call MPI_ALLREDUCE(zonali,adum,mpsi+1,mpi_Rsize,MPI_SUM,toroidal_comm,ierror)
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
! poloidal BC condition
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

end subroutine chargei

