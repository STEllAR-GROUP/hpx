subroutine smooth(ptr,iflag)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters
  use field_array
  use diagnosis_array
  use particle_decomp
  implicit none
  TYPE(C_PTR), INTENT(IN), VALUE :: ptr
  integer iflag,i,j,k,ii,ij,ip,jt,kz,ismooth,mz,mzmax,mzbig,jpe,indp,indt,&
       indp1,indt1,meachtheta,jtp,icount,ierror,idest,isource,isendtag,&
       irecvtag!,istatus(MPI_STATUS_SIZE)
  real(wp) sendl(mgrid),sendr(mgrid),recvl(mgrid),recvr(mgrid),phism(mgrid),&
       ptemp(mzeta),pleft(mthetamax),pright(mthetamax),phitmp(0:mzeta,mgrid),&
       filter(mtdiag/2+1),wt,r,dt,wz,zdum,tdum,pi2_inv,den00(0:mpsi),&
       phiflux(mtdiag/ntoroidal,mtdiag,idiag1:idiag2),&
       eachzeta((idiag2-idiag1+1)*mtdiag*mtdiag/ntoroidal/ntoroidal),&
       allzeta((idiag2-idiag1+1)*mtdiag*mtdiag/ntoroidal),xz(mtdiag)
  complex(wp) y_eigen(mtdiag/ntoroidal*num_mode),yz(mtdiag/2+1),&
       yt(mtdiag*num_mode),ye(mtdiag),yp(mpsi/2+1)
! the dummy array size for thread-safe FFT assume data array size <16384
  real(doubleprec) :: aux1f(25000),aux1b(25000),aux2f(20000),aux2b(20000)
  real(doubleprec) :: aux3f(1),aux3b(1)
  real(wp) :: scale
  real(wp) energy_unit,energy_unit_sq

  scale=1.0_wp
  phitmp=0.

  if(iflag==0)then
!$omp parallel do private(i)
     do i=1,mgrid
        phitmp(1:mzeta,i)=densityi(1:mzeta,i)
     enddo

  elseif(iflag==1)then
!$omp parallel do private(i)
     do i=1,mgrid
        phitmp(1:mzeta,i)=densitye(1:mzeta,i)
     enddo

  else
!$omp parallel do private(i)
     do i=1,mgrid
        phitmp(1:mzeta,i)=phi(1:mzeta,i)
     enddo
  endif

  ismooth=1
  if(nonlinear<0.5)ismooth=0
  do ip=1,ismooth

! radial smoothing
     do i=1,mpsi-1
        phitmp(:,igrid(i))=phitmp(:,igrid(i)+mtheta(i))
     enddo
     do k=1,mzeta
        phism=0.0
!$omp parallel do private(i,j,ij)        
        do i=1,mpsi-1
           do j=1,mtheta(i)
              ij=igrid(i)+j
              phism(ij)=0.25*((1.0-wtp1(1,ij,k))*phitmp(k,jtp1(1,ij,k))+&
                   wtp1(1,ij,k)*phitmp(k,jtp1(1,ij,k)+1)+&
                   (1.0-wtp1(2,ij,k))*phitmp(k,jtp1(2,ij,k))+&
                   wtp1(2,ij,k)*phitmp(k,jtp1(2,ij,k)+1))-&
                   0.0625*((1.0-wtp2(1,ij,k))*phitmp(k,jtp2(1,ij,k))+&
                   wtp2(1,ij,k)*phitmp(k,jtp2(1,ij,k)+1)+&
                   (1.0-wtp2(2,ij,k))*phitmp(k,jtp2(2,ij,k))+&
                   wtp2(2,ij,k)*phitmp(k,jtp2(2,ij,k)+1))
              
           enddo
        enddo
        phitmp(k,:)=0.625*phitmp(k,:)+phism
     enddo

! poloidal smoothing (-0.0625 0.25 0.625 0.25 -0.0625)
!$omp parallel do private(i,j,ii,jt,pright)        
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        do k=1,mzeta
           pright(1:jt)=phitmp(k,ii+1:ii+jt)
           phitmp(k,ii+1:ii+jt)=0.625*pright(1:jt)+&
                0.25*(cshift(pright(1:jt),-1)+cshift(pright(1:jt),1))-&
                0.0625*(cshift(pright(1:jt),-2)+cshift(pright(1:jt),2))
        enddo
     enddo

! parallel smoothing 
! send phi to right and receive from left
     sendr=phitmp(mzeta,:)
     recvl=0.0
     icount=mgrid
     idest=right_pe
     isource=left_pe
     isendtag=myrank_toroidal
     irecvtag=isource
     !call MPI_SENDRECV(sendr,icount,mpi_Rsize,idest,isendtag,recvl,icount,&
     !     mpi_Rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
     call sndright_toroidal_cmm(ptr,sendr,icount)
     call rcvleft_toroidal_cmm(ptr,recvl)
     
! send phi to left and receive from right
     sendl=phitmp(1,:)
     recvr=0.0
     idest=left_pe
     isource=right_pe
     isendtag=myrank_toroidal
     irecvtag=isource
     !call MPI_SENDRECV(sendl,icount,mpi_Rsize,idest,isendtag,recvr,icount,&
     !     mpi_Rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
     call sndleft_toroidal_cmm(ptr,sendl,icount)
     call rcvright_toroidal_cmm(ptr,recvr)
    
!$omp parallel do private(i,ii,jt,j,ij,ptemp,pleft,pright)
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        if(myrank_toroidal==0)then !down-shift for zeta=0
           pleft(1:jt)=cshift(recvl(ii+1:ii+jt),-itran(i))
           pright(1:jt)=recvr(ii+1:ii+jt)
        elseif(myrank_toroidal==ntoroidal-1)then !up-shift for zeta=2*pi
           pright(1:jt)=cshift(recvr(ii+1:ii+jt),itran(i))
           pleft(1:jt)=recvl(ii+1:ii+jt)
        else
           pleft(1:jt)=recvl(ii+1:ii+jt)
           pright(1:jt)=recvr(ii+1:ii+jt)
        endif
        
        do j=1,mtheta(i)
           ij=igrid(i)+j
           ptemp=phitmp(1:mzeta,ij)
           if(mzeta==1)then
              phitmp(1,ij)=0.5*ptemp(1)+0.25*(pleft(j)+pright(j))
           elseif(mzeta==2)then
              phitmp(1,ij)=0.5*ptemp(1)+0.25*(pleft(j)+ptemp(2))
              phitmp(2,ij)=0.5*ptemp(2)+0.25*(ptemp(1)+pright(j))
           else
              phitmp(1,ij)=0.5*ptemp(1)+0.25*(pleft(j)+ptemp(2))
              phitmp(mzeta,ij)=0.5*ptemp(mzeta)+0.25*(ptemp(mzeta-1)+pright(j))
              phitmp(2:mzeta-1,ij)=0.5*ptemp(2:mzeta-1)+&
                   0.25*(ptemp(1:mzeta-2)+ptemp(3:mzeta))
           endif
        enddo
     enddo
  enddo

! toroidal BC: send phi to right and receive from left
  sendr=phitmp(mzeta,:)
  recvl=0.0
  icount=mgrid
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call MPI_SENDRECV(sendr,icount,mpi_Rsize,idest,isendtag,&
  !     recvl,icount,mpi_Rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendr,icount)
  call rcvleft_toroidal_cmm(ptr,recvl)
     
  if(myrank_toroidal==0)then
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        phitmp(0,ii+1:ii+jt)=cshift(recvl(ii+1:ii+jt),-itran(i))
     enddo
  else
     phitmp(0,:)=recvl
  endif

! poloidal BC
  do i=1,mpsi-1
     phitmp(:,igrid(i))=phitmp(:,igrid(i)+mtheta(i))
  enddo

! radial boundary
  phitmp(:,igrid(0):igrid(0)+mtheta(0))=0.0
  phitmp(:,igrid(mpsi):igrid(mpsi)+mtheta(mpsi))=0.0

  if(iflag==0)then
!$omp parallel do private(i)
     do i=1,mgrid
        densityi(:,i)=phitmp(:,i)
     enddo

  elseif(iflag==1)then
!$omp parallel do private(i)
     do i=1,mgrid
        densitye(:,i)=phitmp(:,i)
     enddo

  else
!$omp parallel do private(i)
     do i=1,mgrid
        phi(:,i)=phitmp(:,i)
     enddo
  endif

! solve zonal flow: phi00=r*E_r, E_r(a0)=0. Trapezoid rule
  if(iflag==3)then
     if(nhybrid==0)phip00=qion*zonali
     if(nhybrid>0)phip00=qion*zonali+qelectron*zonale
! (-0.0625 0.25 0.625 0.25 -0.0625) radial smoothing of (0,0) mode density phip00
     do ismooth=1,1
        den00(0)=phip00(0)
        den00(mpsi)=phip00(mpsi)
        den00(1)=phip00(3)
        den00(mpsi-1)=phip00(mpsi-3)
        den00(2:mpsi-2)=phip00(0:mpsi-4)+phip00(4:mpsi)
        den00(1:mpsi-1)=0.625*phip00(1:mpsi-1)+0.25*(phip00(0:mpsi-2)+phip00(2:mpsi))-&
             0.0625*den00(1:mpsi-1)
        phip00=den00
     enddo

     den00=phip00
     phip00=0.0
     do i=1,mpsi
        r=a0+deltar*real(i)
        phip00(i)=phip00(i-1)+0.5*deltar*((r-deltar)*den00(i-1)+r*den00(i))
     enddo

! subtract net momentum
!     phip00=phip00-sum(phip00)/real(mpsi+1)

! d phi/dr, in equilibrium unit
     do i=0,mpsi
        r=a0+deltar*real(i)
        phip00(i)=-phip00(i)/r
     enddo

! add FLR contribution using Pade approximation: b*<phi>=(1+b)*<n>
     phi00=den00*gyroradius*gyroradius ! potential in equilibrium unit
     do i=1,mpsi-1
        phip00(i)=phip00(i)+0.5*(phi00(i+1)-phi00(i-1))/deltar
     enddo

! (0,0) mode potential store in phi00
     phi00=0.0
     do i=1,mpsi
        phi00(i)=phi00(i-1)+0.5*deltar*(phip00(i-1)+phip00(i))
     enddo
     if(mode00==0)phip00=0.0
  endif

! Interpolate on a flux surface from fieldline coordinates to magnetic
! coordinates. Use mtdiag for both poloidal and toroidal grid points.
  if(iflag>1)then
     if(nonlinear<0.5 .or. (iflag==3 .and. idiag==0))then
        xz=0.0
        yz=0.0
#ifdef __ESSL
      ! Initialization of the FFT tables.
        if(wp.eq.singleprec)then
           call srcft(1,xz,0,yz,0,mtdiag,1, 1,1.0,aux1f,25000,&
                      aux2f,20000,aux3f,1)
           call scrft(1,yz,0,xz,0,mtdiag,1,-1,1.0,aux1b,25000,&
                      aux2b,20000,aux3b,1)
        else
           call drcft(1,xz,0,yz,0,mtdiag,1, 1,1.0,aux1f,25000,&
                      aux2f,20000,aux3f,1)
           call dcrft(1,yz,0,xz,0,mtdiag,1,-1,1.0,aux1b,25000,&
                      aux2b,20000,aux3b,1)
        endif
#endif
     
        mzbig=max(1,mtdiag/mzetamax)
        mzmax=mzetamax*mzbig
        mz=mzeta*mzbig
        meachtheta=mtdiag/ntoroidal
        icount=meachtheta*mz*(idiag2-idiag1+1)  
        dt=2.0*pi/real(mtdiag)
        pi2_inv=0.5/pi
        filter=0.0
        filter(nmode+1)=1.0/real(mzmax)
        allzeta=0.0

        do k=1,mzeta

!$omp parallel do private(kz,i,j,wz,zdum,ii,tdum,jt,wt)           
           do kz=1,mzbig
              wz=real(kz)/real(mzbig)
              zdum=zetamin+deltaz*(real(k-1)+wz)
              do i=idiag1,idiag2
                 ii=igrid(i)
                 do j=1,mtdiag
                    tdum=pi2_inv*(dt*real(j)-zdum*qtinv(i))+10.0
                    tdum=(tdum-aint(tdum))*real(mtheta(i))
                    jt=max(0,min(mtheta(i)-1,int(tdum)))
                    wt=tdum-real(jt)
                 
                    phiflux(kz+(k-1)*mzbig,j,i)=((1.0-wt)*phi(k,ii+jt)+wt*phi(k,ii+jt+1))*&
                         wz+(1.0-wz)*((1.0-wt)*phi(k-1,ii+jt)+wt*phi(k-1,ii+jt+1))
                 enddo
              enddo
           enddo           
        enddo

! transpose 2-d matrix from (ntoroidal,mzeta*mzbig) to (1,mzetamax*mzbig)
        do jpe=0,ntoroidal-1

!$omp parallel do private(j,i,k,jt,indt,indp1,indp)
           do j=1,meachtheta
              jt=jpe*meachtheta+j
              indt=(j-1)*mz
              do i=idiag1,idiag2
                 indp1=indt+(i-idiag1)*meachtheta*mz
                 do k=1,mz
                    indp=indp1+k
                    eachzeta(indp)=phiflux(k,jt,i)
                 enddo
              enddo
           enddo
           
           !call MPI_GATHER(eachzeta,icount,mpi_Rsize,allzeta,icount,&
           !          mpi_Rsize,jpe,toroidal_comm,ierror)
           call toroidal_gather_cmm(ptr,eachzeta,icount,jpe)
           call toroidal_gather_receive_cmm(ptr,allzeta,jpe)
        enddo
        
! transform to k space
!$omp parallel do private(j,i,kz,k,indt1,indt,indp,xz,yz) firstprivate(aux2f,aux3f,aux2b,aux3b)
        do j=1,meachtheta
           indt1=(j-1)*mz
           do i=idiag1,idiag2
              indt=indt1+(i-idiag1)*meachtheta*mz
              
              do kz=0,ntoroidal-1
                 do k=1,mz
                    indp=kz*icount+indt+k
                    xz(kz*mz+k)=allzeta(indp)
                 enddo
              enddo
              
#ifdef __ESSL
              if(wp.eq.singleprec)then
                 call srcft(0,xz,0,yz,0,mtdiag,1, 1,1.0,aux1f,25000,&
                            aux2f,20000,aux3f,1)
              else
                 call drcft(0,xz,0,yz,0,mtdiag,1, 1,1.0,aux1f,25000,&
                            aux2f,20000,aux3f,1)
              endif
#else
              ! calling this is generating a segfault on various platforms --
              ! what's up?
              !call fftr1d(1,mtdiag,scale,xz,yz,2)
#endif

! record mode information for diagnostic
              if(i==mpsi/2)then
                 do kz=1,num_mode
                    y_eigen(j+meachtheta*(kz-1))=yz(nmode(kz)+1)
                 enddo
              endif

              if(nonlinear<0.5)then
! linear run only keep a few modes
                 yz=filter*yz

! transform back to real space
#ifdef __ESSL
                 if(wp.eq.singleprec)then
                    call scrft(0,yz,0,xz,0,mtdiag,1,-1,1.0,aux1b,25000,&
                               aux2b,20000,aux3b,1)
                 else
                    call dcrft(0,yz,0,xz,0,mtdiag,1,-1,1.0,aux1b,25000,&
                               aux2b,20000,aux3b,1)
                 endif
#else
                 call fftr1d(-1,mtdiag,scale,xz,yz,2)
#endif

! transpose back to (ntoroidal,mz)
                 do kz=0,ntoroidal-1
                    do k=1,mz
                       indp=kz*icount+indt+k
                       allzeta(indp)=xz(kz*mz+k)
                    enddo
                 enddo
              endif
           enddo
        enddo

        if(nonlinear<0.5)then
           do jpe=0,ntoroidal-1
            !  call MPI_SCATTER(allzeta,icount,mpi_Rsize,eachzeta,&
            !       icount,mpi_Rsize,jpe,toroidal_comm,ierror)
             call toroidal_scatter_cmm(ptr,allzeta,icount,jpe)
             call toroidal_scatter_receive_cmm(ptr,eachzeta,jpe)

!$omp parallel do private(j,i,k,jt,indt,indp1,indp)
              do j=1,meachtheta
                 jt=jpe*meachtheta+j
                 indt=(j-1)*mz
                 do i=idiag1,idiag2
                    indp1=indt+(i-idiag1)*meachtheta*mz
                    do k=1,mz
                       indp=indp1+k
                       phiflux(k,jt,i)=eachzeta(indp)
                    enddo
                 enddo
              enddo              
           enddo
        
! interpolate field from magnetic coordinates to fieldline coordinates
           do k=1,mzeta
              zdum=zetamin+deltaz*real(k)

!$omp parallel do private(i,j,ii,tdum,jt,wt)           
              do i=idiag1,idiag2
                 ii=igrid(i)              
                 do j=1,mtheta(i)
                    tdum=pi2_inv*(deltat(i)*real(j)+zdum*qtinv(i))+10.0
                    tdum=(tdum-aint(tdum))*real(mtdiag)
                    jt=max(0,min(mtdiag-1,int(tdum)))
                    wt=tdum-real(jt)
                    jtp=jt+1
                    if(jt==0)jt=mtdiag

                    phi(k,ii+j)=wt*phiflux(k*mzbig,jtp,i)+(1.0-wt)*phiflux(k*mzbig,jt,i)
                 enddo
              enddo              
           enddo

! toroidal BC: send phi to right and receive from left
           sendr=phi(mzeta,:)
           recvl=0.0
           icount=mgrid
           idest=right_pe
           isource=left_pe
           isendtag=myrank_toroidal
           irecvtag=isource
           !call MPI_SENDRECV(sendr,icount,mpi_Rsize,idest,isendtag,recvl,&
           !     icount,mpi_Rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
           call sndright_toroidal_cmm(ptr,sendr,icount)
           call rcvleft_toroidal_cmm(ptr,recvl)
           
           if(myrank_toroidal==0)then
              do i=idiag1,idiag2
                 ii=igrid(i)
                 jt=mtheta(i)
                 phi(0,ii+1:ii+jt)=cshift(recvl(ii+1:ii+jt),-itran(i))
              enddo
           else
              phi(0,:)=recvl
           endif

! poloidal BC
           do i=idiag1,idiag2
              phi(:,igrid(i))=phi(:,igrid(i)+mtheta(i))
           enddo
        endif
     endif
  endif

  if(iflag==2)then
     j=irk+2*(ihybrid-1)

!$omp parallel do private(i)
     do i=1,mgrid
        phit(:,i)=(phi(:,i)-phisave(:,i,j))/tstep
        phisave(:,i,j)=phi(:,i)
     enddo
  endif

! diagnostic
  if(iflag==3 .and. idiag==0)then

! shear flow amplitude
     eradial=sqrt(sum(phip00(1:mpsi)**2)/real(mpsi))/gyroradius

! RMS of potential
     efield=0.0
     do i=0,mpsi
        efield=efield+sum(phi(1:mzeta,igrid(i)+1:igrid(i)+mtheta(i))**2)
     enddo
     !!!efield=sqrt(efield/real(mzeta*sum(mtheta)))/(gyroradius*gyroradius)
     energy_unit=gyroradius*gyroradius
     energy_unit_sq=energy_unit*energy_unit
     efield=efield/(real(mzeta*sum(mtheta))*energy_unit_sq)

     !!write(0,*)'**** mype=',mype,'  efield=',efield
     
! record zonal flow mode history
     ! segfault problem with this routine
     !call fftr1d(1,mpsi,scale,phip00(1:mpsi),yp,1)

     do i=1,num_mode
        amp_mode(1,i,1)=real(yp(i+1))/(real(mpsi)*gyroradius)
        amp_mode(2,i,1)=aimag(yp(i+1))/(real(mpsi)*gyroradius)
     enddo
     
! dominant (n,m) mode history data for flux surface at i=mpsi/2
     icount=meachtheta*num_mode
     !call MPI_GATHER(y_eigen,icount,mpi_Csize,yt,icount,&
     !     mpi_Csize,0,toroidal_comm,ierror)
     call toroidal_gather_cmm(ptr,y_eigen,icount,0)
     call toroidal_gather_receive_cmm(ptr,yt,0)
     if(myrank_toroidal == 0)then
        do kz=1,num_mode
           
           do i=0,ntoroidal-1
              do j=1,meachtheta
                 ye(j+i*meachtheta)=yt(j+(kz-1)*meachtheta+i*icount)
              enddo
           enddo
           ! segfault problem with this routine
           !call fftc1d(1,mtdiag,scale,ye)
           
           amp_mode(1,kz,2)=real(ye(mtdiag-mmode(kz)+1))/&
                (real(mzmax*mtdiag)*gyroradius**2)
           amp_mode(2,kz,2)=aimag(ye(mtdiag-mmode(kz)+1))/&
                (real(mzmax*mtdiag)*gyroradius**2)
        enddo
     endif
  endif

end subroutine smooth
