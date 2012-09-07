subroutine fieldr_0(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_0
  use field_array_0
  use diagnosis_array_0
  use particle_decomp_0
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,ii,ij,j,k,icount,idest,isource,isendtag,irecvtag,ierror,ip,jt !,&
       !istatus(mpi_status_size)
  real(8) diffr,difft(0:mpsi),diffz,r,drdp,pleft(mthetamax),pright(mthetamax),&
       sendl(mgrid),recvr(mgrid),sendr(mgrid),recvl(mgrid),sendrs(3,mgrid),&
       recvls(3,mgrid),q,delq

! finite difference for e-field in equilibrium unit
  diffr=0.5_wp/deltar
  difft=0.5_wp/deltat
  diffz=0.5_wp/deltaz
!$omp parallel do private(i,j,k)
  do i=1,mgrid
     do j=0,mzeta
        do k=1,3
           evector(k,j,i)=0.0
        enddo
     enddo
  enddo

! d_phi/d_psi
  do k=1,mzeta
!$omp parallel do private(i,j,r,drdp,ij)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        drdp=1.0/r
        do j=1,mtheta(i)
           ij=igrid(i)+j

           evector(1,k,ij)=drdp*diffr*((1.0-wtp1(1,ij,k))*phi(k,jtp1(1,ij,k))+&
                wtp1(1,ij,k)*phi(k,jtp1(1,ij,k)+1)-&
                ((1.0-wtp1(2,ij,k))*phi(k,jtp1(2,ij,k))+&
                wtp1(2,ij,k)*phi(k,jtp1(2,ij,k)+1)))

        enddo
     enddo
  enddo

! d_phi/d_theta
!$omp parallel do private(i,k,j,ij,jt)
  do i=1,mpsi-1
     do k=1,mzeta
        do j=1,mtheta(i)
           ij=igrid(i)+j
           jt=j+1-mtheta(i)*(j/mtheta(i))
           evector(2,k,ij)=difft(i)*(phi(k,igrid(i)+jt)-phi(k,igrid(i)+j-1))
        enddo
     enddo
  enddo
  
! send phi to right and receive from left
  sendr=phi(mzeta,:)
  recvl=0.0
  icount=mgrid
  !!idest=mod(mype+1,numberpe)
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendr,icount,mpi_rsize,idest,isendtag,&
  !     recvl,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendr,icount)
  call rcvleft_toroidal_cmm(ptr,recvl)
  
! send phi to left and receive from right
  sendl=phi(1,:)
  recvr=0.0
  !!idest=mod(mype-1+numberpe,numberpe)
  idest=left_pe
  !!isource=mod(mype+1,numberpe)
  isource=right_pe
  !!isendtag=mype
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)

! unpack phi_boundary and calculate e_zeta at boundaries, mzeta=1
!$omp parallel do private(i,j,ii,jt,ij,pleft,pright)
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

! d_phi/d_zeta
     do j=1,mtheta(i)
        ij=igrid(i)+j
        if(mzeta==1)then           
           evector(3,1,ij)=(pright(j)-pleft(j))*diffz
        elseif(mzeta==2)then
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,2,ij)=(pright(j)-phi(1,ij))*diffz
        else
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,mzeta,ij)=(pright(j)-phi(mzeta-1,ij))*diffz
           evector(3,2:mzeta-1,ij)=(phi(3:mzeta,ij)-phi(1:mzeta-2,ij))*diffz
        endif
     enddo
  enddo

! adjust the difference between safety factor q and qtinv for fieldline coordinate
!omp parallel do private(i,j,r,q,delq,ij)
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     q=q0+q1*r/a+q2*r*r/(a*a)
     delq=(1.0/q-qtinv(i))

     do j=1,mtheta(i)
        ij=igrid(i)+j
        evector(3,:,ij)=evector(3,:,ij)+delq*evector(2,:,ij)
     enddo
  enddo
  
! add (0,0) mode, d phi/d psi
  if(mode00==1)then
!omp parallel do private(i,j,k,ii,jt,r)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        ii=igrid(i)
        jt=mtheta(i)
        do j=ii+1,ii+jt
           do k=1,mzeta
              evector(1,k,j)=evector(1,k,j)+phip00(i)/r
           enddo
        enddo
     enddo
  endif
  
! toroidal end point, pack end point (k=mzeta) data 
! send e to right and receive from left
!$omp parallel do private(i)
  do i=1,mgrid
     sendrs(1,i)=evector(1,mzeta,i)
     sendrs(2,i)=evector(2,mzeta,i)
     sendrs(3,i)=evector(3,mzeta,i)
     recvls(1,i)=0.0
     recvls(2,i)=0.0
     recvls(3,i)=0.0
  enddo
  icount=3*mgrid
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendrs,icount,mpi_rsize,idest,isendtag,&
  !     recvls,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendrs,icount)
  call rcvleft_toroidal_cmm(ptr,recvls)
 
! unpack end point data for k=0
  if(myrank_toroidal==0)then
!$omp parallel do private(i,ii,jt)
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        evector(1,0,ii+1:ii+jt)=cshift(recvls(1,ii+1:ii+jt),-itran(i))
        evector(2,0,ii+1:ii+jt)=cshift(recvls(2,ii+1:ii+jt),-itran(i))
        evector(3,0,ii+1:ii+jt)=cshift(recvls(3,ii+1:ii+jt),-itran(i))
     enddo
  else
!$omp parallel do private(i)
     do i=1,mgrid
        evector(1,0,i)=recvls(1,i)
        evector(2,0,i)=recvls(2,i)
        evector(3,0,i)=recvls(3,i)
     enddo
  endif
  
! poloidal end point
!omp parallel do private(i,j)
  do i=1,mpsi-1
     do j=0,mzeta
        evector(1,j,igrid(i))=evector(1,j,igrid(i)+mtheta(i))
        evector(2,j,igrid(i))=evector(2,j,igrid(i)+mtheta(i))
        evector(3,j,igrid(i))=evector(3,j,igrid(i)+mtheta(i))
     enddo
  enddo
  
end subroutine fieldr_0
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine fieldr_1(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_1
  use field_array_1
  use diagnosis_array_1
  use particle_decomp_1
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,ii,ij,j,k,icount,idest,isource,isendtag,irecvtag,ierror,ip,jt !,&
       !istatus(mpi_status_size)
  real(8) diffr,difft(0:mpsi),diffz,r,drdp,pleft(mthetamax),pright(mthetamax),&
       sendl(mgrid),recvr(mgrid),sendr(mgrid),recvl(mgrid),sendrs(3,mgrid),&
       recvls(3,mgrid),q,delq

! finite difference for e-field in equilibrium unit
  diffr=0.5_wp/deltar
  difft=0.5_wp/deltat
  diffz=0.5_wp/deltaz
!$omp parallel do private(i,j,k)
  do i=1,mgrid
     do j=0,mzeta
        do k=1,3
           evector(k,j,i)=0.0
        enddo
     enddo
  enddo

! d_phi/d_psi
  do k=1,mzeta
!$omp parallel do private(i,j,r,drdp,ij)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        drdp=1.0/r
        do j=1,mtheta(i)
           ij=igrid(i)+j

           evector(1,k,ij)=drdp*diffr*((1.0-wtp1(1,ij,k))*phi(k,jtp1(1,ij,k))+&
                wtp1(1,ij,k)*phi(k,jtp1(1,ij,k)+1)-&
                ((1.0-wtp1(2,ij,k))*phi(k,jtp1(2,ij,k))+&
                wtp1(2,ij,k)*phi(k,jtp1(2,ij,k)+1)))

        enddo
     enddo
  enddo

! d_phi/d_theta
!$omp parallel do private(i,k,j,ij,jt)
  do i=1,mpsi-1
     do k=1,mzeta
        do j=1,mtheta(i)
           ij=igrid(i)+j
           jt=j+1-mtheta(i)*(j/mtheta(i))
           evector(2,k,ij)=difft(i)*(phi(k,igrid(i)+jt)-phi(k,igrid(i)+j-1))
        enddo
     enddo
  enddo
  
! send phi to right and receive from left
  sendr=phi(mzeta,:)
  recvl=0.0
  icount=mgrid
  !!idest=mod(mype+1,numberpe)
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendr,icount,mpi_rsize,idest,isendtag,&
  !     recvl,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendr,icount)
  call rcvleft_toroidal_cmm(ptr,recvl)
  
! send phi to left and receive from right
  sendl=phi(1,:)
  recvr=0.0
  !!idest=mod(mype-1+numberpe,numberpe)
  idest=left_pe
  !!isource=mod(mype+1,numberpe)
  isource=right_pe
  !!isendtag=mype
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)

! unpack phi_boundary and calculate e_zeta at boundaries, mzeta=1
!$omp parallel do private(i,j,ii,jt,ij,pleft,pright)
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

! d_phi/d_zeta
     do j=1,mtheta(i)
        ij=igrid(i)+j
        if(mzeta==1)then           
           evector(3,1,ij)=(pright(j)-pleft(j))*diffz
        elseif(mzeta==2)then
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,2,ij)=(pright(j)-phi(1,ij))*diffz
        else
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,mzeta,ij)=(pright(j)-phi(mzeta-1,ij))*diffz
           evector(3,2:mzeta-1,ij)=(phi(3:mzeta,ij)-phi(1:mzeta-2,ij))*diffz
        endif
     enddo
  enddo

! adjust the difference between safety factor q and qtinv for fieldline coordinate
!omp parallel do private(i,j,r,q,delq,ij)
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     q=q0+q1*r/a+q2*r*r/(a*a)
     delq=(1.0/q-qtinv(i))

     do j=1,mtheta(i)
        ij=igrid(i)+j
        evector(3,:,ij)=evector(3,:,ij)+delq*evector(2,:,ij)
     enddo
  enddo
  
! add (0,0) mode, d phi/d psi
  if(mode00==1)then
!omp parallel do private(i,j,k,ii,jt,r)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        ii=igrid(i)
        jt=mtheta(i)
        do j=ii+1,ii+jt
           do k=1,mzeta
              evector(1,k,j)=evector(1,k,j)+phip00(i)/r
           enddo
        enddo
     enddo
  endif
  
! toroidal end point, pack end point (k=mzeta) data 
! send e to right and receive from left
!$omp parallel do private(i)
  do i=1,mgrid
     sendrs(1,i)=evector(1,mzeta,i)
     sendrs(2,i)=evector(2,mzeta,i)
     sendrs(3,i)=evector(3,mzeta,i)
     recvls(1,i)=0.0
     recvls(2,i)=0.0
     recvls(3,i)=0.0
  enddo
  icount=3*mgrid
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendrs,icount,mpi_rsize,idest,isendtag,&
  !     recvls,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendrs,icount)
  call rcvleft_toroidal_cmm(ptr,recvls)
 
! unpack end point data for k=0
  if(myrank_toroidal==0)then
!$omp parallel do private(i,ii,jt)
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        evector(1,0,ii+1:ii+jt)=cshift(recvls(1,ii+1:ii+jt),-itran(i))
        evector(2,0,ii+1:ii+jt)=cshift(recvls(2,ii+1:ii+jt),-itran(i))
        evector(3,0,ii+1:ii+jt)=cshift(recvls(3,ii+1:ii+jt),-itran(i))
     enddo
  else
!$omp parallel do private(i)
     do i=1,mgrid
        evector(1,0,i)=recvls(1,i)
        evector(2,0,i)=recvls(2,i)
        evector(3,0,i)=recvls(3,i)
     enddo
  endif
  
! poloidal end point
!omp parallel do private(i,j)
  do i=1,mpsi-1
     do j=0,mzeta
        evector(1,j,igrid(i))=evector(1,j,igrid(i)+mtheta(i))
        evector(2,j,igrid(i))=evector(2,j,igrid(i)+mtheta(i))
        evector(3,j,igrid(i))=evector(3,j,igrid(i)+mtheta(i))
     enddo
  enddo
  
end subroutine fieldr_1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine fieldr_2(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_2
  use field_array_2
  use diagnosis_array_2
  use particle_decomp_2
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,ii,ij,j,k,icount,idest,isource,isendtag,irecvtag,ierror,ip,jt !,&
       !istatus(mpi_status_size)
  real(8) diffr,difft(0:mpsi),diffz,r,drdp,pleft(mthetamax),pright(mthetamax),&
       sendl(mgrid),recvr(mgrid),sendr(mgrid),recvl(mgrid),sendrs(3,mgrid),&
       recvls(3,mgrid),q,delq

! finite difference for e-field in equilibrium unit
  diffr=0.5_wp/deltar
  difft=0.5_wp/deltat
  diffz=0.5_wp/deltaz
!$omp parallel do private(i,j,k)
  do i=1,mgrid
     do j=0,mzeta
        do k=1,3
           evector(k,j,i)=0.0
        enddo
     enddo
  enddo

! d_phi/d_psi
  do k=1,mzeta
!$omp parallel do private(i,j,r,drdp,ij)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        drdp=1.0/r
        do j=1,mtheta(i)
           ij=igrid(i)+j

           evector(1,k,ij)=drdp*diffr*((1.0-wtp1(1,ij,k))*phi(k,jtp1(1,ij,k))+&
                wtp1(1,ij,k)*phi(k,jtp1(1,ij,k)+1)-&
                ((1.0-wtp1(2,ij,k))*phi(k,jtp1(2,ij,k))+&
                wtp1(2,ij,k)*phi(k,jtp1(2,ij,k)+1)))

        enddo
     enddo
  enddo

! d_phi/d_theta
!$omp parallel do private(i,k,j,ij,jt)
  do i=1,mpsi-1
     do k=1,mzeta
        do j=1,mtheta(i)
           ij=igrid(i)+j
           jt=j+1-mtheta(i)*(j/mtheta(i))
           evector(2,k,ij)=difft(i)*(phi(k,igrid(i)+jt)-phi(k,igrid(i)+j-1))
        enddo
     enddo
  enddo
  
! send phi to right and receive from left
  sendr=phi(mzeta,:)
  recvl=0.0
  icount=mgrid
  !!idest=mod(mype+1,numberpe)
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendr,icount,mpi_rsize,idest,isendtag,&
  !     recvl,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendr,icount)
  call rcvleft_toroidal_cmm(ptr,recvl)
  
! send phi to left and receive from right
  sendl=phi(1,:)
  recvr=0.0
  !!idest=mod(mype-1+numberpe,numberpe)
  idest=left_pe
  !!isource=mod(mype+1,numberpe)
  isource=right_pe
  !!isendtag=mype
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)

! unpack phi_boundary and calculate e_zeta at boundaries, mzeta=1
!$omp parallel do private(i,j,ii,jt,ij,pleft,pright)
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

! d_phi/d_zeta
     do j=1,mtheta(i)
        ij=igrid(i)+j
        if(mzeta==1)then           
           evector(3,1,ij)=(pright(j)-pleft(j))*diffz
        elseif(mzeta==2)then
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,2,ij)=(pright(j)-phi(1,ij))*diffz
        else
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,mzeta,ij)=(pright(j)-phi(mzeta-1,ij))*diffz
           evector(3,2:mzeta-1,ij)=(phi(3:mzeta,ij)-phi(1:mzeta-2,ij))*diffz
        endif
     enddo
  enddo

! adjust the difference between safety factor q and qtinv for fieldline coordinate
!omp parallel do private(i,j,r,q,delq,ij)
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     q=q0+q1*r/a+q2*r*r/(a*a)
     delq=(1.0/q-qtinv(i))

     do j=1,mtheta(i)
        ij=igrid(i)+j
        evector(3,:,ij)=evector(3,:,ij)+delq*evector(2,:,ij)
     enddo
  enddo
  
! add (0,0) mode, d phi/d psi
  if(mode00==1)then
!omp parallel do private(i,j,k,ii,jt,r)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        ii=igrid(i)
        jt=mtheta(i)
        do j=ii+1,ii+jt
           do k=1,mzeta
              evector(1,k,j)=evector(1,k,j)+phip00(i)/r
           enddo
        enddo
     enddo
  endif
  
! toroidal end point, pack end point (k=mzeta) data 
! send e to right and receive from left
!$omp parallel do private(i)
  do i=1,mgrid
     sendrs(1,i)=evector(1,mzeta,i)
     sendrs(2,i)=evector(2,mzeta,i)
     sendrs(3,i)=evector(3,mzeta,i)
     recvls(1,i)=0.0
     recvls(2,i)=0.0
     recvls(3,i)=0.0
  enddo
  icount=3*mgrid
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendrs,icount,mpi_rsize,idest,isendtag,&
  !     recvls,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendrs,icount)
  call rcvleft_toroidal_cmm(ptr,recvls)
 
! unpack end point data for k=0
  if(myrank_toroidal==0)then
!$omp parallel do private(i,ii,jt)
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        evector(1,0,ii+1:ii+jt)=cshift(recvls(1,ii+1:ii+jt),-itran(i))
        evector(2,0,ii+1:ii+jt)=cshift(recvls(2,ii+1:ii+jt),-itran(i))
        evector(3,0,ii+1:ii+jt)=cshift(recvls(3,ii+1:ii+jt),-itran(i))
     enddo
  else
!$omp parallel do private(i)
     do i=1,mgrid
        evector(1,0,i)=recvls(1,i)
        evector(2,0,i)=recvls(2,i)
        evector(3,0,i)=recvls(3,i)
     enddo
  endif
  
! poloidal end point
!omp parallel do private(i,j)
  do i=1,mpsi-1
     do j=0,mzeta
        evector(1,j,igrid(i))=evector(1,j,igrid(i)+mtheta(i))
        evector(2,j,igrid(i))=evector(2,j,igrid(i)+mtheta(i))
        evector(3,j,igrid(i))=evector(3,j,igrid(i)+mtheta(i))
     enddo
  enddo
  
end subroutine fieldr_2
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine fieldr_3(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_3
  use field_array_3
  use diagnosis_array_3
  use particle_decomp_3
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,ii,ij,j,k,icount,idest,isource,isendtag,irecvtag,ierror,ip,jt !,&
       !istatus(mpi_status_size)
  real(8) diffr,difft(0:mpsi),diffz,r,drdp,pleft(mthetamax),pright(mthetamax),&
       sendl(mgrid),recvr(mgrid),sendr(mgrid),recvl(mgrid),sendrs(3,mgrid),&
       recvls(3,mgrid),q,delq

! finite difference for e-field in equilibrium unit
  diffr=0.5_wp/deltar
  difft=0.5_wp/deltat
  diffz=0.5_wp/deltaz
!$omp parallel do private(i,j,k)
  do i=1,mgrid
     do j=0,mzeta
        do k=1,3
           evector(k,j,i)=0.0
        enddo
     enddo
  enddo

! d_phi/d_psi
  do k=1,mzeta
!$omp parallel do private(i,j,r,drdp,ij)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        drdp=1.0/r
        do j=1,mtheta(i)
           ij=igrid(i)+j

           evector(1,k,ij)=drdp*diffr*((1.0-wtp1(1,ij,k))*phi(k,jtp1(1,ij,k))+&
                wtp1(1,ij,k)*phi(k,jtp1(1,ij,k)+1)-&
                ((1.0-wtp1(2,ij,k))*phi(k,jtp1(2,ij,k))+&
                wtp1(2,ij,k)*phi(k,jtp1(2,ij,k)+1)))

        enddo
     enddo
  enddo

! d_phi/d_theta
!$omp parallel do private(i,k,j,ij,jt)
  do i=1,mpsi-1
     do k=1,mzeta
        do j=1,mtheta(i)
           ij=igrid(i)+j
           jt=j+1-mtheta(i)*(j/mtheta(i))
           evector(2,k,ij)=difft(i)*(phi(k,igrid(i)+jt)-phi(k,igrid(i)+j-1))
        enddo
     enddo
  enddo
  
! send phi to right and receive from left
  sendr=phi(mzeta,:)
  recvl=0.0
  icount=mgrid
  !!idest=mod(mype+1,numberpe)
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendr,icount,mpi_rsize,idest,isendtag,&
  !     recvl,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendr,icount)
  call rcvleft_toroidal_cmm(ptr,recvl)
  
! send phi to left and receive from right
  sendl=phi(1,:)
  recvr=0.0
  !!idest=mod(mype-1+numberpe,numberpe)
  idest=left_pe
  !!isource=mod(mype+1,numberpe)
  isource=right_pe
  !!isendtag=mype
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)

! unpack phi_boundary and calculate e_zeta at boundaries, mzeta=1
!$omp parallel do private(i,j,ii,jt,ij,pleft,pright)
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

! d_phi/d_zeta
     do j=1,mtheta(i)
        ij=igrid(i)+j
        if(mzeta==1)then           
           evector(3,1,ij)=(pright(j)-pleft(j))*diffz
        elseif(mzeta==2)then
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,2,ij)=(pright(j)-phi(1,ij))*diffz
        else
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,mzeta,ij)=(pright(j)-phi(mzeta-1,ij))*diffz
           evector(3,2:mzeta-1,ij)=(phi(3:mzeta,ij)-phi(1:mzeta-2,ij))*diffz
        endif
     enddo
  enddo

! adjust the difference between safety factor q and qtinv for fieldline coordinate
!omp parallel do private(i,j,r,q,delq,ij)
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     q=q0+q1*r/a+q2*r*r/(a*a)
     delq=(1.0/q-qtinv(i))

     do j=1,mtheta(i)
        ij=igrid(i)+j
        evector(3,:,ij)=evector(3,:,ij)+delq*evector(2,:,ij)
     enddo
  enddo
  
! add (0,0) mode, d phi/d psi
  if(mode00==1)then
!omp parallel do private(i,j,k,ii,jt,r)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        ii=igrid(i)
        jt=mtheta(i)
        do j=ii+1,ii+jt
           do k=1,mzeta
              evector(1,k,j)=evector(1,k,j)+phip00(i)/r
           enddo
        enddo
     enddo
  endif
  
! toroidal end point, pack end point (k=mzeta) data 
! send e to right and receive from left
!$omp parallel do private(i)
  do i=1,mgrid
     sendrs(1,i)=evector(1,mzeta,i)
     sendrs(2,i)=evector(2,mzeta,i)
     sendrs(3,i)=evector(3,mzeta,i)
     recvls(1,i)=0.0
     recvls(2,i)=0.0
     recvls(3,i)=0.0
  enddo
  icount=3*mgrid
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendrs,icount,mpi_rsize,idest,isendtag,&
  !     recvls,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendrs,icount)
  call rcvleft_toroidal_cmm(ptr,recvls)
 
! unpack end point data for k=0
  if(myrank_toroidal==0)then
!$omp parallel do private(i,ii,jt)
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        evector(1,0,ii+1:ii+jt)=cshift(recvls(1,ii+1:ii+jt),-itran(i))
        evector(2,0,ii+1:ii+jt)=cshift(recvls(2,ii+1:ii+jt),-itran(i))
        evector(3,0,ii+1:ii+jt)=cshift(recvls(3,ii+1:ii+jt),-itran(i))
     enddo
  else
!$omp parallel do private(i)
     do i=1,mgrid
        evector(1,0,i)=recvls(1,i)
        evector(2,0,i)=recvls(2,i)
        evector(3,0,i)=recvls(3,i)
     enddo
  endif
  
! poloidal end point
!omp parallel do private(i,j)
  do i=1,mpsi-1
     do j=0,mzeta
        evector(1,j,igrid(i))=evector(1,j,igrid(i)+mtheta(i))
        evector(2,j,igrid(i))=evector(2,j,igrid(i)+mtheta(i))
        evector(3,j,igrid(i))=evector(3,j,igrid(i)+mtheta(i))
     enddo
  enddo
  
end subroutine fieldr_3
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine fieldr_4(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_4
  use field_array_4
  use diagnosis_array_4
  use particle_decomp_4
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,ii,ij,j,k,icount,idest,isource,isendtag,irecvtag,ierror,ip,jt !,&
       !istatus(mpi_status_size)
  real(8) diffr,difft(0:mpsi),diffz,r,drdp,pleft(mthetamax),pright(mthetamax),&
       sendl(mgrid),recvr(mgrid),sendr(mgrid),recvl(mgrid),sendrs(3,mgrid),&
       recvls(3,mgrid),q,delq

! finite difference for e-field in equilibrium unit
  diffr=0.5_wp/deltar
  difft=0.5_wp/deltat
  diffz=0.5_wp/deltaz
!$omp parallel do private(i,j,k)
  do i=1,mgrid
     do j=0,mzeta
        do k=1,3
           evector(k,j,i)=0.0
        enddo
     enddo
  enddo

! d_phi/d_psi
  do k=1,mzeta
!$omp parallel do private(i,j,r,drdp,ij)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        drdp=1.0/r
        do j=1,mtheta(i)
           ij=igrid(i)+j

           evector(1,k,ij)=drdp*diffr*((1.0-wtp1(1,ij,k))*phi(k,jtp1(1,ij,k))+&
                wtp1(1,ij,k)*phi(k,jtp1(1,ij,k)+1)-&
                ((1.0-wtp1(2,ij,k))*phi(k,jtp1(2,ij,k))+&
                wtp1(2,ij,k)*phi(k,jtp1(2,ij,k)+1)))

        enddo
     enddo
  enddo

! d_phi/d_theta
!$omp parallel do private(i,k,j,ij,jt)
  do i=1,mpsi-1
     do k=1,mzeta
        do j=1,mtheta(i)
           ij=igrid(i)+j
           jt=j+1-mtheta(i)*(j/mtheta(i))
           evector(2,k,ij)=difft(i)*(phi(k,igrid(i)+jt)-phi(k,igrid(i)+j-1))
        enddo
     enddo
  enddo
  
! send phi to right and receive from left
  sendr=phi(mzeta,:)
  recvl=0.0
  icount=mgrid
  !!idest=mod(mype+1,numberpe)
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendr,icount,mpi_rsize,idest,isendtag,&
  !     recvl,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendr,icount)
  call rcvleft_toroidal_cmm(ptr,recvl)
  
! send phi to left and receive from right
  sendl=phi(1,:)
  recvr=0.0
  !!idest=mod(mype-1+numberpe,numberpe)
  idest=left_pe
  !!isource=mod(mype+1,numberpe)
  isource=right_pe
  !!isendtag=mype
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)

! unpack phi_boundary and calculate e_zeta at boundaries, mzeta=1
!$omp parallel do private(i,j,ii,jt,ij,pleft,pright)
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

! d_phi/d_zeta
     do j=1,mtheta(i)
        ij=igrid(i)+j
        if(mzeta==1)then           
           evector(3,1,ij)=(pright(j)-pleft(j))*diffz
        elseif(mzeta==2)then
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,2,ij)=(pright(j)-phi(1,ij))*diffz
        else
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,mzeta,ij)=(pright(j)-phi(mzeta-1,ij))*diffz
           evector(3,2:mzeta-1,ij)=(phi(3:mzeta,ij)-phi(1:mzeta-2,ij))*diffz
        endif
     enddo
  enddo

! adjust the difference between safety factor q and qtinv for fieldline coordinate
!omp parallel do private(i,j,r,q,delq,ij)
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     q=q0+q1*r/a+q2*r*r/(a*a)
     delq=(1.0/q-qtinv(i))

     do j=1,mtheta(i)
        ij=igrid(i)+j
        evector(3,:,ij)=evector(3,:,ij)+delq*evector(2,:,ij)
     enddo
  enddo
  
! add (0,0) mode, d phi/d psi
  if(mode00==1)then
!omp parallel do private(i,j,k,ii,jt,r)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        ii=igrid(i)
        jt=mtheta(i)
        do j=ii+1,ii+jt
           do k=1,mzeta
              evector(1,k,j)=evector(1,k,j)+phip00(i)/r
           enddo
        enddo
     enddo
  endif
  
! toroidal end point, pack end point (k=mzeta) data 
! send e to right and receive from left
!$omp parallel do private(i)
  do i=1,mgrid
     sendrs(1,i)=evector(1,mzeta,i)
     sendrs(2,i)=evector(2,mzeta,i)
     sendrs(3,i)=evector(3,mzeta,i)
     recvls(1,i)=0.0
     recvls(2,i)=0.0
     recvls(3,i)=0.0
  enddo
  icount=3*mgrid
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendrs,icount,mpi_rsize,idest,isendtag,&
  !     recvls,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendrs,icount)
  call rcvleft_toroidal_cmm(ptr,recvls)
 
! unpack end point data for k=0
  if(myrank_toroidal==0)then
!$omp parallel do private(i,ii,jt)
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        evector(1,0,ii+1:ii+jt)=cshift(recvls(1,ii+1:ii+jt),-itran(i))
        evector(2,0,ii+1:ii+jt)=cshift(recvls(2,ii+1:ii+jt),-itran(i))
        evector(3,0,ii+1:ii+jt)=cshift(recvls(3,ii+1:ii+jt),-itran(i))
     enddo
  else
!$omp parallel do private(i)
     do i=1,mgrid
        evector(1,0,i)=recvls(1,i)
        evector(2,0,i)=recvls(2,i)
        evector(3,0,i)=recvls(3,i)
     enddo
  endif
  
! poloidal end point
!omp parallel do private(i,j)
  do i=1,mpsi-1
     do j=0,mzeta
        evector(1,j,igrid(i))=evector(1,j,igrid(i)+mtheta(i))
        evector(2,j,igrid(i))=evector(2,j,igrid(i)+mtheta(i))
        evector(3,j,igrid(i))=evector(3,j,igrid(i)+mtheta(i))
     enddo
  enddo
  
end subroutine fieldr_4
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine fieldr_5(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_5
  use field_array_5
  use diagnosis_array_5
  use particle_decomp_5
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,ii,ij,j,k,icount,idest,isource,isendtag,irecvtag,ierror,ip,jt !,&
       !istatus(mpi_status_size)
  real(8) diffr,difft(0:mpsi),diffz,r,drdp,pleft(mthetamax),pright(mthetamax),&
       sendl(mgrid),recvr(mgrid),sendr(mgrid),recvl(mgrid),sendrs(3,mgrid),&
       recvls(3,mgrid),q,delq

! finite difference for e-field in equilibrium unit
  diffr=0.5_wp/deltar
  difft=0.5_wp/deltat
  diffz=0.5_wp/deltaz
!$omp parallel do private(i,j,k)
  do i=1,mgrid
     do j=0,mzeta
        do k=1,3
           evector(k,j,i)=0.0
        enddo
     enddo
  enddo

! d_phi/d_psi
  do k=1,mzeta
!$omp parallel do private(i,j,r,drdp,ij)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        drdp=1.0/r
        do j=1,mtheta(i)
           ij=igrid(i)+j

           evector(1,k,ij)=drdp*diffr*((1.0-wtp1(1,ij,k))*phi(k,jtp1(1,ij,k))+&
                wtp1(1,ij,k)*phi(k,jtp1(1,ij,k)+1)-&
                ((1.0-wtp1(2,ij,k))*phi(k,jtp1(2,ij,k))+&
                wtp1(2,ij,k)*phi(k,jtp1(2,ij,k)+1)))

        enddo
     enddo
  enddo

! d_phi/d_theta
!$omp parallel do private(i,k,j,ij,jt)
  do i=1,mpsi-1
     do k=1,mzeta
        do j=1,mtheta(i)
           ij=igrid(i)+j
           jt=j+1-mtheta(i)*(j/mtheta(i))
           evector(2,k,ij)=difft(i)*(phi(k,igrid(i)+jt)-phi(k,igrid(i)+j-1))
        enddo
     enddo
  enddo
  
! send phi to right and receive from left
  sendr=phi(mzeta,:)
  recvl=0.0
  icount=mgrid
  !!idest=mod(mype+1,numberpe)
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendr,icount,mpi_rsize,idest,isendtag,&
  !     recvl,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendr,icount)
  call rcvleft_toroidal_cmm(ptr,recvl)
  
! send phi to left and receive from right
  sendl=phi(1,:)
  recvr=0.0
  !!idest=mod(mype-1+numberpe,numberpe)
  idest=left_pe
  !!isource=mod(mype+1,numberpe)
  isource=right_pe
  !!isendtag=mype
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)

! unpack phi_boundary and calculate e_zeta at boundaries, mzeta=1
!$omp parallel do private(i,j,ii,jt,ij,pleft,pright)
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

! d_phi/d_zeta
     do j=1,mtheta(i)
        ij=igrid(i)+j
        if(mzeta==1)then           
           evector(3,1,ij)=(pright(j)-pleft(j))*diffz
        elseif(mzeta==2)then
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,2,ij)=(pright(j)-phi(1,ij))*diffz
        else
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,mzeta,ij)=(pright(j)-phi(mzeta-1,ij))*diffz
           evector(3,2:mzeta-1,ij)=(phi(3:mzeta,ij)-phi(1:mzeta-2,ij))*diffz
        endif
     enddo
  enddo

! adjust the difference between safety factor q and qtinv for fieldline coordinate
!omp parallel do private(i,j,r,q,delq,ij)
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     q=q0+q1*r/a+q2*r*r/(a*a)
     delq=(1.0/q-qtinv(i))

     do j=1,mtheta(i)
        ij=igrid(i)+j
        evector(3,:,ij)=evector(3,:,ij)+delq*evector(2,:,ij)
     enddo
  enddo
  
! add (0,0) mode, d phi/d psi
  if(mode00==1)then
!omp parallel do private(i,j,k,ii,jt,r)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        ii=igrid(i)
        jt=mtheta(i)
        do j=ii+1,ii+jt
           do k=1,mzeta
              evector(1,k,j)=evector(1,k,j)+phip00(i)/r
           enddo
        enddo
     enddo
  endif
  
! toroidal end point, pack end point (k=mzeta) data 
! send e to right and receive from left
!$omp parallel do private(i)
  do i=1,mgrid
     sendrs(1,i)=evector(1,mzeta,i)
     sendrs(2,i)=evector(2,mzeta,i)
     sendrs(3,i)=evector(3,mzeta,i)
     recvls(1,i)=0.0
     recvls(2,i)=0.0
     recvls(3,i)=0.0
  enddo
  icount=3*mgrid
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendrs,icount,mpi_rsize,idest,isendtag,&
  !     recvls,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendrs,icount)
  call rcvleft_toroidal_cmm(ptr,recvls)
 
! unpack end point data for k=0
  if(myrank_toroidal==0)then
!$omp parallel do private(i,ii,jt)
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        evector(1,0,ii+1:ii+jt)=cshift(recvls(1,ii+1:ii+jt),-itran(i))
        evector(2,0,ii+1:ii+jt)=cshift(recvls(2,ii+1:ii+jt),-itran(i))
        evector(3,0,ii+1:ii+jt)=cshift(recvls(3,ii+1:ii+jt),-itran(i))
     enddo
  else
!$omp parallel do private(i)
     do i=1,mgrid
        evector(1,0,i)=recvls(1,i)
        evector(2,0,i)=recvls(2,i)
        evector(3,0,i)=recvls(3,i)
     enddo
  endif
  
! poloidal end point
!omp parallel do private(i,j)
  do i=1,mpsi-1
     do j=0,mzeta
        evector(1,j,igrid(i))=evector(1,j,igrid(i)+mtheta(i))
        evector(2,j,igrid(i))=evector(2,j,igrid(i)+mtheta(i))
        evector(3,j,igrid(i))=evector(3,j,igrid(i)+mtheta(i))
     enddo
  enddo
  
end subroutine fieldr_5
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine fieldr_6(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_6
  use field_array_6
  use diagnosis_array_6
  use particle_decomp_6
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,ii,ij,j,k,icount,idest,isource,isendtag,irecvtag,ierror,ip,jt !,&
       !istatus(mpi_status_size)
  real(8) diffr,difft(0:mpsi),diffz,r,drdp,pleft(mthetamax),pright(mthetamax),&
       sendl(mgrid),recvr(mgrid),sendr(mgrid),recvl(mgrid),sendrs(3,mgrid),&
       recvls(3,mgrid),q,delq

! finite difference for e-field in equilibrium unit
  diffr=0.5_wp/deltar
  difft=0.5_wp/deltat
  diffz=0.5_wp/deltaz
!$omp parallel do private(i,j,k)
  do i=1,mgrid
     do j=0,mzeta
        do k=1,3
           evector(k,j,i)=0.0
        enddo
     enddo
  enddo

! d_phi/d_psi
  do k=1,mzeta
!$omp parallel do private(i,j,r,drdp,ij)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        drdp=1.0/r
        do j=1,mtheta(i)
           ij=igrid(i)+j

           evector(1,k,ij)=drdp*diffr*((1.0-wtp1(1,ij,k))*phi(k,jtp1(1,ij,k))+&
                wtp1(1,ij,k)*phi(k,jtp1(1,ij,k)+1)-&
                ((1.0-wtp1(2,ij,k))*phi(k,jtp1(2,ij,k))+&
                wtp1(2,ij,k)*phi(k,jtp1(2,ij,k)+1)))

        enddo
     enddo
  enddo

! d_phi/d_theta
!$omp parallel do private(i,k,j,ij,jt)
  do i=1,mpsi-1
     do k=1,mzeta
        do j=1,mtheta(i)
           ij=igrid(i)+j
           jt=j+1-mtheta(i)*(j/mtheta(i))
           evector(2,k,ij)=difft(i)*(phi(k,igrid(i)+jt)-phi(k,igrid(i)+j-1))
        enddo
     enddo
  enddo
  
! send phi to right and receive from left
  sendr=phi(mzeta,:)
  recvl=0.0
  icount=mgrid
  !!idest=mod(mype+1,numberpe)
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendr,icount,mpi_rsize,idest,isendtag,&
  !     recvl,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendr,icount)
  call rcvleft_toroidal_cmm(ptr,recvl)
  
! send phi to left and receive from right
  sendl=phi(1,:)
  recvr=0.0
  !!idest=mod(mype-1+numberpe,numberpe)
  idest=left_pe
  !!isource=mod(mype+1,numberpe)
  isource=right_pe
  !!isendtag=mype
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)

! unpack phi_boundary and calculate e_zeta at boundaries, mzeta=1
!$omp parallel do private(i,j,ii,jt,ij,pleft,pright)
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

! d_phi/d_zeta
     do j=1,mtheta(i)
        ij=igrid(i)+j
        if(mzeta==1)then           
           evector(3,1,ij)=(pright(j)-pleft(j))*diffz
        elseif(mzeta==2)then
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,2,ij)=(pright(j)-phi(1,ij))*diffz
        else
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,mzeta,ij)=(pright(j)-phi(mzeta-1,ij))*diffz
           evector(3,2:mzeta-1,ij)=(phi(3:mzeta,ij)-phi(1:mzeta-2,ij))*diffz
        endif
     enddo
  enddo

! adjust the difference between safety factor q and qtinv for fieldline coordinate
!omp parallel do private(i,j,r,q,delq,ij)
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     q=q0+q1*r/a+q2*r*r/(a*a)
     delq=(1.0/q-qtinv(i))

     do j=1,mtheta(i)
        ij=igrid(i)+j
        evector(3,:,ij)=evector(3,:,ij)+delq*evector(2,:,ij)
     enddo
  enddo
  
! add (0,0) mode, d phi/d psi
  if(mode00==1)then
!omp parallel do private(i,j,k,ii,jt,r)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        ii=igrid(i)
        jt=mtheta(i)
        do j=ii+1,ii+jt
           do k=1,mzeta
              evector(1,k,j)=evector(1,k,j)+phip00(i)/r
           enddo
        enddo
     enddo
  endif
  
! toroidal end point, pack end point (k=mzeta) data 
! send e to right and receive from left
!$omp parallel do private(i)
  do i=1,mgrid
     sendrs(1,i)=evector(1,mzeta,i)
     sendrs(2,i)=evector(2,mzeta,i)
     sendrs(3,i)=evector(3,mzeta,i)
     recvls(1,i)=0.0
     recvls(2,i)=0.0
     recvls(3,i)=0.0
  enddo
  icount=3*mgrid
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendrs,icount,mpi_rsize,idest,isendtag,&
  !     recvls,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendrs,icount)
  call rcvleft_toroidal_cmm(ptr,recvls)
 
! unpack end point data for k=0
  if(myrank_toroidal==0)then
!$omp parallel do private(i,ii,jt)
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        evector(1,0,ii+1:ii+jt)=cshift(recvls(1,ii+1:ii+jt),-itran(i))
        evector(2,0,ii+1:ii+jt)=cshift(recvls(2,ii+1:ii+jt),-itran(i))
        evector(3,0,ii+1:ii+jt)=cshift(recvls(3,ii+1:ii+jt),-itran(i))
     enddo
  else
!$omp parallel do private(i)
     do i=1,mgrid
        evector(1,0,i)=recvls(1,i)
        evector(2,0,i)=recvls(2,i)
        evector(3,0,i)=recvls(3,i)
     enddo
  endif
  
! poloidal end point
!omp parallel do private(i,j)
  do i=1,mpsi-1
     do j=0,mzeta
        evector(1,j,igrid(i))=evector(1,j,igrid(i)+mtheta(i))
        evector(2,j,igrid(i))=evector(2,j,igrid(i)+mtheta(i))
        evector(3,j,igrid(i))=evector(3,j,igrid(i)+mtheta(i))
     enddo
  enddo
  
end subroutine fieldr_6
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine fieldr_7(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_7
  use field_array_7
  use diagnosis_array_7
  use particle_decomp_7
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,ii,ij,j,k,icount,idest,isource,isendtag,irecvtag,ierror,ip,jt !,&
       !istatus(mpi_status_size)
  real(8) diffr,difft(0:mpsi),diffz,r,drdp,pleft(mthetamax),pright(mthetamax),&
       sendl(mgrid),recvr(mgrid),sendr(mgrid),recvl(mgrid),sendrs(3,mgrid),&
       recvls(3,mgrid),q,delq

! finite difference for e-field in equilibrium unit
  diffr=0.5_wp/deltar
  difft=0.5_wp/deltat
  diffz=0.5_wp/deltaz
!$omp parallel do private(i,j,k)
  do i=1,mgrid
     do j=0,mzeta
        do k=1,3
           evector(k,j,i)=0.0
        enddo
     enddo
  enddo

! d_phi/d_psi
  do k=1,mzeta
!$omp parallel do private(i,j,r,drdp,ij)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        drdp=1.0/r
        do j=1,mtheta(i)
           ij=igrid(i)+j

           evector(1,k,ij)=drdp*diffr*((1.0-wtp1(1,ij,k))*phi(k,jtp1(1,ij,k))+&
                wtp1(1,ij,k)*phi(k,jtp1(1,ij,k)+1)-&
                ((1.0-wtp1(2,ij,k))*phi(k,jtp1(2,ij,k))+&
                wtp1(2,ij,k)*phi(k,jtp1(2,ij,k)+1)))

        enddo
     enddo
  enddo

! d_phi/d_theta
!$omp parallel do private(i,k,j,ij,jt)
  do i=1,mpsi-1
     do k=1,mzeta
        do j=1,mtheta(i)
           ij=igrid(i)+j
           jt=j+1-mtheta(i)*(j/mtheta(i))
           evector(2,k,ij)=difft(i)*(phi(k,igrid(i)+jt)-phi(k,igrid(i)+j-1))
        enddo
     enddo
  enddo
  
! send phi to right and receive from left
  sendr=phi(mzeta,:)
  recvl=0.0
  icount=mgrid
  !!idest=mod(mype+1,numberpe)
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendr,icount,mpi_rsize,idest,isendtag,&
  !     recvl,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendr,icount)
  call rcvleft_toroidal_cmm(ptr,recvl)
  
! send phi to left and receive from right
  sendl=phi(1,:)
  recvr=0.0
  !!idest=mod(mype-1+numberpe,numberpe)
  idest=left_pe
  !!isource=mod(mype+1,numberpe)
  isource=right_pe
  !!isendtag=mype
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)

! unpack phi_boundary and calculate e_zeta at boundaries, mzeta=1
!$omp parallel do private(i,j,ii,jt,ij,pleft,pright)
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

! d_phi/d_zeta
     do j=1,mtheta(i)
        ij=igrid(i)+j
        if(mzeta==1)then           
           evector(3,1,ij)=(pright(j)-pleft(j))*diffz
        elseif(mzeta==2)then
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,2,ij)=(pright(j)-phi(1,ij))*diffz
        else
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,mzeta,ij)=(pright(j)-phi(mzeta-1,ij))*diffz
           evector(3,2:mzeta-1,ij)=(phi(3:mzeta,ij)-phi(1:mzeta-2,ij))*diffz
        endif
     enddo
  enddo

! adjust the difference between safety factor q and qtinv for fieldline coordinate
!omp parallel do private(i,j,r,q,delq,ij)
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     q=q0+q1*r/a+q2*r*r/(a*a)
     delq=(1.0/q-qtinv(i))

     do j=1,mtheta(i)
        ij=igrid(i)+j
        evector(3,:,ij)=evector(3,:,ij)+delq*evector(2,:,ij)
     enddo
  enddo
  
! add (0,0) mode, d phi/d psi
  if(mode00==1)then
!omp parallel do private(i,j,k,ii,jt,r)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        ii=igrid(i)
        jt=mtheta(i)
        do j=ii+1,ii+jt
           do k=1,mzeta
              evector(1,k,j)=evector(1,k,j)+phip00(i)/r
           enddo
        enddo
     enddo
  endif
  
! toroidal end point, pack end point (k=mzeta) data 
! send e to right and receive from left
!$omp parallel do private(i)
  do i=1,mgrid
     sendrs(1,i)=evector(1,mzeta,i)
     sendrs(2,i)=evector(2,mzeta,i)
     sendrs(3,i)=evector(3,mzeta,i)
     recvls(1,i)=0.0
     recvls(2,i)=0.0
     recvls(3,i)=0.0
  enddo
  icount=3*mgrid
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendrs,icount,mpi_rsize,idest,isendtag,&
  !     recvls,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendrs,icount)
  call rcvleft_toroidal_cmm(ptr,recvls)
 
! unpack end point data for k=0
  if(myrank_toroidal==0)then
!$omp parallel do private(i,ii,jt)
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        evector(1,0,ii+1:ii+jt)=cshift(recvls(1,ii+1:ii+jt),-itran(i))
        evector(2,0,ii+1:ii+jt)=cshift(recvls(2,ii+1:ii+jt),-itran(i))
        evector(3,0,ii+1:ii+jt)=cshift(recvls(3,ii+1:ii+jt),-itran(i))
     enddo
  else
!$omp parallel do private(i)
     do i=1,mgrid
        evector(1,0,i)=recvls(1,i)
        evector(2,0,i)=recvls(2,i)
        evector(3,0,i)=recvls(3,i)
     enddo
  endif
  
! poloidal end point
!omp parallel do private(i,j)
  do i=1,mpsi-1
     do j=0,mzeta
        evector(1,j,igrid(i))=evector(1,j,igrid(i)+mtheta(i))
        evector(2,j,igrid(i))=evector(2,j,igrid(i)+mtheta(i))
        evector(3,j,igrid(i))=evector(3,j,igrid(i)+mtheta(i))
     enddo
  enddo
  
end subroutine fieldr_7
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine fieldr_8(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_8
  use field_array_8
  use diagnosis_array_8
  use particle_decomp_8
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,ii,ij,j,k,icount,idest,isource,isendtag,irecvtag,ierror,ip,jt !,&
       !istatus(mpi_status_size)
  real(8) diffr,difft(0:mpsi),diffz,r,drdp,pleft(mthetamax),pright(mthetamax),&
       sendl(mgrid),recvr(mgrid),sendr(mgrid),recvl(mgrid),sendrs(3,mgrid),&
       recvls(3,mgrid),q,delq

! finite difference for e-field in equilibrium unit
  diffr=0.5_wp/deltar
  difft=0.5_wp/deltat
  diffz=0.5_wp/deltaz
!$omp parallel do private(i,j,k)
  do i=1,mgrid
     do j=0,mzeta
        do k=1,3
           evector(k,j,i)=0.0
        enddo
     enddo
  enddo

! d_phi/d_psi
  do k=1,mzeta
!$omp parallel do private(i,j,r,drdp,ij)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        drdp=1.0/r
        do j=1,mtheta(i)
           ij=igrid(i)+j

           evector(1,k,ij)=drdp*diffr*((1.0-wtp1(1,ij,k))*phi(k,jtp1(1,ij,k))+&
                wtp1(1,ij,k)*phi(k,jtp1(1,ij,k)+1)-&
                ((1.0-wtp1(2,ij,k))*phi(k,jtp1(2,ij,k))+&
                wtp1(2,ij,k)*phi(k,jtp1(2,ij,k)+1)))

        enddo
     enddo
  enddo

! d_phi/d_theta
!$omp parallel do private(i,k,j,ij,jt)
  do i=1,mpsi-1
     do k=1,mzeta
        do j=1,mtheta(i)
           ij=igrid(i)+j
           jt=j+1-mtheta(i)*(j/mtheta(i))
           evector(2,k,ij)=difft(i)*(phi(k,igrid(i)+jt)-phi(k,igrid(i)+j-1))
        enddo
     enddo
  enddo
  
! send phi to right and receive from left
  sendr=phi(mzeta,:)
  recvl=0.0
  icount=mgrid
  !!idest=mod(mype+1,numberpe)
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendr,icount,mpi_rsize,idest,isendtag,&
  !     recvl,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendr,icount)
  call rcvleft_toroidal_cmm(ptr,recvl)
  
! send phi to left and receive from right
  sendl=phi(1,:)
  recvr=0.0
  !!idest=mod(mype-1+numberpe,numberpe)
  idest=left_pe
  !!isource=mod(mype+1,numberpe)
  isource=right_pe
  !!isendtag=mype
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)

! unpack phi_boundary and calculate e_zeta at boundaries, mzeta=1
!$omp parallel do private(i,j,ii,jt,ij,pleft,pright)
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

! d_phi/d_zeta
     do j=1,mtheta(i)
        ij=igrid(i)+j
        if(mzeta==1)then           
           evector(3,1,ij)=(pright(j)-pleft(j))*diffz
        elseif(mzeta==2)then
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,2,ij)=(pright(j)-phi(1,ij))*diffz
        else
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,mzeta,ij)=(pright(j)-phi(mzeta-1,ij))*diffz
           evector(3,2:mzeta-1,ij)=(phi(3:mzeta,ij)-phi(1:mzeta-2,ij))*diffz
        endif
     enddo
  enddo

! adjust the difference between safety factor q and qtinv for fieldline coordinate
!omp parallel do private(i,j,r,q,delq,ij)
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     q=q0+q1*r/a+q2*r*r/(a*a)
     delq=(1.0/q-qtinv(i))

     do j=1,mtheta(i)
        ij=igrid(i)+j
        evector(3,:,ij)=evector(3,:,ij)+delq*evector(2,:,ij)
     enddo
  enddo
  
! add (0,0) mode, d phi/d psi
  if(mode00==1)then
!omp parallel do private(i,j,k,ii,jt,r)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        ii=igrid(i)
        jt=mtheta(i)
        do j=ii+1,ii+jt
           do k=1,mzeta
              evector(1,k,j)=evector(1,k,j)+phip00(i)/r
           enddo
        enddo
     enddo
  endif
  
! toroidal end point, pack end point (k=mzeta) data 
! send e to right and receive from left
!$omp parallel do private(i)
  do i=1,mgrid
     sendrs(1,i)=evector(1,mzeta,i)
     sendrs(2,i)=evector(2,mzeta,i)
     sendrs(3,i)=evector(3,mzeta,i)
     recvls(1,i)=0.0
     recvls(2,i)=0.0
     recvls(3,i)=0.0
  enddo
  icount=3*mgrid
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendrs,icount,mpi_rsize,idest,isendtag,&
  !     recvls,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendrs,icount)
  call rcvleft_toroidal_cmm(ptr,recvls)
 
! unpack end point data for k=0
  if(myrank_toroidal==0)then
!$omp parallel do private(i,ii,jt)
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        evector(1,0,ii+1:ii+jt)=cshift(recvls(1,ii+1:ii+jt),-itran(i))
        evector(2,0,ii+1:ii+jt)=cshift(recvls(2,ii+1:ii+jt),-itran(i))
        evector(3,0,ii+1:ii+jt)=cshift(recvls(3,ii+1:ii+jt),-itran(i))
     enddo
  else
!$omp parallel do private(i)
     do i=1,mgrid
        evector(1,0,i)=recvls(1,i)
        evector(2,0,i)=recvls(2,i)
        evector(3,0,i)=recvls(3,i)
     enddo
  endif
  
! poloidal end point
!omp parallel do private(i,j)
  do i=1,mpsi-1
     do j=0,mzeta
        evector(1,j,igrid(i))=evector(1,j,igrid(i)+mtheta(i))
        evector(2,j,igrid(i))=evector(2,j,igrid(i)+mtheta(i))
        evector(3,j,igrid(i))=evector(3,j,igrid(i)+mtheta(i))
     enddo
  enddo
  
end subroutine fieldr_8
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine fieldr_9(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_9
  use field_array_9
  use diagnosis_array_9
  use particle_decomp_9
  implicit none
  type(c_ptr), intent(in), value :: ptr

  integer i,ii,ij,j,k,icount,idest,isource,isendtag,irecvtag,ierror,ip,jt !,&
       !istatus(mpi_status_size)
  real(8) diffr,difft(0:mpsi),diffz,r,drdp,pleft(mthetamax),pright(mthetamax),&
       sendl(mgrid),recvr(mgrid),sendr(mgrid),recvl(mgrid),sendrs(3,mgrid),&
       recvls(3,mgrid),q,delq

! finite difference for e-field in equilibrium unit
  diffr=0.5_wp/deltar
  difft=0.5_wp/deltat
  diffz=0.5_wp/deltaz
!$omp parallel do private(i,j,k)
  do i=1,mgrid
     do j=0,mzeta
        do k=1,3
           evector(k,j,i)=0.0
        enddo
     enddo
  enddo

! d_phi/d_psi
  do k=1,mzeta
!$omp parallel do private(i,j,r,drdp,ij)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        drdp=1.0/r
        do j=1,mtheta(i)
           ij=igrid(i)+j

           evector(1,k,ij)=drdp*diffr*((1.0-wtp1(1,ij,k))*phi(k,jtp1(1,ij,k))+&
                wtp1(1,ij,k)*phi(k,jtp1(1,ij,k)+1)-&
                ((1.0-wtp1(2,ij,k))*phi(k,jtp1(2,ij,k))+&
                wtp1(2,ij,k)*phi(k,jtp1(2,ij,k)+1)))

        enddo
     enddo
  enddo

! d_phi/d_theta
!$omp parallel do private(i,k,j,ij,jt)
  do i=1,mpsi-1
     do k=1,mzeta
        do j=1,mtheta(i)
           ij=igrid(i)+j
           jt=j+1-mtheta(i)*(j/mtheta(i))
           evector(2,k,ij)=difft(i)*(phi(k,igrid(i)+jt)-phi(k,igrid(i)+j-1))
        enddo
     enddo
  enddo
  
! send phi to right and receive from left
  sendr=phi(mzeta,:)
  recvl=0.0
  icount=mgrid
  !!idest=mod(mype+1,numberpe)
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendr,icount,mpi_rsize,idest,isendtag,&
  !     recvl,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendr,icount)
  call rcvleft_toroidal_cmm(ptr,recvl)
  
! send phi to left and receive from right
  sendl=phi(1,:)
  recvr=0.0
  !!idest=mod(mype-1+numberpe,numberpe)
  idest=left_pe
  !!isource=mod(mype+1,numberpe)
  isource=right_pe
  !!isendtag=mype
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendl,icount,mpi_rsize,idest,isendtag,&
  !     recvr,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndleft_toroidal_cmm(ptr,sendl,icount)
  call rcvright_toroidal_cmm(ptr,recvr)

! unpack phi_boundary and calculate e_zeta at boundaries, mzeta=1
!$omp parallel do private(i,j,ii,jt,ij,pleft,pright)
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

! d_phi/d_zeta
     do j=1,mtheta(i)
        ij=igrid(i)+j
        if(mzeta==1)then           
           evector(3,1,ij)=(pright(j)-pleft(j))*diffz
        elseif(mzeta==2)then
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,2,ij)=(pright(j)-phi(1,ij))*diffz
        else
           evector(3,1,ij)=(phi(2,ij)-pleft(j))*diffz
           evector(3,mzeta,ij)=(pright(j)-phi(mzeta-1,ij))*diffz
           evector(3,2:mzeta-1,ij)=(phi(3:mzeta,ij)-phi(1:mzeta-2,ij))*diffz
        endif
     enddo
  enddo

! adjust the difference between safety factor q and qtinv for fieldline coordinate
!omp parallel do private(i,j,r,q,delq,ij)
  do i=1,mpsi-1
     r=a0+deltar*real(i)
     q=q0+q1*r/a+q2*r*r/(a*a)
     delq=(1.0/q-qtinv(i))

     do j=1,mtheta(i)
        ij=igrid(i)+j
        evector(3,:,ij)=evector(3,:,ij)+delq*evector(2,:,ij)
     enddo
  enddo
  
! add (0,0) mode, d phi/d psi
  if(mode00==1)then
!omp parallel do private(i,j,k,ii,jt,r)
     do i=1,mpsi-1
        r=a0+deltar*real(i)
        ii=igrid(i)
        jt=mtheta(i)
        do j=ii+1,ii+jt
           do k=1,mzeta
              evector(1,k,j)=evector(1,k,j)+phip00(i)/r
           enddo
        enddo
     enddo
  endif
  
! toroidal end point, pack end point (k=mzeta) data 
! send e to right and receive from left
!$omp parallel do private(i)
  do i=1,mgrid
     sendrs(1,i)=evector(1,mzeta,i)
     sendrs(2,i)=evector(2,mzeta,i)
     sendrs(3,i)=evector(3,mzeta,i)
     recvls(1,i)=0.0
     recvls(2,i)=0.0
     recvls(3,i)=0.0
  enddo
  icount=3*mgrid
  idest=right_pe
  isource=left_pe
  isendtag=myrank_toroidal
  irecvtag=isource
  !call mpi_sendrecv(sendrs,icount,mpi_rsize,idest,isendtag,&
  !     recvls,icount,mpi_rsize,isource,irecvtag,toroidal_comm,istatus,ierror)
  call sndright_toroidal_cmm(ptr,sendrs,icount)
  call rcvleft_toroidal_cmm(ptr,recvls)
 
! unpack end point data for k=0
  if(myrank_toroidal==0)then
!$omp parallel do private(i,ii,jt)
     do i=1,mpsi-1
        ii=igrid(i)
        jt=mtheta(i)
        evector(1,0,ii+1:ii+jt)=cshift(recvls(1,ii+1:ii+jt),-itran(i))
        evector(2,0,ii+1:ii+jt)=cshift(recvls(2,ii+1:ii+jt),-itran(i))
        evector(3,0,ii+1:ii+jt)=cshift(recvls(3,ii+1:ii+jt),-itran(i))
     enddo
  else
!$omp parallel do private(i)
     do i=1,mgrid
        evector(1,0,i)=recvls(1,i)
        evector(2,0,i)=recvls(2,i)
        evector(3,0,i)=recvls(3,i)
     enddo
  endif
  
! poloidal end point
!omp parallel do private(i,j)
  do i=1,mpsi-1
     do j=0,mzeta
        evector(1,j,igrid(i))=evector(1,j,igrid(i)+mtheta(i))
        evector(2,j,igrid(i))=evector(2,j,igrid(i)+mtheta(i))
        evector(3,j,igrid(i))=evector(3,j,igrid(i)+mtheta(i))
     enddo
  enddo
  
end subroutine fieldr_9
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

