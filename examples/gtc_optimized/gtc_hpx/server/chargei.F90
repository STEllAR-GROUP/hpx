subroutine chargei(ptr)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  use global_parameters
  use particle_array
  use field_array
  use diagnosis_array
  use particle_decomp

  implicit none
  TYPE(C_PTR), INTENT(IN), VALUE :: ptr
  integer ij,kk

  integer n1,n2
  !real(wp) dnitmp(0:mzeta,mgrid)
  !real(wp) recvr(mgrid)
  real(wp) dns(0:3,3)
  real(wp) dnitmp(0:3,3)
  real(wp) recvr(3)
  real(wp) adum(0:mpsi)

  n1 = 3
  n2 = 3

  !do ij=1,mgrid
  !  do kk=0,mzeta
  do ij=1,n1
    do kk=0,n2
      dnitmp(kk,ij)=mype*100.0 + ij
      dns(kk,ij)=0.
    enddo
  enddo

  !call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  
  call partd_allreduce_cmm(ptr,dnitmp,dns,n2,n1+1);  

  recvr = 0.0d0
  !call sndleft_toroidal_cmm(ptr,densityi(0,:),mgrid);  
  call sndleft_toroidal_cmm(ptr,dns(0,:),n2);  
  call rcvright_toroidal_cmm(ptr,recvr);  

  zonali = 0.0d0
  call toroidal_allreduce_cmm(ptr,zonali,adum,mpsi+1);  

end subroutine chargei

