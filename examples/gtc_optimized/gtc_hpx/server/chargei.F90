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

  real(wp) dnitmp(0:mzeta,mgrid)
  real(wp) recvr(mgrid)

  do ij=1,mgrid
    do kk=0,mzeta
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  

  recvr = 0.0d0
  call sndleft_toroidal_cmm(ptr,densityi(0,:),recvr,mgrid);  

end subroutine chargei

