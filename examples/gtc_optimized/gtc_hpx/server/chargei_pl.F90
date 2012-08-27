subroutine chargei_0(ptr)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use global_parameters_0
  use particle_array_0
  use field_array_0
  use diagnosis_array_0
  use particle_decomp_0

  implicit none
  type(c_ptr), intent(in), value :: ptr
  integer ij,kk

  integer n1,n2
  !real(8) dnitmp(0:mzeta,mgrid)
  !real(8) recvr(mgrid)
  real(8) dnitmp(0:3,3)
  real(8) recvr(3)

  n1 = 3
  n2 = 3

  !do ij=1,mgrid
  !  do kk=0,mzeta
  do ij=1,n1
    do kk=0,n2
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  !call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  
  call partd_allreduce_cmm(ptr,dnitmp,densityi,n2,n1+1);  

  recvr = 0.0d0
  !call sndleft_toroidal_cmm(ptr,densityi(0,:),mgrid);  
  call sndleft_toroidal_cmm(ptr,densityi(0,:),n2);  
  call rcvright_toroidal_cmm(ptr,recvr);  

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
  integer ij,kk

  integer n1,n2
  !real(8) dnitmp(0:mzeta,mgrid)
  !real(8) recvr(mgrid)
  real(8) dnitmp(0:3,3)
  real(8) recvr(3)

  n1 = 3
  n2 = 3

  !do ij=1,mgrid
  !  do kk=0,mzeta
  do ij=1,n1
    do kk=0,n2
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  !call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  
  call partd_allreduce_cmm(ptr,dnitmp,densityi,n2,n1+1);  

  recvr = 0.0d0
  !call sndleft_toroidal_cmm(ptr,densityi(0,:),mgrid);  
  call sndleft_toroidal_cmm(ptr,densityi(0,:),n2);  
  call rcvright_toroidal_cmm(ptr,recvr);  

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
  integer ij,kk

  integer n1,n2
  !real(8) dnitmp(0:mzeta,mgrid)
  !real(8) recvr(mgrid)
  real(8) dnitmp(0:3,3)
  real(8) recvr(3)

  n1 = 3
  n2 = 3

  !do ij=1,mgrid
  !  do kk=0,mzeta
  do ij=1,n1
    do kk=0,n2
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  !call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  
  call partd_allreduce_cmm(ptr,dnitmp,densityi,n2,n1+1);  

  recvr = 0.0d0
  !call sndleft_toroidal_cmm(ptr,densityi(0,:),mgrid);  
  call sndleft_toroidal_cmm(ptr,densityi(0,:),n2);  
  call rcvright_toroidal_cmm(ptr,recvr);  

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
  integer ij,kk

  integer n1,n2
  !real(8) dnitmp(0:mzeta,mgrid)
  !real(8) recvr(mgrid)
  real(8) dnitmp(0:3,3)
  real(8) recvr(3)

  n1 = 3
  n2 = 3

  !do ij=1,mgrid
  !  do kk=0,mzeta
  do ij=1,n1
    do kk=0,n2
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  !call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  
  call partd_allreduce_cmm(ptr,dnitmp,densityi,n2,n1+1);  

  recvr = 0.0d0
  !call sndleft_toroidal_cmm(ptr,densityi(0,:),mgrid);  
  call sndleft_toroidal_cmm(ptr,densityi(0,:),n2);  
  call rcvright_toroidal_cmm(ptr,recvr);  

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
  integer ij,kk

  integer n1,n2
  !real(8) dnitmp(0:mzeta,mgrid)
  !real(8) recvr(mgrid)
  real(8) dnitmp(0:3,3)
  real(8) recvr(3)

  n1 = 3
  n2 = 3

  !do ij=1,mgrid
  !  do kk=0,mzeta
  do ij=1,n1
    do kk=0,n2
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  !call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  
  call partd_allreduce_cmm(ptr,dnitmp,densityi,n2,n1+1);  

  recvr = 0.0d0
  !call sndleft_toroidal_cmm(ptr,densityi(0,:),mgrid);  
  call sndleft_toroidal_cmm(ptr,densityi(0,:),n2);  
  call rcvright_toroidal_cmm(ptr,recvr);  

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
  integer ij,kk

  integer n1,n2
  !real(8) dnitmp(0:mzeta,mgrid)
  !real(8) recvr(mgrid)
  real(8) dnitmp(0:3,3)
  real(8) recvr(3)

  n1 = 3
  n2 = 3

  !do ij=1,mgrid
  !  do kk=0,mzeta
  do ij=1,n1
    do kk=0,n2
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  !call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  
  call partd_allreduce_cmm(ptr,dnitmp,densityi,n2,n1+1);  

  recvr = 0.0d0
  !call sndleft_toroidal_cmm(ptr,densityi(0,:),mgrid);  
  call sndleft_toroidal_cmm(ptr,densityi(0,:),n2);  
  call rcvright_toroidal_cmm(ptr,recvr);  

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
  integer ij,kk

  integer n1,n2
  !real(8) dnitmp(0:mzeta,mgrid)
  !real(8) recvr(mgrid)
  real(8) dnitmp(0:3,3)
  real(8) recvr(3)

  n1 = 3
  n2 = 3

  !do ij=1,mgrid
  !  do kk=0,mzeta
  do ij=1,n1
    do kk=0,n2
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  !call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  
  call partd_allreduce_cmm(ptr,dnitmp,densityi,n2,n1+1);  

  recvr = 0.0d0
  !call sndleft_toroidal_cmm(ptr,densityi(0,:),mgrid);  
  call sndleft_toroidal_cmm(ptr,densityi(0,:),n2);  
  call rcvright_toroidal_cmm(ptr,recvr);  

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
  integer ij,kk

  integer n1,n2
  !real(8) dnitmp(0:mzeta,mgrid)
  !real(8) recvr(mgrid)
  real(8) dnitmp(0:3,3)
  real(8) recvr(3)

  n1 = 3
  n2 = 3

  !do ij=1,mgrid
  !  do kk=0,mzeta
  do ij=1,n1
    do kk=0,n2
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  !call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  
  call partd_allreduce_cmm(ptr,dnitmp,densityi,n2,n1+1);  

  recvr = 0.0d0
  !call sndleft_toroidal_cmm(ptr,densityi(0,:),mgrid);  
  call sndleft_toroidal_cmm(ptr,densityi(0,:),n2);  
  call rcvright_toroidal_cmm(ptr,recvr);  

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
  integer ij,kk

  integer n1,n2
  !real(8) dnitmp(0:mzeta,mgrid)
  !real(8) recvr(mgrid)
  real(8) dnitmp(0:3,3)
  real(8) recvr(3)

  n1 = 3
  n2 = 3

  !do ij=1,mgrid
  !  do kk=0,mzeta
  do ij=1,n1
    do kk=0,n2
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  !call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  
  call partd_allreduce_cmm(ptr,dnitmp,densityi,n2,n1+1);  

  recvr = 0.0d0
  !call sndleft_toroidal_cmm(ptr,densityi(0,:),mgrid);  
  call sndleft_toroidal_cmm(ptr,densityi(0,:),n2);  
  call rcvright_toroidal_cmm(ptr,recvr);  

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
  integer ij,kk

  integer n1,n2
  !real(8) dnitmp(0:mzeta,mgrid)
  !real(8) recvr(mgrid)
  real(8) dnitmp(0:3,3)
  real(8) recvr(3)

  n1 = 3
  n2 = 3

  !do ij=1,mgrid
  !  do kk=0,mzeta
  do ij=1,n1
    do kk=0,n2
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  !call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  
  call partd_allreduce_cmm(ptr,dnitmp,densityi,n2,n1+1);  

  recvr = 0.0d0
  !call sndleft_toroidal_cmm(ptr,densityi(0,:),mgrid);  
  call sndleft_toroidal_cmm(ptr,densityi(0,:),n2);  
  call rcvright_toroidal_cmm(ptr,recvr);  

end subroutine chargei_9

