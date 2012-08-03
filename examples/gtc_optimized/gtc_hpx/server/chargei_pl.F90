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

  real(wp) dnitmp(0:mzeta,mgrid)

  do ij=1,mgrid
    do kk=0,mzeta
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  print*,' before ',mype, dnitmp(3,3)

  call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  

  print*,' after ', mype, densityi(3,3)

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

  real(wp) dnitmp(0:mzeta,mgrid)

  do ij=1,mgrid
    do kk=0,mzeta
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  print*,' before ',mype, dnitmp(3,3)

  call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  

  print*,' after ', mype, densityi(3,3)

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

  real(wp) dnitmp(0:mzeta,mgrid)

  do ij=1,mgrid
    do kk=0,mzeta
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  print*,' before ',mype, dnitmp(3,3)

  call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  

  print*,' after ', mype, densityi(3,3)

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

  real(wp) dnitmp(0:mzeta,mgrid)

  do ij=1,mgrid
    do kk=0,mzeta
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  print*,' before ',mype, dnitmp(3,3)

  call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  

  print*,' after ', mype, densityi(3,3)

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

  real(wp) dnitmp(0:mzeta,mgrid)

  do ij=1,mgrid
    do kk=0,mzeta
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  print*,' before ',mype, dnitmp(3,3)

  call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  

  print*,' after ', mype, densityi(3,3)

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

  real(wp) dnitmp(0:mzeta,mgrid)

  do ij=1,mgrid
    do kk=0,mzeta
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  print*,' before ',mype, dnitmp(3,3)

  call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  

  print*,' after ', mype, densityi(3,3)

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

  real(wp) dnitmp(0:mzeta,mgrid)

  do ij=1,mgrid
    do kk=0,mzeta
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  print*,' before ',mype, dnitmp(3,3)

  call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  

  print*,' after ', mype, densityi(3,3)

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

  real(wp) dnitmp(0:mzeta,mgrid)

  do ij=1,mgrid
    do kk=0,mzeta
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  print*,' before ',mype, dnitmp(3,3)

  call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  

  print*,' after ', mype, densityi(3,3)

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

  real(wp) dnitmp(0:mzeta,mgrid)

  do ij=1,mgrid
    do kk=0,mzeta
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  print*,' before ',mype, dnitmp(3,3)

  call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  

  print*,' after ', mype, densityi(3,3)

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

  real(wp) dnitmp(0:mzeta,mgrid)

  do ij=1,mgrid
    do kk=0,mzeta
      dnitmp(kk,ij)=mype*100.0 + ij
      densityi(kk,ij)=0.
    enddo
  enddo

  print*,' before ',mype, dnitmp(3,3)

  call partd_allreduce_cmm(ptr,dnitmp,densityi,mgrid,mzeta+1);  

  print*,' after ', mype, densityi(3,3)

end subroutine chargei_9

