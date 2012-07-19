subroutine chargei(ptr)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_PTR
  implicit none
  TYPE(C_PTR), INTENT(IN), VALUE :: ptr

  call partd_allreduce_cmm(ptr);  

end subroutine chargei

