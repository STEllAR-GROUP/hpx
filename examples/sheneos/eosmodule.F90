 module eosmodule

   implicit none
   
   integer,save :: nrho,ntemp,nye

   integer,save :: warn_from !warn from given reflevel

   real*8 :: energy_shift = 0.0d0
   
   real*8 :: precision = 1.0d-9

! min-max values:
   real*8 :: eos_rhomin,eos_rhomax
   real*8 :: eos_yemin,eos_yemax
   real*8 :: eos_tempmin,eos_tempmax

   real*8 :: t_max_hack = 240.0d0

! basics
   integer, parameter :: nvars = 19
   real*8,allocatable :: alltables(:,:,:,:)
  ! index variable mapping:
  !  1 -> logpress
  !  2 -> logenergy
  !  3 -> entropy
  !  4 -> munu
  !  5 -> cs2
  !  6 -> dedT
  !  7 -> dpdrhoe
  !  8 -> dpderho
  !  9 -> muhat
  ! 10 -> mu_e
  ! 11 -> mu_p
  ! 12 -> mu_n
  ! 13 -> xa
  ! 14 -> xh
  ! 15 -> xn
  ! 16 -> xp
  ! 17 -> abar
  ! 18 -> zbar
  ! 19 -> gamma

   real*8,allocatable,save :: logrho(:)
   real*8,allocatable,save :: logtemp(:)
   real*8,allocatable,save :: ye(:)

! constants
   real*8,save :: mev_to_erg = 1.60217733d-6
   real*8,save :: amu_cgs = 1.66053873d-24
   real*8,save :: amu_mev = 931.49432d0
   real*8,save :: pi = 3.14159265358979d0
   real*8,save :: ggrav = 6.672d-8
   real*8,save :: temp_mev_to_kelvin = 1.1604447522806d10
   real*8,save :: clight = 2.99792458d10
   real*8,save :: kb_erg = 1.380658d-16
   real*8,save :: kb_mev = 8.61738568d-11   


 end module eosmodule
