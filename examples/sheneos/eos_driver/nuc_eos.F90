! #########################################################
!
! Copyright C. D. Ott and Evan O'Connor, July 2009
!
! UNITS: density              g/cm^3
!        temperature          MeV
!        ye                   number fraction per baryon
!        energy               erg/g
!        pressure             dyn/cm^2
!        chemical potentials  MeV
!        entropy              k_B / baryon
!        cs2                  cm^2/s^2 (not relativistic)
!
! keyerr --> error output; should be 0
! rfeps --> root finding relative accuracy, set around 1.0d-10
! keytemp: 0 -> coming in with rho,eps,ye (solve for temp)
!          1 -> coming in with rho,temperature,ye
!          2 -> coming in with rho,entropy,ye (solve for temp)
!          3 -> coming in with pressure,temp,ye (solve for rho)
!
subroutine nuc_eos_full(xrho,xtemp,xye,xenr,xprs,xent,xcs2,xdedt,&
     xdpderho,xdpdrhoe,xxa,xxh,xxn,xxp,xabar,xzbar,xmu_e,xmu_n,xmu_p, &
     xmuhat,keytemp,keyerr,rfeps)

  use eosmodule
  implicit none

  real*8, intent(in)    :: xye
  real*8, intent(inout) :: xrho,xtemp,xenr,xent
  real*8, intent(out)   :: xprs,xcs2,xdedt
  real*8, intent(out)   :: xdpderho,xdpdrhoe,xxa,xxh,xxn,xxp
  real*8, intent(out)   :: xabar,xzbar,xmu_e,xmu_n,xmu_p,xmuhat
  real*8, intent(in)    :: rfeps
  integer, intent(in)   :: keytemp
  integer, intent(out)  :: keyerr


  ! local variables
  real*8 :: lr,lt,y,xx,xeps,leps,xs,xpressure
  real*8 :: d1,d2,d3
  real*8 :: ff(nvars)
  integer :: keyerrt = 0
  integer :: keyerrr = 0
  

  if(xrho.gt.eos_rhomax) then
     stop "nuc_eos: rho > rhomax"
  endif

  if(xrho.lt.eos_rhomin) then
     stop "nuc_eos: rho < rhomin"
  endif

  if(xye.gt.eos_yemax) then
     stop "nuc_eos: ye > yemax"
  endif

  if(xye.lt.eos_yemin) then
     stop "nuc_eos: ye < yemin"
  endif

  if(keytemp.eq.1) then
     if(xtemp.gt.eos_tempmax) then
        stop "nuc_eos: temp > tempmax"
     endif
     
     if(xtemp.lt.eos_tempmin) then
        stop "nuc_eos: temp < tempmin"
     endif
  endif

  lr = log10(xrho)
  lt = log10(xtemp)
  y = xye
  xeps = xenr + energy_shift
  leps = log10(max(xeps,1.0d0))

  keyerr = 0

  if(keytemp.eq.0) then
     !need to find temperature based on xeps
     call findtemp(lr,lt,y,leps,keyerrt,rfeps)
     if(keyerrt.ne.0) then
        stop "Did not find temperature"
     endif
     xtemp = 10.0d0**lt

  elseif(keytemp.eq.2) then
     !need to find temperature based on xent
     xs = xent
     call findtemp_entropy(lr,lt,y,xs,keyerrt,rfeps)
     xtemp = 10.0d0**lt

  elseif(keytemp.eq.3) then
     !need to find rho based on xprs
     xpressure = log10(xprs)
     call findrho_press(lr,lt,y,xpressure,keyerrr,rfeps)
     if(keyerrr.ne.0) then
        write(*,*) "Problem in findrho_press:", keyerr
        keyerr = keyerrr
    return
     endif
     xrho = 10.0d0**lr

  endif

  ! have rho,T,ye; proceed:
  call findall(lr,lt,y,ff)

  !unless we want xprs to be constant (keytemp==3), reset xprs
  if(.not.keytemp.eq.3) then
     xprs = 10.0d0**ff(1)
  endif

  !unless we want xenr to be constant (keytemp==0), reset xenr
  if(.not.keytemp.eq.0) then
     xenr = 10.0d0**ff(2) - energy_shift
  endif

  !unless we want xent to be constant (keytemp==2), reset xent
  if(.not.keytemp.eq.2) then
     xent = ff(3)
  endif

  xcs2 = ff(5)

! derivatives
  xdedt = ff(6)

  xdpdrhoe = ff(7)

  xdpderho = ff(8)

! chemical potentials
  xmuhat = ff(9)

  xmu_e = ff(10)

  xmu_p = ff(11)

  xmu_n = ff(12)

! compositions
  xxa = ff(13)

  xxh = ff(14)

  xxn = ff(15)

  xxp = ff(16)

  xabar = ff(17)

  xzbar = ff(18)


end subroutine nuc_eos_full

subroutine nuc_low_eos(xrho,xenr,xprs,xcs2,xdpderho,xdpdrhoe,keytemp)

  implicit none
  real*8 xrho,xenr,xprs,xcs2,xdpderho,xdpdrhoe
  real*8,parameter :: idealK1 =  1.2435d15 * (0.5d0**(4.d0/3.d0))
  real*8,parameter :: idealgamma = 1.41d0
  integer keytemp

  if(keytemp.eq.1) then
!	energy wanted
     xprs=idealK1*xrho**(idealgamma)
     xenr=xprs/xrho/(idealgamma-1.d0)
     xcs2=idealgamma*xprs/xrho
  endif

  xprs = (idealgamma - 1.0d0) * xrho * xenr

  xdpderho = (idealgamma - 1.0d0 ) * xrho
  xdpdrhoe = (idealgamma - 1.0d0 ) * xenr

  xcs2= xdpdrhoe+xdpderho*xprs/xrho**2

end subroutine nuc_low_eos

subroutine nuc_eos_short(xrho,xtemp,xye,xenr,xprs,xent,xcs2,xdedt,&
     xdpderho,xdpdrhoe,xmunu,keytemp,keyerr,rfeps)

  use eosmodule
  implicit none

  real*8, intent(in)    :: xye
  real*8, intent(inout) :: xrho,xtemp,xenr,xent
  real*8, intent(in)    :: rfeps
  real*8, intent(out)   :: xprs,xmunu,xcs2,xdedt
  real*8, intent(out)   :: xdpderho,xdpdrhoe
  integer, intent(in)   :: keytemp
  integer, intent(out)  :: keyerr

  ! local variables
  real*8 :: lr,lt,y,xx,xeps,leps,xs,xpressure
  real*8 :: d1,d2,d3,ff(8)
  integer :: keyerrt = 0
  integer :: keyerrr = 0

  if(xrho.gt.eos_rhomax) then
     stop "nuc_eos: rho > rhomax"
  endif

  if(xrho.lt.eos_rhomin*1.2d0) then
     call nuc_low_eos(xrho,xenr,xprs,xcs2,xdpderho,xdpdrhoe,keytemp)
     xent = 4.0d0
     return
  endif

  if(xye.gt.eos_yemax) then
     stop "nuc_eos: ye > yemax"
  endif

  if(xye.lt.eos_yemin) then
     stop "nuc_eos: ye < yemin"
  endif

  if(keytemp.eq.1) then
     if(xtemp.gt.eos_tempmax) then
        stop "nuc_eos: temp > tempmax"
     endif
     
     if(xtemp.lt.eos_tempmin) then
        call nuc_low_eos(xrho,xenr,xprs,xcs2,xdpderho,xdpdrhoe,keytemp)
        xent = 4.0d0
        return
     endif
  endif

  lr = log10(xrho)
  lt = log10(xtemp)
  y = xye
  xeps = xenr + energy_shift
  leps = log10(max(xeps,1.0d0))

  keyerr = 0

  if(keytemp.eq.0) then
     !need to find temperature based on xeps
     call findtemp(lr,lt,y,leps,keyerrt,rfeps)
     if(keyerrt.ne.0) then
        keyerr = keyerrt
        return
     endif
     xtemp = 10.0d0**lt

  elseif(keytemp.eq.2) then
     !need to find temperature based on xent
     xs = xent
     call findtemp_entropy(lr,lt,y,xs,keyerrt,rfeps)
     if(keyerrt.ne.0) then
        keyerr = keyerrt
        return
     endif
     xtemp = 10.0d0**lt

  elseif(keytemp.eq.3) then
     !need to find rho based on xprs
     xpressure = log10(xprs)
     call findrho_press(lr,lt,y,xpressure,keyerrr,rfeps)
     if (keyerrr.ne.0) then
        keyerr = keyerrr
        write(*,*) "Problem in findrho_press:", keyerr
        return
     endif
     xrho = 10.0d0**lr
  endif

  ! have rho,temp,ye; proceed:
  call findall_short(lr,lt,y,ff)

  !unless we want xprs to be constant (keytemp==3), reset xprs
  if(.not.keytemp.eq.3) then
     xprs = 10.0d0**ff(1)
  endif

  !unless we want xenr to be constant (keytemp==0), reset xenr
  if(.not.keytemp.eq.0) then
     xenr = 10.0d0**ff(2) - energy_shift
  endif

  !unless we want xent to be constant (keytemp==2), reset xent
  if(.not.keytemp.eq.2) then
     xent = ff(3)
  endif

  xmunu = ff(4)

  xcs2 = ff(5)

  xdedt = ff(6)

  xdpdrhoe = ff(7)

  xdpderho = ff(8)

end subroutine nuc_eos_short

subroutine findthis(lr,lt,y,value,array,d1,d2,d3)

  use eosmodule

  implicit none

  integer rip,rim
  integer tip,tim
  integer yip,yim

  real*8 lr,lt,y,value,d1,d2,d3
  real*8 array(*)

! Ewald's interpolator           
  call intp3d(lr,lt,y,value,1,array,nrho,ntemp,nye,logrho,logtemp,ye,d1,d2,d3)


end subroutine findthis


subroutine findall(lr,lt,y,ff)

  use eosmodule
  implicit none

  real*8 ff(nvars)
  real*8 ffx(nvars,1)
  real*8 lr,lt,y
  integer i

! Ewald's interpolator           
  call intp3d_many(lr,lt,y,ffx,1,alltables,&
       nrho,ntemp,nye,nvars,logrho,logtemp,ye)
  ff(:) = ffx(:,1)


end subroutine findall


subroutine findall_short(lr,lt,y,ff)

  use eosmodule
  implicit none

  real*8 ffx(8,1)
  real*8 ff(8)
  real*8 lr,lt,y
  integer i
  integer :: nvarsx = 8


! Ewald's interpolator           
  call intp3d_many(lr,lt,y,ffx,1,alltables(:,:,:,1:8), &
       nrho,ntemp,nye,nvarsx,logrho,logtemp,ye)
  ff(:) = ffx(:,1)

end subroutine findall_short
