!-*-f90-*-
subroutine findrho_press(lr0,lt,y,lpressin,keyerrr,tol)

  use eosmodule

  implicit none

  !given initial guess of density
  real*8 :: lr0
  !working density
  real*8 :: lr, lr_lastguess
  !given T and Ye
  real*8 :: lt,y
  !pressure corresponding to lr1,lt,y
  real*8 :: lpressin
  
  !accuracy we need rho to, ~1 part in tol^{-1}
  real*8 :: tol
  !will be derivatives of pressure wrt rho,T,ye
  real*8 :: d1,d2,d3
  !helpers along the way
  real*8 :: lpress_of_guess,lpress_of_lastguess,first_lpress,ldr,lr_new
  !bounds of density
  real*8 :: lrmax,lrmin
  !counters & flags
  integer :: rl = 0
  integer :: itmax,i,keyerrr
  
  keyerrr=0

  itmax=20 ! use at most 20 iterations, then bisection

  lr=lr0

  lrmax=logrho(nrho)
  lrmin=logrho(1)

  !Note: We are using Ewald's Lagrangian interpolator here!

  !first use initial guess to estimate derivatives
  call findthis(lr,lt,y,lpress_of_guess,alltables(:,:,:,1),d1,d2,d3)
  
  first_lpress = lpress_of_guess

  !preconditioning 1: do we already have the right rho?
  if (abs(lpressin-lpress_of_guess).lt.tol*abs(lpressin)) then
     lr0 = lr
     return
  endif


  !otherwise we must newton-raphson to get the real rho
  do i=1,itmax

     !d1 is the derivative dlpress/dlogrho, use to guess hopefully better rho
!     write(*,*) i,lr,d1,lpressin-lpress_of_guess
     ldr= (lpressin-lpress_of_guess)/d1 
     if (abs(ldr).gt.5.0d0) then
        write(*,*) i,ldr,d1,lr,lr_new
        keyerrr = 473
        write(*,*) "dpdrho very small"
        return
     endif
     lr_new = lr+ldr

     !make sure we do not go out of bounds
     lr_new = min(lr_new,lrmax)
     lr_new = max(lr_new,lrmin)

     !keep old variables incase we want to make our own derivatives
     lpress_of_lastguess = lpress_of_guess
     lr_lastguess = lr
     
     !use to get next iteration of NR
     lr = lr_new     
     call findthis(lr,lt,y,lpress_of_guess,alltables(:,:,:,1),d1,d2,d3)
     if (abs(lpressin - lpress_of_guess).lt.tol*abs(lpressin)) then
        !yes, we got it!
        lr0 = lr
        return
     endif

     ! if we are closer than 10^-3  to the 
     ! root (lpressin-lpress_of_guess)=0, we are switching to 
     ! the secant method, since the table is rather coarse and the
     ! derivatives may be garbage.
     if(abs(lpressin-lpress_of_guess).lt.1.0d-3*abs(lpressin)) then
        d1 = (lpress_of_guess-lpress_of_lastguess)/(lr-lr_lastguess)
     endif

  enddo

  !we may fail in the NR, after itmax reached.  Then we revert to the bisection method.
  if(i.ge.itmax) then
     keyerrr=667
     call bisection_rho(lr0,lt,y,lpressin,lr,alltables(:,:,:,1),keyerrr,3)
     if(keyerrr.eq.667) then
        ! total failure
        call findthis(lr,lt,y,lpress_of_guess,alltables(:,:,:,1),d1,d2,d3)
        write(*,*) "EOS: Did not converge in findrho_press!"
        write(*,*) "iteration,logrho0,logtemp,ye,lpressin,lpress_first,rhoreturn,press_of_rhoreturn"
        write(*,"(i4,1P10E19.10)") i,lr0,lt,y,lpressin,first_lpress,lr,lpress_of_guess
        write(*,*) "Tried calling bisection... didn't help... :-/"
        write(*,*) "Bisection error: ",keyerrr
     endif
  endif
    
  lr0 = lr
  return


end subroutine findrho_press
