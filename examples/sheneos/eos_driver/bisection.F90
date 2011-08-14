subroutine bisection(lr,lt0,y,eps0,lt,bivar,keyerrt,keybisect)

  use eosmodule

  implicit none

  real*8 lr,lt0,y,eps0,lt
  integer keyerrt

  integer keybisect !this doesn't do anything...
  ! 1 -> operate on energy
  ! 2 -> operate on entropy

  !temporary vars
  real*8 lt1,lt2,ltmin,ltmax
  real*8 f1,f2,fmid,dlt,ltmid
  real*8 d1,d2,d3,tol
  real*8 f1a,f2a

  integer bcount,i,itmax,maxbcount

  real*8 bivar

  bcount = 0
  maxbcount = 80

  tol=1.d-9 ! need to find energy to less than 1 in 10^-9
  itmax=50

  ltmax=logtemp(ntemp)
  ltmin=logtemp(1)

  lt = lt0
  lt1 = dlog10(min(10.0d0**ltmax,1.10d0*(10.0d0**lt0)))
  lt2 = dlog10(max(10.0d0**ltmin,0.90d0*(10.0d0**lt0)))

  call findthis(lr,lt1,y,f1a,bivar,d1,d2,d3)
  call findthis(lr,lt2,y,f2a,bivar,d1,d2,d3)

  f1=f1a-eps0
  f2=f2a-eps0

  keyerrt=0
  

  do while(f1*f2.ge.0.0d0)
     bcount=bcount+1
     lt1=dlog10(min(10.0d0**ltmax,1.2d0*(10.0d0**lt1)))
     lt2=dlog10(max(10.0d0**ltmin,0.8d0*(10.0d0**lt2)))
     call findthis(lr,lt1,y,f1a,bivar,d1,d2,d3)
     call findthis(lr,lt2,y,f2a,bivar,d1,d2,d3)
     f1=f1a-eps0
     f2=f2a-eps0
     if(bcount.ge.maxbcount) then
        keyerrt=667
        return
     endif
  enddo

  if(f1.lt.0.0d0) then
     lt=lt1
     dlt=dlog10((10.0D0**lt2)-(10.0d0**lt1))
  else
     lt=lt2
     dlt=dlog10((10.0D0**lt1)-(10.0d0**lt2))
  endif

  do i=1,itmax
     dlt=dlog10((10.0d0**dlt)*0.5d0)
     ltmid=dlog10(10.0d0**lt+10.0d0**dlt)
     call findthis(lr,ltmid,y,f2a,bivar,d1,d2,d3)
     fmid=f2a-eps0
     if(fmid.le.0.0d0) lt=ltmid
     if(abs(1.0d0-f2a/eps0).lt.tol) then
        lt=ltmid
        return
     endif
  enddo



end subroutine bisection

subroutine bisection_rho(lr0,lt,y,lpressin,lr,bivar,keyerrr,keybisect)

  use eosmodule

  implicit none

  real*8 lr,lr0,y,lpressin,lt
  integer keyerrr

  integer keybisect !this doesn't do anything
  ! 3 -> operate on pressure

  !temporary vars
  real*8 lr1,lr2,lrmin,lrmax
  real*8 f1,f2,fmid,dlr,lrmid
  real*8 d1,d2,d3,tol
  real*8 f1a,f2a

  integer bcount,i,itmax,maxbcount

  real*8 bivar

  bcount = 0
  maxbcount = 80

  tol=1.d-9 ! need to find energy to less than 1 in 10^-9
  itmax=50

  lrmax=logrho(nrho)
  lrmin=logrho(1)

  lr = lr0
  lr1 = dlog10(min(10.0d0**lrmax,1.10d0*(10.0d0**lr0)))
  lr2 = dlog10(max(10.0d0**lrmin,0.90d0*(10.0d0**lr0)))

  call findthis(lr1,lt,y,f1a,bivar,d1,d2,d3)
  call findthis(lr2,lt,y,f2a,bivar,d1,d2,d3)

  f1=f1a-lpressin
  f2=f2a-lpressin

  keyerrr=0
  
  do while(f1*f2.ge.0.0d0)
     bcount=bcount+1
     lr1=dlog10(min(10.0d0**lrmax,1.2d0*(10.0d0**lr1)))
     lr2=dlog10(max(10.0d0**lrmin,0.8d0*(10.0d0**lr2)))
     call findthis(lr1,lt,y,f1a,bivar,d1,d2,d3)
     call findthis(lr2,lt,y,f2a,bivar,d1,d2,d3)
     f1=f1a-lpressin
     f2=f2a-lpressin
     if(bcount.ge.maxbcount) then
        keyerrr=667
        return
     endif
  enddo

  if(f1.lt.0.0d0) then
     lr=lr1
     dlr=dlog10((10.0D0**lr2)-(10.0d0**lr1))
  else
     lr=lr2
     dlr=dlog10((10.0D0**lr1)-(10.0d0**lr2))
  endif

  do i=1,itmax
     dlr=dlog10((10.0d0**dlr)*0.5d0)
     lrmid=dlog10(10.0d0**lr+10.0d0**dlr)
     call findthis(lrmid,lt,y,f2a,bivar,d1,d2,d3)
     fmid=f2a-lpressin
     if(fmid.le.0.0d0) lr=lrmid
     if(abs(1.0d0-f2a/lpressin).lt.tol) then
        lr=lrmid
        return
     endif
  enddo



end subroutine bisection_rho
