program one_zone

  use eosmodule
  implicit none

  real*8 rho,y,temp,p,eps,zdedt,s
  real*8 pred_rho,pred_y,pred_temp,pred_p,pred_eps,pred_s
  real*8 pp,ep,rp,tp,sp,yp,dedtime,pred_dedtime
  real*8 xrho,xtemp,xye,xenr,xprs,xent,xcs2,xzdedt,&
       xmunu
  integer keytemp,keyerr

  integer nt
  real*8 dtime,time,drhodt,pred_drhodt
  real*8 xdpderho,xdpdrhoe

  real*8 :: rhoin = 8.655d9
  real*8 :: yin   = 0.1535d0
  real*8 :: tin   = 3.628d10!7.628d9
  real*8 :: rho_max = 6.0d14

  call readtable("EOStable_filename.h5")

  rho = rhoin
  temp = tin/temp_mev_to_kelvin
  y = yin

  ! get start energy
  keytemp = 1
  keyerr =0
  xrho = rho
  xtemp = temp
  xye = y

  call nuc_eos_short(xrho,xtemp,xye,xenr,xprs,xent,xcs2,xzdedt,&
       xdpderho,xdpdrhoe,xmunu,keytemp,keyerr,precision)

  p = xprs
  eps = xenr
  zdedt = xzdedt
  s = xent
  
  pp = p
  ep = eps
  rp = rho
  tp = temp
  sp = s
  yp = y

  nt = 0
  dtime= 1.0d-7
  time = 0.0d0

  write(6,"(i9,1P10E15.6)") nt,time,rho,temp,y,eps,s

  open(unit=654,file="one_zone_output.dat",status="unknown")
  
  do while(rho .lt. rho_max)
     nt = nt + 1

! ############## predictor ##################
     pred_drhodt = rho * sqrt(24.0d0*pi*rho*ggrav*0.0018d0)
     if(drhodt*dtime.gt.rho*1.0d-4) then
        dtime = 1.0d-4/(drhodt*dtime/rho) * dtime
     endif
     pred_rho = rho + dtime*pred_drhodt
     pred_dedtime = p/rho**2 *  pred_drhodt
     pred_eps = eps + pred_dedtime*dtime

     keytemp = 0
     xrho = pred_rho
     xenr = pred_eps
     xtemp = temp
     xye = y

     call nuc_eos_short(xrho,xtemp,xye,xenr,xprs,xent,xcs2,xzdedt,&
          xdpderho,xdpdrhoe,xmunu,keytemp,keyerr,precision)

     pred_temp = xtemp
     pred_p = xprs
     pred_s = xent

! ############## corrector ##################
     drhodt = pred_rho * sqrt(24.0d0*pi*pred_rho*ggrav*0.0018d0)
     rho = rho + dtime/2.0d0 * &
          (drhodt + pred_drhodt)
!     de = - pred_p*(1.0d0/rho - 1.0d0/pred_rho)/dtime
     dedtime = pred_p/pred_rho**2 * drhodt
     eps = eps + dtime/2.0 * (dedtime + pred_dedtime)

     keytemp = 0
     xrho = rho
     xenr = eps
     xtemp = pred_temp
     xye = y

     call nuc_eos_short(xrho,xtemp,xye,xenr,xprs,xent,xcs2,xzdedt,&
          xdpderho,xdpdrhoe,xmunu,keytemp,keyerr,precision)

     temp = xtemp
     p = xprs
     s = xent
     
     pp = p
     ep = eps
     rp = rho
     tp = temp
     sp = s
     yp = y
     
     time = time + dtime
#if 0
     if(mod(nt,500000).eq.0) then
        write(6,*) ""
        write(6,"(A8,1P10A15)") "nt","time","rho", &
             "temp", "ye","eps","s"
     endif
#endif
     if(mod(nt,1000).eq.0) then
        write(6,"(i9,1P10E15.6)") nt,time,rho,temp,y,eps,s
     endif

     if(mod(nt,1000).eq.0) then
        write(654,"(i9,1P10E15.6)") nt,time,rho,temp,y,eps,s
     endif

  enddo

  close(654)

end program one_zone
