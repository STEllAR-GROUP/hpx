subroutine load_0
  use global_parameters_0
  use particle_array_0
  use field_array_0
  use particle_tracking_0
  use particle_decomp_0
  implicit none

  integer i,m,ierr
  real(8) :: c0=2.515517,c1=0.802853,c2=0.010328,&
              d1=1.432788,d2=0.189269,d3=0.001308
  real(8) :: w_initial,energy,momentum,r,rmi,pi2_inv,delr,ainv,vthe,tsqrt,&
              vthi,cost,b,upara,eperp,cmratio,z4tmp

! initialize random number generator
  call rand_num_gen_init_0

! restart from previous runs
  if(irun /= 0)then
     call restart_read_0
     return
  endif

  !!rmi=1.0/real(mi)
  rmi=1.0/real(mi*npartdom)
  pi2_inv=0.5/pi
  delr=1.0/deltar
  ainv=1.0/a
  w_initial=1.0e-3
  if(nonlinear<0.5)w_initial=1.0e-12
  if(track_particles < -10 .and. track_particles > -20)w_initial=0.1_wp*real(-10-track_particles,wp)
  if(mype==0)write(0,*)'w_initial =',w_initial
  ntracer=0
  !if(mype==0)ntracer=mi
  if(mype==0)ntracer=1
      
! radial: uniformly distributed in r^2, later transform to psi
!$omp parallel do private(m)
  do m=1,mi
     !!zion(1,m)=sqrt(a0*a0+(real(m)-0.5)*(a1*a1-a0*a0)*rmi)
     zion(1,m)=sqrt(a0*a0+(real(m+myrank_partd*mi)-0.5)*(a1*a1-a0*a0)*rmi)
  enddo

! if particle tracking is "on", tag each particle with a unique number
  if(track_particles == 1)call tag_particles_0

! set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
  call set_random_zion_0

! poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=2.0*pi*(zion(2,m)-0.5)
     zion0(2,m)=zion(2,m) !zion0(2,:) for temporary storage
  enddo
  do i=1,10
!$omp parallel do private(m)
     do m=1,mi
        zion(2,m)=zion0(2,m)-2.0*zion(1,m)*sin(zion(2,m))
     enddo
  enddo
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
  enddo

! maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
!$omp parallel do private(m)
  do m=1,mi                
     z4tmp=zion(4,m)
     zion(4,m)=zion(4,m)-0.5
     !zion0(4,m)=sign(1.0d+00,zion(4,m))
     zion0(4,m)=sign(one,zion(4,m))
     !zion(4,m)=sqrt(max(1.0d-20,log(1.0/max(1.0d-20,zion(4,m)**2))))
     zion(4,m)=sqrt(max(small,log(1.0/max(small,zion(4,m)**2))))
     zion(4,m)=zion(4,m)-(c0+c1*zion(4,m)+c2*zion(4,m)**2)/&
          (1.0+d1*zion(4,m)+d2*zion(4,m)**2+d3*zion(4,m)**3)
     if(zion(4,m)>umax)zion(4,m)=z4tmp
  enddo

!$omp parallel do private(m)
  do m=1,mi

! toroidal:  uniformly distributed in zeta
     zion(3,m)=zetamin+(zetamax-zetamin)*zion(3,m)

     zion(4,m)=zion0(4,m)*min(umax,zion(4,m))

! initial random weight
     zion(5,m)=2.0*w_initial*(zion(5,m)-0.5)*(1.0+cos(zion(2,m)))

! maxwellian distribution in v_perp, <v_perp^2>=1.0
     !zion(6,m)=max(1.0d-20,min(umax*umax,-log(max(1.0d-20,zion(6,m)))))
     zion(6,m)=max(small,min(umax*umax,-log(max(small,zion(6,m)))))
  enddo

! transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
  vthi=gyroradius*abs(qion)/aion
!$omp parallel do private(m)
  do m=1,mi
     zion0(1,m)=1.0/(1.0+zion(1,m)*cos(zion(2,m))) !b-field
     zion(1,m)=0.5*zion(1,m)*zion(1,m)
     zion(4,m)=vthi*zion(4,m)*aion/(qion*zion0(1,m))
     zion(6,m)=sqrt(aion*vthi*vthi*zion(6,m)/zion0(1,m))
  enddo

! tracer particle initially resides on pe=0
  if(mype == 0)then
     zion(1,ntracer)=0.5*(0.5*(a0+a1))**2
     zion(2,ntracer)=0.0
     zion(3,ntracer)=0.5*(zetamin+zetamax)
     zion(4,ntracer)=0.5*vthi*aion/qion
     zion(5,ntracer)=0.0
     zion(6,ntracer)=sqrt(aion*vthi*vthi)
!     zion0(1:5,ntracer)=zion(1:5,ntracer)
  endif

  if(iload_0==0)then
! uniform load_0ing
!$omp parallel do private(m)
     do m=1,mi
        zion0(6,m)=1.0
     enddo
  else
! true nonuniform for temperature profile, density profile by weight
!$omp parallel do private(m)
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        i=max(0,min(mpsi,int((r-a0)*delr+0.5)))
        zion(4,m)=zion(4,m)*sqrt(rtemi(i))
        zion(6,m)=zion(6,m)*sqrt(rtemi(i))
!        zion0(6,m)=max(0.1d+00,min(10.0d+00,rden(i)))
        zion0(6,m)=max(pt_one,min(ten,rden(i)))
     enddo
  endif
  
! load_0 electron on top of ion if mi=me
  if(nhybrid>0)then
     vthe=sqrt(aion/(aelectron*tite))*(qion/aion)*(aelectron/qelectron)
     tsqrt=sqrt(1.0/tite)

! keep trapped electrons
     cmratio=qion/aion
     me=0
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        cost=cos(zion(2,m))
        b=1.0/(1.0+r*cost)
        upara=zion(4,m)*b*cmratio
        energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
        eperp=zion(6,m)*zion(6,m)/(1.0-r)
        if(eperp>energy)then
           me=me+1
           zelectron(1,me)=zion(1,m)
           zelectron(2,me)=zion(2,m)
           zelectron(3,me)=zion(3,m)
           zelectron(4,me)=zion(4,m)*vthe
           zelectron(5,me)=zion(5,m)
           zelectron(6,me)=zion(6,m)*tsqrt
           zelectron0(6,me)=zion0(6,m)
        endif
     enddo
     if(mype == 0)then
        ntracer=me
!        zelectron0(1:5,ntracer)=zelectron(1:5,ntracer)
     endif
  endif

!  write(mype+40,'(6e15.6)')zion(1:6,1:mi)
!  do m=1,mi
!     write(mype+40,*)zion(:,m)
!  enddo
!  close(mype+40)

end subroutine load_0


subroutine load_1
  use global_parameters_1
  use particle_array_1
  use field_array_1
  use particle_tracking_1
  use particle_decomp_1
  implicit none

  integer i,m,ierr
  real(8) :: c0=2.515517,c1=0.802853,c2=0.010328,&
              d1=1.432788,d2=0.189269,d3=0.001308
  real(8) :: w_initial,energy,momentum,r,rmi,pi2_inv,delr,ainv,vthe,tsqrt,&
              vthi,cost,b,upara,eperp,cmratio,z4tmp

! initialize random number generator
  call rand_num_gen_init_1

! restart from previous runs
  if(irun /= 0)then
     call restart_read_1
     return
  endif

  !!rmi=1.0/real(mi)
  rmi=1.0/real(mi*npartdom)
  pi2_inv=0.5/pi
  delr=1.0/deltar
  ainv=1.0/a
  w_initial=1.0e-3
  if(nonlinear<0.5)w_initial=1.0e-12
  if(track_particles < -10 .and. track_particles > -20)w_initial=0.1_wp*real(-10-track_particles,wp)
  if(mype==0)write(0,*)'w_initial =',w_initial
  ntracer=0
  !if(mype==0)ntracer=mi
  if(mype==0)ntracer=1
      
! radial: uniformly distributed in r^2, later transform to psi
!$omp parallel do private(m)
  do m=1,mi
     !!zion(1,m)=sqrt(a0*a0+(real(m)-0.5)*(a1*a1-a0*a0)*rmi)
     zion(1,m)=sqrt(a0*a0+(real(m+myrank_partd*mi)-0.5)*(a1*a1-a0*a0)*rmi)
  enddo

! if particle tracking is "on", tag each particle with a unique number
  if(track_particles == 1)call tag_particles_1

! set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
  call set_random_zion_1

! poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=2.0*pi*(zion(2,m)-0.5)
     zion0(2,m)=zion(2,m) !zion0(2,:) for temporary storage
  enddo
  do i=1,10
!$omp parallel do private(m)
     do m=1,mi
        zion(2,m)=zion0(2,m)-2.0*zion(1,m)*sin(zion(2,m))
     enddo
  enddo
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
  enddo

! maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
!$omp parallel do private(m)
  do m=1,mi                
     z4tmp=zion(4,m)
     zion(4,m)=zion(4,m)-0.5
     !zion0(4,m)=sign(1.0d+00,zion(4,m))
     zion0(4,m)=sign(one,zion(4,m))
     !zion(4,m)=sqrt(max(1.0d-20,log(1.0/max(1.0d-20,zion(4,m)**2))))
     zion(4,m)=sqrt(max(small,log(1.0/max(small,zion(4,m)**2))))
     zion(4,m)=zion(4,m)-(c0+c1*zion(4,m)+c2*zion(4,m)**2)/&
          (1.0+d1*zion(4,m)+d2*zion(4,m)**2+d3*zion(4,m)**3)
     if(zion(4,m)>umax)zion(4,m)=z4tmp
  enddo

!$omp parallel do private(m)
  do m=1,mi

! toroidal:  uniformly distributed in zeta
     zion(3,m)=zetamin+(zetamax-zetamin)*zion(3,m)

     zion(4,m)=zion0(4,m)*min(umax,zion(4,m))

! initial random weight
     zion(5,m)=2.0*w_initial*(zion(5,m)-0.5)*(1.0+cos(zion(2,m)))

! maxwellian distribution in v_perp, <v_perp^2>=1.0
     !zion(6,m)=max(1.0d-20,min(umax*umax,-log(max(1.0d-20,zion(6,m)))))
     zion(6,m)=max(small,min(umax*umax,-log(max(small,zion(6,m)))))
  enddo

! transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
  vthi=gyroradius*abs(qion)/aion
!$omp parallel do private(m)
  do m=1,mi
     zion0(1,m)=1.0/(1.0+zion(1,m)*cos(zion(2,m))) !b-field
     zion(1,m)=0.5*zion(1,m)*zion(1,m)
     zion(4,m)=vthi*zion(4,m)*aion/(qion*zion0(1,m))
     zion(6,m)=sqrt(aion*vthi*vthi*zion(6,m)/zion0(1,m))
  enddo

! tracer particle initially resides on pe=0
  if(mype == 0)then
     zion(1,ntracer)=0.5*(0.5*(a0+a1))**2
     zion(2,ntracer)=0.0
     zion(3,ntracer)=0.5*(zetamin+zetamax)
     zion(4,ntracer)=0.5*vthi*aion/qion
     zion(5,ntracer)=0.0
     zion(6,ntracer)=sqrt(aion*vthi*vthi)
!     zion0(1:5,ntracer)=zion(1:5,ntracer)
  endif

  if(iload_1==0)then
! uniform load_1ing
!$omp parallel do private(m)
     do m=1,mi
        zion0(6,m)=1.0
     enddo
  else
! true nonuniform for temperature profile, density profile by weight
!$omp parallel do private(m)
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        i=max(0,min(mpsi,int((r-a0)*delr+0.5)))
        zion(4,m)=zion(4,m)*sqrt(rtemi(i))
        zion(6,m)=zion(6,m)*sqrt(rtemi(i))
!        zion0(6,m)=max(0.1d+00,min(10.0d+00,rden(i)))
        zion0(6,m)=max(pt_one,min(ten,rden(i)))
     enddo
  endif
  
! load_1 electron on top of ion if mi=me
  if(nhybrid>0)then
     vthe=sqrt(aion/(aelectron*tite))*(qion/aion)*(aelectron/qelectron)
     tsqrt=sqrt(1.0/tite)

! keep trapped electrons
     cmratio=qion/aion
     me=0
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        cost=cos(zion(2,m))
        b=1.0/(1.0+r*cost)
        upara=zion(4,m)*b*cmratio
        energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
        eperp=zion(6,m)*zion(6,m)/(1.0-r)
        if(eperp>energy)then
           me=me+1
           zelectron(1,me)=zion(1,m)
           zelectron(2,me)=zion(2,m)
           zelectron(3,me)=zion(3,m)
           zelectron(4,me)=zion(4,m)*vthe
           zelectron(5,me)=zion(5,m)
           zelectron(6,me)=zion(6,m)*tsqrt
           zelectron0(6,me)=zion0(6,m)
        endif
     enddo
     if(mype == 0)then
        ntracer=me
!        zelectron0(1:5,ntracer)=zelectron(1:5,ntracer)
     endif
  endif

!  write(mype+40,'(6e15.6)')zion(1:6,1:mi)
!  do m=1,mi
!     write(mype+40,*)zion(:,m)
!  enddo
!  close(mype+40)

end subroutine load_1


subroutine load_2
  use global_parameters_2
  use particle_array_2
  use field_array_2
  use particle_tracking_2
  use particle_decomp_2
  implicit none

  integer i,m,ierr
  real(8) :: c0=2.515517,c1=0.802853,c2=0.010328,&
              d1=1.432788,d2=0.189269,d3=0.001308
  real(8) :: w_initial,energy,momentum,r,rmi,pi2_inv,delr,ainv,vthe,tsqrt,&
              vthi,cost,b,upara,eperp,cmratio,z4tmp

! initialize random number generator
  call rand_num_gen_init_2

! restart from previous runs
  if(irun /= 0)then
     call restart_read_2
     return
  endif

  !!rmi=1.0/real(mi)
  rmi=1.0/real(mi*npartdom)
  pi2_inv=0.5/pi
  delr=1.0/deltar
  ainv=1.0/a
  w_initial=1.0e-3
  if(nonlinear<0.5)w_initial=1.0e-12
  if(track_particles < -10 .and. track_particles > -20)w_initial=0.1_wp*real(-10-track_particles,wp)
  if(mype==0)write(0,*)'w_initial =',w_initial
  ntracer=0
  !if(mype==0)ntracer=mi
  if(mype==0)ntracer=1
      
! radial: uniformly distributed in r^2, later transform to psi
!$omp parallel do private(m)
  do m=1,mi
     !!zion(1,m)=sqrt(a0*a0+(real(m)-0.5)*(a1*a1-a0*a0)*rmi)
     zion(1,m)=sqrt(a0*a0+(real(m+myrank_partd*mi)-0.5)*(a1*a1-a0*a0)*rmi)
  enddo

! if particle tracking is "on", tag each particle with a unique number
  if(track_particles == 1)call tag_particles_2

! set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
  call set_random_zion_2

! poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=2.0*pi*(zion(2,m)-0.5)
     zion0(2,m)=zion(2,m) !zion0(2,:) for temporary storage
  enddo
  do i=1,10
!$omp parallel do private(m)
     do m=1,mi
        zion(2,m)=zion0(2,m)-2.0*zion(1,m)*sin(zion(2,m))
     enddo
  enddo
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
  enddo

! maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
!$omp parallel do private(m)
  do m=1,mi                
     z4tmp=zion(4,m)
     zion(4,m)=zion(4,m)-0.5
     !zion0(4,m)=sign(1.0d+00,zion(4,m))
     zion0(4,m)=sign(one,zion(4,m))
     !zion(4,m)=sqrt(max(1.0d-20,log(1.0/max(1.0d-20,zion(4,m)**2))))
     zion(4,m)=sqrt(max(small,log(1.0/max(small,zion(4,m)**2))))
     zion(4,m)=zion(4,m)-(c0+c1*zion(4,m)+c2*zion(4,m)**2)/&
          (1.0+d1*zion(4,m)+d2*zion(4,m)**2+d3*zion(4,m)**3)
     if(zion(4,m)>umax)zion(4,m)=z4tmp
  enddo

!$omp parallel do private(m)
  do m=1,mi

! toroidal:  uniformly distributed in zeta
     zion(3,m)=zetamin+(zetamax-zetamin)*zion(3,m)

     zion(4,m)=zion0(4,m)*min(umax,zion(4,m))

! initial random weight
     zion(5,m)=2.0*w_initial*(zion(5,m)-0.5)*(1.0+cos(zion(2,m)))

! maxwellian distribution in v_perp, <v_perp^2>=1.0
     !zion(6,m)=max(1.0d-20,min(umax*umax,-log(max(1.0d-20,zion(6,m)))))
     zion(6,m)=max(small,min(umax*umax,-log(max(small,zion(6,m)))))
  enddo

! transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
  vthi=gyroradius*abs(qion)/aion
!$omp parallel do private(m)
  do m=1,mi
     zion0(1,m)=1.0/(1.0+zion(1,m)*cos(zion(2,m))) !b-field
     zion(1,m)=0.5*zion(1,m)*zion(1,m)
     zion(4,m)=vthi*zion(4,m)*aion/(qion*zion0(1,m))
     zion(6,m)=sqrt(aion*vthi*vthi*zion(6,m)/zion0(1,m))
  enddo

! tracer particle initially resides on pe=0
  if(mype == 0)then
     zion(1,ntracer)=0.5*(0.5*(a0+a1))**2
     zion(2,ntracer)=0.0
     zion(3,ntracer)=0.5*(zetamin+zetamax)
     zion(4,ntracer)=0.5*vthi*aion/qion
     zion(5,ntracer)=0.0
     zion(6,ntracer)=sqrt(aion*vthi*vthi)
!     zion0(1:5,ntracer)=zion(1:5,ntracer)
  endif

  if(iload_2==0)then
! uniform load_2ing
!$omp parallel do private(m)
     do m=1,mi
        zion0(6,m)=1.0
     enddo
  else
! true nonuniform for temperature profile, density profile by weight
!$omp parallel do private(m)
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        i=max(0,min(mpsi,int((r-a0)*delr+0.5)))
        zion(4,m)=zion(4,m)*sqrt(rtemi(i))
        zion(6,m)=zion(6,m)*sqrt(rtemi(i))
!        zion0(6,m)=max(0.1d+00,min(10.0d+00,rden(i)))
        zion0(6,m)=max(pt_one,min(ten,rden(i)))
     enddo
  endif
  
! load_2 electron on top of ion if mi=me
  if(nhybrid>0)then
     vthe=sqrt(aion/(aelectron*tite))*(qion/aion)*(aelectron/qelectron)
     tsqrt=sqrt(1.0/tite)

! keep trapped electrons
     cmratio=qion/aion
     me=0
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        cost=cos(zion(2,m))
        b=1.0/(1.0+r*cost)
        upara=zion(4,m)*b*cmratio
        energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
        eperp=zion(6,m)*zion(6,m)/(1.0-r)
        if(eperp>energy)then
           me=me+1
           zelectron(1,me)=zion(1,m)
           zelectron(2,me)=zion(2,m)
           zelectron(3,me)=zion(3,m)
           zelectron(4,me)=zion(4,m)*vthe
           zelectron(5,me)=zion(5,m)
           zelectron(6,me)=zion(6,m)*tsqrt
           zelectron0(6,me)=zion0(6,m)
        endif
     enddo
     if(mype == 0)then
        ntracer=me
!        zelectron0(1:5,ntracer)=zelectron(1:5,ntracer)
     endif
  endif

!  write(mype+40,'(6e15.6)')zion(1:6,1:mi)
!  do m=1,mi
!     write(mype+40,*)zion(:,m)
!  enddo
!  close(mype+40)

end subroutine load_2


subroutine load_3
  use global_parameters_3
  use particle_array_3
  use field_array_3
  use particle_tracking_3
  use particle_decomp_3
  implicit none

  integer i,m,ierr
  real(8) :: c0=2.515517,c1=0.802853,c2=0.010328,&
              d1=1.432788,d2=0.189269,d3=0.001308
  real(8) :: w_initial,energy,momentum,r,rmi,pi2_inv,delr,ainv,vthe,tsqrt,&
              vthi,cost,b,upara,eperp,cmratio,z4tmp

! initialize random number generator
  call rand_num_gen_init_3

! restart from previous runs
  if(irun /= 0)then
     call restart_read_3
     return
  endif

  !!rmi=1.0/real(mi)
  rmi=1.0/real(mi*npartdom)
  pi2_inv=0.5/pi
  delr=1.0/deltar
  ainv=1.0/a
  w_initial=1.0e-3
  if(nonlinear<0.5)w_initial=1.0e-12
  if(track_particles < -10 .and. track_particles > -20)w_initial=0.1_wp*real(-10-track_particles,wp)
  if(mype==0)write(0,*)'w_initial =',w_initial
  ntracer=0
  !if(mype==0)ntracer=mi
  if(mype==0)ntracer=1
      
! radial: uniformly distributed in r^2, later transform to psi
!$omp parallel do private(m)
  do m=1,mi
     !!zion(1,m)=sqrt(a0*a0+(real(m)-0.5)*(a1*a1-a0*a0)*rmi)
     zion(1,m)=sqrt(a0*a0+(real(m+myrank_partd*mi)-0.5)*(a1*a1-a0*a0)*rmi)
  enddo

! if particle tracking is "on", tag each particle with a unique number
  if(track_particles == 1)call tag_particles_3

! set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
  call set_random_zion_3

! poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=2.0*pi*(zion(2,m)-0.5)
     zion0(2,m)=zion(2,m) !zion0(2,:) for temporary storage
  enddo
  do i=1,10
!$omp parallel do private(m)
     do m=1,mi
        zion(2,m)=zion0(2,m)-2.0*zion(1,m)*sin(zion(2,m))
     enddo
  enddo
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
  enddo

! maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
!$omp parallel do private(m)
  do m=1,mi                
     z4tmp=zion(4,m)
     zion(4,m)=zion(4,m)-0.5
     !zion0(4,m)=sign(1.0d+00,zion(4,m))
     zion0(4,m)=sign(one,zion(4,m))
     !zion(4,m)=sqrt(max(1.0d-20,log(1.0/max(1.0d-20,zion(4,m)**2))))
     zion(4,m)=sqrt(max(small,log(1.0/max(small,zion(4,m)**2))))
     zion(4,m)=zion(4,m)-(c0+c1*zion(4,m)+c2*zion(4,m)**2)/&
          (1.0+d1*zion(4,m)+d2*zion(4,m)**2+d3*zion(4,m)**3)
     if(zion(4,m)>umax)zion(4,m)=z4tmp
  enddo

!$omp parallel do private(m)
  do m=1,mi

! toroidal:  uniformly distributed in zeta
     zion(3,m)=zetamin+(zetamax-zetamin)*zion(3,m)

     zion(4,m)=zion0(4,m)*min(umax,zion(4,m))

! initial random weight
     zion(5,m)=2.0*w_initial*(zion(5,m)-0.5)*(1.0+cos(zion(2,m)))

! maxwellian distribution in v_perp, <v_perp^2>=1.0
     !zion(6,m)=max(1.0d-20,min(umax*umax,-log(max(1.0d-20,zion(6,m)))))
     zion(6,m)=max(small,min(umax*umax,-log(max(small,zion(6,m)))))
  enddo

! transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
  vthi=gyroradius*abs(qion)/aion
!$omp parallel do private(m)
  do m=1,mi
     zion0(1,m)=1.0/(1.0+zion(1,m)*cos(zion(2,m))) !b-field
     zion(1,m)=0.5*zion(1,m)*zion(1,m)
     zion(4,m)=vthi*zion(4,m)*aion/(qion*zion0(1,m))
     zion(6,m)=sqrt(aion*vthi*vthi*zion(6,m)/zion0(1,m))
  enddo

! tracer particle initially resides on pe=0
  if(mype == 0)then
     zion(1,ntracer)=0.5*(0.5*(a0+a1))**2
     zion(2,ntracer)=0.0
     zion(3,ntracer)=0.5*(zetamin+zetamax)
     zion(4,ntracer)=0.5*vthi*aion/qion
     zion(5,ntracer)=0.0
     zion(6,ntracer)=sqrt(aion*vthi*vthi)
!     zion0(1:5,ntracer)=zion(1:5,ntracer)
  endif

  if(iload_3==0)then
! uniform load_3ing
!$omp parallel do private(m)
     do m=1,mi
        zion0(6,m)=1.0
     enddo
  else
! true nonuniform for temperature profile, density profile by weight
!$omp parallel do private(m)
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        i=max(0,min(mpsi,int((r-a0)*delr+0.5)))
        zion(4,m)=zion(4,m)*sqrt(rtemi(i))
        zion(6,m)=zion(6,m)*sqrt(rtemi(i))
!        zion0(6,m)=max(0.1d+00,min(10.0d+00,rden(i)))
        zion0(6,m)=max(pt_one,min(ten,rden(i)))
     enddo
  endif
  
! load_3 electron on top of ion if mi=me
  if(nhybrid>0)then
     vthe=sqrt(aion/(aelectron*tite))*(qion/aion)*(aelectron/qelectron)
     tsqrt=sqrt(1.0/tite)

! keep trapped electrons
     cmratio=qion/aion
     me=0
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        cost=cos(zion(2,m))
        b=1.0/(1.0+r*cost)
        upara=zion(4,m)*b*cmratio
        energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
        eperp=zion(6,m)*zion(6,m)/(1.0-r)
        if(eperp>energy)then
           me=me+1
           zelectron(1,me)=zion(1,m)
           zelectron(2,me)=zion(2,m)
           zelectron(3,me)=zion(3,m)
           zelectron(4,me)=zion(4,m)*vthe
           zelectron(5,me)=zion(5,m)
           zelectron(6,me)=zion(6,m)*tsqrt
           zelectron0(6,me)=zion0(6,m)
        endif
     enddo
     if(mype == 0)then
        ntracer=me
!        zelectron0(1:5,ntracer)=zelectron(1:5,ntracer)
     endif
  endif

!  write(mype+40,'(6e15.6)')zion(1:6,1:mi)
!  do m=1,mi
!     write(mype+40,*)zion(:,m)
!  enddo
!  close(mype+40)

end subroutine load_3


subroutine load_4
  use global_parameters_4
  use particle_array_4
  use field_array_4
  use particle_tracking_4
  use particle_decomp_4
  implicit none

  integer i,m,ierr
  real(8) :: c0=2.515517,c1=0.802853,c2=0.010328,&
              d1=1.432788,d2=0.189269,d3=0.001308
  real(8) :: w_initial,energy,momentum,r,rmi,pi2_inv,delr,ainv,vthe,tsqrt,&
              vthi,cost,b,upara,eperp,cmratio,z4tmp

! initialize random number generator
  call rand_num_gen_init_4

! restart from previous runs
  if(irun /= 0)then
     call restart_read_4
     return
  endif

  !!rmi=1.0/real(mi)
  rmi=1.0/real(mi*npartdom)
  pi2_inv=0.5/pi
  delr=1.0/deltar
  ainv=1.0/a
  w_initial=1.0e-3
  if(nonlinear<0.5)w_initial=1.0e-12
  if(track_particles < -10 .and. track_particles > -20)w_initial=0.1_wp*real(-10-track_particles,wp)
  if(mype==0)write(0,*)'w_initial =',w_initial
  ntracer=0
  !if(mype==0)ntracer=mi
  if(mype==0)ntracer=1
      
! radial: uniformly distributed in r^2, later transform to psi
!$omp parallel do private(m)
  do m=1,mi
     !!zion(1,m)=sqrt(a0*a0+(real(m)-0.5)*(a1*a1-a0*a0)*rmi)
     zion(1,m)=sqrt(a0*a0+(real(m+myrank_partd*mi)-0.5)*(a1*a1-a0*a0)*rmi)
  enddo

! if particle tracking is "on", tag each particle with a unique number
  if(track_particles == 1)call tag_particles_4

! set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
  call set_random_zion_4

! poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=2.0*pi*(zion(2,m)-0.5)
     zion0(2,m)=zion(2,m) !zion0(2,:) for temporary storage
  enddo
  do i=1,10
!$omp parallel do private(m)
     do m=1,mi
        zion(2,m)=zion0(2,m)-2.0*zion(1,m)*sin(zion(2,m))
     enddo
  enddo
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
  enddo

! maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
!$omp parallel do private(m)
  do m=1,mi                
     z4tmp=zion(4,m)
     zion(4,m)=zion(4,m)-0.5
     !zion0(4,m)=sign(1.0d+00,zion(4,m))
     zion0(4,m)=sign(one,zion(4,m))
     !zion(4,m)=sqrt(max(1.0d-20,log(1.0/max(1.0d-20,zion(4,m)**2))))
     zion(4,m)=sqrt(max(small,log(1.0/max(small,zion(4,m)**2))))
     zion(4,m)=zion(4,m)-(c0+c1*zion(4,m)+c2*zion(4,m)**2)/&
          (1.0+d1*zion(4,m)+d2*zion(4,m)**2+d3*zion(4,m)**3)
     if(zion(4,m)>umax)zion(4,m)=z4tmp
  enddo

!$omp parallel do private(m)
  do m=1,mi

! toroidal:  uniformly distributed in zeta
     zion(3,m)=zetamin+(zetamax-zetamin)*zion(3,m)

     zion(4,m)=zion0(4,m)*min(umax,zion(4,m))

! initial random weight
     zion(5,m)=2.0*w_initial*(zion(5,m)-0.5)*(1.0+cos(zion(2,m)))

! maxwellian distribution in v_perp, <v_perp^2>=1.0
     !zion(6,m)=max(1.0d-20,min(umax*umax,-log(max(1.0d-20,zion(6,m)))))
     zion(6,m)=max(small,min(umax*umax,-log(max(small,zion(6,m)))))
  enddo

! transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
  vthi=gyroradius*abs(qion)/aion
!$omp parallel do private(m)
  do m=1,mi
     zion0(1,m)=1.0/(1.0+zion(1,m)*cos(zion(2,m))) !b-field
     zion(1,m)=0.5*zion(1,m)*zion(1,m)
     zion(4,m)=vthi*zion(4,m)*aion/(qion*zion0(1,m))
     zion(6,m)=sqrt(aion*vthi*vthi*zion(6,m)/zion0(1,m))
  enddo

! tracer particle initially resides on pe=0
  if(mype == 0)then
     zion(1,ntracer)=0.5*(0.5*(a0+a1))**2
     zion(2,ntracer)=0.0
     zion(3,ntracer)=0.5*(zetamin+zetamax)
     zion(4,ntracer)=0.5*vthi*aion/qion
     zion(5,ntracer)=0.0
     zion(6,ntracer)=sqrt(aion*vthi*vthi)
!     zion0(1:5,ntracer)=zion(1:5,ntracer)
  endif

  if(iload_4==0)then
! uniform load_4ing
!$omp parallel do private(m)
     do m=1,mi
        zion0(6,m)=1.0
     enddo
  else
! true nonuniform for temperature profile, density profile by weight
!$omp parallel do private(m)
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        i=max(0,min(mpsi,int((r-a0)*delr+0.5)))
        zion(4,m)=zion(4,m)*sqrt(rtemi(i))
        zion(6,m)=zion(6,m)*sqrt(rtemi(i))
!        zion0(6,m)=max(0.1d+00,min(10.0d+00,rden(i)))
        zion0(6,m)=max(pt_one,min(ten,rden(i)))
     enddo
  endif
  
! load_4 electron on top of ion if mi=me
  if(nhybrid>0)then
     vthe=sqrt(aion/(aelectron*tite))*(qion/aion)*(aelectron/qelectron)
     tsqrt=sqrt(1.0/tite)

! keep trapped electrons
     cmratio=qion/aion
     me=0
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        cost=cos(zion(2,m))
        b=1.0/(1.0+r*cost)
        upara=zion(4,m)*b*cmratio
        energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
        eperp=zion(6,m)*zion(6,m)/(1.0-r)
        if(eperp>energy)then
           me=me+1
           zelectron(1,me)=zion(1,m)
           zelectron(2,me)=zion(2,m)
           zelectron(3,me)=zion(3,m)
           zelectron(4,me)=zion(4,m)*vthe
           zelectron(5,me)=zion(5,m)
           zelectron(6,me)=zion(6,m)*tsqrt
           zelectron0(6,me)=zion0(6,m)
        endif
     enddo
     if(mype == 0)then
        ntracer=me
!        zelectron0(1:5,ntracer)=zelectron(1:5,ntracer)
     endif
  endif

!  write(mype+40,'(6e15.6)')zion(1:6,1:mi)
!  do m=1,mi
!     write(mype+40,*)zion(:,m)
!  enddo
!  close(mype+40)

end subroutine load_4


subroutine load_5
  use global_parameters_5
  use particle_array_5
  use field_array_5
  use particle_tracking_5
  use particle_decomp_5
  implicit none

  integer i,m,ierr
  real(8) :: c0=2.515517,c1=0.802853,c2=0.010328,&
              d1=1.432788,d2=0.189269,d3=0.001308
  real(8) :: w_initial,energy,momentum,r,rmi,pi2_inv,delr,ainv,vthe,tsqrt,&
              vthi,cost,b,upara,eperp,cmratio,z4tmp

! initialize random number generator
  call rand_num_gen_init_5

! restart from previous runs
  if(irun /= 0)then
     call restart_read_5
     return
  endif

  !!rmi=1.0/real(mi)
  rmi=1.0/real(mi*npartdom)
  pi2_inv=0.5/pi
  delr=1.0/deltar
  ainv=1.0/a
  w_initial=1.0e-3
  if(nonlinear<0.5)w_initial=1.0e-12
  if(track_particles < -10 .and. track_particles > -20)w_initial=0.1_wp*real(-10-track_particles,wp)
  if(mype==0)write(0,*)'w_initial =',w_initial
  ntracer=0
  !if(mype==0)ntracer=mi
  if(mype==0)ntracer=1
      
! radial: uniformly distributed in r^2, later transform to psi
!$omp parallel do private(m)
  do m=1,mi
     !!zion(1,m)=sqrt(a0*a0+(real(m)-0.5)*(a1*a1-a0*a0)*rmi)
     zion(1,m)=sqrt(a0*a0+(real(m+myrank_partd*mi)-0.5)*(a1*a1-a0*a0)*rmi)
  enddo

! if particle tracking is "on", tag each particle with a unique number
  if(track_particles == 1)call tag_particles_5

! set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
  call set_random_zion_5

! poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=2.0*pi*(zion(2,m)-0.5)
     zion0(2,m)=zion(2,m) !zion0(2,:) for temporary storage
  enddo
  do i=1,10
!$omp parallel do private(m)
     do m=1,mi
        zion(2,m)=zion0(2,m)-2.0*zion(1,m)*sin(zion(2,m))
     enddo
  enddo
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
  enddo

! maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
!$omp parallel do private(m)
  do m=1,mi                
     z4tmp=zion(4,m)
     zion(4,m)=zion(4,m)-0.5
     !zion0(4,m)=sign(1.0d+00,zion(4,m))
     zion0(4,m)=sign(one,zion(4,m))
     !zion(4,m)=sqrt(max(1.0d-20,log(1.0/max(1.0d-20,zion(4,m)**2))))
     zion(4,m)=sqrt(max(small,log(1.0/max(small,zion(4,m)**2))))
     zion(4,m)=zion(4,m)-(c0+c1*zion(4,m)+c2*zion(4,m)**2)/&
          (1.0+d1*zion(4,m)+d2*zion(4,m)**2+d3*zion(4,m)**3)
     if(zion(4,m)>umax)zion(4,m)=z4tmp
  enddo

!$omp parallel do private(m)
  do m=1,mi

! toroidal:  uniformly distributed in zeta
     zion(3,m)=zetamin+(zetamax-zetamin)*zion(3,m)

     zion(4,m)=zion0(4,m)*min(umax,zion(4,m))

! initial random weight
     zion(5,m)=2.0*w_initial*(zion(5,m)-0.5)*(1.0+cos(zion(2,m)))

! maxwellian distribution in v_perp, <v_perp^2>=1.0
     !zion(6,m)=max(1.0d-20,min(umax*umax,-log(max(1.0d-20,zion(6,m)))))
     zion(6,m)=max(small,min(umax*umax,-log(max(small,zion(6,m)))))
  enddo

! transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
  vthi=gyroradius*abs(qion)/aion
!$omp parallel do private(m)
  do m=1,mi
     zion0(1,m)=1.0/(1.0+zion(1,m)*cos(zion(2,m))) !b-field
     zion(1,m)=0.5*zion(1,m)*zion(1,m)
     zion(4,m)=vthi*zion(4,m)*aion/(qion*zion0(1,m))
     zion(6,m)=sqrt(aion*vthi*vthi*zion(6,m)/zion0(1,m))
  enddo

! tracer particle initially resides on pe=0
  if(mype == 0)then
     zion(1,ntracer)=0.5*(0.5*(a0+a1))**2
     zion(2,ntracer)=0.0
     zion(3,ntracer)=0.5*(zetamin+zetamax)
     zion(4,ntracer)=0.5*vthi*aion/qion
     zion(5,ntracer)=0.0
     zion(6,ntracer)=sqrt(aion*vthi*vthi)
!     zion0(1:5,ntracer)=zion(1:5,ntracer)
  endif

  if(iload_5==0)then
! uniform load_5ing
!$omp parallel do private(m)
     do m=1,mi
        zion0(6,m)=1.0
     enddo
  else
! true nonuniform for temperature profile, density profile by weight
!$omp parallel do private(m)
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        i=max(0,min(mpsi,int((r-a0)*delr+0.5)))
        zion(4,m)=zion(4,m)*sqrt(rtemi(i))
        zion(6,m)=zion(6,m)*sqrt(rtemi(i))
!        zion0(6,m)=max(0.1d+00,min(10.0d+00,rden(i)))
        zion0(6,m)=max(pt_one,min(ten,rden(i)))
     enddo
  endif
  
! load_5 electron on top of ion if mi=me
  if(nhybrid>0)then
     vthe=sqrt(aion/(aelectron*tite))*(qion/aion)*(aelectron/qelectron)
     tsqrt=sqrt(1.0/tite)

! keep trapped electrons
     cmratio=qion/aion
     me=0
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        cost=cos(zion(2,m))
        b=1.0/(1.0+r*cost)
        upara=zion(4,m)*b*cmratio
        energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
        eperp=zion(6,m)*zion(6,m)/(1.0-r)
        if(eperp>energy)then
           me=me+1
           zelectron(1,me)=zion(1,m)
           zelectron(2,me)=zion(2,m)
           zelectron(3,me)=zion(3,m)
           zelectron(4,me)=zion(4,m)*vthe
           zelectron(5,me)=zion(5,m)
           zelectron(6,me)=zion(6,m)*tsqrt
           zelectron0(6,me)=zion0(6,m)
        endif
     enddo
     if(mype == 0)then
        ntracer=me
!        zelectron0(1:5,ntracer)=zelectron(1:5,ntracer)
     endif
  endif

!  write(mype+40,'(6e15.6)')zion(1:6,1:mi)
!  do m=1,mi
!     write(mype+40,*)zion(:,m)
!  enddo
!  close(mype+40)

end subroutine load_5


subroutine load_6
  use global_parameters_6
  use particle_array_6
  use field_array_6
  use particle_tracking_6
  use particle_decomp_6
  implicit none

  integer i,m,ierr
  real(8) :: c0=2.515517,c1=0.802853,c2=0.010328,&
              d1=1.432788,d2=0.189269,d3=0.001308
  real(8) :: w_initial,energy,momentum,r,rmi,pi2_inv,delr,ainv,vthe,tsqrt,&
              vthi,cost,b,upara,eperp,cmratio,z4tmp

! initialize random number generator
  call rand_num_gen_init_6

! restart from previous runs
  if(irun /= 0)then
     call restart_read_6
     return
  endif

  !!rmi=1.0/real(mi)
  rmi=1.0/real(mi*npartdom)
  pi2_inv=0.5/pi
  delr=1.0/deltar
  ainv=1.0/a
  w_initial=1.0e-3
  if(nonlinear<0.5)w_initial=1.0e-12
  if(track_particles < -10 .and. track_particles > -20)w_initial=0.1_wp*real(-10-track_particles,wp)
  if(mype==0)write(0,*)'w_initial =',w_initial
  ntracer=0
  !if(mype==0)ntracer=mi
  if(mype==0)ntracer=1
      
! radial: uniformly distributed in r^2, later transform to psi
!$omp parallel do private(m)
  do m=1,mi
     !!zion(1,m)=sqrt(a0*a0+(real(m)-0.5)*(a1*a1-a0*a0)*rmi)
     zion(1,m)=sqrt(a0*a0+(real(m+myrank_partd*mi)-0.5)*(a1*a1-a0*a0)*rmi)
  enddo

! if particle tracking is "on", tag each particle with a unique number
  if(track_particles == 1)call tag_particles_6

! set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
  call set_random_zion_6

! poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=2.0*pi*(zion(2,m)-0.5)
     zion0(2,m)=zion(2,m) !zion0(2,:) for temporary storage
  enddo
  do i=1,10
!$omp parallel do private(m)
     do m=1,mi
        zion(2,m)=zion0(2,m)-2.0*zion(1,m)*sin(zion(2,m))
     enddo
  enddo
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
  enddo

! maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
!$omp parallel do private(m)
  do m=1,mi                
     z4tmp=zion(4,m)
     zion(4,m)=zion(4,m)-0.5
     !zion0(4,m)=sign(1.0d+00,zion(4,m))
     zion0(4,m)=sign(one,zion(4,m))
     !zion(4,m)=sqrt(max(1.0d-20,log(1.0/max(1.0d-20,zion(4,m)**2))))
     zion(4,m)=sqrt(max(small,log(1.0/max(small,zion(4,m)**2))))
     zion(4,m)=zion(4,m)-(c0+c1*zion(4,m)+c2*zion(4,m)**2)/&
          (1.0+d1*zion(4,m)+d2*zion(4,m)**2+d3*zion(4,m)**3)
     if(zion(4,m)>umax)zion(4,m)=z4tmp
  enddo

!$omp parallel do private(m)
  do m=1,mi

! toroidal:  uniformly distributed in zeta
     zion(3,m)=zetamin+(zetamax-zetamin)*zion(3,m)

     zion(4,m)=zion0(4,m)*min(umax,zion(4,m))

! initial random weight
     zion(5,m)=2.0*w_initial*(zion(5,m)-0.5)*(1.0+cos(zion(2,m)))

! maxwellian distribution in v_perp, <v_perp^2>=1.0
     !zion(6,m)=max(1.0d-20,min(umax*umax,-log(max(1.0d-20,zion(6,m)))))
     zion(6,m)=max(small,min(umax*umax,-log(max(small,zion(6,m)))))
  enddo

! transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
  vthi=gyroradius*abs(qion)/aion
!$omp parallel do private(m)
  do m=1,mi
     zion0(1,m)=1.0/(1.0+zion(1,m)*cos(zion(2,m))) !b-field
     zion(1,m)=0.5*zion(1,m)*zion(1,m)
     zion(4,m)=vthi*zion(4,m)*aion/(qion*zion0(1,m))
     zion(6,m)=sqrt(aion*vthi*vthi*zion(6,m)/zion0(1,m))
  enddo

! tracer particle initially resides on pe=0
  if(mype == 0)then
     zion(1,ntracer)=0.5*(0.5*(a0+a1))**2
     zion(2,ntracer)=0.0
     zion(3,ntracer)=0.5*(zetamin+zetamax)
     zion(4,ntracer)=0.5*vthi*aion/qion
     zion(5,ntracer)=0.0
     zion(6,ntracer)=sqrt(aion*vthi*vthi)
!     zion0(1:5,ntracer)=zion(1:5,ntracer)
  endif

  if(iload_6==0)then
! uniform load_6ing
!$omp parallel do private(m)
     do m=1,mi
        zion0(6,m)=1.0
     enddo
  else
! true nonuniform for temperature profile, density profile by weight
!$omp parallel do private(m)
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        i=max(0,min(mpsi,int((r-a0)*delr+0.5)))
        zion(4,m)=zion(4,m)*sqrt(rtemi(i))
        zion(6,m)=zion(6,m)*sqrt(rtemi(i))
!        zion0(6,m)=max(0.1d+00,min(10.0d+00,rden(i)))
        zion0(6,m)=max(pt_one,min(ten,rden(i)))
     enddo
  endif
  
! load_6 electron on top of ion if mi=me
  if(nhybrid>0)then
     vthe=sqrt(aion/(aelectron*tite))*(qion/aion)*(aelectron/qelectron)
     tsqrt=sqrt(1.0/tite)

! keep trapped electrons
     cmratio=qion/aion
     me=0
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        cost=cos(zion(2,m))
        b=1.0/(1.0+r*cost)
        upara=zion(4,m)*b*cmratio
        energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
        eperp=zion(6,m)*zion(6,m)/(1.0-r)
        if(eperp>energy)then
           me=me+1
           zelectron(1,me)=zion(1,m)
           zelectron(2,me)=zion(2,m)
           zelectron(3,me)=zion(3,m)
           zelectron(4,me)=zion(4,m)*vthe
           zelectron(5,me)=zion(5,m)
           zelectron(6,me)=zion(6,m)*tsqrt
           zelectron0(6,me)=zion0(6,m)
        endif
     enddo
     if(mype == 0)then
        ntracer=me
!        zelectron0(1:5,ntracer)=zelectron(1:5,ntracer)
     endif
  endif

!  write(mype+40,'(6e15.6)')zion(1:6,1:mi)
!  do m=1,mi
!     write(mype+40,*)zion(:,m)
!  enddo
!  close(mype+40)

end subroutine load_6


subroutine load_7
  use global_parameters_7
  use particle_array_7
  use field_array_7
  use particle_tracking_7
  use particle_decomp_7
  implicit none

  integer i,m,ierr
  real(8) :: c0=2.515517,c1=0.802853,c2=0.010328,&
              d1=1.432788,d2=0.189269,d3=0.001308
  real(8) :: w_initial,energy,momentum,r,rmi,pi2_inv,delr,ainv,vthe,tsqrt,&
              vthi,cost,b,upara,eperp,cmratio,z4tmp

! initialize random number generator
  call rand_num_gen_init_7

! restart from previous runs
  if(irun /= 0)then
     call restart_read_7
     return
  endif

  !!rmi=1.0/real(mi)
  rmi=1.0/real(mi*npartdom)
  pi2_inv=0.5/pi
  delr=1.0/deltar
  ainv=1.0/a
  w_initial=1.0e-3
  if(nonlinear<0.5)w_initial=1.0e-12
  if(track_particles < -10 .and. track_particles > -20)w_initial=0.1_wp*real(-10-track_particles,wp)
  if(mype==0)write(0,*)'w_initial =',w_initial
  ntracer=0
  !if(mype==0)ntracer=mi
  if(mype==0)ntracer=1
      
! radial: uniformly distributed in r^2, later transform to psi
!$omp parallel do private(m)
  do m=1,mi
     !!zion(1,m)=sqrt(a0*a0+(real(m)-0.5)*(a1*a1-a0*a0)*rmi)
     zion(1,m)=sqrt(a0*a0+(real(m+myrank_partd*mi)-0.5)*(a1*a1-a0*a0)*rmi)
  enddo

! if particle tracking is "on", tag each particle with a unique number
  if(track_particles == 1)call tag_particles_7

! set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
  call set_random_zion_7

! poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=2.0*pi*(zion(2,m)-0.5)
     zion0(2,m)=zion(2,m) !zion0(2,:) for temporary storage
  enddo
  do i=1,10
!$omp parallel do private(m)
     do m=1,mi
        zion(2,m)=zion0(2,m)-2.0*zion(1,m)*sin(zion(2,m))
     enddo
  enddo
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
  enddo

! maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
!$omp parallel do private(m)
  do m=1,mi                
     z4tmp=zion(4,m)
     zion(4,m)=zion(4,m)-0.5
     !zion0(4,m)=sign(1.0d+00,zion(4,m))
     zion0(4,m)=sign(one,zion(4,m))
     !zion(4,m)=sqrt(max(1.0d-20,log(1.0/max(1.0d-20,zion(4,m)**2))))
     zion(4,m)=sqrt(max(small,log(1.0/max(small,zion(4,m)**2))))
     zion(4,m)=zion(4,m)-(c0+c1*zion(4,m)+c2*zion(4,m)**2)/&
          (1.0+d1*zion(4,m)+d2*zion(4,m)**2+d3*zion(4,m)**3)
     if(zion(4,m)>umax)zion(4,m)=z4tmp
  enddo

!$omp parallel do private(m)
  do m=1,mi

! toroidal:  uniformly distributed in zeta
     zion(3,m)=zetamin+(zetamax-zetamin)*zion(3,m)

     zion(4,m)=zion0(4,m)*min(umax,zion(4,m))

! initial random weight
     zion(5,m)=2.0*w_initial*(zion(5,m)-0.5)*(1.0+cos(zion(2,m)))

! maxwellian distribution in v_perp, <v_perp^2>=1.0
     !zion(6,m)=max(1.0d-20,min(umax*umax,-log(max(1.0d-20,zion(6,m)))))
     zion(6,m)=max(small,min(umax*umax,-log(max(small,zion(6,m)))))
  enddo

! transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
  vthi=gyroradius*abs(qion)/aion
!$omp parallel do private(m)
  do m=1,mi
     zion0(1,m)=1.0/(1.0+zion(1,m)*cos(zion(2,m))) !b-field
     zion(1,m)=0.5*zion(1,m)*zion(1,m)
     zion(4,m)=vthi*zion(4,m)*aion/(qion*zion0(1,m))
     zion(6,m)=sqrt(aion*vthi*vthi*zion(6,m)/zion0(1,m))
  enddo

! tracer particle initially resides on pe=0
  if(mype == 0)then
     zion(1,ntracer)=0.5*(0.5*(a0+a1))**2
     zion(2,ntracer)=0.0
     zion(3,ntracer)=0.5*(zetamin+zetamax)
     zion(4,ntracer)=0.5*vthi*aion/qion
     zion(5,ntracer)=0.0
     zion(6,ntracer)=sqrt(aion*vthi*vthi)
!     zion0(1:5,ntracer)=zion(1:5,ntracer)
  endif

  if(iload_7==0)then
! uniform load_7ing
!$omp parallel do private(m)
     do m=1,mi
        zion0(6,m)=1.0
     enddo
  else
! true nonuniform for temperature profile, density profile by weight
!$omp parallel do private(m)
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        i=max(0,min(mpsi,int((r-a0)*delr+0.5)))
        zion(4,m)=zion(4,m)*sqrt(rtemi(i))
        zion(6,m)=zion(6,m)*sqrt(rtemi(i))
!        zion0(6,m)=max(0.1d+00,min(10.0d+00,rden(i)))
        zion0(6,m)=max(pt_one,min(ten,rden(i)))
     enddo
  endif
  
! load_7 electron on top of ion if mi=me
  if(nhybrid>0)then
     vthe=sqrt(aion/(aelectron*tite))*(qion/aion)*(aelectron/qelectron)
     tsqrt=sqrt(1.0/tite)

! keep trapped electrons
     cmratio=qion/aion
     me=0
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        cost=cos(zion(2,m))
        b=1.0/(1.0+r*cost)
        upara=zion(4,m)*b*cmratio
        energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
        eperp=zion(6,m)*zion(6,m)/(1.0-r)
        if(eperp>energy)then
           me=me+1
           zelectron(1,me)=zion(1,m)
           zelectron(2,me)=zion(2,m)
           zelectron(3,me)=zion(3,m)
           zelectron(4,me)=zion(4,m)*vthe
           zelectron(5,me)=zion(5,m)
           zelectron(6,me)=zion(6,m)*tsqrt
           zelectron0(6,me)=zion0(6,m)
        endif
     enddo
     if(mype == 0)then
        ntracer=me
!        zelectron0(1:5,ntracer)=zelectron(1:5,ntracer)
     endif
  endif

!  write(mype+40,'(6e15.6)')zion(1:6,1:mi)
!  do m=1,mi
!     write(mype+40,*)zion(:,m)
!  enddo
!  close(mype+40)

end subroutine load_7


subroutine load_8
  use global_parameters_8
  use particle_array_8
  use field_array_8
  use particle_tracking_8
  use particle_decomp_8
  implicit none

  integer i,m,ierr
  real(8) :: c0=2.515517,c1=0.802853,c2=0.010328,&
              d1=1.432788,d2=0.189269,d3=0.001308
  real(8) :: w_initial,energy,momentum,r,rmi,pi2_inv,delr,ainv,vthe,tsqrt,&
              vthi,cost,b,upara,eperp,cmratio,z4tmp

! initialize random number generator
  call rand_num_gen_init_8

! restart from previous runs
  if(irun /= 0)then
     call restart_read_8
     return
  endif

  !!rmi=1.0/real(mi)
  rmi=1.0/real(mi*npartdom)
  pi2_inv=0.5/pi
  delr=1.0/deltar
  ainv=1.0/a
  w_initial=1.0e-3
  if(nonlinear<0.5)w_initial=1.0e-12
  if(track_particles < -10 .and. track_particles > -20)w_initial=0.1_wp*real(-10-track_particles,wp)
  if(mype==0)write(0,*)'w_initial =',w_initial
  ntracer=0
  !if(mype==0)ntracer=mi
  if(mype==0)ntracer=1
      
! radial: uniformly distributed in r^2, later transform to psi
!$omp parallel do private(m)
  do m=1,mi
     !!zion(1,m)=sqrt(a0*a0+(real(m)-0.5)*(a1*a1-a0*a0)*rmi)
     zion(1,m)=sqrt(a0*a0+(real(m+myrank_partd*mi)-0.5)*(a1*a1-a0*a0)*rmi)
  enddo

! if particle tracking is "on", tag each particle with a unique number
  if(track_particles == 1)call tag_particles_8

! set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
  call set_random_zion_8

! poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=2.0*pi*(zion(2,m)-0.5)
     zion0(2,m)=zion(2,m) !zion0(2,:) for temporary storage
  enddo
  do i=1,10
!$omp parallel do private(m)
     do m=1,mi
        zion(2,m)=zion0(2,m)-2.0*zion(1,m)*sin(zion(2,m))
     enddo
  enddo
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
  enddo

! maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
!$omp parallel do private(m)
  do m=1,mi                
     z4tmp=zion(4,m)
     zion(4,m)=zion(4,m)-0.5
     !zion0(4,m)=sign(1.0d+00,zion(4,m))
     zion0(4,m)=sign(one,zion(4,m))
     !zion(4,m)=sqrt(max(1.0d-20,log(1.0/max(1.0d-20,zion(4,m)**2))))
     zion(4,m)=sqrt(max(small,log(1.0/max(small,zion(4,m)**2))))
     zion(4,m)=zion(4,m)-(c0+c1*zion(4,m)+c2*zion(4,m)**2)/&
          (1.0+d1*zion(4,m)+d2*zion(4,m)**2+d3*zion(4,m)**3)
     if(zion(4,m)>umax)zion(4,m)=z4tmp
  enddo

!$omp parallel do private(m)
  do m=1,mi

! toroidal:  uniformly distributed in zeta
     zion(3,m)=zetamin+(zetamax-zetamin)*zion(3,m)

     zion(4,m)=zion0(4,m)*min(umax,zion(4,m))

! initial random weight
     zion(5,m)=2.0*w_initial*(zion(5,m)-0.5)*(1.0+cos(zion(2,m)))

! maxwellian distribution in v_perp, <v_perp^2>=1.0
     !zion(6,m)=max(1.0d-20,min(umax*umax,-log(max(1.0d-20,zion(6,m)))))
     zion(6,m)=max(small,min(umax*umax,-log(max(small,zion(6,m)))))
  enddo

! transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
  vthi=gyroradius*abs(qion)/aion
!$omp parallel do private(m)
  do m=1,mi
     zion0(1,m)=1.0/(1.0+zion(1,m)*cos(zion(2,m))) !b-field
     zion(1,m)=0.5*zion(1,m)*zion(1,m)
     zion(4,m)=vthi*zion(4,m)*aion/(qion*zion0(1,m))
     zion(6,m)=sqrt(aion*vthi*vthi*zion(6,m)/zion0(1,m))
  enddo

! tracer particle initially resides on pe=0
  if(mype == 0)then
     zion(1,ntracer)=0.5*(0.5*(a0+a1))**2
     zion(2,ntracer)=0.0
     zion(3,ntracer)=0.5*(zetamin+zetamax)
     zion(4,ntracer)=0.5*vthi*aion/qion
     zion(5,ntracer)=0.0
     zion(6,ntracer)=sqrt(aion*vthi*vthi)
!     zion0(1:5,ntracer)=zion(1:5,ntracer)
  endif

  if(iload_8==0)then
! uniform load_8ing
!$omp parallel do private(m)
     do m=1,mi
        zion0(6,m)=1.0
     enddo
  else
! true nonuniform for temperature profile, density profile by weight
!$omp parallel do private(m)
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        i=max(0,min(mpsi,int((r-a0)*delr+0.5)))
        zion(4,m)=zion(4,m)*sqrt(rtemi(i))
        zion(6,m)=zion(6,m)*sqrt(rtemi(i))
!        zion0(6,m)=max(0.1d+00,min(10.0d+00,rden(i)))
        zion0(6,m)=max(pt_one,min(ten,rden(i)))
     enddo
  endif
  
! load_8 electron on top of ion if mi=me
  if(nhybrid>0)then
     vthe=sqrt(aion/(aelectron*tite))*(qion/aion)*(aelectron/qelectron)
     tsqrt=sqrt(1.0/tite)

! keep trapped electrons
     cmratio=qion/aion
     me=0
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        cost=cos(zion(2,m))
        b=1.0/(1.0+r*cost)
        upara=zion(4,m)*b*cmratio
        energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
        eperp=zion(6,m)*zion(6,m)/(1.0-r)
        if(eperp>energy)then
           me=me+1
           zelectron(1,me)=zion(1,m)
           zelectron(2,me)=zion(2,m)
           zelectron(3,me)=zion(3,m)
           zelectron(4,me)=zion(4,m)*vthe
           zelectron(5,me)=zion(5,m)
           zelectron(6,me)=zion(6,m)*tsqrt
           zelectron0(6,me)=zion0(6,m)
        endif
     enddo
     if(mype == 0)then
        ntracer=me
!        zelectron0(1:5,ntracer)=zelectron(1:5,ntracer)
     endif
  endif

!  write(mype+40,'(6e15.6)')zion(1:6,1:mi)
!  do m=1,mi
!     write(mype+40,*)zion(:,m)
!  enddo
!  close(mype+40)

end subroutine load_8


subroutine load_9
  use global_parameters_9
  use particle_array_9
  use field_array_9
  use particle_tracking_9
  use particle_decomp_9
  implicit none

  integer i,m,ierr
  real(8) :: c0=2.515517,c1=0.802853,c2=0.010328,&
              d1=1.432788,d2=0.189269,d3=0.001308
  real(8) :: w_initial,energy,momentum,r,rmi,pi2_inv,delr,ainv,vthe,tsqrt,&
              vthi,cost,b,upara,eperp,cmratio,z4tmp

! initialize random number generator
  call rand_num_gen_init_9

! restart from previous runs
  if(irun /= 0)then
     call restart_read_9
     return
  endif

  !!rmi=1.0/real(mi)
  rmi=1.0/real(mi*npartdom)
  pi2_inv=0.5/pi
  delr=1.0/deltar
  ainv=1.0/a
  w_initial=1.0e-3
  if(nonlinear<0.5)w_initial=1.0e-12
  if(track_particles < -10 .and. track_particles > -20)w_initial=0.1_wp*real(-10-track_particles,wp)
  if(mype==0)write(0,*)'w_initial =',w_initial
  ntracer=0
  !if(mype==0)ntracer=mi
  if(mype==0)ntracer=1
      
! radial: uniformly distributed in r^2, later transform to psi
!$omp parallel do private(m)
  do m=1,mi
     !!zion(1,m)=sqrt(a0*a0+(real(m)-0.5)*(a1*a1-a0*a0)*rmi)
     zion(1,m)=sqrt(a0*a0+(real(m+myrank_partd*mi)-0.5)*(a1*a1-a0*a0)*rmi)
  enddo

! if particle tracking is "on", tag each particle with a unique number
  if(track_particles == 1)call tag_particles_9

! set zion(2:6,1:mi) to uniformly distributed random values between 0 and 1
  call set_random_zion_9

! poloidal: uniform in alpha=theta_0+r*sin(alpha_0), theta_0=theta+r*sin(theta)
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=2.0*pi*(zion(2,m)-0.5)
     zion0(2,m)=zion(2,m) !zion0(2,:) for temporary storage
  enddo
  do i=1,10
!$omp parallel do private(m)
     do m=1,mi
        zion(2,m)=zion0(2,m)-2.0*zion(1,m)*sin(zion(2,m))
     enddo
  enddo
!$omp parallel do private(m)
  do m=1,mi                
     zion(2,m)=zion(2,m)*pi2_inv+10.0 !period of 1
     zion(2,m)=2.0*pi*(zion(2,m)-aint(zion(2,m))) ![0,2*pi)
  enddo

! maxwellian distribution in v_para, <v_para^2>=1.0, use zion0(4,:) as temporary storage
!$omp parallel do private(m)
  do m=1,mi                
     z4tmp=zion(4,m)
     zion(4,m)=zion(4,m)-0.5
     !zion0(4,m)=sign(1.0d+00,zion(4,m))
     zion0(4,m)=sign(one,zion(4,m))
     !zion(4,m)=sqrt(max(1.0d-20,log(1.0/max(1.0d-20,zion(4,m)**2))))
     zion(4,m)=sqrt(max(small,log(1.0/max(small,zion(4,m)**2))))
     zion(4,m)=zion(4,m)-(c0+c1*zion(4,m)+c2*zion(4,m)**2)/&
          (1.0+d1*zion(4,m)+d2*zion(4,m)**2+d3*zion(4,m)**3)
     if(zion(4,m)>umax)zion(4,m)=z4tmp
  enddo

!$omp parallel do private(m)
  do m=1,mi

! toroidal:  uniformly distributed in zeta
     zion(3,m)=zetamin+(zetamax-zetamin)*zion(3,m)

     zion(4,m)=zion0(4,m)*min(umax,zion(4,m))

! initial random weight
     zion(5,m)=2.0*w_initial*(zion(5,m)-0.5)*(1.0+cos(zion(2,m)))

! maxwellian distribution in v_perp, <v_perp^2>=1.0
     !zion(6,m)=max(1.0d-20,min(umax*umax,-log(max(1.0d-20,zion(6,m)))))
     zion(6,m)=max(small,min(umax*umax,-log(max(small,zion(6,m)))))
  enddo

! transform zion(1,:) to psi, zion(4,:) to rho_para, zion(6,:) to sqrt(mu) 
  vthi=gyroradius*abs(qion)/aion
!$omp parallel do private(m)
  do m=1,mi
     zion0(1,m)=1.0/(1.0+zion(1,m)*cos(zion(2,m))) !b-field
     zion(1,m)=0.5*zion(1,m)*zion(1,m)
     zion(4,m)=vthi*zion(4,m)*aion/(qion*zion0(1,m))
     zion(6,m)=sqrt(aion*vthi*vthi*zion(6,m)/zion0(1,m))
  enddo

! tracer particle initially resides on pe=0
  if(mype == 0)then
     zion(1,ntracer)=0.5*(0.5*(a0+a1))**2
     zion(2,ntracer)=0.0
     zion(3,ntracer)=0.5*(zetamin+zetamax)
     zion(4,ntracer)=0.5*vthi*aion/qion
     zion(5,ntracer)=0.0
     zion(6,ntracer)=sqrt(aion*vthi*vthi)
!     zion0(1:5,ntracer)=zion(1:5,ntracer)
  endif

  if(iload_9==0)then
! uniform load_9ing
!$omp parallel do private(m)
     do m=1,mi
        zion0(6,m)=1.0
     enddo
  else
! true nonuniform for temperature profile, density profile by weight
!$omp parallel do private(m)
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        i=max(0,min(mpsi,int((r-a0)*delr+0.5)))
        zion(4,m)=zion(4,m)*sqrt(rtemi(i))
        zion(6,m)=zion(6,m)*sqrt(rtemi(i))
!        zion0(6,m)=max(0.1d+00,min(10.0d+00,rden(i)))
        zion0(6,m)=max(pt_one,min(ten,rden(i)))
     enddo
  endif
  
! load_9 electron on top of ion if mi=me
  if(nhybrid>0)then
     vthe=sqrt(aion/(aelectron*tite))*(qion/aion)*(aelectron/qelectron)
     tsqrt=sqrt(1.0/tite)

! keep trapped electrons
     cmratio=qion/aion
     me=0
     do m=1,mi
        r=sqrt(2.0*zion(1,m))
        cost=cos(zion(2,m))
        b=1.0/(1.0+r*cost)
        upara=zion(4,m)*b*cmratio
        energy=0.5*aion*upara*upara+zion(6,m)*zion(6,m)*b
        eperp=zion(6,m)*zion(6,m)/(1.0-r)
        if(eperp>energy)then
           me=me+1
           zelectron(1,me)=zion(1,m)
           zelectron(2,me)=zion(2,m)
           zelectron(3,me)=zion(3,m)
           zelectron(4,me)=zion(4,m)*vthe
           zelectron(5,me)=zion(5,m)
           zelectron(6,me)=zion(6,m)*tsqrt
           zelectron0(6,me)=zion0(6,m)
        endif
     enddo
     if(mype == 0)then
        ntracer=me
!        zelectron0(1:5,ntracer)=zelectron(1:5,ntracer)
     endif
  endif

!  write(mype+40,'(6e15.6)')zion(1:6,1:mi)
!  do m=1,mi
!     write(mype+40,*)zion(:,m)
!  enddo
!  close(mype+40)

end subroutine load_9


