subroutine diagnosis(hpx4_bti,&
             t_gids, p_gids,&
! global parameters
                       ihistory,snapout,maxmpsi,&
          mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile,&
           nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron,&
                                       mtheta, &
                                       deltat, &
           do_collision, &
! particle array
                                      kzion,kzelectron,jtelectron0,jtelectron1,&
                                        jtion0,jtion1,&
                                       wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1, &
                                        wpion,wtion0,wtion1,&
                                         zion,zion0,zelectron,&
        zelectron0,zelectron1, &
! particle decomp
              ntoroidal,npartdom,&
              partd_comm,nproc_partd,myrank_partd,&
              toroidal_comm,nproc_toroidal,myrank_toroidal,&
              left_pe,right_pe,&
              toroidal_domain_location,particle_domain_location, &
! field array
                      mmpsi, &
                                      itran,igrid, &
                                          jtp1,jtp2,&
                                       phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt,&
                                         phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit,&
                                           evector,wtp1,wtp2,phisave,&
              Total_field_energy, &
! diagnosis array
                       mflux,num_mode,m_poloidal,&
          nmode,mmode,&
           efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy,eflux,&
       rmarker,rdtemi,rdteme,pfluxpsi,&
       amp_mode, &
                                       hfluxpsi,&
                                           eigenmode,&
           etracer,ptracer)

  !use global_parameters
  !use particle_array
  !use particle_decomp
  !use field_array
  !use diagnosis_array
  use, intrinsic :: iso_c_binding, only : c_ptr
  use precision
  implicit none
  interface ! {{{
    subroutine err_check(f1,f2,numpe,&
! global parameters
                          ihistory,snapout,maxmpsi,&
             mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
          istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
          mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
          ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile,&
              nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
          flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
          zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron,&
                                          mtheta, &
                                          deltat, &
              do_collision)
      !use global_parameters
      use precision
      implicit none
!     global parameters
     integer :: ihistory,snapout,maxmpsi
     integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
          istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
          mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
          ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile
     real(kind=wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
          flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
          zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
     integer,dimension(:),allocatable :: mtheta
     real(kind=wp),dimension(:),allocatable :: deltat
     logical  do_collision

      real(kind=wp) f1, f2, FOM, expected
      integer numpe, ierr
    end subroutine err_check
  end interface ! }}}

  integer,dimension(:),allocatable :: t_gids
  integer,dimension(:),allocatable :: p_gids
  TYPE(C_PTR), INTENT(IN), VALUE :: hpx4_bti

!  global parameters
  integer :: ihistory,snapout,maxmpsi
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile
  real(kind=wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(kind=wp),dimension(:),allocatable :: deltat
  logical  do_collision

! particle array
  integer,dimension(:),allocatable :: kzion,kzelectron,jtelectron0,jtelectron1
  integer,dimension(:,:),allocatable :: jtion0,jtion1
  real(kind=wp),dimension(:),allocatable :: wzion,wzelectron,wpelectron,&
       wtelectron0,wtelectron1
  real(kind=wp),dimension(:,:),allocatable :: wpion,wtion0,wtion1
  real(kind=wp),dimension(:,:),allocatable :: zion,zion0,zelectron,&
        zelectron0,zelectron1

! particle decomp
  integer  :: ntoroidal,npartdom
  integer  :: partd_comm,nproc_partd,myrank_partd
  integer  :: toroidal_comm,nproc_toroidal,myrank_toroidal
  integer  :: left_pe,right_pe
  integer  :: toroidal_domain_location,particle_domain_location

! field array
  integer :: mmpsi
  integer,dimension(:),allocatable :: itran,igrid
  integer,dimension(:,:,:),allocatable :: jtp1,jtp2
  real(kind=wp),dimension(:),allocatable :: phi00,phip00,rtemi,rteme,rden,qtinv,&
       pmarki,pmarke,zonali,zonale,gradt
  real(kind=wp),dimension(:,:),allocatable :: phi,densityi,densitye,markeri,&
       markere,pgyro,tgyro,dtemper,heatflux,phit
  real(kind=wp),dimension(:,:,:),allocatable :: evector,wtp1,wtp2,phisave
  real(kind=wp) :: Total_field_energy(3)

! diagnosis array
  integer :: mflux,num_mode,m_poloidal
  integer nmode(num_mode),mmode(num_mode)
  real(kind=wp) efluxi,efluxe,pfluxi,pfluxe,ddeni,ddene,dflowi,dflowe,&
       entropyi,entropye,efield,eradial,particles_energy(2),eflux(mflux),&
       rmarker(mflux),rdtemi(mflux),rdteme(mflux),pfluxpsi(mflux),&
       amp_mode(2,num_mode,2)
  real(kind=wp),dimension(:),allocatable :: hfluxpsi
  real(kind=wp),dimension(:,:,:),allocatable :: eigenmode
  real(kind=wp) etracer,ptracer(4)

  real(kind=wp) f1, f2, FOM, expected
  integer numpe, ierr

  integer,parameter :: mquantity=20
  integer i,j,k,ij,m,ierror
  real(kind=wp) r,b,g,q,kappa,fdum(21),adum(21),ddum(mpsi),ddumtmp(mpsi),tracer(6),&
       tmarker,xnormal(mflux),rfac,dum,epotential,vthi,vthe,tem_inv
  real(kind=wp),external :: boozer2x,boozer2z,bfield,psi2r
  character(len=11) cdum

#ifdef __NERSC
#define FLUSH flush_
#else
#define FLUSH flush
#endif

! send tracer information to PE=0
! Make sure that PE 0 does not send to itself if it already holds the tracer.
! The processor holding the tracer particle has ntracer>0 while all the
! other processors have ntracer=0.
  tracer=0.0
  if(ntracer>0)then
     if(nhybrid==0)then
        tracer(1:5)=zion0(1:5,ntracer)
        tracer(6)=zion(6,ntracer)
     else
        tracer(1:5)=zelectron0(1:5,ntracer)
        tracer(6)=zelectron(6,ntracer)
     endif
     if(mype/=0) then
       !call MPI_SEND(tracer,6,mpi_Rsize,0,1,MPI_COMM_WORLD,ierror)
       call send_cmm(hpx4_bti,tracer,6,0)
     endif
  endif
  if(mype==0 .and. ntracer==0) then 
    !call MPI_RECV(tracer,6,mpi_Rsize,MPI_ANY_SOURCE,&
    !              1,MPI_COMM_WORLD,istatus,ierror)
    call receive_cmm(hpx4_bti,tracer,6,0)
  endif

! initialize output file
  if(istep==ndiag)then
     if(mype==0)then
        if(irun==0)then
! first time run 
!hjw       open(ihistory,file='history.out',status='replace')            
!hjw       write(ihistory,101)istep
!hjw       write(ihistory,101)irun,mquantity,mflux,num_mode,mstep/ndiag
!hjw       write(ihistory,102)tstep*real(ndiag)
           mstepall=0

! E X B history
           i=3
!hjw       open(444,file='sheareb.out',status='replace')
!hjw       write(444,101)istep
!hjw       write(444,101)i
!hjw       write(444,101)mpsi

! tracer particle energy and angular momentum
! S.Ethier 4/15/04  The calculation of "etracer" and "ptracer" is now done
! only for irun=0 since we save the values of these variables in the
! restart files. This ensures continuity of the tracer particle energy
! and momentum plots.
           b=bfield(tracer(1),tracer(2))
           g=1.0
           r=sqrt(2.0*tracer(1))
           q=q0+q1*r/a+q2*r*r/(a*a)
!           epotential=gyroradius*r*r*(0.5*flow0+flow1*r/(3.0*a)+0.25*flow2*r*r/(a*a))
           epotential=0.0
           if(nhybrid==0)etracer=tracer(6)*tracer(6)*b &
                +0.5*(tracer(4)*b*qion)**2/aion+epotential
           if(nhybrid>0)etracer=tracer(6)*tracer(6)*b &
                +0.5*(tracer(4)*b*qelectron)**2/aelectron+epotential
           ptracer(1)=tracer(1)
           ptracer(2)=tracer(4)*g
           ptracer(3)=q
           ptracer(4)=0.0
           write(stdout,*)'etracer =',etracer

        else
! E X B history
!hjw       open(444,file='sheareb.out',status='old',position='append')
           
! restart, copy old data to backup file to protect previous run data
!hjw       open(555,file='history.out',status='old')
! # of run
!hjw       read(555,101)j
           
           write(cdum,'("histry",i1,".bak")')j-10*(j/10)
!hjw       open(ihistory,file=cdum,status='replace')
!hjw       write(ihistory,101)j
           do i=1,3
!hjw          read(555,101)j
!hjw          write(ihistory,101)j
           enddo
! # of time step
!hjw       read(555,101)j
!hjw       write(ihistory,101)j
           do i=0,(mquantity+mflux+4*num_mode)*j !i=0 for tstep
!hjw          read(555,102)dum
!hjw          write(ihistory,102)dum
           enddo
!hjw       close(555)
!hjw       close(ihistory)

! if everything goes OK, copy old data to history.out
!hjw       open(666,file=cdum,status='old')
!hjw       open(ihistory,file='history.out',status='replace')            
!hjw       read(666,101)j
           irun=j+1
!hjw       write(ihistory,101)irun               
           do i=1,3
!hjw          read(666,101)j
!hjw          write(ihistory,101)j
           enddo
!hjw       read(666,101)mstepall
!hjw       write(ihistory,101)mstepall+mstep/ndiag
           do i=0,(mquantity+mflux+4*num_mode)*mstepall
!hjw          read(666,102)dum
!hjw          write(ihistory,102)dum
           enddo
!hjw       close(666)
           mstepall=mstepall*ndiag
        endif

! flux normalization
        do i=1,mflux
           r=a0+(a1-a0)/real(mflux)*(real(i)-0.5)

! divided by local Kappa_T to obtain chi_i=c_i rho_i
           rfac=rw*(r-rc)
           rfac=rfac*rfac
           rfac=rfac*rfac*rfac
           rfac=max(0.1_wp,exp(-rfac))
           
           kappa=1.0
           if(nbound==0)kappa=0.0
           kappa=1.0-kappa+kappa*rfac
           
           xnormal(i)=1.0/(kappa*kappati*gyroradius)
           write(stdout,*)"kappa_T at radial_bin=",i,kappa*kappati
        enddo
        do i=1,mpsi
           r=a0+(a1-a0)/real(mpsi)*(real(i)-0.5)

! divided by local Kappa_T to obtain chi_i=c_i rho_i
           rfac=rw*(r-rc)
           rfac=rfac*rfac
           rfac=rfac*rfac*rfac
           rfac=max(0.1_wp,exp(-rfac))

           kappa=1.0
           if(nbound==0)kappa=0.0
           kappa=1.0-kappa+kappa*rfac
           
           gradt(i)=1.0/(kappa*kappati*gyroradius)
        enddo
     endif
     !call MPI_BCAST(mstepall,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierror)
     call broadcast_int_cmm(hpx4_bti,mstepall,1);

  endif

! global sum of fluxes
! All these quantities come from summing up contributions from the particles
! so the MPI_REDUCE involves all the MPI processes.
  vthi=gyroradius*abs(qion)/aion
  vthe=vthi*sqrt(aion/(aelectron*tite))
  tem_inv=1.0_wp/(aion*vthi*vthi)
  fdum(1)=efield
  fdum(2)=entropyi
  fdum(3)=entropye
  fdum(4)=dflowi/vthi
  fdum(5)=dflowe/vthe
  fdum(6)=pfluxi/vthi
  fdum(7)=pfluxe/vthi
  fdum(8)=efluxi*tem_inv/vthi
  fdum(9)=efluxe*tem_inv/vthi
  fdum(10:14)=eflux*tem_inv/vthi
  fdum(15:19)=rmarker
! S.Ethier 9/21/04 Add the total energy in the particles to the list
! of variables to reduce.
  fdum(20)=particles_energy(1)
  fdum(21)=particles_energy(2)

  !call MPI_REDUCE(fdum,adum,21,mpi_Rsize,MPI_SUM,0,MPI_COMM_WORLD,ierror)
  call comm_reduce_cmm(hpx4_bti,fdum,adum,21,0)

! radial envelope of fluctuation intensity
  ddum=0.0_wp
  do i=1,mpsi
           ddum(i)=ddum(i)+sum((phi(1:mzeta,igrid(i)+1:igrid(i)+mtheta(i)))**2)
  enddo
  if(myrank_partd==0) then
    !call MPI_REDUCE(ddum,ddumtmp,mpsi,mpi_Rsize,MPI_SUM,0,toroidal_comm,ierror)
    call toroidal_reduce_cmm(hpx4_bti,ddum,ddumtmp,mpsi,0)
  endif

  if(mype==0)then  !mype=0 is also myrank_partd=0 and myrank_toroidal=0
     fdum=adum
     do i=1,mpsi
        ddum(i)=tem_inv*sqrt(ddumtmp(i)/real(mzetamax*mtheta(i)))
     enddo

! record tracer information
!hjw write(ihistory,101)istep
!hjw write(ihistory,102)boozer2x(tracer(1),tracer(2))-1.0
!hjw write(ihistory,102)boozer2z(tracer(1),tracer(2))
!hjw write(ihistory,102)tracer(2)
!hjw write(ihistory,102)tracer(3)
!hjw write(ihistory,102)tracer(5) ! tracer weight
     b=bfield(tracer(1),tracer(2))
     g=1.0
     r=sqrt(2.0*tracer(1))
     q=q0+q1*r/a+q2*r*r/(a*a)
!hjw if(nhybrid==0)write(ihistory,102)tracer(4)*b*qion/(aion*vthi)
!hjw if(nhybrid>0)write(ihistory,102)tracer(4)*b*qelectron/(aelectron*vthe)

! energy error
!     epotential=gyroradius*r*r*(0.5*flow0+flow1*r/(3.0*a)+0.25*flow2*r*r/(a*a))
     epotential=0.0
!hjw if(nhybrid==0)write(ihistory,102)(tracer(6)*tracer(6)*b &
!hjw      +0.5*(tracer(4)*b*qion)**2/aion+epotential)/etracer-1.0
!hjw if(nhybrid>0)write(ihistory,102)(tracer(6)*tracer(6)*b &
!hjw      +0.5*(tracer(4)*b*qelectron)**2/aelectron+epotential)/etracer-1.0

     dum=(tracer(1)-ptracer(1))/(0.5*(ptracer(3)+q))-(tracer(4)*g-ptracer(2))
     ptracer(1)=tracer(1)
     ptracer(2)=tracer(4)*g
     ptracer(3)=q
     ptracer(4)=ptracer(4)+dum
!hjw write(ihistory,102)ptracer(4) ! momentum error

! normalization
     fdum(15:19)=max(1.0_wp,fdum(15:19))
   !!  fdum(1)=fdum(1)/real(numberpe)
   ! At this point, fdum(1)/real(numberpe) is the volume average of phi**2.
   ! Since sqrt(y1)+sqrt(y2) is not equal to sqrt(y1+y2), we need to take
   ! the square root after the MPI_Reduce operation and final volume average.
   ! Modifications to the calculation of efield were also made in smooth.
     fdum(1)=sqrt(fdum(1)/real(numberpe))
     tmarker=sum(fdum(15:19))
     fdum(2:9)=fdum(2:9)/tmarker
     fdum(10:14)=fdum(10:14)*xnormal(1:5)/fdum(15:19)

! write fluxes to history file
!hjw write(ihistory,102)ddeni,ddene,eradial,fdum(1:14)

! write mode amplitude history
     do i=1,2
        do j=1,num_mode
           do k=1,2
!hjw          write(ihistory,102)amp_mode(k,j,i)
           enddo
        enddo
     enddo
     !call FLUSH(ihistory)
     write(stdout,*)istep,fdum(1),eradial,fdum(12),fdum(20),fdum(21)
     write(stdout,*)' Step ',istep,' thermal flux: ',fdum(10)
     write(stdout,*)istep,Total_field_energy(1:3),fdum(15),fdum(16),fdum(17),fdum(18),fdum(19)
     if(istep>=mstep)then
        !call err_check(fdum(1),fdum(19),numberpe)
        call err_check(f1,f2,numpe,&
! global parameters
                       ihistory,snapout,maxmpsi,&
          mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile,&
           nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron,&
                                       mtheta, &
                                       deltat, &
           do_collision)
     endif
         
     !call FLUSH(stdout)

! radial-time 2D data
!hjw write(444,101)istep
!hjw write(444,102)phip00(1:mpsi)/vthi
!     write(444,102)phi00*tem_inv
!hjw write(444,102)ddum
!hjw write(444,102)hfluxpsi(1:mpsi)*gradt*tem_inv/vthi
!hjw if(istep==mstep)close(444)
  endif

101 format(i6)
102 format(e12.6)

end subroutine diagnosis

   subroutine err_check(f1,f2,numpe,&
! global parameters
                       ihistory,snapout,maxmpsi,&
          mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile,&
           nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron,&
                                       mtheta, &
                                       deltat, &
           do_collision)
   !use global_parameters
   use precision
   implicit none
!  global parameters
  integer :: ihistory,snapout,maxmpsi
  integer mi,mimax,me,me1,memax,mgrid,mpsi,mthetamax,mzeta,mzetamax,&
       istep,ndiag,ntracer,msnap,mstep,mstepall,stdout,mype,numberpe,&
       mode00,nbound,irun,iload,irk,idiag,ncycle,mtdiag,idiag1,idiag2,&
       ntracer1,nhybrid,ihybrid,nparam,rng_control,limit_vpara,fixed_Tprofile
  real(kind=wp) nonlinear,paranl,a0,a1,a,q0,q1,q2,pi,tstep,kappati,kappate,kappan,&
       flow0,flow1,flow2,ulength,utime,gyroradius,deltar,deltaz,zetamax,&
       zetamin,umax,tite,rc,rw,tauii,qion,qelectron,aion,aelectron
  integer,dimension(:),allocatable :: mtheta
  real(kind=wp),dimension(:),allocatable :: deltat
  logical  do_collision

   real(kind=wp) f1, f2, FOM, expected
   integer numpe, ierr
   real(kind=wp) correct(3), diff

   data correct /563.023477896237, 1403.47902959535, 3117.02934556827/

   FOM = f1 * f2
   if (numberpe .eq. 64) then
    expected = correct(1)
   else if (numberpe .eq. 512) then
    expected = correct(2)
   else if (numberpe .eq.2048) then
    expected = correct(3)
   else 
      write (*,*) ' illegal value for numberpe in err_check! '
      !call MPI_Abort(MPI_COMM_WORLD,1,ierr)
   endif

   diff = abs(expected-FOM)

   write(stdout,99) 
   write(stdout,100) 
   write(stdout,101) expected
   write(stdout,102) FOM
   write(stdout,103) diff
 99 format(' ')
100 format(' Checking answers after last time step.... ')
101 format(' Expected Answer: ',e15.8)
102 format(' This Run       : ',e15.8)
103 format(' Difference:    : ',e15.8)
end subroutine err_check
