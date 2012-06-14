subroutine restart_write
  use global_parameters
  use particle_array
  use field_array
  use diagnosis_array
  implicit none

  character(len=18) cdum
  character(len=10) restart_dir
  character(len=60) file_name
  real(wp) dum
  integer i,j,mquantity,mflx,n_mode,mstepfinal,noutputs
  integer save_restart_files,ierr

  !save_restart_files=1
  save_restart_files=0

  write(cdum,'("DATA_RESTART.",i4.4)')mype

  if(save_restart_files==1)then
     write(restart_dir,'("STEP_",i0)')(mstepall+istep)
     if(mype==0)call system("mkdir "//restart_dir)
     !call MPI_BARRIER(MPI_COMM_WORLD,ierr)
     file_name=trim(restart_dir)//'/'//trim(cdum)
     open(222,file=file_name,status='replace',form='unformatted')
  else
     open(222,file=cdum,status='replace',form='unformatted')
  endif

! record particle information for future restart run

  write(222)mi,me,ntracer,rdtemi,rdteme,pfluxpsi,phi,phip00,zonali,zonale
  if(mype==0)write(222)etracer,ptracer
  write(222)zion(1:nparam,1:mi),zion0(6,1:mi)
  if(nhybrid>0)write(222)phisave,zelectron(1:6,1:me),zelectron0(6,1:me)
  close(222)

! S.Ethier 01/30/04 Save a copy of history.out and sheareb.out for restart
  if(mype==0 .and. istep<mstep)then
     open(777,file='history_restart.out',status='unknown')
     rewind(ihistory)
     read(ihistory,101)j
     write(777,101)j
     read(ihistory,101)mquantity
     write(777,101)mquantity
     read(ihistory,101)mflx
     write(777,101)mflx
     read(ihistory,101)n_mode
     write(777,101)n_mode
     read(ihistory,101)mstepfinal
     noutputs=mstepfinal-mstep/ndiag+istep/ndiag
     write(777,101)noutputs
     do i=0,(mquantity+mflx+4*n_mode)*noutputs
        read(ihistory,102)dum
        write(777,102)dum
     enddo
     close(777)

   ! Now do sheareb.out
     open(777,file='sheareb_restart.out',status='unknown')
     rewind(444)
     read(444,101)j
     write(777,101)j
     do i=1,mpsi*noutputs
        read(444,102)dum
        write(777,102)dum
     enddo
     close(777)
  endif

101 format(i6)
102 format(e12.6)

end subroutine restart_write

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine restart_read
  use global_parameters
  use particle_array
  use field_array
  use diagnosis_array
  implicit none

  integer m
  character(len=18) cdum
  
  write(cdum,'("DATA_RESTART.",i4.4)')mype
  open(333,file=cdum,status='old',form='unformatted')

! read particle information to restart previous run
  read(333)mi,me,ntracer,rdtemi,rdteme,pfluxpsi,phi,phip00,zonali,zonale
  if(mype==0)read(333)etracer,ptracer
  read(333)zion(1:nparam,1:mi),zion0(6,1:mi)
  if(nhybrid>0)read(333)phisave,zelectron(1:6,1:me),zelectron0(6,1:me)
  close(333)

  return

! test domain decomposition
  do m=1,mi
     if(zion(3,m)>zetamax+1.0e-10 .or. zion(3,m)<zetamin-1.0e-10)then
        print *, 'PE=',mype, ' m=',m, ' zion=',zion(3,m)
        stop
     endif
  enddo
  if(nhybrid>0)then
     do m=1,me
        if(zelectron(3,m)>zetamax+1.0e-10 .or. zelectron(3,m)<zetamin-1.0e-10)then
           print *, 'PE=',mype, ' m=',m, ' zelectron=',zelectron(3,m)
           stop
        endif
     enddo
  endif

end subroutine restart_read

