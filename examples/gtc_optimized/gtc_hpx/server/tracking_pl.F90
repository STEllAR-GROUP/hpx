!========================================

subroutine tag_particles_0

!========================================

  use global_parameters_0
  use particle_array_0
  use particle_tracking_0
  implicit none
  integer :: m,np

! we tag each particle with a unique number that will be carried along
! with it as it moves from one processor/domain to another. to facilitate
! the search for the tracked particles, we give the non-tracked particles
! a negative value.
  do m=1,mi
     zion(7,m)=-real(m+mype*mi)
     zion0(7,m)=zion(7,m)
  enddo
  if(mype==0)then
  ! on the master process (mype=0), we pick "nptrack" particles that will
  ! be followed at every time step. to facilitate the search of those
  ! particles among all the others in the zion arrays of each processor,
  ! we give them a positive number from 1 to nptrack. the particles are
  ! picked around r=0.5, which is mi/2.
    np=0
    do m=(mi-nptrack)/2,(mi+nptrack)/2-1
       np=np+1
       zion(7,m)=np
       zion0(7,m)=np
    enddo
  endif

end subroutine tag_particles_0

!========================================

subroutine locate_tracked_particles_0

!========================================

  use global_parameters_0
  use particle_array_0
  use particle_tracking_0
  implicit none
  integer :: i,m,npp,iout

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
  iout=mod(istep,isnap)
  if(iout==0)iout=isnap
  npp=0
  do m=1,mi
     if(zion(7,m)>0.0)then
       npp=npp+1
       ptracked(1:nparam,npp,iout)=zion(1:nparam,m)
     endif
  enddo
  ntrackp(iout)=npp

end subroutine locate_tracked_particles_0

!========================================

subroutine write_tracked_particles_0

!========================================

  use global_parameters_0
  use particle_tracking_0
  implicit none
 
  integer :: i,j
  character(len=10) :: cdum

  if(mype < 10)then
     write(cdum,'("trackp.00",i1)')mype
  elseif(mype < 100)then
     write(cdum,'("trackp.0",i2)')mype
  else
     write(cdum,'("trackp.",i3)')mype
  endif

  open(57,file=cdum,status='unknown',position='append')
  write(57,*)istep,isnap,nptrack
  write(57,*)ntrackp
  do i=1,isnap
     do j=1,ntrackp(i)
        write(57,*)ptracked(1:nparam,j,i)
     enddo
  enddo
  close(57)

end subroutine write_tracked_particles_0

!========================================

subroutine hdf5out_tracked_particles_0

!========================================
  use global_parameters_0
#ifdef __hdf5
  use particle_array_0
  use particle_tracking_0
  use hdf5
  implicit none

  integer  :: i,j,m,ipp,npp,ntpart(0:numberpe-1)
  real     :: r,q,theta,theta0,zeta,x,y,z

! hdf5 declarations
  integer(hid_t),save :: file_id       ! file identifier
  integer(hid_t) :: dset_id       ! dataset identifier
  integer(hid_t) :: filespace     ! dataspace identifier in file
  integer(hid_t) :: memspace      ! dataspace identifier in memory
  integer(hid_t) :: plist_id      ! property list identifier
  integer(hid_t) :: grp_id        ! group identifier
  integer(hid_t) :: time_id       ! attribute identifier
  integer(hid_t) :: mpsi_id       ! attribute identifier
  integer(hid_t) :: npe_id        ! attribute identifier
  integer(hid_t) :: mtheta_id     ! attribute identifier
  integer(hid_t) :: radial_id     ! attribute identifier
  integer(hid_t) :: itran_id      ! attribute identifier
  integer(hid_t) :: aspace_id     ! attribute dataspace identifier
  integer(hid_t) :: a1dspace_id   ! attribute dataspace identifier
  integer     :: ierr             ! return error flag
  integer     :: comm,info
  real        :: curr_time
  character(len=20),save :: fdum
  character(len=20) :: vdum
  integer :: rank
  integer(hsize_t), dimension(2) :: count,dimsf
  integer(hssize_t), dimension(2) :: offset
  integer, dimension(7) :: dimsfi = (/0,0,0,0,0,0,0/)

  comm = mpi_comm_world
  info = mpi_info_null


! this routine is called when idiag=0, right after the first runge-kutta
! half step in pushi (irk=1). for this reason, we need to use zion0()
! instead of zion(). at this point, the array zion0() holds the particle
! quantities at the end of the previous full step (irk=2). zion0() is
! also used in pushi at irk=1 to do the particle diagnostics, so using
! the same array ensures synchronization between the ouput files.

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
! we transform the coordinates of the particle, which are in magnetic
! coordinates (psi,theta,zeta), to x, y, z coordinates.
!
!       r=sqrt(2.0*zion(1,m))
!       q=q0+q1*r/a+q2*r*r/(a*a)
  npp=0
  do m=1,mi
     if(zion0(7,m)>0.0)then
       npp=npp+1
       r=sqrt(2.0*zion0(1,m))
       q=q0+q1*r/a+q2*r*r/(a*a)
       zeta=zion0(3,m)
       !!!theta=zion0(2,m)
       !!!theta0=theta+r*sin(theta)
       theta0=zion0(2,m)
       x=cos(zeta)*(1.0+r*cos(theta0))
       y=sin(zeta)*(1.0+r*cos(theta0))
       z=r*sin(theta0)
       ptracked(1,npp,1)=x
       ptracked(2,npp,1)=y
       ptracked(3,npp,1)=z
       ptracked(4,npp,1)=zion0(4,m)
       ptracked(5,npp,1)=zion0(5,m)
       ptracked(6,npp,1)=zion(6,m)
       ptracked(7,npp,1)=zion(7,m)
     endif
  enddo

! we need to know how many tracked particles each process has. this will
! be required to calculate the offset in the hdf5 file.
  ntpart=0
  call mpi_allgather(npp,1,mpi_integer,ntpart,1,mpi_integer,comm,ierr)

  if(mod((istep-ndiag),isnap)==0)then
   ! we open a new hdf5 file at the first diagnostic call after an output
   ! to the snapshot files.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_0 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! filename for grid quantities
     write(fdum,'("trackp_",i0,".h5")')(1+(mstepall+istep-ndiag)/isnap)

   ! create the file collectively.
     call h5fcreate_f(fdum,h5f_acc_trunc_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  else
   ! reopen the hdf5 file used during the last call to this subroutine.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_0 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! reopen the file collectively.
     call h5fopen_f(fdum,h5f_acc_rdwr_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  endif

! we now write the current calculation time as an attribute to the
! dataset. we need create the attribute data space, data type, etc.
! only one processor needs to write the attribute.

! create the data space for the 2d dataset:
!  1st dim: 7 quantities = particle phase space coordinates + identifier
!  2nd dim: number of tracked particles
  rank=2
  dimsf(1)=nparam
  dimsf(2)=nptrack
  call h5screate_simple_f(rank,dimsf,filespace,ierr)

! write dataset name into variable
  write(vdum,'("particles_t",i0)')(mstepall+istep)/ndiag

! create the dataset with default properties.
  call h5dcreate_f(file_id,vdum,h5t_native_real,filespace,dset_id,ierr)
  call h5sclose_f(filespace,ierr)

! create scalar data space for the time attribute that will be attached to
! the particle dataset.
  call h5screate_f(h5s_scalar_f,aspace_id,ierr)

! create dataset attribute
  call h5acreate_f(dset_id,"time",h5t_native_real,aspace_id,time_id,ierr)

  curr_time=real(mstepall+istep)*tstep  ! attribute data

! write the time attribute data. here dimsfi is just a dummy argument.
  if(mype.eq.0)then
    call h5awrite_f(time_id,h5t_native_real,curr_time,dimsfi,ierr)
  endif

! release attribute id, attribute data space
  call h5aclose_f(time_id,ierr)
  call h5sclose_f(aspace_id,ierr)

! each process defines a dataset in memory and writes it to the hyperslab
! in the file.
! first dimension is the particle quantities.
  count(1)=nparam
  offset(1)=0

! second dimension is the number of tracked particles.
! the offset is determine by the sum of tracked particles on pes < mype
  ipp=0
  do i=0,mype-1
     ipp=ipp+ntpart(i)
  enddo
  count(2)=npp
  offset(2)=ipp

  dimsfi(1)=count(1)
  dimsfi(2)=count(2)

! only the processes holding tracked particles participate in the output
! to the hdf5 file:
  if(npp > 0)then
    call h5screate_simple_f(rank,count,memspace,ierr)

  ! select hyperslab in the file.
    call h5dget_space_f(dset_id,filespace,ierr)
    call h5sselect_hyperslab_f(filespace,h5s_select_set_f,offset,count,ierr)
  
  ! create property list for collective dataset write
    call h5pcreate_f(h5p_dataset_xfer_f, plist_id, ierr)
  !!!  call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_collective_f, ierr)
    call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_independent_f, ierr)

  ! write the dataset
    call h5dwrite_f(dset_id,h5t_native_real,ptracked, dimsfi,ierr, &
         file_space_id=filespace,mem_space_id=memspace,xfer_prp=plist_id)

  ! close property list and dataspaces.
    call h5pclose_f(plist_id,ierr)
    call h5sclose_f(filespace,ierr)
    call h5sclose_f(memspace,ierr)
  endif

! close dataset.
  call h5dclose_f(dset_id,ierr)

! close hdf5 file to insure that we do not loose the hdf5 file if the code 
! is terminated abruptly.
  call h5fclose_f(file_id,ierr)

! close the hdf5 library and the fortran90 interface
  call h5close_f(ierr)

#else
  if(istep==ndiag)write(0,*)mype,' **** the code was compiled without hdf5 support ***'
#endif
end subroutine hdf5out_tracked_particles_0
!========================================

subroutine tag_particles_1

!========================================

  use global_parameters_1
  use particle_array_1
  use particle_tracking_1
  implicit none
  integer :: m,np

! we tag each particle with a unique number that will be carried along
! with it as it moves from one processor/domain to another. to facilitate
! the search for the tracked particles, we give the non-tracked particles
! a negative value.
  do m=1,mi
     zion(7,m)=-real(m+mype*mi)
     zion0(7,m)=zion(7,m)
  enddo
  if(mype==0)then
  ! on the master process (mype=0), we pick "nptrack" particles that will
  ! be followed at every time step. to facilitate the search of those
  ! particles among all the others in the zion arrays of each processor,
  ! we give them a positive number from 1 to nptrack. the particles are
  ! picked around r=0.5, which is mi/2.
    np=0
    do m=(mi-nptrack)/2,(mi+nptrack)/2-1
       np=np+1
       zion(7,m)=np
       zion0(7,m)=np
    enddo
  endif

end subroutine tag_particles_1

!========================================

subroutine locate_tracked_particles_1

!========================================

  use global_parameters_1
  use particle_array_1
  use particle_tracking_1
  implicit none
  integer :: i,m,npp,iout

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
  iout=mod(istep,isnap)
  if(iout==0)iout=isnap
  npp=0
  do m=1,mi
     if(zion(7,m)>0.0)then
       npp=npp+1
       ptracked(1:nparam,npp,iout)=zion(1:nparam,m)
     endif
  enddo
  ntrackp(iout)=npp

end subroutine locate_tracked_particles_1

!========================================

subroutine write_tracked_particles_1

!========================================

  use global_parameters_1
  use particle_tracking_1
  implicit none
 
  integer :: i,j
  character(len=10) :: cdum

  if(mype < 10)then
     write(cdum,'("trackp.00",i1)')mype
  elseif(mype < 100)then
     write(cdum,'("trackp.0",i2)')mype
  else
     write(cdum,'("trackp.",i3)')mype
  endif

  open(57,file=cdum,status='unknown',position='append')
  write(57,*)istep,isnap,nptrack
  write(57,*)ntrackp
  do i=1,isnap
     do j=1,ntrackp(i)
        write(57,*)ptracked(1:nparam,j,i)
     enddo
  enddo
  close(57)

end subroutine write_tracked_particles_1

!========================================

subroutine hdf5out_tracked_particles_1

!========================================
  use global_parameters_1
#ifdef __hdf5
  use particle_array_1
  use particle_tracking_1
  use hdf5
  implicit none

  integer  :: i,j,m,ipp,npp,ntpart(0:numberpe-1)
  real     :: r,q,theta,theta0,zeta,x,y,z

! hdf5 declarations
  integer(hid_t),save :: file_id       ! file identifier
  integer(hid_t) :: dset_id       ! dataset identifier
  integer(hid_t) :: filespace     ! dataspace identifier in file
  integer(hid_t) :: memspace      ! dataspace identifier in memory
  integer(hid_t) :: plist_id      ! property list identifier
  integer(hid_t) :: grp_id        ! group identifier
  integer(hid_t) :: time_id       ! attribute identifier
  integer(hid_t) :: mpsi_id       ! attribute identifier
  integer(hid_t) :: npe_id        ! attribute identifier
  integer(hid_t) :: mtheta_id     ! attribute identifier
  integer(hid_t) :: radial_id     ! attribute identifier
  integer(hid_t) :: itran_id      ! attribute identifier
  integer(hid_t) :: aspace_id     ! attribute dataspace identifier
  integer(hid_t) :: a1dspace_id   ! attribute dataspace identifier
  integer     :: ierr             ! return error flag
  integer     :: comm,info
  real        :: curr_time
  character(len=20),save :: fdum
  character(len=20) :: vdum
  integer :: rank
  integer(hsize_t), dimension(2) :: count,dimsf
  integer(hssize_t), dimension(2) :: offset
  integer, dimension(7) :: dimsfi = (/0,0,0,0,0,0,0/)

  comm = mpi_comm_world
  info = mpi_info_null


! this routine is called when idiag=0, right after the first runge-kutta
! half step in pushi (irk=1). for this reason, we need to use zion0()
! instead of zion(). at this point, the array zion0() holds the particle
! quantities at the end of the previous full step (irk=2). zion0() is
! also used in pushi at irk=1 to do the particle diagnostics, so using
! the same array ensures synchronization between the ouput files.

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
! we transform the coordinates of the particle, which are in magnetic
! coordinates (psi,theta,zeta), to x, y, z coordinates.
!
!       r=sqrt(2.0*zion(1,m))
!       q=q0+q1*r/a+q2*r*r/(a*a)
  npp=0
  do m=1,mi
     if(zion0(7,m)>0.0)then
       npp=npp+1
       r=sqrt(2.0*zion0(1,m))
       q=q0+q1*r/a+q2*r*r/(a*a)
       zeta=zion0(3,m)
       !!!theta=zion0(2,m)
       !!!theta0=theta+r*sin(theta)
       theta0=zion0(2,m)
       x=cos(zeta)*(1.0+r*cos(theta0))
       y=sin(zeta)*(1.0+r*cos(theta0))
       z=r*sin(theta0)
       ptracked(1,npp,1)=x
       ptracked(2,npp,1)=y
       ptracked(3,npp,1)=z
       ptracked(4,npp,1)=zion0(4,m)
       ptracked(5,npp,1)=zion0(5,m)
       ptracked(6,npp,1)=zion(6,m)
       ptracked(7,npp,1)=zion(7,m)
     endif
  enddo

! we need to know how many tracked particles each process has. this will
! be required to calculate the offset in the hdf5 file.
  ntpart=0
  call mpi_allgather(npp,1,mpi_integer,ntpart,1,mpi_integer,comm,ierr)

  if(mod((istep-ndiag),isnap)==0)then
   ! we open a new hdf5 file at the first diagnostic call after an output
   ! to the snapshot files.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_1 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! filename for grid quantities
     write(fdum,'("trackp_",i0,".h5")')(1+(mstepall+istep-ndiag)/isnap)

   ! create the file collectively.
     call h5fcreate_f(fdum,h5f_acc_trunc_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  else
   ! reopen the hdf5 file used during the last call to this subroutine.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_1 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! reopen the file collectively.
     call h5fopen_f(fdum,h5f_acc_rdwr_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  endif

! we now write the current calculation time as an attribute to the
! dataset. we need create the attribute data space, data type, etc.
! only one processor needs to write the attribute.

! create the data space for the 2d dataset:
!  1st dim: 7 quantities = particle phase space coordinates + identifier
!  2nd dim: number of tracked particles
  rank=2
  dimsf(1)=nparam
  dimsf(2)=nptrack
  call h5screate_simple_f(rank,dimsf,filespace,ierr)

! write dataset name into variable
  write(vdum,'("particles_t",i0)')(mstepall+istep)/ndiag

! create the dataset with default properties.
  call h5dcreate_f(file_id,vdum,h5t_native_real,filespace,dset_id,ierr)
  call h5sclose_f(filespace,ierr)

! create scalar data space for the time attribute that will be attached to
! the particle dataset.
  call h5screate_f(h5s_scalar_f,aspace_id,ierr)

! create dataset attribute
  call h5acreate_f(dset_id,"time",h5t_native_real,aspace_id,time_id,ierr)

  curr_time=real(mstepall+istep)*tstep  ! attribute data

! write the time attribute data. here dimsfi is just a dummy argument.
  if(mype.eq.0)then
    call h5awrite_f(time_id,h5t_native_real,curr_time,dimsfi,ierr)
  endif

! release attribute id, attribute data space
  call h5aclose_f(time_id,ierr)
  call h5sclose_f(aspace_id,ierr)

! each process defines a dataset in memory and writes it to the hyperslab
! in the file.
! first dimension is the particle quantities.
  count(1)=nparam
  offset(1)=0

! second dimension is the number of tracked particles.
! the offset is determine by the sum of tracked particles on pes < mype
  ipp=0
  do i=0,mype-1
     ipp=ipp+ntpart(i)
  enddo
  count(2)=npp
  offset(2)=ipp

  dimsfi(1)=count(1)
  dimsfi(2)=count(2)

! only the processes holding tracked particles participate in the output
! to the hdf5 file:
  if(npp > 0)then
    call h5screate_simple_f(rank,count,memspace,ierr)

  ! select hyperslab in the file.
    call h5dget_space_f(dset_id,filespace,ierr)
    call h5sselect_hyperslab_f(filespace,h5s_select_set_f,offset,count,ierr)
  
  ! create property list for collective dataset write
    call h5pcreate_f(h5p_dataset_xfer_f, plist_id, ierr)
  !!!  call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_collective_f, ierr)
    call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_independent_f, ierr)

  ! write the dataset
    call h5dwrite_f(dset_id,h5t_native_real,ptracked, dimsfi,ierr, &
         file_space_id=filespace,mem_space_id=memspace,xfer_prp=plist_id)

  ! close property list and dataspaces.
    call h5pclose_f(plist_id,ierr)
    call h5sclose_f(filespace,ierr)
    call h5sclose_f(memspace,ierr)
  endif

! close dataset.
  call h5dclose_f(dset_id,ierr)

! close hdf5 file to insure that we do not loose the hdf5 file if the code 
! is terminated abruptly.
  call h5fclose_f(file_id,ierr)

! close the hdf5 library and the fortran90 interface
  call h5close_f(ierr)

#else
  if(istep==ndiag)write(0,*)mype,' **** the code was compiled without hdf5 support ***'
#endif
end subroutine hdf5out_tracked_particles_1
!========================================

subroutine tag_particles_2

!========================================

  use global_parameters_2
  use particle_array_2
  use particle_tracking_2
  implicit none
  integer :: m,np

! we tag each particle with a unique number that will be carried along
! with it as it moves from one processor/domain to another. to facilitate
! the search for the tracked particles, we give the non-tracked particles
! a negative value.
  do m=1,mi
     zion(7,m)=-real(m+mype*mi)
     zion0(7,m)=zion(7,m)
  enddo
  if(mype==0)then
  ! on the master process (mype=0), we pick "nptrack" particles that will
  ! be followed at every time step. to facilitate the search of those
  ! particles among all the others in the zion arrays of each processor,
  ! we give them a positive number from 1 to nptrack. the particles are
  ! picked around r=0.5, which is mi/2.
    np=0
    do m=(mi-nptrack)/2,(mi+nptrack)/2-1
       np=np+1
       zion(7,m)=np
       zion0(7,m)=np
    enddo
  endif

end subroutine tag_particles_2

!========================================

subroutine locate_tracked_particles_2

!========================================

  use global_parameters_2
  use particle_array_2
  use particle_tracking_2
  implicit none
  integer :: i,m,npp,iout

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
  iout=mod(istep,isnap)
  if(iout==0)iout=isnap
  npp=0
  do m=1,mi
     if(zion(7,m)>0.0)then
       npp=npp+1
       ptracked(1:nparam,npp,iout)=zion(1:nparam,m)
     endif
  enddo
  ntrackp(iout)=npp

end subroutine locate_tracked_particles_2

!========================================

subroutine write_tracked_particles_2

!========================================

  use global_parameters_2
  use particle_tracking_2
  implicit none
 
  integer :: i,j
  character(len=10) :: cdum

  if(mype < 10)then
     write(cdum,'("trackp.00",i1)')mype
  elseif(mype < 100)then
     write(cdum,'("trackp.0",i2)')mype
  else
     write(cdum,'("trackp.",i3)')mype
  endif

  open(57,file=cdum,status='unknown',position='append')
  write(57,*)istep,isnap,nptrack
  write(57,*)ntrackp
  do i=1,isnap
     do j=1,ntrackp(i)
        write(57,*)ptracked(1:nparam,j,i)
     enddo
  enddo
  close(57)

end subroutine write_tracked_particles_2

!========================================

subroutine hdf5out_tracked_particles_2

!========================================
  use global_parameters_2
#ifdef __hdf5
  use particle_array_2
  use particle_tracking_2
  use hdf5
  implicit none

  integer  :: i,j,m,ipp,npp,ntpart(0:numberpe-1)
  real     :: r,q,theta,theta0,zeta,x,y,z

! hdf5 declarations
  integer(hid_t),save :: file_id       ! file identifier
  integer(hid_t) :: dset_id       ! dataset identifier
  integer(hid_t) :: filespace     ! dataspace identifier in file
  integer(hid_t) :: memspace      ! dataspace identifier in memory
  integer(hid_t) :: plist_id      ! property list identifier
  integer(hid_t) :: grp_id        ! group identifier
  integer(hid_t) :: time_id       ! attribute identifier
  integer(hid_t) :: mpsi_id       ! attribute identifier
  integer(hid_t) :: npe_id        ! attribute identifier
  integer(hid_t) :: mtheta_id     ! attribute identifier
  integer(hid_t) :: radial_id     ! attribute identifier
  integer(hid_t) :: itran_id      ! attribute identifier
  integer(hid_t) :: aspace_id     ! attribute dataspace identifier
  integer(hid_t) :: a1dspace_id   ! attribute dataspace identifier
  integer     :: ierr             ! return error flag
  integer     :: comm,info
  real        :: curr_time
  character(len=20),save :: fdum
  character(len=20) :: vdum
  integer :: rank
  integer(hsize_t), dimension(2) :: count,dimsf
  integer(hssize_t), dimension(2) :: offset
  integer, dimension(7) :: dimsfi = (/0,0,0,0,0,0,0/)

  comm = mpi_comm_world
  info = mpi_info_null


! this routine is called when idiag=0, right after the first runge-kutta
! half step in pushi (irk=1). for this reason, we need to use zion0()
! instead of zion(). at this point, the array zion0() holds the particle
! quantities at the end of the previous full step (irk=2). zion0() is
! also used in pushi at irk=1 to do the particle diagnostics, so using
! the same array ensures synchronization between the ouput files.

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
! we transform the coordinates of the particle, which are in magnetic
! coordinates (psi,theta,zeta), to x, y, z coordinates.
!
!       r=sqrt(2.0*zion(1,m))
!       q=q0+q1*r/a+q2*r*r/(a*a)
  npp=0
  do m=1,mi
     if(zion0(7,m)>0.0)then
       npp=npp+1
       r=sqrt(2.0*zion0(1,m))
       q=q0+q1*r/a+q2*r*r/(a*a)
       zeta=zion0(3,m)
       !!!theta=zion0(2,m)
       !!!theta0=theta+r*sin(theta)
       theta0=zion0(2,m)
       x=cos(zeta)*(1.0+r*cos(theta0))
       y=sin(zeta)*(1.0+r*cos(theta0))
       z=r*sin(theta0)
       ptracked(1,npp,1)=x
       ptracked(2,npp,1)=y
       ptracked(3,npp,1)=z
       ptracked(4,npp,1)=zion0(4,m)
       ptracked(5,npp,1)=zion0(5,m)
       ptracked(6,npp,1)=zion(6,m)
       ptracked(7,npp,1)=zion(7,m)
     endif
  enddo

! we need to know how many tracked particles each process has. this will
! be required to calculate the offset in the hdf5 file.
  ntpart=0
  call mpi_allgather(npp,1,mpi_integer,ntpart,1,mpi_integer,comm,ierr)

  if(mod((istep-ndiag),isnap)==0)then
   ! we open a new hdf5 file at the first diagnostic call after an output
   ! to the snapshot files.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_2 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! filename for grid quantities
     write(fdum,'("trackp_",i0,".h5")')(1+(mstepall+istep-ndiag)/isnap)

   ! create the file collectively.
     call h5fcreate_f(fdum,h5f_acc_trunc_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  else
   ! reopen the hdf5 file used during the last call to this subroutine.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_2 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! reopen the file collectively.
     call h5fopen_f(fdum,h5f_acc_rdwr_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  endif

! we now write the current calculation time as an attribute to the
! dataset. we need create the attribute data space, data type, etc.
! only one processor needs to write the attribute.

! create the data space for the 2d dataset:
!  1st dim: 7 quantities = particle phase space coordinates + identifier
!  2nd dim: number of tracked particles
  rank=2
  dimsf(1)=nparam
  dimsf(2)=nptrack
  call h5screate_simple_f(rank,dimsf,filespace,ierr)

! write dataset name into variable
  write(vdum,'("particles_t",i0)')(mstepall+istep)/ndiag

! create the dataset with default properties.
  call h5dcreate_f(file_id,vdum,h5t_native_real,filespace,dset_id,ierr)
  call h5sclose_f(filespace,ierr)

! create scalar data space for the time attribute that will be attached to
! the particle dataset.
  call h5screate_f(h5s_scalar_f,aspace_id,ierr)

! create dataset attribute
  call h5acreate_f(dset_id,"time",h5t_native_real,aspace_id,time_id,ierr)

  curr_time=real(mstepall+istep)*tstep  ! attribute data

! write the time attribute data. here dimsfi is just a dummy argument.
  if(mype.eq.0)then
    call h5awrite_f(time_id,h5t_native_real,curr_time,dimsfi,ierr)
  endif

! release attribute id, attribute data space
  call h5aclose_f(time_id,ierr)
  call h5sclose_f(aspace_id,ierr)

! each process defines a dataset in memory and writes it to the hyperslab
! in the file.
! first dimension is the particle quantities.
  count(1)=nparam
  offset(1)=0

! second dimension is the number of tracked particles.
! the offset is determine by the sum of tracked particles on pes < mype
  ipp=0
  do i=0,mype-1
     ipp=ipp+ntpart(i)
  enddo
  count(2)=npp
  offset(2)=ipp

  dimsfi(1)=count(1)
  dimsfi(2)=count(2)

! only the processes holding tracked particles participate in the output
! to the hdf5 file:
  if(npp > 0)then
    call h5screate_simple_f(rank,count,memspace,ierr)

  ! select hyperslab in the file.
    call h5dget_space_f(dset_id,filespace,ierr)
    call h5sselect_hyperslab_f(filespace,h5s_select_set_f,offset,count,ierr)
  
  ! create property list for collective dataset write
    call h5pcreate_f(h5p_dataset_xfer_f, plist_id, ierr)
  !!!  call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_collective_f, ierr)
    call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_independent_f, ierr)

  ! write the dataset
    call h5dwrite_f(dset_id,h5t_native_real,ptracked, dimsfi,ierr, &
         file_space_id=filespace,mem_space_id=memspace,xfer_prp=plist_id)

  ! close property list and dataspaces.
    call h5pclose_f(plist_id,ierr)
    call h5sclose_f(filespace,ierr)
    call h5sclose_f(memspace,ierr)
  endif

! close dataset.
  call h5dclose_f(dset_id,ierr)

! close hdf5 file to insure that we do not loose the hdf5 file if the code 
! is terminated abruptly.
  call h5fclose_f(file_id,ierr)

! close the hdf5 library and the fortran90 interface
  call h5close_f(ierr)

#else
  if(istep==ndiag)write(0,*)mype,' **** the code was compiled without hdf5 support ***'
#endif
end subroutine hdf5out_tracked_particles_2
!========================================

subroutine tag_particles_3

!========================================

  use global_parameters_3
  use particle_array_3
  use particle_tracking_3
  implicit none
  integer :: m,np

! we tag each particle with a unique number that will be carried along
! with it as it moves from one processor/domain to another. to facilitate
! the search for the tracked particles, we give the non-tracked particles
! a negative value.
  do m=1,mi
     zion(7,m)=-real(m+mype*mi)
     zion0(7,m)=zion(7,m)
  enddo
  if(mype==0)then
  ! on the master process (mype=0), we pick "nptrack" particles that will
  ! be followed at every time step. to facilitate the search of those
  ! particles among all the others in the zion arrays of each processor,
  ! we give them a positive number from 1 to nptrack. the particles are
  ! picked around r=0.5, which is mi/2.
    np=0
    do m=(mi-nptrack)/2,(mi+nptrack)/2-1
       np=np+1
       zion(7,m)=np
       zion0(7,m)=np
    enddo
  endif

end subroutine tag_particles_3

!========================================

subroutine locate_tracked_particles_3

!========================================

  use global_parameters_3
  use particle_array_3
  use particle_tracking_3
  implicit none
  integer :: i,m,npp,iout

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
  iout=mod(istep,isnap)
  if(iout==0)iout=isnap
  npp=0
  do m=1,mi
     if(zion(7,m)>0.0)then
       npp=npp+1
       ptracked(1:nparam,npp,iout)=zion(1:nparam,m)
     endif
  enddo
  ntrackp(iout)=npp

end subroutine locate_tracked_particles_3

!========================================

subroutine write_tracked_particles_3

!========================================

  use global_parameters_3
  use particle_tracking_3
  implicit none
 
  integer :: i,j
  character(len=10) :: cdum

  if(mype < 10)then
     write(cdum,'("trackp.00",i1)')mype
  elseif(mype < 100)then
     write(cdum,'("trackp.0",i2)')mype
  else
     write(cdum,'("trackp.",i3)')mype
  endif

  open(57,file=cdum,status='unknown',position='append')
  write(57,*)istep,isnap,nptrack
  write(57,*)ntrackp
  do i=1,isnap
     do j=1,ntrackp(i)
        write(57,*)ptracked(1:nparam,j,i)
     enddo
  enddo
  close(57)

end subroutine write_tracked_particles_3

!========================================

subroutine hdf5out_tracked_particles_3

!========================================
  use global_parameters_3
#ifdef __hdf5
  use particle_array_3
  use particle_tracking_3
  use hdf5
  implicit none

  integer  :: i,j,m,ipp,npp,ntpart(0:numberpe-1)
  real     :: r,q,theta,theta0,zeta,x,y,z

! hdf5 declarations
  integer(hid_t),save :: file_id       ! file identifier
  integer(hid_t) :: dset_id       ! dataset identifier
  integer(hid_t) :: filespace     ! dataspace identifier in file
  integer(hid_t) :: memspace      ! dataspace identifier in memory
  integer(hid_t) :: plist_id      ! property list identifier
  integer(hid_t) :: grp_id        ! group identifier
  integer(hid_t) :: time_id       ! attribute identifier
  integer(hid_t) :: mpsi_id       ! attribute identifier
  integer(hid_t) :: npe_id        ! attribute identifier
  integer(hid_t) :: mtheta_id     ! attribute identifier
  integer(hid_t) :: radial_id     ! attribute identifier
  integer(hid_t) :: itran_id      ! attribute identifier
  integer(hid_t) :: aspace_id     ! attribute dataspace identifier
  integer(hid_t) :: a1dspace_id   ! attribute dataspace identifier
  integer     :: ierr             ! return error flag
  integer     :: comm,info
  real        :: curr_time
  character(len=20),save :: fdum
  character(len=20) :: vdum
  integer :: rank
  integer(hsize_t), dimension(2) :: count,dimsf
  integer(hssize_t), dimension(2) :: offset
  integer, dimension(7) :: dimsfi = (/0,0,0,0,0,0,0/)

  comm = mpi_comm_world
  info = mpi_info_null


! this routine is called when idiag=0, right after the first runge-kutta
! half step in pushi (irk=1). for this reason, we need to use zion0()
! instead of zion(). at this point, the array zion0() holds the particle
! quantities at the end of the previous full step (irk=2). zion0() is
! also used in pushi at irk=1 to do the particle diagnostics, so using
! the same array ensures synchronization between the ouput files.

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
! we transform the coordinates of the particle, which are in magnetic
! coordinates (psi,theta,zeta), to x, y, z coordinates.
!
!       r=sqrt(2.0*zion(1,m))
!       q=q0+q1*r/a+q2*r*r/(a*a)
  npp=0
  do m=1,mi
     if(zion0(7,m)>0.0)then
       npp=npp+1
       r=sqrt(2.0*zion0(1,m))
       q=q0+q1*r/a+q2*r*r/(a*a)
       zeta=zion0(3,m)
       !!!theta=zion0(2,m)
       !!!theta0=theta+r*sin(theta)
       theta0=zion0(2,m)
       x=cos(zeta)*(1.0+r*cos(theta0))
       y=sin(zeta)*(1.0+r*cos(theta0))
       z=r*sin(theta0)
       ptracked(1,npp,1)=x
       ptracked(2,npp,1)=y
       ptracked(3,npp,1)=z
       ptracked(4,npp,1)=zion0(4,m)
       ptracked(5,npp,1)=zion0(5,m)
       ptracked(6,npp,1)=zion(6,m)
       ptracked(7,npp,1)=zion(7,m)
     endif
  enddo

! we need to know how many tracked particles each process has. this will
! be required to calculate the offset in the hdf5 file.
  ntpart=0
  call mpi_allgather(npp,1,mpi_integer,ntpart,1,mpi_integer,comm,ierr)

  if(mod((istep-ndiag),isnap)==0)then
   ! we open a new hdf5 file at the first diagnostic call after an output
   ! to the snapshot files.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_3 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! filename for grid quantities
     write(fdum,'("trackp_",i0,".h5")')(1+(mstepall+istep-ndiag)/isnap)

   ! create the file collectively.
     call h5fcreate_f(fdum,h5f_acc_trunc_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  else
   ! reopen the hdf5 file used during the last call to this subroutine.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_3 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! reopen the file collectively.
     call h5fopen_f(fdum,h5f_acc_rdwr_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  endif

! we now write the current calculation time as an attribute to the
! dataset. we need create the attribute data space, data type, etc.
! only one processor needs to write the attribute.

! create the data space for the 2d dataset:
!  1st dim: 7 quantities = particle phase space coordinates + identifier
!  2nd dim: number of tracked particles
  rank=2
  dimsf(1)=nparam
  dimsf(2)=nptrack
  call h5screate_simple_f(rank,dimsf,filespace,ierr)

! write dataset name into variable
  write(vdum,'("particles_t",i0)')(mstepall+istep)/ndiag

! create the dataset with default properties.
  call h5dcreate_f(file_id,vdum,h5t_native_real,filespace,dset_id,ierr)
  call h5sclose_f(filespace,ierr)

! create scalar data space for the time attribute that will be attached to
! the particle dataset.
  call h5screate_f(h5s_scalar_f,aspace_id,ierr)

! create dataset attribute
  call h5acreate_f(dset_id,"time",h5t_native_real,aspace_id,time_id,ierr)

  curr_time=real(mstepall+istep)*tstep  ! attribute data

! write the time attribute data. here dimsfi is just a dummy argument.
  if(mype.eq.0)then
    call h5awrite_f(time_id,h5t_native_real,curr_time,dimsfi,ierr)
  endif

! release attribute id, attribute data space
  call h5aclose_f(time_id,ierr)
  call h5sclose_f(aspace_id,ierr)

! each process defines a dataset in memory and writes it to the hyperslab
! in the file.
! first dimension is the particle quantities.
  count(1)=nparam
  offset(1)=0

! second dimension is the number of tracked particles.
! the offset is determine by the sum of tracked particles on pes < mype
  ipp=0
  do i=0,mype-1
     ipp=ipp+ntpart(i)
  enddo
  count(2)=npp
  offset(2)=ipp

  dimsfi(1)=count(1)
  dimsfi(2)=count(2)

! only the processes holding tracked particles participate in the output
! to the hdf5 file:
  if(npp > 0)then
    call h5screate_simple_f(rank,count,memspace,ierr)

  ! select hyperslab in the file.
    call h5dget_space_f(dset_id,filespace,ierr)
    call h5sselect_hyperslab_f(filespace,h5s_select_set_f,offset,count,ierr)
  
  ! create property list for collective dataset write
    call h5pcreate_f(h5p_dataset_xfer_f, plist_id, ierr)
  !!!  call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_collective_f, ierr)
    call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_independent_f, ierr)

  ! write the dataset
    call h5dwrite_f(dset_id,h5t_native_real,ptracked, dimsfi,ierr, &
         file_space_id=filespace,mem_space_id=memspace,xfer_prp=plist_id)

  ! close property list and dataspaces.
    call h5pclose_f(plist_id,ierr)
    call h5sclose_f(filespace,ierr)
    call h5sclose_f(memspace,ierr)
  endif

! close dataset.
  call h5dclose_f(dset_id,ierr)

! close hdf5 file to insure that we do not loose the hdf5 file if the code 
! is terminated abruptly.
  call h5fclose_f(file_id,ierr)

! close the hdf5 library and the fortran90 interface
  call h5close_f(ierr)

#else
  if(istep==ndiag)write(0,*)mype,' **** the code was compiled without hdf5 support ***'
#endif
end subroutine hdf5out_tracked_particles_3
!========================================

subroutine tag_particles_4

!========================================

  use global_parameters_4
  use particle_array_4
  use particle_tracking_4
  implicit none
  integer :: m,np

! we tag each particle with a unique number that will be carried along
! with it as it moves from one processor/domain to another. to facilitate
! the search for the tracked particles, we give the non-tracked particles
! a negative value.
  do m=1,mi
     zion(7,m)=-real(m+mype*mi)
     zion0(7,m)=zion(7,m)
  enddo
  if(mype==0)then
  ! on the master process (mype=0), we pick "nptrack" particles that will
  ! be followed at every time step. to facilitate the search of those
  ! particles among all the others in the zion arrays of each processor,
  ! we give them a positive number from 1 to nptrack. the particles are
  ! picked around r=0.5, which is mi/2.
    np=0
    do m=(mi-nptrack)/2,(mi+nptrack)/2-1
       np=np+1
       zion(7,m)=np
       zion0(7,m)=np
    enddo
  endif

end subroutine tag_particles_4

!========================================

subroutine locate_tracked_particles_4

!========================================

  use global_parameters_4
  use particle_array_4
  use particle_tracking_4
  implicit none
  integer :: i,m,npp,iout

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
  iout=mod(istep,isnap)
  if(iout==0)iout=isnap
  npp=0
  do m=1,mi
     if(zion(7,m)>0.0)then
       npp=npp+1
       ptracked(1:nparam,npp,iout)=zion(1:nparam,m)
     endif
  enddo
  ntrackp(iout)=npp

end subroutine locate_tracked_particles_4

!========================================

subroutine write_tracked_particles_4

!========================================

  use global_parameters_4
  use particle_tracking_4
  implicit none
 
  integer :: i,j
  character(len=10) :: cdum

  if(mype < 10)then
     write(cdum,'("trackp.00",i1)')mype
  elseif(mype < 100)then
     write(cdum,'("trackp.0",i2)')mype
  else
     write(cdum,'("trackp.",i3)')mype
  endif

  open(57,file=cdum,status='unknown',position='append')
  write(57,*)istep,isnap,nptrack
  write(57,*)ntrackp
  do i=1,isnap
     do j=1,ntrackp(i)
        write(57,*)ptracked(1:nparam,j,i)
     enddo
  enddo
  close(57)

end subroutine write_tracked_particles_4

!========================================

subroutine hdf5out_tracked_particles_4

!========================================
  use global_parameters_4
#ifdef __hdf5
  use particle_array_4
  use particle_tracking_4
  use hdf5
  implicit none

  integer  :: i,j,m,ipp,npp,ntpart(0:numberpe-1)
  real     :: r,q,theta,theta0,zeta,x,y,z

! hdf5 declarations
  integer(hid_t),save :: file_id       ! file identifier
  integer(hid_t) :: dset_id       ! dataset identifier
  integer(hid_t) :: filespace     ! dataspace identifier in file
  integer(hid_t) :: memspace      ! dataspace identifier in memory
  integer(hid_t) :: plist_id      ! property list identifier
  integer(hid_t) :: grp_id        ! group identifier
  integer(hid_t) :: time_id       ! attribute identifier
  integer(hid_t) :: mpsi_id       ! attribute identifier
  integer(hid_t) :: npe_id        ! attribute identifier
  integer(hid_t) :: mtheta_id     ! attribute identifier
  integer(hid_t) :: radial_id     ! attribute identifier
  integer(hid_t) :: itran_id      ! attribute identifier
  integer(hid_t) :: aspace_id     ! attribute dataspace identifier
  integer(hid_t) :: a1dspace_id   ! attribute dataspace identifier
  integer     :: ierr             ! return error flag
  integer     :: comm,info
  real        :: curr_time
  character(len=20),save :: fdum
  character(len=20) :: vdum
  integer :: rank
  integer(hsize_t), dimension(2) :: count,dimsf
  integer(hssize_t), dimension(2) :: offset
  integer, dimension(7) :: dimsfi = (/0,0,0,0,0,0,0/)

  comm = mpi_comm_world
  info = mpi_info_null


! this routine is called when idiag=0, right after the first runge-kutta
! half step in pushi (irk=1). for this reason, we need to use zion0()
! instead of zion(). at this point, the array zion0() holds the particle
! quantities at the end of the previous full step (irk=2). zion0() is
! also used in pushi at irk=1 to do the particle diagnostics, so using
! the same array ensures synchronization between the ouput files.

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
! we transform the coordinates of the particle, which are in magnetic
! coordinates (psi,theta,zeta), to x, y, z coordinates.
!
!       r=sqrt(2.0*zion(1,m))
!       q=q0+q1*r/a+q2*r*r/(a*a)
  npp=0
  do m=1,mi
     if(zion0(7,m)>0.0)then
       npp=npp+1
       r=sqrt(2.0*zion0(1,m))
       q=q0+q1*r/a+q2*r*r/(a*a)
       zeta=zion0(3,m)
       !!!theta=zion0(2,m)
       !!!theta0=theta+r*sin(theta)
       theta0=zion0(2,m)
       x=cos(zeta)*(1.0+r*cos(theta0))
       y=sin(zeta)*(1.0+r*cos(theta0))
       z=r*sin(theta0)
       ptracked(1,npp,1)=x
       ptracked(2,npp,1)=y
       ptracked(3,npp,1)=z
       ptracked(4,npp,1)=zion0(4,m)
       ptracked(5,npp,1)=zion0(5,m)
       ptracked(6,npp,1)=zion(6,m)
       ptracked(7,npp,1)=zion(7,m)
     endif
  enddo

! we need to know how many tracked particles each process has. this will
! be required to calculate the offset in the hdf5 file.
  ntpart=0
  call mpi_allgather(npp,1,mpi_integer,ntpart,1,mpi_integer,comm,ierr)

  if(mod((istep-ndiag),isnap)==0)then
   ! we open a new hdf5 file at the first diagnostic call after an output
   ! to the snapshot files.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_4 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! filename for grid quantities
     write(fdum,'("trackp_",i0,".h5")')(1+(mstepall+istep-ndiag)/isnap)

   ! create the file collectively.
     call h5fcreate_f(fdum,h5f_acc_trunc_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  else
   ! reopen the hdf5 file used during the last call to this subroutine.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_4 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! reopen the file collectively.
     call h5fopen_f(fdum,h5f_acc_rdwr_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  endif

! we now write the current calculation time as an attribute to the
! dataset. we need create the attribute data space, data type, etc.
! only one processor needs to write the attribute.

! create the data space for the 2d dataset:
!  1st dim: 7 quantities = particle phase space coordinates + identifier
!  2nd dim: number of tracked particles
  rank=2
  dimsf(1)=nparam
  dimsf(2)=nptrack
  call h5screate_simple_f(rank,dimsf,filespace,ierr)

! write dataset name into variable
  write(vdum,'("particles_t",i0)')(mstepall+istep)/ndiag

! create the dataset with default properties.
  call h5dcreate_f(file_id,vdum,h5t_native_real,filespace,dset_id,ierr)
  call h5sclose_f(filespace,ierr)

! create scalar data space for the time attribute that will be attached to
! the particle dataset.
  call h5screate_f(h5s_scalar_f,aspace_id,ierr)

! create dataset attribute
  call h5acreate_f(dset_id,"time",h5t_native_real,aspace_id,time_id,ierr)

  curr_time=real(mstepall+istep)*tstep  ! attribute data

! write the time attribute data. here dimsfi is just a dummy argument.
  if(mype.eq.0)then
    call h5awrite_f(time_id,h5t_native_real,curr_time,dimsfi,ierr)
  endif

! release attribute id, attribute data space
  call h5aclose_f(time_id,ierr)
  call h5sclose_f(aspace_id,ierr)

! each process defines a dataset in memory and writes it to the hyperslab
! in the file.
! first dimension is the particle quantities.
  count(1)=nparam
  offset(1)=0

! second dimension is the number of tracked particles.
! the offset is determine by the sum of tracked particles on pes < mype
  ipp=0
  do i=0,mype-1
     ipp=ipp+ntpart(i)
  enddo
  count(2)=npp
  offset(2)=ipp

  dimsfi(1)=count(1)
  dimsfi(2)=count(2)

! only the processes holding tracked particles participate in the output
! to the hdf5 file:
  if(npp > 0)then
    call h5screate_simple_f(rank,count,memspace,ierr)

  ! select hyperslab in the file.
    call h5dget_space_f(dset_id,filespace,ierr)
    call h5sselect_hyperslab_f(filespace,h5s_select_set_f,offset,count,ierr)
  
  ! create property list for collective dataset write
    call h5pcreate_f(h5p_dataset_xfer_f, plist_id, ierr)
  !!!  call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_collective_f, ierr)
    call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_independent_f, ierr)

  ! write the dataset
    call h5dwrite_f(dset_id,h5t_native_real,ptracked, dimsfi,ierr, &
         file_space_id=filespace,mem_space_id=memspace,xfer_prp=plist_id)

  ! close property list and dataspaces.
    call h5pclose_f(plist_id,ierr)
    call h5sclose_f(filespace,ierr)
    call h5sclose_f(memspace,ierr)
  endif

! close dataset.
  call h5dclose_f(dset_id,ierr)

! close hdf5 file to insure that we do not loose the hdf5 file if the code 
! is terminated abruptly.
  call h5fclose_f(file_id,ierr)

! close the hdf5 library and the fortran90 interface
  call h5close_f(ierr)

#else
  if(istep==ndiag)write(0,*)mype,' **** the code was compiled without hdf5 support ***'
#endif
end subroutine hdf5out_tracked_particles_4
!========================================

subroutine tag_particles_5

!========================================

  use global_parameters_5
  use particle_array_5
  use particle_tracking_5
  implicit none
  integer :: m,np

! we tag each particle with a unique number that will be carried along
! with it as it moves from one processor/domain to another. to facilitate
! the search for the tracked particles, we give the non-tracked particles
! a negative value.
  do m=1,mi
     zion(7,m)=-real(m+mype*mi)
     zion0(7,m)=zion(7,m)
  enddo
  if(mype==0)then
  ! on the master process (mype=0), we pick "nptrack" particles that will
  ! be followed at every time step. to facilitate the search of those
  ! particles among all the others in the zion arrays of each processor,
  ! we give them a positive number from 1 to nptrack. the particles are
  ! picked around r=0.5, which is mi/2.
    np=0
    do m=(mi-nptrack)/2,(mi+nptrack)/2-1
       np=np+1
       zion(7,m)=np
       zion0(7,m)=np
    enddo
  endif

end subroutine tag_particles_5

!========================================

subroutine locate_tracked_particles_5

!========================================

  use global_parameters_5
  use particle_array_5
  use particle_tracking_5
  implicit none
  integer :: i,m,npp,iout

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
  iout=mod(istep,isnap)
  if(iout==0)iout=isnap
  npp=0
  do m=1,mi
     if(zion(7,m)>0.0)then
       npp=npp+1
       ptracked(1:nparam,npp,iout)=zion(1:nparam,m)
     endif
  enddo
  ntrackp(iout)=npp

end subroutine locate_tracked_particles_5

!========================================

subroutine write_tracked_particles_5

!========================================

  use global_parameters_5
  use particle_tracking_5
  implicit none
 
  integer :: i,j
  character(len=10) :: cdum

  if(mype < 10)then
     write(cdum,'("trackp.00",i1)')mype
  elseif(mype < 100)then
     write(cdum,'("trackp.0",i2)')mype
  else
     write(cdum,'("trackp.",i3)')mype
  endif

  open(57,file=cdum,status='unknown',position='append')
  write(57,*)istep,isnap,nptrack
  write(57,*)ntrackp
  do i=1,isnap
     do j=1,ntrackp(i)
        write(57,*)ptracked(1:nparam,j,i)
     enddo
  enddo
  close(57)

end subroutine write_tracked_particles_5

!========================================

subroutine hdf5out_tracked_particles_5

!========================================
  use global_parameters_5
#ifdef __hdf5
  use particle_array_5
  use particle_tracking_5
  use hdf5
  implicit none

  integer  :: i,j,m,ipp,npp,ntpart(0:numberpe-1)
  real     :: r,q,theta,theta0,zeta,x,y,z

! hdf5 declarations
  integer(hid_t),save :: file_id       ! file identifier
  integer(hid_t) :: dset_id       ! dataset identifier
  integer(hid_t) :: filespace     ! dataspace identifier in file
  integer(hid_t) :: memspace      ! dataspace identifier in memory
  integer(hid_t) :: plist_id      ! property list identifier
  integer(hid_t) :: grp_id        ! group identifier
  integer(hid_t) :: time_id       ! attribute identifier
  integer(hid_t) :: mpsi_id       ! attribute identifier
  integer(hid_t) :: npe_id        ! attribute identifier
  integer(hid_t) :: mtheta_id     ! attribute identifier
  integer(hid_t) :: radial_id     ! attribute identifier
  integer(hid_t) :: itran_id      ! attribute identifier
  integer(hid_t) :: aspace_id     ! attribute dataspace identifier
  integer(hid_t) :: a1dspace_id   ! attribute dataspace identifier
  integer     :: ierr             ! return error flag
  integer     :: comm,info
  real        :: curr_time
  character(len=20),save :: fdum
  character(len=20) :: vdum
  integer :: rank
  integer(hsize_t), dimension(2) :: count,dimsf
  integer(hssize_t), dimension(2) :: offset
  integer, dimension(7) :: dimsfi = (/0,0,0,0,0,0,0/)

  comm = mpi_comm_world
  info = mpi_info_null


! this routine is called when idiag=0, right after the first runge-kutta
! half step in pushi (irk=1). for this reason, we need to use zion0()
! instead of zion(). at this point, the array zion0() holds the particle
! quantities at the end of the previous full step (irk=2). zion0() is
! also used in pushi at irk=1 to do the particle diagnostics, so using
! the same array ensures synchronization between the ouput files.

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
! we transform the coordinates of the particle, which are in magnetic
! coordinates (psi,theta,zeta), to x, y, z coordinates.
!
!       r=sqrt(2.0*zion(1,m))
!       q=q0+q1*r/a+q2*r*r/(a*a)
  npp=0
  do m=1,mi
     if(zion0(7,m)>0.0)then
       npp=npp+1
       r=sqrt(2.0*zion0(1,m))
       q=q0+q1*r/a+q2*r*r/(a*a)
       zeta=zion0(3,m)
       !!!theta=zion0(2,m)
       !!!theta0=theta+r*sin(theta)
       theta0=zion0(2,m)
       x=cos(zeta)*(1.0+r*cos(theta0))
       y=sin(zeta)*(1.0+r*cos(theta0))
       z=r*sin(theta0)
       ptracked(1,npp,1)=x
       ptracked(2,npp,1)=y
       ptracked(3,npp,1)=z
       ptracked(4,npp,1)=zion0(4,m)
       ptracked(5,npp,1)=zion0(5,m)
       ptracked(6,npp,1)=zion(6,m)
       ptracked(7,npp,1)=zion(7,m)
     endif
  enddo

! we need to know how many tracked particles each process has. this will
! be required to calculate the offset in the hdf5 file.
  ntpart=0
  call mpi_allgather(npp,1,mpi_integer,ntpart,1,mpi_integer,comm,ierr)

  if(mod((istep-ndiag),isnap)==0)then
   ! we open a new hdf5 file at the first diagnostic call after an output
   ! to the snapshot files.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_5 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! filename for grid quantities
     write(fdum,'("trackp_",i0,".h5")')(1+(mstepall+istep-ndiag)/isnap)

   ! create the file collectively.
     call h5fcreate_f(fdum,h5f_acc_trunc_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  else
   ! reopen the hdf5 file used during the last call to this subroutine.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_5 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! reopen the file collectively.
     call h5fopen_f(fdum,h5f_acc_rdwr_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  endif

! we now write the current calculation time as an attribute to the
! dataset. we need create the attribute data space, data type, etc.
! only one processor needs to write the attribute.

! create the data space for the 2d dataset:
!  1st dim: 7 quantities = particle phase space coordinates + identifier
!  2nd dim: number of tracked particles
  rank=2
  dimsf(1)=nparam
  dimsf(2)=nptrack
  call h5screate_simple_f(rank,dimsf,filespace,ierr)

! write dataset name into variable
  write(vdum,'("particles_t",i0)')(mstepall+istep)/ndiag

! create the dataset with default properties.
  call h5dcreate_f(file_id,vdum,h5t_native_real,filespace,dset_id,ierr)
  call h5sclose_f(filespace,ierr)

! create scalar data space for the time attribute that will be attached to
! the particle dataset.
  call h5screate_f(h5s_scalar_f,aspace_id,ierr)

! create dataset attribute
  call h5acreate_f(dset_id,"time",h5t_native_real,aspace_id,time_id,ierr)

  curr_time=real(mstepall+istep)*tstep  ! attribute data

! write the time attribute data. here dimsfi is just a dummy argument.
  if(mype.eq.0)then
    call h5awrite_f(time_id,h5t_native_real,curr_time,dimsfi,ierr)
  endif

! release attribute id, attribute data space
  call h5aclose_f(time_id,ierr)
  call h5sclose_f(aspace_id,ierr)

! each process defines a dataset in memory and writes it to the hyperslab
! in the file.
! first dimension is the particle quantities.
  count(1)=nparam
  offset(1)=0

! second dimension is the number of tracked particles.
! the offset is determine by the sum of tracked particles on pes < mype
  ipp=0
  do i=0,mype-1
     ipp=ipp+ntpart(i)
  enddo
  count(2)=npp
  offset(2)=ipp

  dimsfi(1)=count(1)
  dimsfi(2)=count(2)

! only the processes holding tracked particles participate in the output
! to the hdf5 file:
  if(npp > 0)then
    call h5screate_simple_f(rank,count,memspace,ierr)

  ! select hyperslab in the file.
    call h5dget_space_f(dset_id,filespace,ierr)
    call h5sselect_hyperslab_f(filespace,h5s_select_set_f,offset,count,ierr)
  
  ! create property list for collective dataset write
    call h5pcreate_f(h5p_dataset_xfer_f, plist_id, ierr)
  !!!  call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_collective_f, ierr)
    call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_independent_f, ierr)

  ! write the dataset
    call h5dwrite_f(dset_id,h5t_native_real,ptracked, dimsfi,ierr, &
         file_space_id=filespace,mem_space_id=memspace,xfer_prp=plist_id)

  ! close property list and dataspaces.
    call h5pclose_f(plist_id,ierr)
    call h5sclose_f(filespace,ierr)
    call h5sclose_f(memspace,ierr)
  endif

! close dataset.
  call h5dclose_f(dset_id,ierr)

! close hdf5 file to insure that we do not loose the hdf5 file if the code 
! is terminated abruptly.
  call h5fclose_f(file_id,ierr)

! close the hdf5 library and the fortran90 interface
  call h5close_f(ierr)

#else
  if(istep==ndiag)write(0,*)mype,' **** the code was compiled without hdf5 support ***'
#endif
end subroutine hdf5out_tracked_particles_5
!========================================

subroutine tag_particles_6

!========================================

  use global_parameters_6
  use particle_array_6
  use particle_tracking_6
  implicit none
  integer :: m,np

! we tag each particle with a unique number that will be carried along
! with it as it moves from one processor/domain to another. to facilitate
! the search for the tracked particles, we give the non-tracked particles
! a negative value.
  do m=1,mi
     zion(7,m)=-real(m+mype*mi)
     zion0(7,m)=zion(7,m)
  enddo
  if(mype==0)then
  ! on the master process (mype=0), we pick "nptrack" particles that will
  ! be followed at every time step. to facilitate the search of those
  ! particles among all the others in the zion arrays of each processor,
  ! we give them a positive number from 1 to nptrack. the particles are
  ! picked around r=0.5, which is mi/2.
    np=0
    do m=(mi-nptrack)/2,(mi+nptrack)/2-1
       np=np+1
       zion(7,m)=np
       zion0(7,m)=np
    enddo
  endif

end subroutine tag_particles_6

!========================================

subroutine locate_tracked_particles_6

!========================================

  use global_parameters_6
  use particle_array_6
  use particle_tracking_6
  implicit none
  integer :: i,m,npp,iout

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
  iout=mod(istep,isnap)
  if(iout==0)iout=isnap
  npp=0
  do m=1,mi
     if(zion(7,m)>0.0)then
       npp=npp+1
       ptracked(1:nparam,npp,iout)=zion(1:nparam,m)
     endif
  enddo
  ntrackp(iout)=npp

end subroutine locate_tracked_particles_6

!========================================

subroutine write_tracked_particles_6

!========================================

  use global_parameters_6
  use particle_tracking_6
  implicit none
 
  integer :: i,j
  character(len=10) :: cdum

  if(mype < 10)then
     write(cdum,'("trackp.00",i1)')mype
  elseif(mype < 100)then
     write(cdum,'("trackp.0",i2)')mype
  else
     write(cdum,'("trackp.",i3)')mype
  endif

  open(57,file=cdum,status='unknown',position='append')
  write(57,*)istep,isnap,nptrack
  write(57,*)ntrackp
  do i=1,isnap
     do j=1,ntrackp(i)
        write(57,*)ptracked(1:nparam,j,i)
     enddo
  enddo
  close(57)

end subroutine write_tracked_particles_6

!========================================

subroutine hdf5out_tracked_particles_6

!========================================
  use global_parameters_6
#ifdef __hdf5
  use particle_array_6
  use particle_tracking_6
  use hdf5
  implicit none

  integer  :: i,j,m,ipp,npp,ntpart(0:numberpe-1)
  real     :: r,q,theta,theta0,zeta,x,y,z

! hdf5 declarations
  integer(hid_t),save :: file_id       ! file identifier
  integer(hid_t) :: dset_id       ! dataset identifier
  integer(hid_t) :: filespace     ! dataspace identifier in file
  integer(hid_t) :: memspace      ! dataspace identifier in memory
  integer(hid_t) :: plist_id      ! property list identifier
  integer(hid_t) :: grp_id        ! group identifier
  integer(hid_t) :: time_id       ! attribute identifier
  integer(hid_t) :: mpsi_id       ! attribute identifier
  integer(hid_t) :: npe_id        ! attribute identifier
  integer(hid_t) :: mtheta_id     ! attribute identifier
  integer(hid_t) :: radial_id     ! attribute identifier
  integer(hid_t) :: itran_id      ! attribute identifier
  integer(hid_t) :: aspace_id     ! attribute dataspace identifier
  integer(hid_t) :: a1dspace_id   ! attribute dataspace identifier
  integer     :: ierr             ! return error flag
  integer     :: comm,info
  real        :: curr_time
  character(len=20),save :: fdum
  character(len=20) :: vdum
  integer :: rank
  integer(hsize_t), dimension(2) :: count,dimsf
  integer(hssize_t), dimension(2) :: offset
  integer, dimension(7) :: dimsfi = (/0,0,0,0,0,0,0/)

  comm = mpi_comm_world
  info = mpi_info_null


! this routine is called when idiag=0, right after the first runge-kutta
! half step in pushi (irk=1). for this reason, we need to use zion0()
! instead of zion(). at this point, the array zion0() holds the particle
! quantities at the end of the previous full step (irk=2). zion0() is
! also used in pushi at irk=1 to do the particle diagnostics, so using
! the same array ensures synchronization between the ouput files.

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
! we transform the coordinates of the particle, which are in magnetic
! coordinates (psi,theta,zeta), to x, y, z coordinates.
!
!       r=sqrt(2.0*zion(1,m))
!       q=q0+q1*r/a+q2*r*r/(a*a)
  npp=0
  do m=1,mi
     if(zion0(7,m)>0.0)then
       npp=npp+1
       r=sqrt(2.0*zion0(1,m))
       q=q0+q1*r/a+q2*r*r/(a*a)
       zeta=zion0(3,m)
       !!!theta=zion0(2,m)
       !!!theta0=theta+r*sin(theta)
       theta0=zion0(2,m)
       x=cos(zeta)*(1.0+r*cos(theta0))
       y=sin(zeta)*(1.0+r*cos(theta0))
       z=r*sin(theta0)
       ptracked(1,npp,1)=x
       ptracked(2,npp,1)=y
       ptracked(3,npp,1)=z
       ptracked(4,npp,1)=zion0(4,m)
       ptracked(5,npp,1)=zion0(5,m)
       ptracked(6,npp,1)=zion(6,m)
       ptracked(7,npp,1)=zion(7,m)
     endif
  enddo

! we need to know how many tracked particles each process has. this will
! be required to calculate the offset in the hdf5 file.
  ntpart=0
  call mpi_allgather(npp,1,mpi_integer,ntpart,1,mpi_integer,comm,ierr)

  if(mod((istep-ndiag),isnap)==0)then
   ! we open a new hdf5 file at the first diagnostic call after an output
   ! to the snapshot files.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_6 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! filename for grid quantities
     write(fdum,'("trackp_",i0,".h5")')(1+(mstepall+istep-ndiag)/isnap)

   ! create the file collectively.
     call h5fcreate_f(fdum,h5f_acc_trunc_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  else
   ! reopen the hdf5 file used during the last call to this subroutine.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_6 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! reopen the file collectively.
     call h5fopen_f(fdum,h5f_acc_rdwr_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  endif

! we now write the current calculation time as an attribute to the
! dataset. we need create the attribute data space, data type, etc.
! only one processor needs to write the attribute.

! create the data space for the 2d dataset:
!  1st dim: 7 quantities = particle phase space coordinates + identifier
!  2nd dim: number of tracked particles
  rank=2
  dimsf(1)=nparam
  dimsf(2)=nptrack
  call h5screate_simple_f(rank,dimsf,filespace,ierr)

! write dataset name into variable
  write(vdum,'("particles_t",i0)')(mstepall+istep)/ndiag

! create the dataset with default properties.
  call h5dcreate_f(file_id,vdum,h5t_native_real,filespace,dset_id,ierr)
  call h5sclose_f(filespace,ierr)

! create scalar data space for the time attribute that will be attached to
! the particle dataset.
  call h5screate_f(h5s_scalar_f,aspace_id,ierr)

! create dataset attribute
  call h5acreate_f(dset_id,"time",h5t_native_real,aspace_id,time_id,ierr)

  curr_time=real(mstepall+istep)*tstep  ! attribute data

! write the time attribute data. here dimsfi is just a dummy argument.
  if(mype.eq.0)then
    call h5awrite_f(time_id,h5t_native_real,curr_time,dimsfi,ierr)
  endif

! release attribute id, attribute data space
  call h5aclose_f(time_id,ierr)
  call h5sclose_f(aspace_id,ierr)

! each process defines a dataset in memory and writes it to the hyperslab
! in the file.
! first dimension is the particle quantities.
  count(1)=nparam
  offset(1)=0

! second dimension is the number of tracked particles.
! the offset is determine by the sum of tracked particles on pes < mype
  ipp=0
  do i=0,mype-1
     ipp=ipp+ntpart(i)
  enddo
  count(2)=npp
  offset(2)=ipp

  dimsfi(1)=count(1)
  dimsfi(2)=count(2)

! only the processes holding tracked particles participate in the output
! to the hdf5 file:
  if(npp > 0)then
    call h5screate_simple_f(rank,count,memspace,ierr)

  ! select hyperslab in the file.
    call h5dget_space_f(dset_id,filespace,ierr)
    call h5sselect_hyperslab_f(filespace,h5s_select_set_f,offset,count,ierr)
  
  ! create property list for collective dataset write
    call h5pcreate_f(h5p_dataset_xfer_f, plist_id, ierr)
  !!!  call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_collective_f, ierr)
    call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_independent_f, ierr)

  ! write the dataset
    call h5dwrite_f(dset_id,h5t_native_real,ptracked, dimsfi,ierr, &
         file_space_id=filespace,mem_space_id=memspace,xfer_prp=plist_id)

  ! close property list and dataspaces.
    call h5pclose_f(plist_id,ierr)
    call h5sclose_f(filespace,ierr)
    call h5sclose_f(memspace,ierr)
  endif

! close dataset.
  call h5dclose_f(dset_id,ierr)

! close hdf5 file to insure that we do not loose the hdf5 file if the code 
! is terminated abruptly.
  call h5fclose_f(file_id,ierr)

! close the hdf5 library and the fortran90 interface
  call h5close_f(ierr)

#else
  if(istep==ndiag)write(0,*)mype,' **** the code was compiled without hdf5 support ***'
#endif
end subroutine hdf5out_tracked_particles_6
!========================================

subroutine tag_particles_7

!========================================

  use global_parameters_7
  use particle_array_7
  use particle_tracking_7
  implicit none
  integer :: m,np

! we tag each particle with a unique number that will be carried along
! with it as it moves from one processor/domain to another. to facilitate
! the search for the tracked particles, we give the non-tracked particles
! a negative value.
  do m=1,mi
     zion(7,m)=-real(m+mype*mi)
     zion0(7,m)=zion(7,m)
  enddo
  if(mype==0)then
  ! on the master process (mype=0), we pick "nptrack" particles that will
  ! be followed at every time step. to facilitate the search of those
  ! particles among all the others in the zion arrays of each processor,
  ! we give them a positive number from 1 to nptrack. the particles are
  ! picked around r=0.5, which is mi/2.
    np=0
    do m=(mi-nptrack)/2,(mi+nptrack)/2-1
       np=np+1
       zion(7,m)=np
       zion0(7,m)=np
    enddo
  endif

end subroutine tag_particles_7

!========================================

subroutine locate_tracked_particles_7

!========================================

  use global_parameters_7
  use particle_array_7
  use particle_tracking_7
  implicit none
  integer :: i,m,npp,iout

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
  iout=mod(istep,isnap)
  if(iout==0)iout=isnap
  npp=0
  do m=1,mi
     if(zion(7,m)>0.0)then
       npp=npp+1
       ptracked(1:nparam,npp,iout)=zion(1:nparam,m)
     endif
  enddo
  ntrackp(iout)=npp

end subroutine locate_tracked_particles_7

!========================================

subroutine write_tracked_particles_7

!========================================

  use global_parameters_7
  use particle_tracking_7
  implicit none
 
  integer :: i,j
  character(len=10) :: cdum

  if(mype < 10)then
     write(cdum,'("trackp.00",i1)')mype
  elseif(mype < 100)then
     write(cdum,'("trackp.0",i2)')mype
  else
     write(cdum,'("trackp.",i3)')mype
  endif

  open(57,file=cdum,status='unknown',position='append')
  write(57,*)istep,isnap,nptrack
  write(57,*)ntrackp
  do i=1,isnap
     do j=1,ntrackp(i)
        write(57,*)ptracked(1:nparam,j,i)
     enddo
  enddo
  close(57)

end subroutine write_tracked_particles_7

!========================================

subroutine hdf5out_tracked_particles_7

!========================================
  use global_parameters_7
#ifdef __hdf5
  use particle_array_7
  use particle_tracking_7
  use hdf5
  implicit none

  integer  :: i,j,m,ipp,npp,ntpart(0:numberpe-1)
  real     :: r,q,theta,theta0,zeta,x,y,z

! hdf5 declarations
  integer(hid_t),save :: file_id       ! file identifier
  integer(hid_t) :: dset_id       ! dataset identifier
  integer(hid_t) :: filespace     ! dataspace identifier in file
  integer(hid_t) :: memspace      ! dataspace identifier in memory
  integer(hid_t) :: plist_id      ! property list identifier
  integer(hid_t) :: grp_id        ! group identifier
  integer(hid_t) :: time_id       ! attribute identifier
  integer(hid_t) :: mpsi_id       ! attribute identifier
  integer(hid_t) :: npe_id        ! attribute identifier
  integer(hid_t) :: mtheta_id     ! attribute identifier
  integer(hid_t) :: radial_id     ! attribute identifier
  integer(hid_t) :: itran_id      ! attribute identifier
  integer(hid_t) :: aspace_id     ! attribute dataspace identifier
  integer(hid_t) :: a1dspace_id   ! attribute dataspace identifier
  integer     :: ierr             ! return error flag
  integer     :: comm,info
  real        :: curr_time
  character(len=20),save :: fdum
  character(len=20) :: vdum
  integer :: rank
  integer(hsize_t), dimension(2) :: count,dimsf
  integer(hssize_t), dimension(2) :: offset
  integer, dimension(7) :: dimsfi = (/0,0,0,0,0,0,0/)

  comm = mpi_comm_world
  info = mpi_info_null


! this routine is called when idiag=0, right after the first runge-kutta
! half step in pushi (irk=1). for this reason, we need to use zion0()
! instead of zion(). at this point, the array zion0() holds the particle
! quantities at the end of the previous full step (irk=2). zion0() is
! also used in pushi at irk=1 to do the particle diagnostics, so using
! the same array ensures synchronization between the ouput files.

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
! we transform the coordinates of the particle, which are in magnetic
! coordinates (psi,theta,zeta), to x, y, z coordinates.
!
!       r=sqrt(2.0*zion(1,m))
!       q=q0+q1*r/a+q2*r*r/(a*a)
  npp=0
  do m=1,mi
     if(zion0(7,m)>0.0)then
       npp=npp+1
       r=sqrt(2.0*zion0(1,m))
       q=q0+q1*r/a+q2*r*r/(a*a)
       zeta=zion0(3,m)
       !!!theta=zion0(2,m)
       !!!theta0=theta+r*sin(theta)
       theta0=zion0(2,m)
       x=cos(zeta)*(1.0+r*cos(theta0))
       y=sin(zeta)*(1.0+r*cos(theta0))
       z=r*sin(theta0)
       ptracked(1,npp,1)=x
       ptracked(2,npp,1)=y
       ptracked(3,npp,1)=z
       ptracked(4,npp,1)=zion0(4,m)
       ptracked(5,npp,1)=zion0(5,m)
       ptracked(6,npp,1)=zion(6,m)
       ptracked(7,npp,1)=zion(7,m)
     endif
  enddo

! we need to know how many tracked particles each process has. this will
! be required to calculate the offset in the hdf5 file.
  ntpart=0
  call mpi_allgather(npp,1,mpi_integer,ntpart,1,mpi_integer,comm,ierr)

  if(mod((istep-ndiag),isnap)==0)then
   ! we open a new hdf5 file at the first diagnostic call after an output
   ! to the snapshot files.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_7 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! filename for grid quantities
     write(fdum,'("trackp_",i0,".h5")')(1+(mstepall+istep-ndiag)/isnap)

   ! create the file collectively.
     call h5fcreate_f(fdum,h5f_acc_trunc_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  else
   ! reopen the hdf5 file used during the last call to this subroutine.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_7 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! reopen the file collectively.
     call h5fopen_f(fdum,h5f_acc_rdwr_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  endif

! we now write the current calculation time as an attribute to the
! dataset. we need create the attribute data space, data type, etc.
! only one processor needs to write the attribute.

! create the data space for the 2d dataset:
!  1st dim: 7 quantities = particle phase space coordinates + identifier
!  2nd dim: number of tracked particles
  rank=2
  dimsf(1)=nparam
  dimsf(2)=nptrack
  call h5screate_simple_f(rank,dimsf,filespace,ierr)

! write dataset name into variable
  write(vdum,'("particles_t",i0)')(mstepall+istep)/ndiag

! create the dataset with default properties.
  call h5dcreate_f(file_id,vdum,h5t_native_real,filespace,dset_id,ierr)
  call h5sclose_f(filespace,ierr)

! create scalar data space for the time attribute that will be attached to
! the particle dataset.
  call h5screate_f(h5s_scalar_f,aspace_id,ierr)

! create dataset attribute
  call h5acreate_f(dset_id,"time",h5t_native_real,aspace_id,time_id,ierr)

  curr_time=real(mstepall+istep)*tstep  ! attribute data

! write the time attribute data. here dimsfi is just a dummy argument.
  if(mype.eq.0)then
    call h5awrite_f(time_id,h5t_native_real,curr_time,dimsfi,ierr)
  endif

! release attribute id, attribute data space
  call h5aclose_f(time_id,ierr)
  call h5sclose_f(aspace_id,ierr)

! each process defines a dataset in memory and writes it to the hyperslab
! in the file.
! first dimension is the particle quantities.
  count(1)=nparam
  offset(1)=0

! second dimension is the number of tracked particles.
! the offset is determine by the sum of tracked particles on pes < mype
  ipp=0
  do i=0,mype-1
     ipp=ipp+ntpart(i)
  enddo
  count(2)=npp
  offset(2)=ipp

  dimsfi(1)=count(1)
  dimsfi(2)=count(2)

! only the processes holding tracked particles participate in the output
! to the hdf5 file:
  if(npp > 0)then
    call h5screate_simple_f(rank,count,memspace,ierr)

  ! select hyperslab in the file.
    call h5dget_space_f(dset_id,filespace,ierr)
    call h5sselect_hyperslab_f(filespace,h5s_select_set_f,offset,count,ierr)
  
  ! create property list for collective dataset write
    call h5pcreate_f(h5p_dataset_xfer_f, plist_id, ierr)
  !!!  call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_collective_f, ierr)
    call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_independent_f, ierr)

  ! write the dataset
    call h5dwrite_f(dset_id,h5t_native_real,ptracked, dimsfi,ierr, &
         file_space_id=filespace,mem_space_id=memspace,xfer_prp=plist_id)

  ! close property list and dataspaces.
    call h5pclose_f(plist_id,ierr)
    call h5sclose_f(filespace,ierr)
    call h5sclose_f(memspace,ierr)
  endif

! close dataset.
  call h5dclose_f(dset_id,ierr)

! close hdf5 file to insure that we do not loose the hdf5 file if the code 
! is terminated abruptly.
  call h5fclose_f(file_id,ierr)

! close the hdf5 library and the fortran90 interface
  call h5close_f(ierr)

#else
  if(istep==ndiag)write(0,*)mype,' **** the code was compiled without hdf5 support ***'
#endif
end subroutine hdf5out_tracked_particles_7
!========================================

subroutine tag_particles_8

!========================================

  use global_parameters_8
  use particle_array_8
  use particle_tracking_8
  implicit none
  integer :: m,np

! we tag each particle with a unique number that will be carried along
! with it as it moves from one processor/domain to another. to facilitate
! the search for the tracked particles, we give the non-tracked particles
! a negative value.
  do m=1,mi
     zion(7,m)=-real(m+mype*mi)
     zion0(7,m)=zion(7,m)
  enddo
  if(mype==0)then
  ! on the master process (mype=0), we pick "nptrack" particles that will
  ! be followed at every time step. to facilitate the search of those
  ! particles among all the others in the zion arrays of each processor,
  ! we give them a positive number from 1 to nptrack. the particles are
  ! picked around r=0.5, which is mi/2.
    np=0
    do m=(mi-nptrack)/2,(mi+nptrack)/2-1
       np=np+1
       zion(7,m)=np
       zion0(7,m)=np
    enddo
  endif

end subroutine tag_particles_8

!========================================

subroutine locate_tracked_particles_8

!========================================

  use global_parameters_8
  use particle_array_8
  use particle_tracking_8
  implicit none
  integer :: i,m,npp,iout

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
  iout=mod(istep,isnap)
  if(iout==0)iout=isnap
  npp=0
  do m=1,mi
     if(zion(7,m)>0.0)then
       npp=npp+1
       ptracked(1:nparam,npp,iout)=zion(1:nparam,m)
     endif
  enddo
  ntrackp(iout)=npp

end subroutine locate_tracked_particles_8

!========================================

subroutine write_tracked_particles_8

!========================================

  use global_parameters_8
  use particle_tracking_8
  implicit none
 
  integer :: i,j
  character(len=10) :: cdum

  if(mype < 10)then
     write(cdum,'("trackp.00",i1)')mype
  elseif(mype < 100)then
     write(cdum,'("trackp.0",i2)')mype
  else
     write(cdum,'("trackp.",i3)')mype
  endif

  open(57,file=cdum,status='unknown',position='append')
  write(57,*)istep,isnap,nptrack
  write(57,*)ntrackp
  do i=1,isnap
     do j=1,ntrackp(i)
        write(57,*)ptracked(1:nparam,j,i)
     enddo
  enddo
  close(57)

end subroutine write_tracked_particles_8

!========================================

subroutine hdf5out_tracked_particles_8

!========================================
  use global_parameters_8
#ifdef __hdf5
  use particle_array_8
  use particle_tracking_8
  use hdf5
  implicit none

  integer  :: i,j,m,ipp,npp,ntpart(0:numberpe-1)
  real     :: r,q,theta,theta0,zeta,x,y,z

! hdf5 declarations
  integer(hid_t),save :: file_id       ! file identifier
  integer(hid_t) :: dset_id       ! dataset identifier
  integer(hid_t) :: filespace     ! dataspace identifier in file
  integer(hid_t) :: memspace      ! dataspace identifier in memory
  integer(hid_t) :: plist_id      ! property list identifier
  integer(hid_t) :: grp_id        ! group identifier
  integer(hid_t) :: time_id       ! attribute identifier
  integer(hid_t) :: mpsi_id       ! attribute identifier
  integer(hid_t) :: npe_id        ! attribute identifier
  integer(hid_t) :: mtheta_id     ! attribute identifier
  integer(hid_t) :: radial_id     ! attribute identifier
  integer(hid_t) :: itran_id      ! attribute identifier
  integer(hid_t) :: aspace_id     ! attribute dataspace identifier
  integer(hid_t) :: a1dspace_id   ! attribute dataspace identifier
  integer     :: ierr             ! return error flag
  integer     :: comm,info
  real        :: curr_time
  character(len=20),save :: fdum
  character(len=20) :: vdum
  integer :: rank
  integer(hsize_t), dimension(2) :: count,dimsf
  integer(hssize_t), dimension(2) :: offset
  integer, dimension(7) :: dimsfi = (/0,0,0,0,0,0,0/)

  comm = mpi_comm_world
  info = mpi_info_null


! this routine is called when idiag=0, right after the first runge-kutta
! half step in pushi (irk=1). for this reason, we need to use zion0()
! instead of zion(). at this point, the array zion0() holds the particle
! quantities at the end of the previous full step (irk=2). zion0() is
! also used in pushi at irk=1 to do the particle diagnostics, so using
! the same array ensures synchronization between the ouput files.

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
! we transform the coordinates of the particle, which are in magnetic
! coordinates (psi,theta,zeta), to x, y, z coordinates.
!
!       r=sqrt(2.0*zion(1,m))
!       q=q0+q1*r/a+q2*r*r/(a*a)
  npp=0
  do m=1,mi
     if(zion0(7,m)>0.0)then
       npp=npp+1
       r=sqrt(2.0*zion0(1,m))
       q=q0+q1*r/a+q2*r*r/(a*a)
       zeta=zion0(3,m)
       !!!theta=zion0(2,m)
       !!!theta0=theta+r*sin(theta)
       theta0=zion0(2,m)
       x=cos(zeta)*(1.0+r*cos(theta0))
       y=sin(zeta)*(1.0+r*cos(theta0))
       z=r*sin(theta0)
       ptracked(1,npp,1)=x
       ptracked(2,npp,1)=y
       ptracked(3,npp,1)=z
       ptracked(4,npp,1)=zion0(4,m)
       ptracked(5,npp,1)=zion0(5,m)
       ptracked(6,npp,1)=zion(6,m)
       ptracked(7,npp,1)=zion(7,m)
     endif
  enddo

! we need to know how many tracked particles each process has. this will
! be required to calculate the offset in the hdf5 file.
  ntpart=0
  call mpi_allgather(npp,1,mpi_integer,ntpart,1,mpi_integer,comm,ierr)

  if(mod((istep-ndiag),isnap)==0)then
   ! we open a new hdf5 file at the first diagnostic call after an output
   ! to the snapshot files.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_8 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! filename for grid quantities
     write(fdum,'("trackp_",i0,".h5")')(1+(mstepall+istep-ndiag)/isnap)

   ! create the file collectively.
     call h5fcreate_f(fdum,h5f_acc_trunc_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  else
   ! reopen the hdf5 file used during the last call to this subroutine.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_8 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! reopen the file collectively.
     call h5fopen_f(fdum,h5f_acc_rdwr_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  endif

! we now write the current calculation time as an attribute to the
! dataset. we need create the attribute data space, data type, etc.
! only one processor needs to write the attribute.

! create the data space for the 2d dataset:
!  1st dim: 7 quantities = particle phase space coordinates + identifier
!  2nd dim: number of tracked particles
  rank=2
  dimsf(1)=nparam
  dimsf(2)=nptrack
  call h5screate_simple_f(rank,dimsf,filespace,ierr)

! write dataset name into variable
  write(vdum,'("particles_t",i0)')(mstepall+istep)/ndiag

! create the dataset with default properties.
  call h5dcreate_f(file_id,vdum,h5t_native_real,filespace,dset_id,ierr)
  call h5sclose_f(filespace,ierr)

! create scalar data space for the time attribute that will be attached to
! the particle dataset.
  call h5screate_f(h5s_scalar_f,aspace_id,ierr)

! create dataset attribute
  call h5acreate_f(dset_id,"time",h5t_native_real,aspace_id,time_id,ierr)

  curr_time=real(mstepall+istep)*tstep  ! attribute data

! write the time attribute data. here dimsfi is just a dummy argument.
  if(mype.eq.0)then
    call h5awrite_f(time_id,h5t_native_real,curr_time,dimsfi,ierr)
  endif

! release attribute id, attribute data space
  call h5aclose_f(time_id,ierr)
  call h5sclose_f(aspace_id,ierr)

! each process defines a dataset in memory and writes it to the hyperslab
! in the file.
! first dimension is the particle quantities.
  count(1)=nparam
  offset(1)=0

! second dimension is the number of tracked particles.
! the offset is determine by the sum of tracked particles on pes < mype
  ipp=0
  do i=0,mype-1
     ipp=ipp+ntpart(i)
  enddo
  count(2)=npp
  offset(2)=ipp

  dimsfi(1)=count(1)
  dimsfi(2)=count(2)

! only the processes holding tracked particles participate in the output
! to the hdf5 file:
  if(npp > 0)then
    call h5screate_simple_f(rank,count,memspace,ierr)

  ! select hyperslab in the file.
    call h5dget_space_f(dset_id,filespace,ierr)
    call h5sselect_hyperslab_f(filespace,h5s_select_set_f,offset,count,ierr)
  
  ! create property list for collective dataset write
    call h5pcreate_f(h5p_dataset_xfer_f, plist_id, ierr)
  !!!  call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_collective_f, ierr)
    call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_independent_f, ierr)

  ! write the dataset
    call h5dwrite_f(dset_id,h5t_native_real,ptracked, dimsfi,ierr, &
         file_space_id=filespace,mem_space_id=memspace,xfer_prp=plist_id)

  ! close property list and dataspaces.
    call h5pclose_f(plist_id,ierr)
    call h5sclose_f(filespace,ierr)
    call h5sclose_f(memspace,ierr)
  endif

! close dataset.
  call h5dclose_f(dset_id,ierr)

! close hdf5 file to insure that we do not loose the hdf5 file if the code 
! is terminated abruptly.
  call h5fclose_f(file_id,ierr)

! close the hdf5 library and the fortran90 interface
  call h5close_f(ierr)

#else
  if(istep==ndiag)write(0,*)mype,' **** the code was compiled without hdf5 support ***'
#endif
end subroutine hdf5out_tracked_particles_8
!========================================

subroutine tag_particles_9

!========================================

  use global_parameters_9
  use particle_array_9
  use particle_tracking_9
  implicit none
  integer :: m,np

! we tag each particle with a unique number that will be carried along
! with it as it moves from one processor/domain to another. to facilitate
! the search for the tracked particles, we give the non-tracked particles
! a negative value.
  do m=1,mi
     zion(7,m)=-real(m+mype*mi)
     zion0(7,m)=zion(7,m)
  enddo
  if(mype==0)then
  ! on the master process (mype=0), we pick "nptrack" particles that will
  ! be followed at every time step. to facilitate the search of those
  ! particles among all the others in the zion arrays of each processor,
  ! we give them a positive number from 1 to nptrack. the particles are
  ! picked around r=0.5, which is mi/2.
    np=0
    do m=(mi-nptrack)/2,(mi+nptrack)/2-1
       np=np+1
       zion(7,m)=np
       zion0(7,m)=np
    enddo
  endif

end subroutine tag_particles_9

!========================================

subroutine locate_tracked_particles_9

!========================================

  use global_parameters_9
  use particle_array_9
  use particle_tracking_9
  implicit none
  integer :: i,m,npp,iout

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
  iout=mod(istep,isnap)
  if(iout==0)iout=isnap
  npp=0
  do m=1,mi
     if(zion(7,m)>0.0)then
       npp=npp+1
       ptracked(1:nparam,npp,iout)=zion(1:nparam,m)
     endif
  enddo
  ntrackp(iout)=npp

end subroutine locate_tracked_particles_9

!========================================

subroutine write_tracked_particles_9

!========================================

  use global_parameters_9
  use particle_tracking_9
  implicit none
 
  integer :: i,j
  character(len=10) :: cdum

  if(mype < 10)then
     write(cdum,'("trackp.00",i1)')mype
  elseif(mype < 100)then
     write(cdum,'("trackp.0",i2)')mype
  else
     write(cdum,'("trackp.",i3)')mype
  endif

  open(57,file=cdum,status='unknown',position='append')
  write(57,*)istep,isnap,nptrack
  write(57,*)ntrackp
  do i=1,isnap
     do j=1,ntrackp(i)
        write(57,*)ptracked(1:nparam,j,i)
     enddo
  enddo
  close(57)

end subroutine write_tracked_particles_9

!========================================

subroutine hdf5out_tracked_particles_9

!========================================
  use global_parameters_9
#ifdef __hdf5
  use particle_array_9
  use particle_tracking_9
  use hdf5
  implicit none

  integer  :: i,j,m,ipp,npp,ntpart(0:numberpe-1)
  real     :: r,q,theta,theta0,zeta,x,y,z

! hdf5 declarations
  integer(hid_t),save :: file_id       ! file identifier
  integer(hid_t) :: dset_id       ! dataset identifier
  integer(hid_t) :: filespace     ! dataspace identifier in file
  integer(hid_t) :: memspace      ! dataspace identifier in memory
  integer(hid_t) :: plist_id      ! property list identifier
  integer(hid_t) :: grp_id        ! group identifier
  integer(hid_t) :: time_id       ! attribute identifier
  integer(hid_t) :: mpsi_id       ! attribute identifier
  integer(hid_t) :: npe_id        ! attribute identifier
  integer(hid_t) :: mtheta_id     ! attribute identifier
  integer(hid_t) :: radial_id     ! attribute identifier
  integer(hid_t) :: itran_id      ! attribute identifier
  integer(hid_t) :: aspace_id     ! attribute dataspace identifier
  integer(hid_t) :: a1dspace_id   ! attribute dataspace identifier
  integer     :: ierr             ! return error flag
  integer     :: comm,info
  real        :: curr_time
  character(len=20),save :: fdum
  character(len=20) :: vdum
  integer :: rank
  integer(hsize_t), dimension(2) :: count,dimsf
  integer(hssize_t), dimension(2) :: offset
  integer, dimension(7) :: dimsfi = (/0,0,0,0,0,0,0/)

  comm = mpi_comm_world
  info = mpi_info_null


! this routine is called when idiag=0, right after the first runge-kutta
! half step in pushi (irk=1). for this reason, we need to use zion0()
! instead of zion(). at this point, the array zion0() holds the particle
! quantities at the end of the previous full step (irk=2). zion0() is
! also used in pushi at irk=1 to do the particle diagnostics, so using
! the same array ensures synchronization between the ouput files.

! check if tracked particles are located on this processor. particles that
! are tracked at every time step have a positive zion(7,m) value. all the
! others have a negative value.
! we transform the coordinates of the particle, which are in magnetic
! coordinates (psi,theta,zeta), to x, y, z coordinates.
!
!       r=sqrt(2.0*zion(1,m))
!       q=q0+q1*r/a+q2*r*r/(a*a)
  npp=0
  do m=1,mi
     if(zion0(7,m)>0.0)then
       npp=npp+1
       r=sqrt(2.0*zion0(1,m))
       q=q0+q1*r/a+q2*r*r/(a*a)
       zeta=zion0(3,m)
       !!!theta=zion0(2,m)
       !!!theta0=theta+r*sin(theta)
       theta0=zion0(2,m)
       x=cos(zeta)*(1.0+r*cos(theta0))
       y=sin(zeta)*(1.0+r*cos(theta0))
       z=r*sin(theta0)
       ptracked(1,npp,1)=x
       ptracked(2,npp,1)=y
       ptracked(3,npp,1)=z
       ptracked(4,npp,1)=zion0(4,m)
       ptracked(5,npp,1)=zion0(5,m)
       ptracked(6,npp,1)=zion(6,m)
       ptracked(7,npp,1)=zion(7,m)
     endif
  enddo

! we need to know how many tracked particles each process has. this will
! be required to calculate the offset in the hdf5 file.
  ntpart=0
  call mpi_allgather(npp,1,mpi_integer,ntpart,1,mpi_integer,comm,ierr)

  if(mod((istep-ndiag),isnap)==0)then
   ! we open a new hdf5 file at the first diagnostic call after an output
   ! to the snapshot files.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_9 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! filename for grid quantities
     write(fdum,'("trackp_",i0,".h5")')(1+(mstepall+istep-ndiag)/isnap)

   ! create the file collectively.
     call h5fcreate_f(fdum,h5f_acc_trunc_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  else
   ! reopen the hdf5 file used during the last call to this subroutine.

   ! initializes the hdf5 library and the fortran90 interface
     call h5open_f(ierr)

   ! setup_9 file access property list with parallel i/o access.
     call h5pcreate_f(h5p_file_access_f,plist_id,ierr)
     call h5pset_fapl_mpio_f(plist_id,comm,info,ierr)

   ! reopen the file collectively.
     call h5fopen_f(fdum,h5f_acc_rdwr_f,file_id,ierr,access_prp=plist_id)

   ! close access to a property list
     call h5pclose_f(plist_id,ierr)

  endif

! we now write the current calculation time as an attribute to the
! dataset. we need create the attribute data space, data type, etc.
! only one processor needs to write the attribute.

! create the data space for the 2d dataset:
!  1st dim: 7 quantities = particle phase space coordinates + identifier
!  2nd dim: number of tracked particles
  rank=2
  dimsf(1)=nparam
  dimsf(2)=nptrack
  call h5screate_simple_f(rank,dimsf,filespace,ierr)

! write dataset name into variable
  write(vdum,'("particles_t",i0)')(mstepall+istep)/ndiag

! create the dataset with default properties.
  call h5dcreate_f(file_id,vdum,h5t_native_real,filespace,dset_id,ierr)
  call h5sclose_f(filespace,ierr)

! create scalar data space for the time attribute that will be attached to
! the particle dataset.
  call h5screate_f(h5s_scalar_f,aspace_id,ierr)

! create dataset attribute
  call h5acreate_f(dset_id,"time",h5t_native_real,aspace_id,time_id,ierr)

  curr_time=real(mstepall+istep)*tstep  ! attribute data

! write the time attribute data. here dimsfi is just a dummy argument.
  if(mype.eq.0)then
    call h5awrite_f(time_id,h5t_native_real,curr_time,dimsfi,ierr)
  endif

! release attribute id, attribute data space
  call h5aclose_f(time_id,ierr)
  call h5sclose_f(aspace_id,ierr)

! each process defines a dataset in memory and writes it to the hyperslab
! in the file.
! first dimension is the particle quantities.
  count(1)=nparam
  offset(1)=0

! second dimension is the number of tracked particles.
! the offset is determine by the sum of tracked particles on pes < mype
  ipp=0
  do i=0,mype-1
     ipp=ipp+ntpart(i)
  enddo
  count(2)=npp
  offset(2)=ipp

  dimsfi(1)=count(1)
  dimsfi(2)=count(2)

! only the processes holding tracked particles participate in the output
! to the hdf5 file:
  if(npp > 0)then
    call h5screate_simple_f(rank,count,memspace,ierr)

  ! select hyperslab in the file.
    call h5dget_space_f(dset_id,filespace,ierr)
    call h5sselect_hyperslab_f(filespace,h5s_select_set_f,offset,count,ierr)
  
  ! create property list for collective dataset write
    call h5pcreate_f(h5p_dataset_xfer_f, plist_id, ierr)
  !!!  call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_collective_f, ierr)
    call h5pset_dxpl_mpio_f(plist_id, h5fd_mpio_independent_f, ierr)

  ! write the dataset
    call h5dwrite_f(dset_id,h5t_native_real,ptracked, dimsfi,ierr, &
         file_space_id=filespace,mem_space_id=memspace,xfer_prp=plist_id)

  ! close property list and dataspaces.
    call h5pclose_f(plist_id,ierr)
    call h5sclose_f(filespace,ierr)
    call h5sclose_f(memspace,ierr)
  endif

! close dataset.
  call h5dclose_f(dset_id,ierr)

! close hdf5 file to insure that we do not loose the hdf5 file if the code 
! is terminated abruptly.
  call h5fclose_f(file_id,ierr)

! close the hdf5 library and the fortran90 interface
  call h5close_f(ierr)

#else
  if(istep==ndiag)write(0,*)mype,' **** the code was compiled without hdf5 support ***'
#endif
end subroutine hdf5out_tracked_particles_9
