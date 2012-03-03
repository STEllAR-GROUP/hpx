subroutine readtable(eos_filename)
! This routine reads the table and initializes
! all variables in the module. 

  use eosmodule
  use hdf5 

  implicit none

  character(*) eos_filename

  character(len=100) message

! HDF5 vars
  integer(HID_T) file_id,dset_id,dspace_id
  integer(HSIZE_T) dims1(1), dims3(3)
  integer error,rank,accerr
  integer i,j,k

  real*8 amu_cgs_andi
  real*8 buffer1,buffer2,buffer3,buffer4
  accerr=0

  write(*,*) "Reading Ott EOS Table"

  call h5open_f(error)

  call h5fopen_f (trim(adjustl(eos_filename)), H5F_ACC_RDONLY_F, file_id, error)

  write(6,*) trim(adjustl(eos_filename))

! read scalars
  dims1(1)=1
  call h5dopen_f(file_id, "pointsrho", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_INTEGER, nrho, dims1, error)
  call h5dclose_f(dset_id,error)

  if(error.ne.0) then
     stop "Could not read EOS table file"
  endif

  dims1(1)=1
  call h5dopen_f(file_id, "pointstemp", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_INTEGER, ntemp, dims1, error)
  call h5dclose_f(dset_id,error)

  if(error.ne.0) then
     stop "Could not read EOS table file"
  endif

  dims1(1)=1
  call h5dopen_f(file_id, "pointsye", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_INTEGER, nye, dims1, error)
  call h5dclose_f(dset_id,error)

  if(error.ne.0) then
     stop "Could not read EOS table file"
  endif

  write(message,"(a25,i5,i5,i5)") "We have nrho ntemp nye: ", nrho,ntemp,nye
  write(*,*) message

  allocate(alltables(nrho,ntemp,nye,nvars))

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


  dims3(1)=nrho
  dims3(2)=ntemp
  dims3(3)=nye
  call h5dopen_f(file_id, "logpress", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,1), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error
  call h5dopen_f(file_id, "logenergy", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,2), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error
  call h5dopen_f(file_id, "entropy", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,3), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error
  call h5dopen_f(file_id, "munu", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,4), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error
  call h5dopen_f(file_id, "cs2", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,5), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error
  call h5dopen_f(file_id, "dedt", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,6), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error
  call h5dopen_f(file_id, "dpdrhoe", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,7), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error
  call h5dopen_f(file_id, "dpderho", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,8), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

! chemical potentials
  call h5dopen_f(file_id, "muhat", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,9), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

  call h5dopen_f(file_id, "mu_e", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,10), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

  call h5dopen_f(file_id, "mu_p", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,11), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

  call h5dopen_f(file_id, "mu_n", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,12), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

! compositions
  call h5dopen_f(file_id, "Xa", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,13), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

  call h5dopen_f(file_id, "Xh", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,14), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

  call h5dopen_f(file_id, "Xn", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,15), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

  call h5dopen_f(file_id, "Xp", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,16), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error


! average nucleus
  call h5dopen_f(file_id, "Abar", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,17), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

  call h5dopen_f(file_id, "Zbar", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,18), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

! Gamma
  call h5dopen_f(file_id, "gamma", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, alltables(:,:,:,19), dims3, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

  allocate(logrho(nrho))
  dims1(1)=nrho
  call h5dopen_f(file_id, "logrho", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, logrho, dims1, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

  allocate(logtemp(ntemp))
  dims1(1)=ntemp
  call h5dopen_f(file_id, "logtemp", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, logtemp, dims1, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

  allocate(ye(nye))
  dims1(1)=nye
  call h5dopen_f(file_id, "ye", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, ye, dims1, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error


  call h5dopen_f(file_id, "energy_shift", dset_id, error)
  call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, energy_shift, dims1, error)
  call h5dclose_f(dset_id,error)
  accerr=accerr+error

  if(accerr.ne.0) then
    stop "Problem reading EOS table file"
  endif


  call h5fclose_f (file_id,error)

  call h5close_f (error)

  ! set min-max values:

  eos_rhomin = 10.0d0**logrho(1)
  eos_rhomax = 10.0d0**logrho(nrho)

  eos_yemin = ye(1)
  eos_yemax = ye(nye)

  eos_tempmin = 10.0d0**logtemp(1)
  eos_tempmax = 10.0d0**logtemp(ntemp)

  write(6,*) "Done reading eos tables"


end subroutine readtable


