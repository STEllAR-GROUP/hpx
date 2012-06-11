  subroutine read_nuc_table()
    implicit none
    integer rc
    character*128 ::  file = '/home/manderson/parallex/install/bin/myshen_test_220r_180t_50y_extT_analmu_20100322_SVNr28.h5'
    logical, parameter :: ltrace = .true.

    if (ltrace) then
      write(0,*)'Calling readtable to load nuclear EOS.'
    end if
    
    call readtable(file)

    return
  end subroutine
