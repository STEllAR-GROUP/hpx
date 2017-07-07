# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES_HDF5)
  if(NOT HDF5_CXX_LIBRARIES OR NOT HDF5_IS_PARALLEL)
    hpx_error("The ShenEos example requires HDF C++ library enabled for thread-safety")
  endif()

  include_directories(${HDF5_INCLUDE_DIRS})

  # add a custom target for this example
  add_hpx_pseudo_target(examples.sheneos.sheneos)

  add_subdirectory(sheneos)

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.sheneos examples.sheneos.sheneos)

  ###############################################################################
  # add example executable
  add_hpx_executable(sheneos_client
    SOURCES sheneos_client.cpp
    DEPENDENCIES sheneos_component
    FOLDER "Examples/ShenEOS")

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.sheneos sheneos_client_exe)

  add_hpx_executable(sheneos_test
    SOURCES sheneos_test.cpp
    DEPENDENCIES sheneos_component
    FOLDER "Examples/ShenEOS")

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.sheneos sheneos_test_exe)

  if(HPX_WITH_FORTRAN)
    if(CMAKE_Fortran_COMPILER AND HDF5_FORTRAN_FOUND)
      add_hpx_executable(sheneos_compare
        SOURCES
            sheneos_compare.cpp
            eos_driver/read_nuc_table.F90
            eos_driver/eosmodule.F90
            eos_driver/readtable.F90
            eos_driver/nuc_eos.F90
            eos_driver/findrho.F90
            eos_driver/findtemp.F90
            eos_driver/linterp.f
            eos_driver/linterp_many.F90
            eos_driver/bisection.F90
        DEPENDENCIES sheneos_component ${HDF5_FORTRAN_LIBRARY} ${HDF5_LIBRARY}
        FOLDER "Examples/ShenEOS")

      set_target_properties(sheneos_compare_exe
          PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
      set_target_properties(sheneos_compare_exe
          PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")

      # add dependencies to pseudo-target
      add_hpx_pseudo_dependencies(examples.sheneos sheneos_compare_exe)
    endif()
  endif()
endif()
