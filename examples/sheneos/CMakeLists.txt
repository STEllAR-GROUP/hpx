# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_EXAMPLES_HDF5)
  if(NOT HDF5_CXX_LIBRARIES)
    hpx_error("The ShenEos example requires HDF C++ library")
  endif()

  # add a custom target for this example
  add_hpx_pseudo_target(examples.sheneos.sheneos_)

  add_subdirectory(sheneos)

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.sheneos examples.sheneos.sheneos_)

  # ############################################################################
  # add example executable
  add_hpx_executable(
    sheneos_client INTERNAL_FLAGS
    SOURCES sheneos_client.cpp
    DEPENDENCIES sheneos_component
    FOLDER "Examples/ShenEOS"
  )

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.sheneos sheneos_client)

  add_hpx_executable(
    sheneos_test INTERNAL_FLAGS
    SOURCES sheneos_test.cpp
    DEPENDENCIES sheneos_component
    FOLDER "Examples/ShenEOS"
  )

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.sheneos sheneos_test)

  # TODO: Fix example. Not added to unit tests until fixed.

  if(HPX_WITH_FORTRAN)
    if(CMAKE_Fortran_COMPILER AND HDF5_FORTRAN_FOUND)
      add_hpx_executable(
        sheneos_compare INTERNAL_FLAGS
        SOURCES sheneos_compare.cpp
                eos_driver/read_nuc_table.F90
                eos_driver/eosmodule.F90
                eos_driver/readtable.F90
                eos_driver/nuc_eos.F90
                eos_driver/findrho.F90
                eos_driver/findtemp.F90
                eos_driver/linterp.f
                eos_driver/linterp_many.F90
                eos_driver/bisection.F90
        DEPENDENCIES sheneos_component
        FOLDER "Examples/ShenEOS"
      )

      target_include_directories(sheneos_compare PRIVATE ${HDF5_INCLUDE_DIRS})
      target_link_libraries(
        sheneos_compare PRIVATE ${HDF5_FORTRAN_LIBRARY} ${HDF5_LIBRARY}
      )

      target_compile_options(sheneos_compare ${OpenMP_CXX_FLAGS})
      set_target_properties(
        sheneos_compare PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}"
      )

      # add dependencies to pseudo-target
      add_hpx_pseudo_dependencies(examples.sheneos sheneos_compare)
    endif()
  endif()
endif()
