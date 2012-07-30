# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there

find_package(HPX_HDF5)

if(CMAKE_Fortran_COMPILER AND HDF5_FORTRAN_LIBRARY)
  set(CMAKE_Fortran_FLAGS "-DDOUBLE_PRECISION ${CMAKE_Fortran_FLAGS} -fopenmp")

  add_subdirectory(tsf)

  # for all targets specified above
  set(sources
    tsf_client.cpp
    )

  source_group("Source Files" FILES ${sources})

  set(dependencies
    ${HDF5_FORTRAN_LIBRARY}
    distributing_factory_component)

  # add example executable
  add_hpx_executable(tsf_client
    SOURCES ${sources}
    DEPENDENCIES tsf_component ${dependencies}
    FOLDER "Examples/Applications/Fortran Threadsafe")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.fortran_threadsafe.tsf)

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.fortran_threadsafe
                              examples.fortran_threadsafe.tsf)

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.fortran_threadsafe.tsf
                              tsf_client_exe)
endif()
