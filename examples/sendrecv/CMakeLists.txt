# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there

find_package(HPX_HDF5)

if(HDF5_CPP_FOUND AND HDF5_THREAD_SAFE AND CMAKE_Fortran_COMPILER AND HDF5_FORTRAN_FOUND)

  if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lgfortran")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -lgfortran")
  endif()

  add_subdirectory(sendrecv_hpx)

  # for all targets specified above
  set(sources
    sendrecv_client.cpp
  )

  source_group("Source Files" FILES ${sources})

  set(dependencies
    ${HDF5_FORTRAN_LIBRARY}
    distributing_factory_component)

  # add example executable
  add_hpx_executable(sendrecv_client
    SOURCES ${sources}
    DEPENDENCIES sendrecv_hpx_component ${dependencies}
    FOLDER "Examples/Applications/GTCX")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.sendrecv.sendrecv_hpx)

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.sendrecv
                              examples.sendrecv.sendrecv_hpx)

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.sendrecv.sendrecv_hpx
                              sendrecv_client_exe)

endif()
