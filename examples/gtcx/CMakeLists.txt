# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there

find_package(HPX_HDF5)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=6)

if(HDF5_CPP_FOUND AND HDF5_THREAD_SAFE AND CMAKE_Fortran_COMPILER AND HDF5_FORTRAN_FOUND)

  if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lgfortran")
    set(CMAKE_Fortran_FLAGS "-DDOUBLE_PRECISION ${CMAKE_Fortran_FLAGS} -lgfortran")
  endif()

  add_subdirectory(gtcx_hpx)

  # for all targets specified above
  set(sources
    gtcx_client.cpp
  )

  source_group("Source Files" FILES ${sources})

  set(dependencies
    ${HDF5_FORTRAN_LIBRARY}
    distributing_factory_component)

  # add example executable
  add_hpx_executable(gtcx_client
    SOURCES ${sources}
    DEPENDENCIES gtcx_hpx_component ${dependencies}
    FOLDER "Examples/Applications/GTCX")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.gtcx.gtcx_hpx)

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.gtcx
                              examples.gtcx.gtcx_hpx)

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.gtcx.gtcx_hpx
                              gtcx_client_exe)

endif()
