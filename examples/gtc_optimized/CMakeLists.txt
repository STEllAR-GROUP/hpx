# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# just recurse into all components subdirectories and execute the scripts there

set(CMAKE_Fortran_FLAGS "-fdefault-real-8 -DDOUBLE_PRECISION ${CMAKE_Fortran_FLAGS}")

add_subdirectory(gtc_hpx)

# for all targets specified above
set(sources
    gtc_hpx_client.cpp
    )

source_group("Source Files" FILES ${sources})

set(dependencies
    ${HDF5_FORTRAN_LIBRARY}
    distributing_factory_component)

  # add example executable
if (CMAKE_Fortran_COMPILER)
  add_hpx_executable(gtc_hpx_client
    SOURCES ${sources}
    DEPENDENCIES gtc_hpx_component ${dependencies}
    FOLDER "Examples/GTC Optimized")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.gtc_optimized.gtc_hpx)

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.gtc_optimized
                              examples.gtc_optimized.gtc_hpx)

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.gtc_optimized.gtc_hpx
                              gtc_hpx_client_exe)
endif()
