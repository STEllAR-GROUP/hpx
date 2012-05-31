
# Copyright (c) 2012      Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=10)
add_definitions(-DHPX_COMPONENT_CREATE_ARGUMENT_LIMIT=10)
add_definitions(-DHPX_FUNCTION_LIMIT=15)

set(jacobi_programs jacobi)

set(jacobi_sources
    grid.cpp
    row.cpp
)
  
set(jacobi_FLAGS DEPENDENCIES dataflow_component iostreams_component remote_object_component)

foreach(jacobi_program ${jacobi_programs})
  set(sources
       ${${jacobi_program}_sources} ${jacobi_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add jacobi executable
  add_hpx_executable(${jacobi_program}
                     MODULE jacobi
                     SOURCES ${sources}
                     ${${jacobi_program}_FLAGS}
                     FOLDER "Examples/BrightFuture")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.jacobi.${jacobi_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.jacobi
                              examples.jacobi.${jacobi_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.jacobi.${jacobi_program}
                              ${jacobi_program}_exe)
endforeach()
