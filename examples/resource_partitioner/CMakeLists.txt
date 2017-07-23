# Copyright (c) 2017 Shoshana Jakobovits
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
  simple_resource_partitioner
)

set(simple_resource_partitioner_FLAGS DEPENDENCIES iostreams_component)

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/ResourcePartitioner")

  if(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
  endif()

  # add a custom target for this example
  add_hpx_pseudo_target(examples.resource_partitioner.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.resource_partitioner
                              examples.resource_partitioner.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.resource_partitioner.${example_program}
                              ${example_program}_exe)
endforeach()

