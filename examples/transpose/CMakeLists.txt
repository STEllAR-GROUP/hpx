# Copyright (c) 2014-2015 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
  transpose_serial
  transpose_serial_block
  transpose_smp
  transpose_smp_block
  transpose_block
  transpose_block_numa
  transpose_serial_vector
)

if(HPX_WITH_AWAIT)
  set(example_programs
    ${example_programs}
    transpose_await)
endif()

set(transpose_serial_vector_FLAGS DEPENDENCIES partitioned_vector_component)

foreach(example_program ${example_programs})

  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/Transpose/${example_program}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.transpose.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.transpose
                              examples.transpose.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.transpose.${example_program}
                              ${example_program}_exe)
endforeach()
