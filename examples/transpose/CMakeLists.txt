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

set(transpose_smp_PARAMETERS THREADS_PER_LOCALITY 4)
set(transpose_smp_block_PARAMETERS THREADS_PER_LOCALITY 4)
set(transpose_block_PARAMETERS THREADS_PER_LOCALITY 4)
set(transpose_block_numa_PARAMETERS THREADS_PER_LOCALITY 4
    "--transpose-threads=4" "--transpose-numa-domains=1")

foreach(example_program ${example_programs})

  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/Transpose")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.transpose.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.transpose
                              examples.transpose.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.transpose.${example_program}
                              ${example_program}_exe)

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test("transpose" ${example_program} ${${example_program}_PARAMETERS})
    add_hpx_pseudo_target(tests.examples.transpose.${example_program})
    add_hpx_pseudo_dependencies(tests.examples.transpose
                                tests.examples.transpose.${example_program})
    add_hpx_pseudo_dependencies(tests.examples.transpose.${example_program}
                                ${example_program}_exe)
  endif()
endforeach()
