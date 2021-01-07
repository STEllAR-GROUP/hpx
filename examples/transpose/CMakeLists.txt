# Copyright (c) 2014-2015 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs transpose_serial transpose_serial_block transpose_smp
                     transpose_smp_block
)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(example_programs ${example_programs} transpose_block
                       transpose_serial_vector
  )
  if(HPX_WITH_THREAD_EXECUTORS_COMPATIBILITY)
    set(example_programs ${example_programs} transpose_block_numa)
  endif()
endif()

if(HPX_WITH_AWAIT OR HPX_WITH_CXX20_COROUTINES)
  set(example_programs ${example_programs} transpose_await)
endif()

set(transpose_serial_vector_FLAGS DEPENDENCIES partitioned_vector_component)

set(transpose_smp_PARAMETERS THREADS_PER_LOCALITY 4)
set(transpose_smp_block_PARAMETERS THREADS_PER_LOCALITY 4)
set(transpose_block_PARAMETERS THREADS_PER_LOCALITY 4)
set(transpose_block_numa_PARAMETERS
    THREADS_PER_LOCALITY 4 "--transpose-threads=4" "--transpose-numa-domains=1"
    RUN_SERIAL
)

foreach(example_program ${example_programs})

  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Transpose"
  )

  add_hpx_example_target_dependencies("transpose" ${example_program})

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test(
      "transpose" ${example_program} ${${example_program}_PARAMETERS}
    )
  endif()
endforeach()
