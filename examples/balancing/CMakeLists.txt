# Copyright (c) 2011 Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs hpx_thread_phase os_thread_num)

set(os_thread_num_PARAMETERS THREADS_PER_LOCALITY 4)
set(hpx_thread_phase_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(example_program ${example_programs})
  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  hpx_local_add_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Balancing/${example_program}"
  )

  hpx_local_add_example_target_dependencies("balancing" ${example_program})

  if(HPXLocal_WITH_TESTS AND HPXLocal_WITH_TESTS_EXAMPLES)
    hpx_local_add_example_test(
      "balancing" ${example_program} ${${example_program}_PARAMETERS}
    )
  endif()
endforeach()
