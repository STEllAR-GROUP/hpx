# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    apex_policy_engine_periodic apex_policy_engine_events
    apex_policy_engine_active_thread_count apex_balance apex_fibonacci
)

foreach(example_program ${example_programs})
  set(${example_program}_PARAMETERS THREADS_PER_LOCALITY 4)

  set(sources ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Apex/${example_program}"
  )

  add_hpx_example_target_dependencies("apex" ${example_program})

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test(
      "apex" ${example_program} ${${example_program}_PARAMETERS}
    )
  endif()

endforeach()
