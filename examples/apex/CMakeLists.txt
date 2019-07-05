# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2011      Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    apex_policy_engine_periodic
    apex_policy_engine_events
    apex_policy_engine_active_thread_count
    apex_balance
    apex_fibonacci
   )

set(apex_policy_engine_periodic_PARAMETERS THREADS_PER_LOCALITY 4)
set(apex_policy_engine_events_PARAMETERS THREADS_PER_LOCALITY 4)
set(apex_policy_engine_active_thread_coutn_PARAMETERS THREADS_PER_LOCALITY 4)
set(apex_balance_PARAMETERS THREADS_PER_LOCALITY 4)
set(apex_fibonacci_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/Apex/${example_program}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.apex.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.apex
                              examples.apex.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.apex.${example_program}
                              ${example_program})

  if(HPX_WITH_TESTS AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_example_test("apex" ${example_program} ${${example_program}_PARAMETERS})
    add_hpx_pseudo_target(tests.examples.apex.${example_program})
    add_hpx_pseudo_dependencies(tests.examples.apex
                                tests.examples.apex.${example_program})
    add_hpx_pseudo_dependencies(tests.examples.apex.${example_program}
                                ${example_program})
  endif()
endforeach()
