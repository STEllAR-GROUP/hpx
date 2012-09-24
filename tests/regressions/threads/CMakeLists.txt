# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    thread_rescheduling
    thread_suspend_pending
    thread_suspend_duration
   )

set(thread_rescheduling_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_suspend_pending_PARAMETERS THREADS_PER_LOCALITY 1)
set(thread_suspend_duration_PARAMETERS THREADS_PER_LOCALITY 1)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Regressions/Threads")

  add_hpx_regression_test(${test} ${${test}_PARAMETERS})  

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.threads.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.threads
                              tests.regressions.threads.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.threads.${test}
                              ${test}_test_exe)
endforeach()

