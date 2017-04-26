# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    block_os_threads_1036
    resume_priority
    thread_data_1111
    thread_pool_executor_1112
    thread_rescheduling
    thread_suspend_pending
    thread_suspend_duration
    threads_all_1422
   )

if(HPX_WITH_LOCAL_SCHEDULER OR HPX_WITH_ALL_SCHEDULERS)
  set(tests ${tests}
    thread_pool_executor_1114
    )
endif()

set(block_os_threads_1036_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_pool_executor_1112_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_rescheduling_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_suspend_pending_PARAMETERS THREADS_PER_LOCALITY 1)
set(thread_suspend_duration_PARAMETERS THREADS_PER_LOCALITY 1)
set(threads_all_1422_PARAMETERS THREADS_PER_LOCALITY -1)        # -1 means 'all'

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER "Tests/Regressions/Threads")

  add_hpx_regression_test("threads" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.regressions.threads_dir.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.threads_dir
                              tests.regressions.threads_dir.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.regressions.threads_dir.${test}
                              ${test}_test_exe)
endforeach()

