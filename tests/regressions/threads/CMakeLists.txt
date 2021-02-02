# Copyright (c) 2007-2018 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    block_os_threads_1036
    run_as_hpx_thread_exceptions_3304
    run_as_os_thread_lockup_2991
    stackless_self_4155
    thread_data_1111
    thread_suspend_pending
    thread_suspend_duration
    threads_all_1422
)

if((HPX_WITH_LOCAL_SCHEDULER OR HPX_WITH_ALL_SCHEDULERS)
   AND HPX_WITH_THREAD_POOL_OS_EXECUTORS_COMPATIBILITY
)
  set(tests ${tests} thread_pool_executor_1114)
endif()

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests ${tests} resume_priority thread_rescheduling)
endif()

if(HPX_WITH_THREAD_POOL_OS_EXECUTORS_COMPATIBILITY)
  set(tests ${tests} thread_pool_executor_1112)
endif()

set(block_os_threads_1036_PARAMETERS THREADS_PER_LOCALITY 4 RUN_SERIAL)
set(thread_pool_executor_1112_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_rescheduling_PARAMETERS THREADS_PER_LOCALITY 4)
set(thread_suspend_pending_PARAMETERS THREADS_PER_LOCALITY 1)
set(thread_suspend_duration_PARAMETERS THREADS_PER_LOCALITY 1)
set(threads_all_1422_PARAMETERS THREADS_PER_LOCALITY -1) # -1 means 'all'

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/Threads"
  )

  add_hpx_regression_test("threads" ${test} ${${test}_PARAMETERS})

endforeach()
