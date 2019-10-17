# Copyright (c) 2014-2017 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    bulk_async
    created_executor
    executor_parameters
    executor_parameters_timer_hooks
    minimal_async_executor
    minimal_sync_executor
    minimal_timed_async_executor
    minimal_timed_sync_executor
    parallel_executor
    parallel_fork_executor
    parallel_policy_executor
    persistent_executor_parameters
    sequenced_executor
    service_executors
    shared_parallel_executor
    this_thread_executors
    thread_pool_attached_executors
    thread_pool_executors
    thread_pool_os_executors
    timed_parallel_executor
    timed_this_thread_executors
    timed_thread_pool_executors
    standalone_thread_pool_os_executors
    standalone_thread_pool_executor
   )

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Execution")

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     EXCLUDE_FROM_ALL
                     HPX_PREFIX ${HPX_BUILD_PREFIX}
                     FOLDER ${folder_name})

  add_hpx_unit_test("modules.execution" ${test} ${${test}_PARAMETERS})

endforeach()
