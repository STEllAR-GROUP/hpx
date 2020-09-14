# Copyright (c) 2014-2020 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    bulk_async
    created_executor
    executor_parameters
    executor_parameters_dispatching
    executor_parameters_timer_hooks
    future_then_executor
    minimal_async_executor
    minimal_sync_executor
    parallel_executor
    parallel_fork_executor
    parallel_policy_executor
    persistent_executor_parameters
    polymorphic_executor
    shared_parallel_executor
    standalone_thread_pool_executor
)

set(future_then_executor_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Parallelism/Execution")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("modules.execution" ${test} ${${test}_PARAMETERS})

endforeach()
