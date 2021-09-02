# Copyright (c) 2014-2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    algorithm_bulk
    algorithm_detach
    algorithm_ensure_started
    algorithm_execute
    algorithm_just
    algorithm_just_on
    algorithm_let_value
    algorithm_let_error
    algorithm_on
    algorithm_split
    algorithm_sync_wait
    algorithm_transform
    algorithm_when_all
    bulk_async
    executor_parameters
    executor_parameters_dispatching
    executor_parameters_timer_hooks
    future_then_executor
    minimal_async_executor
    minimal_sync_executor
    persistent_executor_parameters
)

set(future_then_executor_PARAMETERS THREADS_PER_LOCALITY 4)

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Core/Execution")

  # add example executable
  hpx_local_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
  )

  hpx_local_add_unit_test("modules.execution" ${test} ${${test}_PARAMETERS})

endforeach()
