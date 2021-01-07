# Copyright (c) 2014-2018 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests annotated_minmax_2593 chunk_size_4118 future_then_async_executor
          is_executor_1691 parallel_executor_1781
)

if(HPX_WITH_DATAPAR)
  set(tests ${tests} lambda_arguments_2403 lambda_return_type_2402)
endif()

if(HPX_WITH_THREAD_POOL_OS_EXECUTOR_COMPATIBILITY)
  set(tests ${tests} handled_exception_2959)
endif()

if(HPX_WITH_POOL_EXECUTOR_COMPATIBILITY)
  set(tests ${tests} bulk_then_execute_3182)
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  set(${test}_PARAMETERS THREADS_PER_LOCALITY 4)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/Modules/Parallelism/Execution"
  )

  add_hpx_regression_test("modules.execution" ${test} ${${test}_PARAMETERS})

endforeach()
