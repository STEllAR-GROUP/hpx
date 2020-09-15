# Copyright (c) 2007-2016 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests for_each_value_proxy)
set(tests ${tests} parallel_fill_4132)

set(for_each_value_proxy_PARAMETERS THREADS_PER_LOCALITY 4)
set(parallel_fill_4132_PARAMETERS THREADS_PER_LOCALITY 4 RUN_SERIAL)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/Modules/Full/Compute"
  )

  target_include_directories(${test}_test SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
  add_hpx_regression_test("modules.compute" ${test} ${${test}_PARAMETERS})
endforeach()
