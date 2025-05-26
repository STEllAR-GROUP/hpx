# Copyright (c) 2020-2025 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests multiple_init multiple_init_2918 shutdown_hang_6699
          unhandled_exception_582
)

set(unhandled_exception_582_PARAMETERS THREADS_PER_LOCALITY 1)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/Modules/Full/RuntimeDistributed"
  )

  add_hpx_regression_test(
    "modules.runtime_distributed" ${test} ${${test}_PARAMETERS}
  )
endforeach()
