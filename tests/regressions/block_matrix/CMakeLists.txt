# Copyright (c) 2013 Erik Schnetter
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(tests block_matrix)

  set(sources algorithms.cpp block_matrix.cpp main.cpp matrix.cpp
              matrix_hpx.cpp tests.cpp
  )

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(
    block_matrix_test INTERNAL_FLAGS
    SOURCES ${sources} ${block_matrix_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Regressions/Block-Matrix"
  )

  add_hpx_regression_test(
    "block_matrix" block_matrix ${block_matrix_PARAMETERS}
  )
endif()
