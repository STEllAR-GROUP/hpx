# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_NETWORKING)
  set(tests barrier_3792 barrier_hang broadcast_unwrap_future_2885
            broadcast_wait_for_2822 multiple_gather_ops_2001
  )

  set(broadcast_unwrap_future_2885_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY
                                              4
  )
  set(broadcast_wait_for_2822_PARAMETERS LOCALITIES 2 THREADS_PER_LOCALITY 4)
  set(multiple_gather_ops_2001_PARAMETERS LOCALITIES 2)
  set(barrier_3792_PARAMETERS LOCALITIES 3 THREADS_PER_LOCALITY 1)

  foreach(test ${tests})
    set(sources ${test}.cpp)

    source_group("Source Files" FILES ${sources})

    # add example executable
    add_hpx_executable(
      ${test}_test INTERNAL_FLAGS
      SOURCES ${sources} ${${test}_FLAGS}
      EXCLUDE_FROM_ALL
      HPX_PREFIX ${HPX_BUILD_PREFIX}
      FOLDER "Tests/Regressions/Modules/Full/Collectives"
    )

    add_hpx_regression_test("modules.collectives" ${test} ${${test}_PARAMETERS})
  endforeach()

endif()
