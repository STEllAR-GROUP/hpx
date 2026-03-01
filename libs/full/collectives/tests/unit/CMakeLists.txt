# Copyright (c) 2019-2025 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    all_gather
    all_gather_sync
    all_reduce
    all_reduce_sync
    all_to_all
    all_to_all_sync
    barrier
    barrier_synchronize
    broadcast
    broadcast_component
    broadcast_hierarchical
    broadcast_post
    broadcast_sync
    channel_communicator
    fold
    global_spmd_block
    reduce_direct
    remote_latch
)

if(HPX_WITH_NETWORKING)
  set(tests
      ${tests}
      broadcast_direct
      concurrent_collectives
      exclusive_scan_
      exclusive_scan_sync
      gather
      gather_hierarchical
      gather_sync
      inclusive_scan_
      inclusive_scan_sync
      reduce
      reduce_hierarchical
      reduce_sync
      scatter
      scatter_hierarchical
      scatter_sync
  )

  foreach(test ${tests})
    set(${test}_PARAMETERS LOCALITIES 2)
  endforeach()
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Full/Collectives"
  )

  add_hpx_unit_test("modules.collectives" ${test} ${${test}_PARAMETERS})
endforeach()
