# Copyright (c) 2019-2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    all_gather
    all_reduce
    all_to_all
    barrier
    broadcast_apply
    broadcast_component
    channel_communicator
    communication_set
    exclusive_scan_
    fold
    global_spmd_block
    inclusive_scan_
    reduce
    reduce_direct
    remote_latch
)

if(HPX_WITH_NETWORKING)
  set(tests ${tests} broadcast broadcast_direct gather scatter)
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

  add_hpx_unit_test(
    "modules.collectives" ${test} ${${test}_PARAMETERS} LOCALITIES 2
  )
endforeach()
