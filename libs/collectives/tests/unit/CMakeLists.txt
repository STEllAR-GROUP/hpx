# Copyright (c) 2019 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
  all_reduce
  all_to_all
  barrier
  broadcast
  broadcast_apply
  broadcast_component
  fold
  global_spmd_block
  reduce
  remote_latch
)

set(all_reduce_PARAMETERS LOCALITIES 2)
set(all_to_all_PARAMETERS LOCALITIES 2)
set(broadcast_PARAMETERS LOCALITIES 2)
set(broadcast_apply_PARAMETERS LOCALITIES 2)
set(broadcast_component_PARAMETERS LOCALITIES 2)
set(remote_latch_PARAMETERS LOCALITIES 2)
set(reduce_PARAMETERS LOCALITIES 2)
set(global_spmd_block_PARAMETERS LOCALITIES 2)

foreach(test ${tests})
  set(sources
      ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
    INTERNAL_FLAGS
    SOURCES ${sources}
    ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER "Tests/Unit/Modules/Collectives")

  add_hpx_unit_test("modules.collectives" ${test} ${${test}_PARAMETERS})
endforeach()


