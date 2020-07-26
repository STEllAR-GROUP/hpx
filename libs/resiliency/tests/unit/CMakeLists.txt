# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    async_replay_executor
    async_replay_plain
    async_replicate_executor
    async_replicate_plain
    async_replicate_vote_executor
    async_replicate_vote_plain
    dataflow_replay_executor
    dataflow_replay_plain
    dataflow_replicate_executor
    dataflow_replicate_plain
)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Resiliency")

  # add test executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("modules.resiliency" ${test} ${${test}_PARAMETERS})
endforeach()
