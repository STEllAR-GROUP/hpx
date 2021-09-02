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
    replay_executor
    replicate_executor
)

# The Intel compiler version 19.1.1.217 does not manage to compile these. Other
# versions have not been tested.
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  list(APPEND tests dataflow_replay_executor dataflow_replay_plain
       dataflow_replicate_executor dataflow_replicate_plain
  )
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Core/Resiliency")

  # add test executable
  hpx_local_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  hpx_local_add_unit_test("modules.resiliency" ${test} ${${test}_PARAMETERS})
endforeach()
