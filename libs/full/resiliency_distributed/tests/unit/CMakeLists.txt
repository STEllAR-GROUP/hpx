# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests)

if(HPX_WITH_NETWORKING)
  set(tests ${tests} async_replay_distributed_plain
            async_replicate_distributed_plain
  )
  set(async_replay_distributed_plain_PARAMETERS LOCALITIES 2)
  set(async_replicate_distributed_plain_PARAMETERS LOCALITIES 2)
endif()

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Full/Resiliency")

  # add test executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  add_hpx_unit_test(
    "modules.resiliency_distributed" ${test} ${${test}_PARAMETERS}
  )
endforeach()
