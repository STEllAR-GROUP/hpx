set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HPX_SOURCE_DIR}/cmake")

include(HPX_Message)
include(HPX_Option)
include(HPX_Utils)
include(HPX_AddTest)

# Copyright (c)      2018 Mikael Simberg Copyright (c) 2007-2025 Hartmut Kaiser
# Copyright (c) 2024 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0 Distributed under the Boost Software License,
# Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)

set(tests run_on_all_exec_policy)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Core/Algorithms/Block")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    FOLDER ${folder_name}
    DEPENDENCIES hpx_testing hpx_execution hpx_executors
  )

  add_hpx_unit_test("modules.algorithms.block" ${test})
endforeach()
