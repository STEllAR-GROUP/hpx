# Copyright (c) 2020 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests apply_colocated async_cb_colocated async_colocated
          async_continue_cb_colocated async_continue_colocated
)

set(apply_colocated_PARAMETERS LOCALITIES 2)
set(async_cb_colocated_PARAMETERS LOCALITIES 2)

set(async_continue_cb_colocated_PARAMETERS LOCALITIES 2)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Full/AsyncColocated")

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("modules.async_colocated" ${test} ${${test}_PARAMETERS})
endforeach()
