# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests format)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_executable(${test}_test EXCLUDE_FROM_ALL ${sources})
  target_link_libraries(${test}_test hpx_format hpx_testing)
  set_target_properties(
    ${test}_test PROPERTIES FOLDER "Tests/Unit/Modules/Core/Format"
  )

  add_hpx_unit_test("modules.format" ${test} ${${test}_PARAMETERS})

endforeach()
