# Copyright (c) 2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Function tests
set(function_tests tag_invoke)

foreach(test ${function_tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    NOLIBS
    DEPENDENCIES hpx_core
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Core/TagInvoke"
  )

  add_hpx_unit_test("modules.tag_invoke" ${test})

endforeach()
