# Copyright (c) 2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Function tests
set(function_tests tag_dispatch)

foreach(test ${function_tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  hpx_local_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    NOLIBS
    DEPENDENCIES hpx_local
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Core/TagDispatch"
  )

  hpx_local_add_unit_test("modules.tag_dispatch" ${test})

endforeach()
