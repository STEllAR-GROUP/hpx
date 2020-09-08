# Copyright (c) 2020 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests print)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Debugging")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS} ${${test}_LIBS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
    NOLIBS
    DEPENDENCIES hpx_errors hpx_debugging hpx_threading_base hpx_testing
  )

  add_hpx_unit_test("modules.debugging" ${test} ${${test}_PARAMETERS})
endforeach()
