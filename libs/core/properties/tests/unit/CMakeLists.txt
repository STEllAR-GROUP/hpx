# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests properties)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  hpx_local_add_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    NOLIBS
    DEPENDENCIES hpx_local
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Core/Properties"
  )

  hpx_local_add_unit_test("modules.properties" ${test} ${${test}_PARAMETERS})
endforeach()
