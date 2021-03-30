# Copyright (c) 2020-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    can_query
    can_require
    can_require_concept
    prefer
    query_free
    query_member
    query_static
    require
    require_concept
    require_concept_free
    require_concept_member
    require_concept_static
    require_free
    require_member
    require_static
)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    NOLIBS
    DEPENDENCIES hpx_core
    EXCLUDE_FROM_ALL
    FOLDER "Tests/Unit/Modules/Core/Properties"
  )

  add_hpx_unit_test("modules.properties" ${test} ${${test}_PARAMETERS})
endforeach()
