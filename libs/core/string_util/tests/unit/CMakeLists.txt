# Copyright (c) 2020 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    string_to_lower
    string_split
    string_trim
    tokenizer
    tokenizer_1
    tokenizer_2
    tokenizer_3
    tokenizer_4
    tokenizer_5
)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Core/StringUtil")

  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL NOLIBS
    DEPENDENCIES hpx_core
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("modules.string_util" ${test} ${${test}_PARAMETERS})
endforeach()
