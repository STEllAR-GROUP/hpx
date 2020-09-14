# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests any boost_any is_tuple_like unique_any tagged tuple)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  set(folder_name "Tests/Unit/Modules/Core/Datastructures")

  # add example executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources} ${${test}_FLAGS}
    EXCLUDE_FROM_ALL
    FOLDER ${folder_name}
  )

  add_hpx_unit_test("modules.datastructures" ${test} ${${test}_PARAMETERS})
endforeach()
