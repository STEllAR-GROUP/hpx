# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    range
)

foreach(test ${tests})
  set(sources ${test}.cpp)

  source_group("Source Files" FILES ${sources})

  add_executable(${test}_test ${sources})
  target_link_libraries(${test}_test hpx_assertion hpx_testing hpx_util)

  add_hpx_unit_test("modules.util" ${test} ${${test}_PARAMETERS})

endforeach()
