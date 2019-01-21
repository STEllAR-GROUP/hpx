# Copyright (c) 2018 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_executable(hpx.assert.test.assert_fail assert_fail.cpp)
target_link_libraries(hpx.assert.test.assert_fail hpx.assert)
add_test(NAME tests.unit.assert.test.assert_fail
  COMMAND "${PYTHON_EXECUTABLE}" "${CMAKE_BINARY_DIR}/bin/hpxrun.py"
  $<TARGET_FILE:hpx.assert.test.assert_fail>)

set_tests_properties(tests.unit.assert.test.assert_fail
  PROPERTIES WILL_FAIL $<$<CONFIG:Debug>:true>)

add_executable(hpx.assert.test.assert_succeed assert_succeed.cpp)
target_link_libraries(hpx.assert.test.assert_succeed hpx.assert)
add_test(NAME tests.unit.assert.test.assert_succeed
  COMMAND $<TARGET_FILE:hpx.assert.test.assert_succeed>)

add_dependencies(tests.unit.assert
  hpx.assert.test.assert_fail
  hpx.assert.test.assert_succeed
)
