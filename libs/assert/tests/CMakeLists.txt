# Copyright (c) 2018 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if (NOT HPX_WITH_TESTS AND HPX_TOP_LEVEL)
  return()
endif()
if (NOT HPX_ASSERT_WITH_TESTS)
  message(STATUS "Tests for assert disabled")
  return()
endif()

add_custom_target(tests.unit.assert)
add_dependencies(tests.unit tests.unit.assert)
add_subdirectory(unit)

add_custom_target(tests.regressions.assert)
add_dependencies(tests.regressions tests.regressions.assert)
add_subdirectory(regressions)

add_custom_target(tests.regressions.performance)
add_dependencies(tests.regressions tests.regressions.performance)
add_subdirectory(performance)
