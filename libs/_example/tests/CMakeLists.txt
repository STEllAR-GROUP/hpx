# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if (NOT HPX_WITH_TESTS AND HPX_TOP_LEVEL)
  return()
endif()
if (NOT HPX__EXAMPLE_WITH_TESTS)
  message(STATUS "Tests for example disabled")
  return()
endif()

if (HPX_WITH_TESTS_UNIT)
  add_custom_target(tests.unit.example)
  add_dependencies(tests.unit tests.unit.example)
  add_subdirectory(unit)
endif()

if (HPX_WITH_TESTS_REGRESSIONS)
  add_custom_target(tests.regressions.example)
  add_dependencies(tests.regressions tests.regressions.example)
  add_subdirectory(regressions)
endif()

if (HPX_WITH_TESTS_BENCHMARKS)
  add_custom_target(tests.performance.example)
  add_dependencies(tests.performance tests.performance.example)
  add_subdirectory(performance)
endif()

if (HPX_WITH_TESTS_HEADERS)
  add_hpx_lib_header_tests(example)
endif()
