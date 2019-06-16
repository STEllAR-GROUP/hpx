# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_Option)

if (NOT HPX_WITH_TESTS AND HPX_TOP_LEVEL)
  hpx_set_option(HPX_CONFIG_WITH_TESTS VALUE OFF FORCE)
  return()
endif()
if (NOT HPX_CONFIG_WITH_TESTS)
  message(STATUS "Tests for config disabled")
  return()
endif()

if (HPX_WITH_TESTS_UNIT)
  add_hpx_pseudo_target(tests.unit.config)
  add_hpx_pseudo_dependencies(tests.unit tests.unit.config)
  add_subdirectory(unit)
endif()

if (HPX_WITH_TESTS_REGRESSIONS)
  add_hpx_pseudo_target(tests.regressions.config)
  add_hpx_pseudo_dependencies(tests.regressions tests.regressions.config)
  add_subdirectory(regressions)
endif()

if (HPX_WITH_TESTS_BENCHMARKS)
  add_hpx_pseudo_target(tests.performance.config)
  add_hpx_pseudo_dependencies(tests.performance tests.performance.config)
  add_subdirectory(performance)
endif()

if (HPX_WITH_TESTS_HEADERS)
  add_hpx_lib_header_tests(config)
endif()
