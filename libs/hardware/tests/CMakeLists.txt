# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if (NOT HPX_WITH_TESTS AND HPX_TOP_LEVEL)
  hpx_set_option(HPX_HARDWARE_WITH_TESTS VALUE OFF FORCE)
  return()
endif()
if (NOT HPX_HARDWARE_WITH_TESTS)
  message(STATUS "Tests for hardware disabled")
  return()
endif()

if (HPX_WITH_TESTS_UNIT)
  add_hpx_pseudo_target(tests.unit.modules.hardware)
  add_hpx_pseudo_dependencies(tests.unit.modules tests.unit.modules.hardware)
  add_subdirectory(unit)
endif()

if (HPX_WITH_TESTS_REGRESSIONS)
  add_hpx_pseudo_target(tests.regressions.modules.hardware)
  add_hpx_pseudo_dependencies(tests.regressions.modules tests.regressions.modules.hardware)
  add_subdirectory(regressions)
endif()

if (HPX_WITH_TESTS_BENCHMARKS)
  add_hpx_pseudo_target(tests.performance.modules.hardware)
  add_hpx_pseudo_dependencies(tests.performance.modules tests.performance.modules.hardware)
  add_subdirectory(performance)
endif()

if (HPX_WITH_TESTS_HEADERS)
  add_hpx_module_header_tests(hardware
     EXCLUDE
    "hardware/timestamp/msvc.hpp"
    "hardware/timestamp/linux_x86_32.hpp"
    "hardware/timestamp/linux_x86_64.hpp"
    "hardware/timestamp/linux_generic.hpp"
    "hardware/timestamp/bgq.hpp")
endif()
