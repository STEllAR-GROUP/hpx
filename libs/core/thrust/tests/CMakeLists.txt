# Copyright (c) 2025 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_Message)

if(HPX_WITH_TESTS)
  if(HPX_WITH_TESTS_UNIT)
    add_hpx_pseudo_target(tests.unit.modules.thrust)
    add_hpx_pseudo_dependencies(
      tests.unit.modules tests.unit.modules.thrust
    )
    add_subdirectory(unit)
  endif()

  if(HPX_WITH_TESTS_REGRESSIONS)
    add_hpx_pseudo_target(tests.regressions.modules.thrust)
    add_hpx_pseudo_dependencies(
      tests.regressions.modules tests.regressions.modules.thrust
    )
    add_subdirectory(regressions)
  endif()

  if(HPX_WITH_TESTS_BENCHMARKS)
    add_hpx_pseudo_target(tests.performance.modules.thrust)
    add_hpx_pseudo_dependencies(
      tests.performance.modules tests.performance.modules.thrust
    )
    add_subdirectory(performance)
  endif()

  if(HPX_WITH_TESTS_HEADERS)
    add_hpx_header_tests(
      modules.thrust
      HEADERS ${thrust_headers}
      HEADER_ROOT ${PROJECT_SOURCE_DIR}/include
      DEPENDENCIES hpx_thrust
    )
  endif()
endif()
