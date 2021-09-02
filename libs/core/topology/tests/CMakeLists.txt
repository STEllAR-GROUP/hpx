# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPXLocal_Message)
include(HPXLocal_Option)

if(HPXLocal_WITH_TESTS)
  if(HPXLocal_WITH_TESTS_UNIT)
    hpx_local_add_pseudo_target(tests.unit.modules.topology)
    hpx_local_add_pseudo_dependencies(
      tests.unit.modules tests.unit.modules.topology
    )
    add_subdirectory(unit)
  endif()

  if(HPXLocal_WITH_TESTS_REGRESSIONS)
    hpx_local_add_pseudo_target(tests.regressions.modules.topology)
    hpx_local_add_pseudo_dependencies(
      tests.regressions.modules tests.regressions.modules.topology
    )
    add_subdirectory(regressions)
  endif()

  if(HPXLocal_WITH_TESTS_BENCHMARKS)
    hpx_local_add_pseudo_target(tests.performance.modules.topology)
    hpx_local_add_pseudo_dependencies(
      tests.performance.modules tests.performance.modules.topology
    )
    add_subdirectory(performance)
  endif()

  if(HPXLocal_WITH_TESTS_HEADERS)
    hpx_local_add_header_tests(
      modules.topology
      HEADERS ${topology_headers}
      HEADER_ROOT ${PROJECT_SOURCE_DIR}/include
      NOLIBS
      DEPENDENCIES hpx_topology
    )
  endif()
endif()
