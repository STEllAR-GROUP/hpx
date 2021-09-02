# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPXLocal_Message)
include(HPXLocal_Option)

if(HPXLocal_WITH_TESTS)
  if(HPXLocal_WITH_TESTS_UNIT)
    hpx_local_add_pseudo_target(tests.unit.modules.allocator_support)
    hpx_local_add_pseudo_dependencies(
      tests.unit.modules tests.unit.modules.allocator_support
    )
    add_subdirectory(unit)
  endif()

  if(HPXLocal_WITH_TESTS_REGRESSIONS)
    hpx_local_add_pseudo_target(tests.regressions.modules.allocator_support)
    hpx_local_add_pseudo_dependencies(
      tests.regressions.modules tests.regressions.modules.allocator_support
    )
    add_subdirectory(regressions)
  endif()

  if(HPXLocal_WITH_TESTS_BENCHMARKS)
    hpx_local_add_pseudo_target(tests.performance.modules.allocator_support)
    hpx_local_add_pseudo_dependencies(
      tests.performance.modules tests.performance.modules.allocator_support
    )
    add_subdirectory(performance)
  endif()

  if(HPXLocal_WITH_TESTS_HEADERS)
    hpx_local_add_header_tests(
      modules.allocator_support
      HEADERS ${allocator_support_headers}
      HEADER_ROOT ${PROJECT_SOURCE_DIR}/include NOHPX_INIT
      NOLIBS
      DEPENDENCIES hpx_allocator_support
    )
  endif()
endif()
