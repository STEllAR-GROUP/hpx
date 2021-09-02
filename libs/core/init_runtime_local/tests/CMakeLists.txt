# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPXLocal_Message)
include(HPXLocal_Option)

if(HPXLocal_WITH_TESTS)
  if(HPXLocal_WITH_TESTS_UNIT)
    hpx_local_add_pseudo_target(tests.unit.modules.init_runtime_local)
    hpx_local_add_pseudo_dependencies(
      tests.unit.modules tests.unit.modules.init_runtime_local
    )
    add_subdirectory(unit)
  endif()
  if(HPXLocal_WITH_TESTS_REGRESSIONS)
    hpx_local_add_pseudo_target(tests.regressions.modules.init_runtime_local)
    hpx_local_add_pseudo_dependencies(
      tests.regressions.modules tests.regressions.modules.init_runtime_local
    )
    add_subdirectory(regressions)
  endif()

  if(HPXLocal_WITH_TESTS_BENCHMARKS)
    hpx_local_add_pseudo_target(tests.performance.modules.init_runtime_local)
    hpx_local_add_pseudo_dependencies(
      tests.performance.modules tests.performance.modules.init_runtime_local
    )
    add_subdirectory(performance)
  endif()

  if(HPXLocal_WITH_TESTS_HEADERS)
    hpx_local_add_header_tests(
      modules.init_runtime_local
      HEADERS ${init_runtime_local_headers}
      HEADER_ROOT ${PROJECT_SOURCE_DIR}/include
      NOLIBS
      DEPENDENCIES hpx_init_runtime_local
    )
  endif()
endif()
