# Copyright (c) 2019-2022 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_Message)
include(HPX_Option)

if(HPX_WITH_TESTS)
  add_library(hpx_execution_test_utilities INTERFACE)
  target_include_directories(hpx_execution_test_utilities INTERFACE include/)

  if(HPX_WITH_TESTS_UNIT)
    add_hpx_pseudo_target(tests.unit.modules.execution_base)
    add_hpx_pseudo_dependencies(
      tests.unit.modules tests.unit.modules.execution_base
    )
    add_subdirectory(unit)
  endif()

  if(HPX_WITH_TESTS_REGRESSIONS)
    add_hpx_pseudo_target(tests.regressions.modules.execution_base)
    add_hpx_pseudo_dependencies(
      tests.regressions.modules tests.regressions.modules.execution_base
    )
    add_subdirectory(regressions)
  endif()

  if(HPX_WITH_TESTS_BENCHMARKS)
    add_hpx_pseudo_target(tests.performance.modules.execution_base)
    add_hpx_pseudo_dependencies(
      tests.performance.modules tests.performance.modules.execution_base
    )
    add_subdirectory(performance)
  endif()

  if(HPX_WITH_TESTS_HEADERS)
    add_hpx_header_tests(
      modules.execution_base
      HEADERS ${execution_base_headers}
      HEADER_ROOT ${PROJECT_SOURCE_DIR}/include
      EXCLUDE "hpx/parallel/executors/timed_execution_fwd.hpp"
              "hpx/parallel/executors/parallel_executor.hpp"
      DEPENDENCIES hpx_execution_base
      NOLIBS
    )
  endif()
endif()
