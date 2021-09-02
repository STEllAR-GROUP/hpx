# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPXLocal_Message)
include(HPXLocal_Option)

if(HPXLocal_WITH_TESTS)
  if(HPXLocal_WITH_TESTS_UNIT)
    hpx_local_add_pseudo_target(tests.unit.modules.futures)
    hpx_local_add_pseudo_dependencies(
      tests.unit.modules tests.unit.modules.futures
    )
    add_subdirectory(unit)
  endif()

  if(HPXLocal_WITH_TESTS_REGRESSIONS)
    hpx_local_add_pseudo_target(tests.regressions.modules.futures)
    hpx_local_add_pseudo_dependencies(
      tests.regressions.modules tests.regressions.modules.futures
    )
    add_subdirectory(regressions)
  endif()

  if(HPXLocal_WITH_TESTS_BENCHMARKS)
    hpx_local_add_pseudo_target(tests.performance.modules.futures)
    hpx_local_add_pseudo_dependencies(
      tests.performance.modules tests.performance.modules.futures
    )
    add_subdirectory(performance)
  endif()

  if(HPXLocal_WITH_TESTS_HEADERS)
    # packaged_continuation.hpp has to be at the end of future.hpp.
    set(excluded_headers hpx/futures/packaged_continuation.hpp)
    hpx_local_add_header_tests(
      modules.futures
      HEADERS ${futures_headers}
      HEADER_ROOT ${PROJECT_SOURCE_DIR}/include
      EXCLUDE ${excluded_headers}
      NOLIBS
      DEPENDENCIES hpx_futures
    )
  endif()
endif()
