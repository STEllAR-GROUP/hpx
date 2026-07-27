# Copyright (c) 2026 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests sentinel_start)

foreach(test ${tests})
  set(sources ${test}.cpp)

  # add executable
  add_hpx_executable(
    ${test}_test INTERNAL_FLAGS
    SOURCES ${sources}
    EXCLUDE_FROM_ALL
    HPX_PREFIX ${HPX_BUILD_PREFIX}
    COMPONENT_DEPENDENCIES supervision_dispatch
    FOLDER "Tests/Unit/Components/Supervision"
  )

  add_hpx_unit_test(
    "components.supervision_dispatch" ${test} ${${test}_PARAMETERS}
  )

  # MSVC V1951 and V1952 expose ICE compiling this in C++20 module mode
  if(HPX_WITH_CXX_MODULES
     AND (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
     AND (MSVC_VERSION LESS_EQUAL 1952)
  )
    target_compile_definitions(
      ${test}_test PRIVATE HPX_HAVE_FORCE_NO_CXX_MODULES
    )
  endif()
endforeach()
