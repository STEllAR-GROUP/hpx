# Copyright (c) 2019-2022 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/assertion/include
set(assertion_headers
    hpx/assert.hpp hpx/assertion/assertion.hpp
    hpx/assertion/current_function.hpp hpx/assertion/evaluate_assert.hpp
    hpx/assertion/source_location.hpp
    hpx/assertion/detail/evaluate_assert_impl.hpp
    hpx/assertion/detail/source_location_impl.hpp
)

# Default location is $HPX_ROOT/libs/assertion/include_compatibility
# cmake-format: off
set(assertion_compat_headers
    hpx/assertion.hpp => hpx/assert.hpp
    hpx/util/assert.hpp => hpx/assert.hpp
)
# cmake-format: on

# Default location is $HPX_ROOT/libs/assertion/src
set(assertion_sources assertion.cpp source_location.cpp)

if(HPX_WITH_CXX20_MODULES)
  set(assertion_modulefiles hpx.core.assertion.cppm)
endif()

include(HPX_AddModule)
add_hpx_module(
  core assertion
  GLOBAL_HEADER_GEN MODULE
  EXCLUDE_FROM_GLOBAL_HEADER "hpx/assert.hpp"
  GENERATE_EXPORT_DEFINITIONS
  SOURCES ${assertion_sources}
  HEADERS ${assertion_headers}
  MODULES ${assertion_modulefiles}
  COMPAT_HEADERS ${assertion_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_preprocessor
  CMAKE_SUBDIRS examples tests
)
