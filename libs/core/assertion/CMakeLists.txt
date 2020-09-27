# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/assertion/include
set(assertion_headers
    hpx/assert.hpp hpx/modules/assertion.hpp hpx/assertion/current_function.hpp
    hpx/assertion/evaluate_assert.hpp hpx/assertion/source_location.hpp
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

include(HPX_AddModule)
add_hpx_module(
  core assertion
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN OFF
  SOURCES ${assertion_sources}
  HEADERS ${assertion_headers}
  COMPAT_HEADERS ${assertion_compat_headers}
  MODULE_DEPENDENCIES hpx_config hpx_preprocessor
  CMAKE_SUBDIRS examples tests
)
