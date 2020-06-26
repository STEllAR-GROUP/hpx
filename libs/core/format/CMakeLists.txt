# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/format/include
set(format_headers
    hpx/format.hpp hpx/modules/format.hpp hpx/util/bad_lexical_cast.hpp
    hpx/util/from_string.hpp hpx/util/to_string.hpp
)

# Default location is $HPX_ROOT/libs/format/include_compatibility
set(format_compat_headers hpx/util/format.hpp)

# Default location is $HPX_ROOT/libs/format/src
set(format_sources format.cpp util/bad_lexical_cast.cpp)

include(HPX_AddModule)
add_hpx_module(
  core format DEPRECATION_WARNINGS
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  GLOBAL_HEADER_GEN OFF
  SOURCES ${format_sources}
  HEADERS ${format_headers}
  COMPAT_HEADERS ${format_compat_headers}
  MODULE_DEPENDENCIES hpx_assertion hpx_config
  CMAKE_SUBDIRS examples tests
)
