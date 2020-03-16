# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/checkpoint/include
set(checkpoint_headers
    hpx/checkpoint/checkpoint.hpp
  )

# Default location is $HPX_ROOT/libs/checkpoint/include_compatibility
set(checkpoint_compat_headers
    hpx/util/checkpoint.hpp
  )

set(checkpoint_sources)

include(HPX_AddModule)
add_hpx_module(checkpoint
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  SOURCES ${checkpoint_sources}
  HEADERS ${checkpoint_headers}
  COMPAT_HEADERS ${checkpoint_compat_headers}
  DEPENDENCIES hpx_serialization
  CMAKE_SUBDIRS examples tests
)
