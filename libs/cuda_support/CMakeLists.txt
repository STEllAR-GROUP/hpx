# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/cuda_support/include
set(cuda_support_headers)

# Default location is $HPX_ROOT/libs/cuda_support/include_compatibility
set(cuda_support_compat_headers)

set(cuda_support_sources)

include(HPX_AddModule)
add_hpx_module(cuda_support
  COMPATIBILITY_HEADERS OFF
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  SOURCES ${cuda_support_sources}
  HEADERS ${cuda_support_headers}
  COMPAT_HEADERS ${cuda_support_compat_headers}
  DEPENDENCIES
  CMAKE_SUBDIRS examples tests
)
