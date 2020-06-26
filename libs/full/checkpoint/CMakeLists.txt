# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/checkpoint/include
set(checkpoint_headers hpx/checkpoint/checkpoint.hpp)

# Default location is $HPX_ROOT/libs/checkpoint/include_compatibility
set(checkpoint_compat_headers hpx/checkpoint.hpp hpx/util/checkpoint.hpp)

set(checkpoint_sources)

include(HPX_AddModule)
add_hpx_module(
  full checkpoint
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${checkpoint_sources}
  HEADERS ${checkpoint_headers}
  COMPAT_HEADERS ${checkpoint_compat_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  MODULE_DEPENDENCIES hpx_async_distributed hpx_checkpoint_base
  CMAKE_SUBDIRS examples tests
)
