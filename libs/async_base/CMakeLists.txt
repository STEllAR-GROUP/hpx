# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_base_headers
  hpx/async_base/apply.hpp
  hpx/async_base/async.hpp
  hpx/async_base/dataflow.hpp
  hpx/async_base/sync.hpp

  # TODO: Move to async_base directory
  hpx/runtime/launch_policy.hpp
  hpx/traits/is_launch_policy.hpp
  )

set(async_base_sources
  launch_policy.cpp
  )

include(HPX_AddModule)
add_hpx_module(
  async_base FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  HEADERS ${async_base_headers}
  SOURCES ${async_base_sources}
  DEPENDENCIES
    hpx_allocator_support
    hpx_config
    hpx_coroutines
    hpx_type_support
    hpx_serialization
  CMAKE_SUBDIRS examples tests
)
