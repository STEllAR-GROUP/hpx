# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(local_async_headers hpx/local_async/apply.hpp hpx/local_async/async.hpp
                        hpx/local_async/dataflow.hpp hpx/local_async/sync.hpp
)

include(HPX_AddModule)
add_hpx_module(
  local_async FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  HEADERS ${local_async_headers}
  DEPENDENCIES
    hpx_allocator_support
    hpx_async_base
    hpx_config
    hpx_coroutines
    hpx_datastructures
    hpx_execution
    hpx_executors
    hpx_functional
    hpx_memory
    hpx_threading_base
    hpx_type_support
    hpx_util
  CMAKE_SUBDIRS examples tests
)
