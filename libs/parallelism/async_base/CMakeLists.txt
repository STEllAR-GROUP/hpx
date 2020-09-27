# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_base_headers
    hpx/async_base/apply.hpp hpx/async_base/async.hpp
    hpx/async_base/dataflow.hpp hpx/async_base/launch_policy.hpp
    hpx/async_base/sync.hpp hpx/async_base/traits/is_launch_policy.hpp
)

set(async_base_sources launch_policy.cpp)

# cmake-format: off
set(async_base_compat_headers
    hpx/runtime/launch_policy.hpp => hpx/future.hpp
    hpx/traits/is_launch_policy.hpp => hpx/future.hpp
)
# cmake-format: on

include(HPX_AddModule)
add_hpx_module(
  parallelism async_base
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  HEADERS ${async_base_headers}
  COMPAT_HEADERS ${async_base_compat_headers}
  SOURCES ${async_base_sources}
  DEPENDENCIES hpx_core
  CMAKE_SUBDIRS examples tests
)
