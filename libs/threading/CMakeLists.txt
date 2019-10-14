# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threading_headers
  hpx/threading/force_linking.hpp
  hpx/threading/jthread.hpp
  hpx/threading/stop_token.hpp
  hpx/threading/thread.hpp)

set(threading_compat_headers
  hpx/runtime/threads/thread.hpp)

set(threading_sources
  force_linking.cpp
  stop_token.cpp
  thread.cpp)

include(HPX_AddModule)
add_hpx_module(threading
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${threading_sources}
  HEADERS ${threading_headers}
  COMPAT_HEADERS ${threading_compat_headers}
    EXCLUDE_FROM_GLOBAL_HEADER
      hpx/threading/force_linking.hpp
  DEPENDENCIES
    hpx_assertion
    hpx_basic_execution
    hpx_config
    hpx_coroutines
    hpx_errors
    hpx_execution
    hpx_functional
    hpx_memory
    hpx_threading_base
    hpx_thread_support
    hpx_timing
  CMAKE_SUBDIRS examples tests
)
