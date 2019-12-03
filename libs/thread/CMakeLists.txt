# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(thread_headers
  hpx/thread/thread.hpp)

set(thread_compat_headers
  hpx/runtime/threads/thread.hpp)

set(thread_sources
  thread.cpp)

include(HPX_AddModule)
add_hpx_module(thread
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  SOURCES ${thread_sources}
  HEADERS ${thread_headers}
  COMPAT_HEADERS ${thread_compat_headers}
  DEPENDENCIES
    hpx_assertion
    hpx_config
    hpx_coroutines
    hpx_errors
    hpx_execution
    # hpx_local_lcos
    hpx_functional
    hpx_thread_data
    hpx_timing
  CMAKE_SUBDIRS examples tests
)

include(HPX_PrintSummary)
create_configuration_summary(
  "  Module configuration summary (thread):" "thread")
