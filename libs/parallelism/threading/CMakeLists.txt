# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threading_headers hpx/thread.hpp hpx/threading/jthread.hpp
                      hpx/threading/thread.hpp
)

# cmake-format: off
set(threading_compat_headers
    hpx/runtime/threads/thread.hpp => hpx/thread.hpp
)
# cmake-format: on

set(threading_sources thread.cpp)

include(HPX_AddModule)
add_hpx_module(
  parallelism threading
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${threading_sources}
  HEADERS ${threading_headers}
  COMPAT_HEADERS ${threading_compat_headers}
  DEPENDENCIES hpx_core
  MODULE_DEPENDENCIES hpx_execution hpx_futures
  CMAKE_SUBDIRS examples tests
)
