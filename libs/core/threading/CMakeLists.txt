# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threading_headers hpx/threading/jthread.hpp hpx/threading/thread.hpp)

# cmake-format: off
set(threading_compat_headers
    hpx/runtime/threads/thread.hpp => hpx/thread.hpp
)
# cmake-format: on

set(threading_sources thread.cpp)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local threading
  GLOBAL_HEADER_GEN ON
  SOURCES ${threading_sources}
  HEADERS ${threading_headers}
  COMPAT_HEADERS ${threading_compat_headers}
  MODULE_DEPENDENCIES hpx_futures hpx_errors hpx_memory
  CMAKE_SUBDIRS examples tests
)
