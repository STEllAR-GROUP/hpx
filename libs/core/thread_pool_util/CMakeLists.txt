# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(thread_pool_util_headers
    hpx/thread_pool_util/thread_pool_suspension_helpers.hpp
)

set(thread_pool_util_compat_headers)

set(thread_pool_util_sources thread_pool_suspension_helpers.cpp)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local thread_pool_util
  GLOBAL_HEADER_GEN ON
  SOURCES ${thread_pool_util_sources}
  HEADERS ${thread_pool_util_headers}
  MODULE_DEPENDENCIES hpx_async_local hpx_futures hpx_functional
  CMAKE_SUBDIRS examples tests
)
