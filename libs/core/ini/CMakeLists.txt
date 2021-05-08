# Copyright (c) 2019-2021 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(ini_headers hpx/ini/ini.hpp)
set(ini_sources ini.cpp)

include(HPX_AddModule)
add_hpx_module(
  core ini
  GLOBAL_HEADER_GEN ON
  SOURCES ${ini_sources}
  HEADERS ${ini_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config
    hpx_errors
    hpx_serialization
    hpx_string_util
    hpx_thread_support
    hpx_concurrency
    hpx_functional
    hpx_util
  CMAKE_SUBDIRS examples tests
)
