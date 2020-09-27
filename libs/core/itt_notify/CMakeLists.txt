# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/itt_notify/include
set(itt_notify_headers hpx/modules/itt_notify.hpp
                       hpx/itt_notify/thread_name.hpp
)

# Default location is $HPX_ROOT/libs/itt_notify/include_compatibility
# cmake-format: off
set(itt_notify_compat_headers
    hpx/util/itt_notify.hpp => hpx/modules/itt_notify.hpp
)
# cmake-format: on

set(itt_notify_sources itt_notify.cpp thread_name.cpp)

include(HPX_AddModule)
add_hpx_module(
  core itt_notify
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN OFF
  SOURCES ${itt_notify_sources}
  HEADERS ${itt_notify_headers}
  COMPAT_HEADERS ${itt_notify_compat_headers}
  MODULE_DEPENDENCIES hpx_config
  CMAKE_SUBDIRS examples tests
)
