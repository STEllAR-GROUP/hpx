# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(io_service_headers hpx/io_service/io_service_pool.hpp
                       hpx/io_service/io_service_thread_pool.hpp
)

# cmake-format: off
set(io_service_compat_headers
    hpx/util/io_service_pool.hpp => hpx/modules/io_service.hpp
)
# cmake-format: on

set(io_service_sources io_service_pool.cpp io_service_thread_pool.cpp)

include(HPX_AddModule)
add_hpx_module(
  core io_service
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${io_service_sources}
  HEADERS ${io_service_headers}
  COMPAT_HEADERS ${io_service_compat_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_concurrency
    hpx_config
    hpx_errors
    hpx_functional
    hpx_logging
    hpx_threading_base
  CMAKE_SUBDIRS examples tests
)
