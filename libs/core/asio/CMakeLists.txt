# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(asio_headers hpx/asio/asio_util.hpp hpx/asio/map_hostnames.hpp)

set(asio_compat_headers hpx/util/asio_util.hpp hpx/util/map_hostnames.hpp)

set(asio_sources asio_util.cpp map_hostnames.cpp)

include(HPX_AddModule)
add_hpx_module(
  core asio
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${asio_sources}
  HEADERS ${asio_headers}
  COMPAT_HEADERS ${asio_compat_headers}
  MODULE_DEPENDENCIES hpx_assertion hpx_config hpx_errors hpx_functional
  CMAKE_SUBDIRS examples tests
)
