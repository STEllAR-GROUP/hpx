# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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

include(HPXLocal_AddModule)
hpx_local_add_module(
  local io_service
  GLOBAL_HEADER_GEN ON
  SOURCES ${io_service_sources}
  HEADERS ${io_service_headers}
  COMPAT_HEADERS ${io_service_compat_headers}
  DEPENDENCIES Asio::asio
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_concurrency
    hpx_config_local
    hpx_errors
    hpx_functional
    hpx_logging
    hpx_threading_base
    hpx_timing
  CMAKE_SUBDIRS examples tests
)
