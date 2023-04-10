# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(init_runtime_local_headers hpx/init_runtime_local/detail/init_logging.hpp
                               hpx/init_runtime_local/init_runtime_local.hpp
)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  set(init_runtime_local_headers ${init_runtime_local_headers} hpx/init.hpp)
endif()

set(init_runtime_local_sources init_logging.cpp init_runtime_local.cpp)

# The headers in hpx/local/ were deprecated in HPX V1.9.0
# cmake-format: off
set(init_runtime_local_compat_headers
    hpx/local/init.hpp => hpx/init.hpp
)
# cmake-format: on

include(HPX_AddModule)
add_hpx_module(
  core init_runtime_local
  GLOBAL_HEADER_GEN ON
  SOURCES ${init_runtime_local_sources}
  HEADERS ${init_runtime_local_headers}
  COMPAT_HEADERS ${init_runtime_local_compat_headers}
  MODULE_DEPENDENCIES
    hpx_algorithms
    hpx_command_line_handling_local
    hpx_program_options
    hpx_runtime_local
    hpx_errors
    hpx_filesystem
    hpx_format
    hpx_lock_registration
    hpx_logging
    hpx_schedulers
    hpx_testing
    hpx_threading_base
    hpx_timing
  CMAKE_SUBDIRS examples tests
)
