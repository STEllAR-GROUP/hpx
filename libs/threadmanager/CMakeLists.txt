# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threadmanager_headers hpx/modules/threadmanager.hpp)

set(threadmanager_compat_headers hpx/threadmanager.hpp
                                 hpx/runtime/threads/threadmanager.hpp
)

set(threadmanager_sources threadmanager.cpp)

include(HPX_AddModule)
add_hpx_module(
  threadmanager
  COMPATIBILITY_HEADERS ON # Added in 1.4.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN OFF
  SOURCES ${threadmanager_sources}
  HEADERS ${threadmanager_headers}
  COMPAT_HEADERS ${threadmanager_compat_headers}
  DEPENDENCIES
    hpx_allocator_support
    hpx_assertion
    hpx_command_line_handling
    hpx_concurrency
    hpx_concurrency
    hpx_config
    hpx_coroutines
    hpx_errors
    hpx_execution
    hpx_format
    hpx_functional
    hpx_hardware
    hpx_io_service
    hpx_logging
    hpx_plugin
    hpx_program_options
    hpx_resource_partitioner
    hpx_runtime_configuration
    hpx_schedulers
    hpx_serialization
    hpx_thread_executors
    hpx_topology
  CMAKE_SUBDIRS examples tests
)
