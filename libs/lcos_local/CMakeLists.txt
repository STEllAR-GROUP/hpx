# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/lcos_local/include
set(lcos_local_headers
    hpx/lcos_local/and_gate.hpp
    hpx/lcos_local/channel.hpp
    hpx/lcos_local/composable_guard.hpp
    hpx/lcos_local/conditional_trigger.hpp
    hpx/lcos_local/packaged_task.hpp
    hpx/lcos_local/promise.hpp
    hpx/lcos_local/receive_buffer.hpp
    hpx/lcos_local/trigger.hpp
    hpx/local/channel.hpp
)

# Default location is $HPX_ROOT/libs/lcos_local/include_compatibility
set(lcos_local_compat_headers
    hpx/lcos/local/and_gate.hpp
    hpx/lcos/local/channel.hpp
    hpx/lcos/local/composable_guard.hpp
    hpx/lcos/local/conditional_trigger.hpp
    hpx/lcos/local/packaged_task.hpp
    hpx/lcos/local/promise.hpp
    hpx/lcos/local/receive_buffer.hpp
    hpx/lcos/local/spmd_block.hpp
    hpx/lcos/local/trigger.hpp
)

set(lcos_local_sources composable_guard.cpp)

include(HPX_AddModule)
add_hpx_module(
  lcos_local
  COMPATIBILITY_HEADERS ON # Added in 1.4.0
  DEPRECATION_WARNINGS
  SOURCES ${lcos_local_sources}
  HEADERS ${lcos_local_headers}
  COMPAT_HEADERS ${lcos_local_compat_headers}
  DEPENDENCIES
    hpx_config
    hpx_execution_base
    hpx_allocator_support
    hpx_assertion
    hpx_errors
    hpx_execution
    hpx_executors
    hpx_functional
    hpx_iterator_support
    hpx_memory
    hpx_pack_traversal
    hpx_synchronization
    hpx_thread_support
    hpx_type_support
  CMAKE_SUBDIRS examples tests
)

# Temporary because of the dependency to main hpx/ includes
target_link_libraries(
  hpx_lcos_local PUBLIC hpx_plugin hpx_topology hpx_threadmanager
)
