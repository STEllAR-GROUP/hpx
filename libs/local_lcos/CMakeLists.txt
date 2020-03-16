# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/local_lcos/include
set(local_lcos_headers
  hpx/local_lcos/and_gate.hpp
  hpx/local_lcos/channel.hpp
  hpx/local_lcos/composable_guard.hpp
  hpx/local_lcos/conditional_trigger.hpp
  hpx/local_lcos/force_linking.hpp
  hpx/local_lcos/packaged_task.hpp
  hpx/local_lcos/promise.hpp
  hpx/local_lcos/receive_buffer.hpp
  hpx/local_lcos/trigger.hpp
  )

# Default location is $HPX_ROOT/libs/local_lcos/include_compatibility
set(local_lcos_compat_headers
  hpx/lcos/local/and_gate.hpp
  hpx/lcos/local/channel.hpp
  hpx/lcos/local/composable_guard.hpp
  hpx/lcos/local/conditional_trigger.hpp
  hpx/lcos/local/packaged_task.hpp
  hpx/lcos/local/promise.hpp
  hpx/lcos/local/receive_buffer.hpp
  hpx/lcos/local/trigger.hpp
  )

set(local_lcos_sources
  composable_guard.cpp
  force_linking.cpp
  )

include(HPX_AddModule)
add_hpx_module(local_lcos
  COMPATIBILITY_HEADERS ON # Added in 1.4.0
  DEPRECATION_WARNINGS
  SOURCES ${local_lcos_sources}
  HEADERS ${local_lcos_headers}
  COMPAT_HEADERS ${local_lcos_compat_headers}
  DEPENDENCIES
    hpx_config
    hpx_basic_execution
    hpx_allocator_support
    hpx_assertion
    hpx_errors
    hpx_functional
    hpx_iterator_support
    hpx_memory
    hpx_synchronization
    hpx_thread_support
    hpx_type_support
  CMAKE_SUBDIRS examples tests
)

# Temporary because of the dependency to main hpx/ includes
target_link_libraries(hpx_local_lcos PUBLIC
  hpx_execution
  hpx_plugin
  hpx_topology
  hpx_threadmanager
  )
