# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/executors/include
set(executors_headers
  hpx/execution/executors/current_executor.hpp
  hpx/execution/executors/guided_pool_executor.hpp
  hpx/execution/executors.hpp
  hpx/execution/executors/limiting_executor.hpp
  hpx/execution/executors/parallel_executor_aggregated.hpp
  hpx/execution/executors/parallel_executor.hpp
  hpx/execution/executors/pool_executor.hpp
  hpx/execution/executors/restricted_thread_pool_executor.hpp
  hpx/execution/executors/sequenced_executor.hpp
  hpx/execution/executors/service_executors.hpp
  hpx/execution/executors/thread_pool_attached_executors.hpp
  hpx/executors/timed_executors.hpp
  )

# Default location is $HPX_ROOT/libs/executors/include_compatibility
set(executors_compat_headers
  hpx/parallel/executors.hpp
  hpx/parallel/executors/parallel_executor_aggregated.hpp
  hpx/parallel/executors/parallel_executor.hpp
  hpx/parallel/executors/pool_executor.hpp
  hpx/parallel/executors/sequenced_executor.hpp
  hpx/parallel/executors/service_executors.hpp
  hpx/parallel/executors/thread_pool_attached_executors.hpp
  )

set(executors_sources
  current_executor.cpp
  )

include(HPX_AddModule)
add_hpx_module(executors
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  SOURCES ${executors_sources}
  HEADERS ${executors_headers}
  COMPAT_HEADERS ${executors_compat_headers}
  DEPENDENCIES
    hpx_allocator_support
    hpx_assertion
    hpx_basic_execution
    hpx_config
    hpx_execution
    hpx_functional
    hpx_iterator_support
    hpx_serialization
    hpx_synchronization
    hpx_threading_base
    hpx_threadmanager
    hpx_timing
    hpx_type_support
    hpx_util
  CMAKE_SUBDIRS examples tests
)
