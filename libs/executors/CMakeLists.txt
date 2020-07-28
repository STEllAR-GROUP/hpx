# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/executors/include
set(executors_headers
    hpx/executors/current_executor.hpp
    hpx/executors/datapar/execution_policy_fwd.hpp
    hpx/executors/datapar/execution_policy.hpp
    hpx/executors/guided_pool_executor.hpp
    hpx/executors/apply.hpp
    hpx/executors/async.hpp
    hpx/executors/dataflow.hpp
    hpx/executors/exception_list.hpp
    hpx/executors/execution_policy_fwd.hpp
    hpx/executors/execution_policy.hpp
    hpx/executors/limiting_executor.hpp
    hpx/executors/parallel_executor_aggregated.hpp
    hpx/executors/parallel_executor.hpp
    hpx/executors/restricted_thread_pool_executor.hpp
    hpx/executors/sequenced_executor.hpp
    hpx/executors/service_executors.hpp
    hpx/executors/sync.hpp
    hpx/executors/thread_pool_attached_executors.hpp
    hpx/executors/thread_pool_executor.hpp
)

# Default location is $HPX_ROOT/libs/executors/include_compatibility
set(executors_compat_headers
    hpx/execution/datapar/execution_policy_fwd.hpp
    hpx/execution/datapar/execution_policy.hpp
    hpx/execution/execution_policy_fwd.hpp
    hpx/execution/execution_policy.hpp
    hpx/execution/executors/current_executor.hpp
    hpx/execution/executors/guided_pool_executor.hpp
    hpx/execution/executors.hpp
    hpx/execution/executors/limiting_executor.hpp
    hpx/execution/executors/parallel_executor_aggregated.hpp
    hpx/execution/executors/parallel_executor.hpp
    hpx/execution/executors/restricted_thread_pool_executor.hpp
    hpx/execution/executors/sequenced_executor.hpp
    hpx/execution/executors/thread_pool_attached_executors.hpp
    hpx/parallel/datapar/execution_policy_fwd.hpp
    hpx/parallel/datapar/execution_policy.hpp
    hpx/parallel/exception_list.hpp
    hpx/parallel/execution_policy_fwd.hpp
    hpx/parallel/execution_policy.hpp
    hpx/parallel/executors.hpp
    hpx/parallel/executors/parallel_executor_aggregated.hpp
    hpx/parallel/executors/parallel_executor.hpp
    hpx/parallel/executors/sequenced_executor.hpp
    hpx/parallel/executors/thread_pool_attached_executors.hpp
    hpx/parallel/executors/thread_pool_executor.hpp
)

set(executors_sources current_executor.cpp exception_list_callbacks.cpp)

include(HPX_AddModule)
add_hpx_module(
  executors
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${executors_sources}
  HEADERS ${executors_headers}
  COMPAT_HEADERS ${executors_compat_headers}
  DEPENDENCIES
    hpx_allocator_support
    hpx_assertion
    hpx_execution_base
    hpx_config
    hpx_errors
    hpx_execution
    hpx_functional
    hpx_iterator_support
    hpx_memory
    hpx_serialization
    hpx_synchronization
    hpx_threading_base
    hpx_threadmanager
    hpx_timing
    hpx_type_support
    hpx_util
  CMAKE_SUBDIRS examples tests
)
