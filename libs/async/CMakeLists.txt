# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(async_headers
    hpx/async.hpp
    hpx/async/applier/applier.hpp
    hpx/async/applier/apply_callback.hpp
    hpx/async/applier/apply_continue_callback.hpp
    hpx/async/applier/apply_continue_fwd.hpp
    hpx/async/applier/apply_continue.hpp
    hpx/async/applier/apply_helper.hpp
    hpx/async/applier/apply.hpp
    hpx/async/applier/bind_naming_wrappers.hpp
    hpx/async/applier/detail/apply_colocated_callback_fwd.hpp
    hpx/async/applier/detail/apply_colocated_callback.hpp
    hpx/async/applier/detail/apply_colocated_fwd.hpp
    hpx/async/applier/detail/apply_colocated.hpp
    hpx/async/applier/detail/apply_implementations_fwd.hpp
    hpx/async/applier/detail/apply_implementations.hpp
    hpx/async/applier_fwd.hpp
    hpx/async/applier/register_apply_colocated.hpp
    hpx/async/applier/trigger.hpp
    hpx/async/apply.hpp
    hpx/async/async_callback_fwd.hpp
    hpx/async/async_callback.hpp
    hpx/async/async_continue_callback_fwd.hpp
    hpx/async/async_continue_callback.hpp
    hpx/async/async_continue_fwd.hpp
    hpx/async/async_continue.hpp
    hpx/async/async_fwd.hpp
    hpx/async/async.hpp
    hpx/async/dataflow.hpp
    hpx/async/detail/async_colocated_callback_fwd.hpp
    hpx/async/detail/async_colocated_callback.hpp
    hpx/async/detail/async_colocated_fwd.hpp
    hpx/async/detail/async_colocated.hpp
    hpx/async/detail/async_implementations_fwd.hpp
    hpx/async/detail/async_implementations.hpp
    hpx/async/detail/async_unwrap_result_implementations_fwd.hpp
    hpx/async/detail/async_unwrap_result_implementations.hpp
    hpx/async/detail/sync_implementations_fwd.hpp
    hpx/async/detail/sync_implementations.hpp
    hpx/async/sync_fwd.hpp
    hpx/async/sync.hpp
)

set(async_compat_headers
    hpx/apply.hpp
    hpx/dataflow.hpp
    hpx/lcos/async_callback_fwd.hpp
    hpx/lcos/async_callback.hpp
    hpx/lcos/async_continue_callback_fwd.hpp
    hpx/lcos/async_continue_callback.hpp
    hpx/lcos/async_continue_fwd.hpp
    hpx/lcos/async_continue.hpp
    hpx/lcos/async_fwd.hpp
    hpx/lcos/async.hpp
    hpx/lcos/dataflow.hpp
    hpx/lcos/detail/async_colocated_callback_fwd.hpp
    hpx/lcos/detail/async_colocated_callback.hpp
    hpx/lcos/detail/async_colocated_fwd.hpp
    hpx/lcos/detail/async_colocated.hpp
    hpx/lcos/detail/async_implementations_fwd.hpp
    hpx/lcos/detail/async_implementations.hpp
    hpx/lcos/detail/async_unwrap_result_implementations_fwd.hpp
    hpx/lcos/detail/async_unwrap_result_implementations.hpp
    hpx/lcos/detail/sync_implementations_fwd.hpp
    hpx/lcos/detail/sync_implementations.hpp
    hpx/lcos/sync_fwd.hpp
    hpx/lcos/sync.hpp
    hpx/runtime/applier/applier.hpp
    hpx/runtime/applier/apply_callback.hpp
    hpx/runtime/applier/apply_continue_callback.hpp
    hpx/runtime/applier/apply_continue_fwd.hpp
    hpx/runtime/applier/apply_continue.hpp
    hpx/runtime/applier/apply_helper.hpp
    hpx/runtime/applier/apply.hpp
    hpx/runtime/applier/bind_naming_wrappers.hpp
    hpx/runtime/applier/detail/apply_colocated_callback_fwd.hpp
    hpx/runtime/applier/detail/apply_colocated_callback.hpp
    hpx/runtime/applier/detail/apply_colocated_fwd.hpp
    hpx/runtime/applier/detail/apply_colocated.hpp
    hpx/runtime/applier/detail/apply_implementations_fwd.hpp
    hpx/runtime/applier/detail/apply_implementations.hpp
    hpx/runtime/applier_fwd.hpp
    hpx/runtime/applier/register_apply_colocated.hpp
    hpx/runtime/applier/trigger.hpp
    hpx/sync.hpp
)

include(HPX_AddModule)
add_hpx_module(
  async
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN OFF
  HEADERS ${async_headers}
  COMPAT_HEADERS ${async_compat_headers}
  DEPENDENCIES
    hpx_allocator_support
    hpx_assertion
    hpx_config
    hpx_coroutines
    hpx_datastructures
    hpx_errors
    hpx_execution
    hpx_format
    hpx_functional
    hpx_futures
    hpx_async_local
    hpx_local_lcos
    hpx_memory
    hpx_runtime_configuration
    hpx_threading_base
    hpx_threadmanager
    hpx_type_support
    hpx_util
  CMAKE_SUBDIRS examples tests
)
