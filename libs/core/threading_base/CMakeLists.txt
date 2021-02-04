# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(threading_base_headers
    hpx/threading_base/annotated_function.hpp
    hpx/threading_base/callback_notifier.hpp
    hpx/threading_base/create_thread.hpp
    hpx/threading_base/create_work.hpp
    hpx/threading_base/detail/reset_backtrace.hpp
    hpx/threading_base/detail/reset_lco_description.hpp
    hpx/threading_base/execution_agent.hpp
    hpx/threading_base/external_timer.hpp
    hpx/threading_base/network_background_callback.hpp
    hpx/threading_base/print.hpp
    hpx/threading_base/register_thread.hpp
    hpx/threading_base/scheduler_base.hpp
    hpx/threading_base/scheduler_mode.hpp
    hpx/threading_base/scheduler_state.hpp
    hpx/threading_base/set_thread_state.hpp
    hpx/threading_base/thread_data.hpp
    hpx/threading_base/thread_data_stackful.hpp
    hpx/threading_base/thread_data_stackless.hpp
    hpx/threading_base/thread_description.hpp
    hpx/threading_base/thread_helpers.hpp
    hpx/threading_base/thread_init_data.hpp
    hpx/threading_base/thread_num_tss.hpp
    hpx/threading_base/thread_pool_base.hpp
    hpx/threading_base/thread_queue_init_parameters.hpp
    hpx/threading_base/thread_specific_ptr.hpp
    hpx/threading_base/threading_base_fwd.hpp
)

# cmake-format: off
set(threading_base_compat_headers
    hpx/runtime/threads/execution_agent.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/policies/callback_notifier.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/policies/scheduler_base.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/policies/scheduler_mode.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/policies/thread_queue_init_parameters.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/register_thread.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/thread_data.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/thread_data_stackful.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/thread_data_stackless.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/thread_helpers.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/thread_init_data.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/thread_pool_base.hpp => hpx/modules/threading_base.hpp
    hpx/runtime/threads/thread_specific_ptr.hpp => hpx/modules/threading_base.hpp
    hpx/util/annotated_function.hpp => hpx/modules/threading_base.hpp
    hpx/util/external_timer.hpp => hpx/modules/threading_base.hpp
    hpx/util/thread_description.hpp => hpx/modules/threading_base.hpp
)
# cmake-format: on

set(threading_base_sources
    annotated_function.cpp
    execution_agent.cpp
    external_timer.cpp
    print.cpp
    register_thread.cpp
    scheduler_base.cpp
    thread_data.cpp
    thread_data_stackful.cpp
    thread_data_stackless.cpp
    thread_description.cpp
    thread_helpers.cpp
    thread_num_tss.cpp
    thread_pool_base.cpp
)

if(HPX_WITH_THREAD_BACKTRACE_ON_SUSPENSION)
  set(additional_dependencies hpx_debugging)
endif()

include(HPX_AddModule)
add_hpx_module(
  core threading_base
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  EXCLUDE_FROM_GLOBAL_HEADER "hpx/threading_base/thread_data_stackful.hpp"
                             "hpx/threading_base/thread_data_stackless.hpp"
  SOURCES ${threading_base_sources}
  HEADERS ${threading_base_headers}
  COMPAT_HEADERS ${threading_base_compat_headers}
  MODULE_DEPENDENCIES
    hpx_affinity
    hpx_allocator_support
    hpx_assertion
    hpx_execution_base
    hpx_config
    hpx_concurrency
    hpx_coroutines
    hpx_debugging
    hpx_errors
    hpx_format
    hpx_functional
    hpx_hardware
    hpx_itt_notify
    hpx_logging
    hpx_memory
    hpx_type_support
    ${additional_dependencies}
  CMAKE_SUBDIRS examples tests
)
