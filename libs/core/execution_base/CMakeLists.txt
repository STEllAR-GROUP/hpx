# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(execution_base_headers
    hpx/execution_base/agent_base.hpp
    hpx/execution_base/agent_ref.hpp
    hpx/execution_base/context_base.hpp
    hpx/execution_base/detail/spinlock_deadlock_detection.hpp
    hpx/execution_base/execution.hpp
    hpx/execution_base/operation_state.hpp
    hpx/execution_base/receiver.hpp
    hpx/execution_base/register_locks.hpp
    hpx/execution_base/resource_base.hpp
    hpx/execution_base/sender.hpp
    hpx/execution_base/this_thread.hpp
)

# cmake-format: off
set(execution_base_compat_headers
    hpx/basic_execution.hpp => hpx/modules/execution_base.hpp
    hpx/parallel/executors/execution_fwd.hpp => hpx/modules/execution_base.hpp
    hpx/util/register_locks.hpp => hpx/modules/execution_base.hpp
    hpx/util/yield_while.hpp => hpx/modules/execution_base.hpp
    hpx/basic_execution/agent_base.hpp => hpx/modules/execution_base.hpp
    hpx/basic_execution/agent_ref.hpp => hpx/modules/execution_base.hpp
    hpx/basic_execution/context_base.hpp => hpx/modules/execution_base.hpp
    hpx/basic_execution/execution.hpp => hpx/modules/execution_base.hpp
    hpx/basic_execution/register_locks.hpp => hpx/modules/execution_base.hpp
    hpx/basic_execution/resource_base.hpp => hpx/modules/execution_base.hpp
    hpx/basic_execution/this_thread.hpp => hpx/modules/execution_base.hpp
)
# cmake-format: on

set(execution_base_sources agent_ref.cpp register_locks.cpp
                           spinlock_deadlock_detection.cpp this_thread.cpp
)

include(HPX_AddModule)
add_hpx_module(
  core execution_base
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${execution_base_sources}
  HEADERS ${execution_base_headers}
  COMPAT_HEADERS ${execution_base_compat_headers}
  MODULE_DEPENDENCIES
    hpx_assertion
    hpx_config
    hpx_errors
    hpx_format
    hpx_functional
    hpx_iterator_support
    hpx_timing
    hpx_type_support
  CMAKE_SUBDIRS examples tests
)
