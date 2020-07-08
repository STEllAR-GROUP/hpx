# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/actions_base/include
set(actions_base_headers
    hpx/actions_base/actions_base_fwd.hpp
    hpx/actions_base/action_base_support.hpp
    hpx/actions_base/action_priority.hpp
    hpx/actions_base/basic_action.hpp
    hpx/actions_base/basic_action_fwd.hpp
    hpx/actions_base/component_action.hpp
    hpx/actions_base/plain_action.hpp
    hpx/actions_base/traits/action_priority.hpp
    hpx/actions_base/traits/action_select_direct_execution.hpp
    hpx/actions_base/traits/action_stacksize.hpp
    hpx/actions_base/traits/action_remote_result.hpp
    hpx/actions_base/traits/extract_action.hpp
)

# Default location is $HPX_ROOT/libs/actions_base/include_compatibility
set(actions_base_compat_headers
    hpx/runtime/actions_base_fwd.hpp
    hpx/runtime/actions/action_priority.hpp
    hpx/runtime/actions/basic_action.hpp
    hpx/runtime/actions/basic_action_fwd.hpp
    hpx/runtime/actions/component_action.hpp
    hpx/runtime/actions/plain_action.hpp
    hpx/traits/action_priority.hpp
    hpx/traits/extract_action.hpp
    hpx/traits/action_select_direct_execution.hpp
    hpx/traits/action_remote_result.hpp
    hpx/traits/action_stacksize.hpp
)

set(actions_base_sources)

include(HPX_AddModule)
add_hpx_module(
  actions_base
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${actions_base_sources}
  HEADERS ${actions_base_headers}
  COMPAT_HEADERS ${actions_base_compat_headers}
  DEPENDENCIES
    hpx_async_base
    hpx_components_base
    hpx_concurrency
    hpx_config
    hpx_coroutines
    hpx_datastructures
    hpx_errors
    hpx_functional
    hpx_futures
    hpx_itt_notify
    hpx_logging
    hpx_naming_base
    hpx_preprocessor
    hpx_threading_base
    hpx_type_support
  CMAKE_SUBDIRS examples tests
)
