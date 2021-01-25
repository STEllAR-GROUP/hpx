# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/actions_base/include
set(actions_base_headers
    hpx/actions_base/actions_base_fwd.hpp
    hpx/actions_base/actions_base_support.hpp
    hpx/actions_base/action_priority.hpp
    hpx/actions_base/basic_action.hpp
    hpx/actions_base/basic_action_fwd.hpp
    hpx/actions_base/component_action.hpp
    hpx/actions_base/continuation_fwd.hpp
    hpx/actions_base/detail/action_factory.hpp
    hpx/actions_base/detail/invocation_count_registry.hpp
    hpx/actions_base/detail/per_action_data_counter_registry.hpp
    hpx/actions_base/plain_action.hpp
    hpx/actions_base/preassigned_action_id.hpp
    hpx/actions_base/traits/action_continuation.hpp
    hpx/actions_base/traits/action_priority.hpp
    hpx/actions_base/traits/action_select_direct_execution.hpp
    hpx/actions_base/traits/action_stacksize.hpp
    hpx/actions_base/traits/action_remote_result.hpp
    hpx/actions_base/traits/extract_action.hpp
)

# Default location is $HPX_ROOT/libs/actions_base/include_compatibility
# cmake-format: off
set(actions_base_compat_headers
    hpx/runtime/actions_base_fwd.hpp => hpx/modules/actions_base.hpp
    hpx/runtime/actions/action_priority.hpp => hpx/modules/actions_base.hpp
    hpx/runtime/actions/action_support.hpp => hpx/modules/actions_base.hpp
    hpx/runtime/actions/basic_action.hpp => hpx/modules/actions_base.hpp
    hpx/runtime/actions/basic_action_fwd.hpp => hpx/modules/actions_base.hpp
    hpx/runtime/actions/component_action.hpp => hpx/modules/actions_base.hpp
    hpx/runtime/actions/continuation_fwd.hpp => hpx/modules/actions_base.hpp
    hpx/runtime/actions/detail/action_factory.hpp => hpx/modules/actions_base.hpp
    hpx/runtime/actions/detail/invocation_count_registry.hpp => hpx/actions_base/detail/invocation_count_registry.hpp
    hpx/runtime/actions/plain_action.hpp => hpx/modules/actions_base.hpp
    hpx/runtime/actions/preassigned_action_id.hpp => hpx/modules/actions_base.hpp
    hpx/runtime/parcelset/detail/per_action_data_counter_registry.hpp => hpx/actions_base/detail/per_action_data_counter_registry.hpp
    hpx/traits/action_continuation.hpp => hpx/modules/actions_base.hpp
    hpx/traits/action_priority.hpp => hpx/modules/actions_base.hpp
    hpx/traits/action_select_direct_execution.hpp => hpx/modules/actions_base.hpp
    hpx/traits/action_remote_result.hpp => hpx/modules/actions_base.hpp
    hpx/traits/action_stacksize.hpp => hpx/modules/actions_base.hpp
    hpx/traits/extract_action.hpp => hpx/modules/actions_base.hpp
)
# cmake-format: on

set(actions_base_sources
    detail/action_factory.cpp detail/invocation_count_registry.cpp
    detail/per_action_data_counter_registry.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full actions_base
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${actions_base_sources}
  HEADERS ${actions_base_headers}
  COMPAT_HEADERS ${actions_base_compat_headers}
  MODULE_DEPENDENCIES hpx_components_base hpx_naming
  DEPENDENCIES hpx_core
  CMAKE_SUBDIRS examples tests
)
