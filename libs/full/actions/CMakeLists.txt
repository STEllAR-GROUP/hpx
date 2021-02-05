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

# Default location is $HPX_ROOT/libs/actions/include
set(actions_headers
    hpx/actions/action_support.hpp
    hpx/actions/actions_fwd.hpp
    hpx/actions/apply_helper_fwd.hpp
    hpx/actions/apply_helper.hpp
    hpx/actions/base_action.hpp
    hpx/actions/continuation_fwd.hpp
    hpx/actions/continuation.hpp
    hpx/actions/register_action.hpp
    hpx/actions/set_lco_value_continuation.hpp
    hpx/actions/transfer_base_action.hpp
    hpx/actions/transfer_action.hpp
    hpx/actions/transfer_continuation_action.hpp
    hpx/actions/trigger.hpp
    hpx/actions/traits/action_continuation.hpp
    hpx/actions/traits/action_decorate_continuation.hpp
    hpx/actions/traits/action_trigger_continuation.hpp
)

# Default location is $HPX_ROOT/libs/actions/include_compatibility
# cmake-format: off
set(actions_compat_headers
    hpx/async_distributed/applier/apply_helper.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions_fwd.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/action_support.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/continuation_fwd.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/continuation.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/base_action.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/set_lco_value_continuation.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/transfer_action.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/transfer_base_action.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/transfer_continuation_action.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/trigger.hpp => hpx/modules/actions.hpp
    hpx/traits/action_continuation.hpp => hpx/modules/actions.hpp
    hpx/traits/action_decorate_continuation.hpp => hpx/modules/actions.hpp
)
# cmake-format: on

set(actions_sources base_action.cpp continuation.cpp)

include(HPX_AddModule)
add_hpx_module(
  full actions
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${actions_sources}
  HEADERS ${actions_headers}
  COMPAT_HEADERS ${actions_compat_headers}
  MODULE_DEPENDENCIES hpx_actions_base hpx_naming hpx_runtime_local
  DEPENDENCIES hpx_core
  CMAKE_SUBDIRS examples tests
)
