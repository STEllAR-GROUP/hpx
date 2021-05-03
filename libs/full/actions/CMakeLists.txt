# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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
    hpx/actions/invoke_function.hpp
    hpx/actions/register_action.hpp
    hpx/actions/transfer_base_action.hpp
    hpx/actions/transfer_action.hpp
)

# Default location is $HPX_ROOT/libs/actions/include_compatibility
# cmake-format: off
set(actions_compat_headers
    hpx/async_distributed/applier/apply_helper.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions_fwd.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/action_support.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/base_action.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/transfer_action.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/transfer_base_action.hpp => hpx/modules/actions.hpp
    hpx/runtime/actions/trigger.hpp => hpx/modules/actions.hpp
    hpx/runtime/components/server/invoke_function.hpp => hpx/modules/actions.hpp
)
# cmake-format: on

set(actions_sources base_action.cpp)

include(HPX_AddModule)
add_hpx_module(
  full actions
  GLOBAL_HEADER_GEN ON
  SOURCES ${actions_sources}
  HEADERS ${actions_headers}
  COMPAT_HEADERS ${actions_compat_headers}
  MODULE_DEPENDENCIES hpx_actions_base hpx_agas hpx_async_distributed hpx_naming
  DEPENDENCIES hpx_core
  CMAKE_SUBDIRS examples tests
)
