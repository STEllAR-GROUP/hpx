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
    hpx/actions/base_action.hpp
    hpx/actions/register_action.hpp
    hpx/actions/transfer_base_action.hpp
    hpx/actions/transfer_action.hpp
    hpx/actions/transfer_continuation_action.hpp
)

# Default location is $HPX_ROOT/libs/actions/include_compatibility
set(actions_compat_headers
    hpx/runtime/actions_fwd.hpp
    hpx/runtime/actions/action_support.hpp
    hpx/runtime/actions/base_action.hpp
    hpx/runtime/actions/transfer_base_action.hpp
    hpx/runtime/actions/transfer_action.hpp
    hpx/runtime/actions/transfer_continuation_action.hpp
)

set(actions_sources base_action.cpp)

include(HPX_AddModule)
add_hpx_module(
  full actions
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${actions_sources}
  HEADERS ${actions_headers}
  COMPAT_HEADERS ${actions_compat_headers}
  MODULE_DEPENDENCIES hpx_actions_base hpx_async_distributed hpx_runtime_local
  DEPENDENCIES hpx_core
  CMAKE_SUBDIRS examples tests
)
