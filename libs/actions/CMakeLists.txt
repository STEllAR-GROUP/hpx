# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/actions/include
set(actions_headers
    hpx/actions/actions_fwd.hpp hpx/actions/action_support.hpp
    hpx/actions/base_action.hpp hpx/actions/transfer_base_action.hpp
)

# Default location is $HPX_ROOT/libs/actions/include_compatibility
set(actions_compat_headers
    hpx/runtime/actions_fwd.hpp hpx/runtime/actions/action_support.hpp
    hpx/runtime/actions/base_action.hpp
    hpx/runtime/actions/transfer_base_action.hpp
)

set(actions_sources)

include(HPX_AddModule)
add_hpx_module(
  actions
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${actions_sources}
  HEADERS ${actions_headers}
  COMPAT_HEADERS ${actions_compat_headers}
  DEPENDENCIES
    hpx_actions_base
    hpx_assertion
    hpx_config
    hpx_concurrency
    hpx_coroutines
    hpx_datastructures
    hpx_debugging
    hpx_itt_notify
    hpx_serialization
  CMAKE_SUBDIRS examples tests
)
