# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(basic_execution_headers
  hpx/basic_execution/agent_base.hpp
  hpx/basic_execution/agent_ref.hpp
  hpx/basic_execution/context_base.hpp
  hpx/basic_execution/register_locks.hpp
  hpx/basic_execution/resource_base.hpp
  hpx/basic_execution/this_thread.hpp
)

set(basic_execution_compat_headers
  hpx/synchronization/detail/yield_k.hpp
  hpx/util/register_locks.hpp
  hpx/util/yield_while.hpp
)

set(basic_execution_sources
  agent_ref.cpp
  register_locks.cpp
  this_thread.cpp
)

include(HPX_AddModule)
add_hpx_module(basic_execution
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  SOURCES ${basic_execution_sources}
  HEADERS ${basic_execution_headers}
  COMPAT_HEADERS ${basic_execution_compat_headers}
  DEPENDENCIES
    hpx_assertion
    hpx_config
    hpx_errors
    hpx_format
    hpx_timing
  CMAKE_SUBDIRS examples tests
)
