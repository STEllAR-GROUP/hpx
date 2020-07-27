# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Default location is $HPX_ROOT/libs/resiliency/include
set(resiliency_headers
    hpx/resiliency/async_replay.hpp
    hpx/resiliency/async_replay_executor.hpp
    hpx/resiliency/async_replicate.hpp
    hpx/resiliency/async_replicate_executor.hpp
    hpx/resiliency/config.hpp
    hpx/resiliency/version.hpp
    hpx/resiliency/resiliency.hpp
    hpx/resiliency/resiliency_cpos.hpp
)

# Default location is $HPX_ROOT/libs/resiliency/src
set(resiliency_sources resiliency.cpp)

include(HPX_AddModule)
add_hpx_module(
  resiliency
  GLOBAL_HEADER_GEN ON
  SOURCES ${resiliency_sources}
  HEADERS ${resiliency_headers}
  DEPENDENCIES
  DEPENDENCIES
    hpx_assertion
    hpx_async_local
    hpx_config
    hpx_concepts
    hpx_datastructures
    hpx_execution
    hpx_functional
    hpx_futures
    hpx_type_support
  CMAKE_SUBDIRS examples tests
)
