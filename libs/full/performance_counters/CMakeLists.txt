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

set(performance_counters_headers
    hpx/performance_counters/apex_sample_value.hpp
    hpx/performance_counters/base_performance_counter.hpp
    hpx/performance_counters/counter_creators.hpp
    hpx/performance_counters/counter_parser.hpp
    hpx/performance_counters/counters.hpp
    hpx/performance_counters/counters_fwd.hpp
    hpx/performance_counters/manage_counter.hpp
    hpx/performance_counters/manage_counter_type.hpp
    hpx/performance_counters/performance_counter.hpp
    hpx/performance_counters/performance_counter_base.hpp
    hpx/performance_counters/performance_counter_set.hpp
    hpx/performance_counters/registry.hpp
    hpx/performance_counters/parcels/data_point.hpp
    hpx/performance_counters/parcels/gatherer.hpp
    hpx/performance_counters/server/arithmetics_counter.hpp
    hpx/performance_counters/server/arithmetics_counter_extended.hpp
    hpx/performance_counters/server/base_performance_counter.hpp
    hpx/performance_counters/server/elapsed_time_counter.hpp
    hpx/performance_counters/server/raw_counter.hpp
    hpx/performance_counters/server/raw_values_counter.hpp
    hpx/performance_counters/server/statistics_counter.hpp
)

set(performance_counters_sources
    counter_creators.cpp
    counter_parser.cpp
    counters.cpp
    manage_counter.cpp
    manage_counter_type.cpp
    registry.cpp
    performance_counter.cpp
    performance_counter_set.cpp
    server/component_instance_counter.cpp
    server/arithmetics_counter_extended.cpp
    server/raw_values_counter.cpp
    server/raw_counter.cpp
    server/per_action_data_counters.cpp
    server/elapsed_time_counter.cpp
    server/action_invocation_counter.cpp
    server/arithmetics_counter.cpp
    server/statistics_counter.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full performance_counters
  GLOBAL_HEADER_GEN ON
  SOURCES ${performance_counters_sources}
  HEADERS ${performance_counters_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  MODULE_DEPENDENCIES hpx_actions hpx_async_distributed hpx_runtime_local
  CMAKE_SUBDIRS examples tests
)
