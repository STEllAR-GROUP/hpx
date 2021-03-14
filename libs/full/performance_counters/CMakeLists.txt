# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(performance_counters_headers
    hpx/performance_counters/action_invocation_counter_discoverer.hpp
    hpx/performance_counters/agas_counter_types.hpp
    hpx/performance_counters/agas_namespace_action_code.hpp
    hpx/performance_counters/apex_sample_value.hpp
    hpx/performance_counters/base_performance_counter.hpp
    hpx/performance_counters/component_namespace_counters.hpp
    hpx/performance_counters/counter_creators.hpp
    hpx/performance_counters/counter_interface.hpp
    hpx/performance_counters/counter_parser.hpp
    hpx/performance_counters/counters.hpp
    hpx/performance_counters/counters_fwd.hpp
    hpx/performance_counters/detail/counter_interface_functions.hpp
    hpx/performance_counters/locality_namespace_counters.hpp
    hpx/performance_counters/manage_counter.hpp
    hpx/performance_counters/manage_counter_type.hpp
    hpx/performance_counters/per_action_data_counter_discoverer.hpp
    hpx/performance_counters/performance_counter.hpp
    hpx/performance_counters/performance_counter_base.hpp
    hpx/performance_counters/performance_counter_set.hpp
    hpx/performance_counters/primary_namespace_counters.hpp
    hpx/performance_counters/query_counters.hpp
    hpx/performance_counters/registry.hpp
    hpx/performance_counters/symbol_namespace_counters.hpp
    hpx/performance_counters/threadmanager_counter_types.hpp
    hpx/performance_counters/server/arithmetics_counter.hpp
    hpx/performance_counters/server/arithmetics_counter_extended.hpp
    hpx/performance_counters/server/base_performance_counter.hpp
    hpx/performance_counters/server/component_namespace_counters.hpp
    hpx/performance_counters/server/elapsed_time_counter.hpp
    hpx/performance_counters/server/locality_namespace_counters.hpp
    hpx/performance_counters/server/primary_namespace_counters.hpp
    hpx/performance_counters/server/raw_counter.hpp
    hpx/performance_counters/server/raw_values_counter.hpp
    hpx/performance_counters/server/statistics_counter.hpp
    hpx/performance_counters/server/symbol_namespace_counters.hpp
)

set(performance_counters_sources
    action_invocation_counter_discoverer.cpp
    agas_counter_types.cpp
    agas_namespace_action_code.cpp
    component_namespace_counters.cpp
    counter_creators.cpp
    counter_interface.cpp
    counter_parser.cpp
    counters.cpp
    detail/counter_interface_functions.cpp
    locality_namespace_counters.cpp
    manage_counter.cpp
    manage_counter_type.cpp
    per_action_data_counter_discoverer.cpp
    performance_counter.cpp
    performance_counter_set.cpp
    primary_namespace_counters.cpp
    query_counters.cpp
    registry.cpp
    symbol_namespace_counters.cpp
    threadmanager_counter_types.cpp
    server/action_invocation_counter.cpp
    server/arithmetics_counter.cpp
    server/arithmetics_counter_extended.cpp
    server/component_instance_counter.cpp
    server/elapsed_time_counter.cpp
    server/per_action_data_counters.cpp
    server/raw_values_counter.cpp
    server/raw_counter.cpp
    server/statistics_counter.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full performance_counters
  GLOBAL_HEADER_GEN ON
  SOURCES ${performance_counters_sources}
  HEADERS ${performance_counters_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  MODULE_DEPENDENCIES
    hpx_actions
    hpx_agas
    hpx_async_colocated
    hpx_async_distributed
    hpx_naming
    hpx_runtime_components
    hpx_runtime_local
    hpx_runtime_distributed
    hpx_threadmanager
  CMAKE_SUBDIRS examples tests
)
