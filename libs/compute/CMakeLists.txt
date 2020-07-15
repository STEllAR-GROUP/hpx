# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(compute_headers
    hpx/compute/detail/get_proxy_type.hpp
    hpx/compute/detail/iterator.hpp
    hpx/compute/detail/new.hpp
    hpx/compute/detail/target_distribution_policy.hpp
    hpx/compute/host/block_allocator.hpp
    hpx/compute/host/block_executor.hpp
    hpx/compute/host/get_targets.hpp
    hpx/compute/host.hpp
    hpx/compute/host/numa_allocator.hpp
    hpx/compute/host/numa_binding_allocator.hpp
    hpx/compute/host/numa_domains.hpp
    hpx/compute/host/target_distribution_policy.hpp
    hpx/compute/host/target.hpp
    hpx/compute/host/traits/access_target.hpp
    hpx/compute/serialization/vector.hpp
    hpx/compute/traits/access_target.hpp
    hpx/compute/traits/allocator_traits.hpp
    hpx/compute/traits.hpp
    hpx/compute/vector.hpp
)

set(compute_compat_headers
    hpx/compute.hpp hpx/parallel/util/numa_allocator.hpp
    hpx/parallel/util/numa_binding_allocator.hpp hpx/util/functional/new.hpp
)

set(compute_sources get_host_targets.cpp host_target.cpp numa_domains.cpp)

include(HPX_AddModule)
add_hpx_module(
  compute
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${compute_sources}
  HEADERS ${compute_headers}
  COMPAT_HEADERS ${compute_compat_headers}
  DEPENDENCIES
    hpx_actions_base
    hpx_affinity
    hpx_algorithms
    hpx_assertion
    hpx_config
    hpx_coroutines
    hpx_execution
    hpx_executors
    hpx_futures
    hpx_hardware
    hpx_memory
    hpx_pack_traversal
    hpx_plugin
    hpx_program_options
    hpx_resource_partitioner
    hpx_runtime_configuration
    hpx_runtime_local
    hpx_serialization
    hpx_synchronization
    hpx_threading
    hpx_threading_base
    hpx_threadmanager
    hpx_topology
    hpx_type_support
  CMAKE_SUBDIRS examples tests
)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  target_link_libraries(
    hpx_compute PUBLIC hpx_performance_counters hpx_async_distributed
  )
endif()

# Temporary since the module still depends on some runtime files
target_link_libraries(hpx_compute PUBLIC hpx_static_reinit hpx_lcos_local)
