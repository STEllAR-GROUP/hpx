# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(resource_partitioner_headers
    hpx/resource_partitioner/detail/create_partitioner.hpp
    hpx/resource_partitioner/detail/partitioner.hpp
    hpx/resource_partitioner/partitioner.hpp
    hpx/resource_partitioner/partitioner_fwd.hpp
)

set(resource_partitioner_compat_headers
    hpx/resource_partitioner.hpp hpx/runtime/resource/partitioner.hpp
    hpx/runtime/resource/partitioner_fwd.hpp
)

set(resource_partitioner_sources detail_partitioner.cpp partitioner.cpp)

if((HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI) OR HPX_WITH_ASYNC_MPI)
  set(MPI_DEPS hpx_mpi_base)
endif()

include(HPX_AddModule)
add_hpx_module(
  resource_partitioner
  COMPATIBILITY_HEADERS ON # Added in 1.4.0
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${resource_partitioner_sources}
  HEADERS ${resource_partitioner_headers}
  COMPAT_HEADERS ${resource_partitioner_compat_headers}
  DEPENDENCIES
    hpx_affinity
    hpx_allocator_support
    hpx_assertion
    hpx_cache
    hpx_command_line_handling
    hpx_concurrency
    hpx_config
    hpx_coroutines
    hpx_errors
    hpx_execution
    hpx_format
    hpx_functional
    hpx_hardware
    hpx_memory
    ${MPI_DEPS}
    hpx_pack_traversal
    hpx_plugin
    hpx_program_options
    hpx_runtime_configuration
    hpx_serialization
    hpx_synchronization
    hpx_thread_pools
    hpx_threading_base
    hpx_topology
  CMAKE_SUBDIRS examples tests
)
