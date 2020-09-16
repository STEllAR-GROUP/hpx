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

# Default location is $HPX_ROOT/libs/collectives/include
set(collectives_headers
    hpx/collectives/all_gather.hpp
    hpx/collectives/all_reduce.hpp
    hpx/collectives/all_to_all.hpp
    hpx/collectives/barrier.hpp
    hpx/collectives/broadcast.hpp
    hpx/collectives/broadcast_direct.hpp
    hpx/collectives/communication_set.hpp
    hpx/collectives/detail/communication_set_node.hpp
    hpx/collectives/detail/communicator.hpp
    hpx/collectives/fold.hpp
    hpx/collectives/gather.hpp
    hpx/collectives/latch.hpp
    hpx/collectives/reduce.hpp
    hpx/collectives/scatter.hpp
    hpx/collectives/spmd_block.hpp
    hpx/collectives/detail/barrier_node.hpp
    hpx/collectives/detail/latch.hpp
    hpx/distributed/barrier.hpp
    hpx/distributed/latch.hpp
)

# Default location is $HPX_ROOT/libs/collectives/include_compatibility
set(collectives_compat_headers
    hpx/lcos/barrier.hpp
    hpx/lcos/broadcast.hpp
    hpx/lcos/fold.hpp
    hpx/lcos/gather.hpp
    hpx/lcos/latch.hpp
    hpx/lcos/reduce.hpp
    hpx/lcos/spmd_block.hpp
)

# Default location is $HPX_ROOT/libs/collectives/src
set(collectives_sources
    barrier.cpp create_communication_set.cpp latch.cpp detail/barrier_node.cpp
    detail/communication_set_node.cpp detail/communicator.cpp
)

include(HPX_AddModule)
add_hpx_module(
  full collectives
  COMPATIBILITY_HEADERS ON # Added in 1.5.0
  DEPRECATION_WARNINGS
  SOURCES ${collectives_sources}
  HEADERS ${collectives_headers}
  COMPAT_HEADERS ${collectives_compat_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  MODULE_DEPENDENCIES
    hpx_actions
    hpx_async_distributed
    hpx_init_runtime
    hpx_performance_counters
    hpx_program_options
    hpx_runtime_configuration
    hpx_runtime_local
    hpx_threadmanager
  CMAKE_SUBDIRS examples tests
)
