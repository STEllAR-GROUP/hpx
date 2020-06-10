# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(init_runtime_headers
    hpx/init.hpp
    hpx/hpx_finalize.hpp
    hpx/hpx_init.hpp
    hpx/hpx_init_impl.hpp
    hpx/hpx_init_params.hpp
    hpx/hpx_main_winsocket.hpp
    hpx/hpx_start.hpp
    hpx/hpx_start_impl.hpp
    hpx/hpx_suspend.hpp
    hpx/hpx_user_main_config.hpp
    hpx/init_runtime/detail/run_or_start.hpp
)

set(init_runtime_sources hpx_init.cpp hpx_main_winsocket.cpp)

include(HPX_AddModule)
add_hpx_module(
  init_runtime
  GLOBAL_HEADER_GEN OFF
  SOURCES ${init_runtime_sources}
  HEADERS ${init_runtime_headers}
  DEPENDENCIES hpx_actions_base hpx_algorithms hpx_config hpx_runtime_local
               hpx_testing
  CMAKE_SUBDIRS examples tests
)

set(_hpx_init_runtime_extra_dependencies)
if((HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI) OR HPX_WITH_ASYNC_MPI)
  list(APPEND _hpx_init_runtime_extra_dependencies hpx_mpi_base)
endif()

if(HPX_WITH_LIBCDS)
    hpx_warn("Adding hpx_libcds as a dependency to hpx_init_runtime")
  list(APPEND _hpx_init_runtime_extra_dependencies hpx_libcds)
endif()

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  target_link_libraries(
    hpx_init_runtime
    PUBLIC hpx_async_distributed
           hpx_errors
           hpx_filesystem
           hpx_format
           hpx_logging
           hpx_performance_counters
           hpx_program_options
           hpx_schedulers
           hpx_timing
           ${_hpx_init_runtime_extra_dependencies}
  )
endif()
