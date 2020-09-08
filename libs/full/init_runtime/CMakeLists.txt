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

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(init_runtime_optional_module_dependencies hpx_async_distributed
                                                hpx_performance_counters
  )
endif()

if((HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI) OR HPX_WITH_ASYNC_MPI)
  list(APPEND init_runtime_optional_module_dependencies hpx_mpi_base)
endif()

include(HPX_AddModule)
add_hpx_module(
  full init_runtime
  GLOBAL_HEADER_GEN OFF
  SOURCES ${init_runtime_sources}
  HEADERS ${init_runtime_headers}
  DEPENDENCIES hpx_core hpx_parallelism
  MODULE_DEPENDENCIES hpx_program_options hpx_runtime_local
                      ${init_runtime_optional_module_dependencies}
  CMAKE_SUBDIRS examples tests
)
