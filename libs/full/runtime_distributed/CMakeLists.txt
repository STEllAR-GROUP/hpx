# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_DISTRIBUTED_RUNTIME)
  return()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(runtime_distributed_headers
    hpx/runtime_distributed/big_boot_barrier.hpp
    hpx/runtime_distributed/copy_component.hpp
    hpx/runtime_distributed.hpp
    hpx/runtime_distributed/get_num_localities.hpp
    hpx/runtime_distributed/find_localities.hpp
    hpx/runtime_distributed/migrate_component.hpp
    hpx/runtime_distributed/runtime_fwd.hpp
    hpx/runtime_distributed/runtime_support.hpp
    hpx/runtime_distributed/server/copy_component.hpp
    hpx/runtime_distributed/server/migrate_component.hpp
    hpx/runtime_distributed/server/runtime_support.hpp
    hpx/runtime_distributed/stubs/runtime_support.hpp
)

# cmake-format: off
set(runtime_distributed_compat_headers
    hpx/runtime/find_localities.hpp => hpx/runtime_distributed.hpp
    hpx/runtime/get_num_localities.hpp => hpx/runtime_distributed.hpp
    hpx/runtime/agas/big_boot_barrier.hpp => hpx/runtime_distributed.hpp
    hpx/runtime/components/copy_component.hpp => hpx/modules/runtime_components.hpp
    hpx/runtime/components/migrate_component.hpp => hpx/modules/runtime_components.hpp
)
# cmake-format: on

set(runtime_distributed_sources
    big_boot_barrier.cpp runtime_support.cpp server/runtime_support_server.cpp
    stubs/runtime_support_stubs.cpp
)

set(MPI_DEPS)
if(HPX_WITH_PARCELPORT_MPI OR HPX_WITH_ASYNC_MPI)
  set(MPI_DEPS hpx_mpi_base)
endif()

include(HPX_AddModule)
add_hpx_module(
  full runtime_distributed
  GLOBAL_HEADER_GEN OFF
  SOURCES ${runtime_distributed_sources}
  HEADERS ${runtime_distributed_headers}
  COMPAT_HEADERS ${runtime_distributed_compat_headers}
  DEPENDENCIES hpx_core
  MODULE_DEPENDENCIES
    hpx_agas
    hpx_agas_base
    hpx_async_colocated
    hpx_async_distributed
    hpx_collectives
    ${MPI_DEPS}
    hpx_naming
    hpx_naming_base
    hpx_performance_counters
    hpx_program_options
    hpx_runtime_components
    hpx_runtime_local
    hpx_threadmanager
  CMAKE_SUBDIRS examples tests
)
