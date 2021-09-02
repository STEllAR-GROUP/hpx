# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT HPXLocal_WITH_ASYNC_MPI)
  return()
endif()

include(HPXLocal_SetupMPI)
hpx_local_setup_mpi()
set(additional_dependencies Mpi::mpi)

set(mpi_base_headers hpx/mpi_base/mpi.hpp hpx/mpi_base/mpi_environment.hpp)
set(mpi_base_sources mpi_environment.cpp)

include(HPXLocal_AddModule)
hpx_local_add_module(
  local mpi_base
  GLOBAL_HEADER_GEN ON
  SOURCES ${mpi_base_sources}
  HEADERS ${mpi_base_headers}
  MODULE_DEPENDENCIES hpx_runtime_configuration hpx_util
  DEPENDENCIES ${additional_dependencies}
  CMAKE_SUBDIRS examples tests
)
