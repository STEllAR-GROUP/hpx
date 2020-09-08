# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# Note: HPX_WITH_ASYNC_MPI is handled in the main CMakeLists.txt

# if the user does not want mpi async futures, quit - the module will not be
# enabled
if(NOT ${HPX_WITH_ASYNC_MPI})
  return()
endif()

# setup MPI, if necessary
if(NOT TARGET Mpi::mpi)
  include(HPX_SetupMPI)
  setup_mpi()
endif()

# Default location is $HPX_ROOT/libs/mpi/include
set(async_mpi_headers hpx/async_mpi/mpi_executor.hpp
                      hpx/async_mpi/mpi_future.hpp
)

# Default location is $HPX_ROOT/libs/mpi/src
set(mpi_sources mpi_future.cpp)

include(HPX_AddModule)
add_hpx_module(
  full async_mpi
  GLOBAL_HEADER_GEN ON
  SOURCES ${mpi_sources}
  HEADERS ${async_mpi_headers}
  DEPENDENCIES hpx_core hpx_parallelism Mpi::mpi
  MODULE_DEPENDENCIES hpx_mpi_base hpx_runtime_local
  CMAKE_SUBDIRS examples tests
)
