# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# Note: HPX_MPI_WITH_FUTURES is handled in the main CMakeLists.txt

# if the user does not want mpi async futures, quit - the module will not be
# enabled
if(NOT ${HPX_MPI_WITH_FUTURES})
  return()
endif()

# setup MPI, if necessary
if(NOT TARGET Mpi::mpi)
  include(HPX_SetupMPI)
  setup_mpi()
endif()

# Default location is $HPX_ROOT/libs/mpi/include
set(mpi_headers hpx/mpi/force_linking.hpp hpx/mpi/mpi_executor.hpp
                hpx/mpi/mpi_future.hpp
)

# Default location is $HPX_ROOT/libs/mpi/src
set(mpi_sources force_linking.cpp mpi_future.cpp)

include(HPX_AddModule)
add_hpx_module(
  mpi
  GLOBAL_HEADER_GEN ON
  SOURCES ${mpi_sources}
  HEADERS ${mpi_headers}
  DEPENDENCIES
    hpx_assertion
    hpx_basic_execution
    hpx_concurrency
    hpx_config
    hpx_errors
    hpx_execution
    hpx_local_lcos
    hpx_memory
    hpx_mpi_base
    hpx_threading_base
    Mpi::mpi
  CMAKE_SUBDIRS examples tests
)
