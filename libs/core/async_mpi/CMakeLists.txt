# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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
  core async_mpi
  GLOBAL_HEADER_GEN ON
  SOURCES ${mpi_sources}
  HEADERS ${async_mpi_headers}
  DEPENDENCIES Mpi::mpi
  MODULE_DEPENDENCIES
    hpx_concurrency
    hpx_errors
    hpx_execution_base
    hpx_memory
    hpx_threading_base
    hpx_mpi_base
    hpx_runtime_local
    hpx_config
  CMAKE_SUBDIRS examples tests
)
