# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if((HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI) OR HPX_WITH_ASYNC_MPI)
  # setup MPI, if necessary
  include(HPX_SetupMPI)
  setup_mpi()
  set(additional_dependencies Mpi::mpi)
else()
  return()
endif()

# Default location is $HPX_ROOT/libs/mpi_base/include
set(mpi_base_headers hpx/mpi_base/mpi.hpp hpx/mpi_base/mpi_environment.hpp)

# Default location is $HPX_ROOT/libs/mpi_base/include_compatibility
# cmake-format: off
set(mpi_base_compat_headers
    hpx/plugins/parcelport/mpi/mpi.hpp => hpx/modules/mpi_base.hpp
    hpx/plugins/parcelport/mpi/mpi_environment.hpp => hpx/modules/mpi_base.hpp
)
# cmake-format: on

set(mpi_base_sources mpi_environment.cpp)

include(HPX_AddModule)
add_hpx_module(
  full mpi_base
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  GLOBAL_HEADER_GEN ON
  SOURCES ${mpi_base_sources}
  HEADERS ${mpi_base_headers}
  COMPAT_HEADERS ${mpi_base_compat_headers}
  MODULE_DEPENDENCIES hpx_runtime_configuration
  DEPENDENCIES hpx_core hpx_parallelism ${additional_dependencies}
  CMAKE_SUBDIRS examples tests
)
