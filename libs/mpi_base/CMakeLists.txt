# Copyright (c) 2019-2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if((HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI) OR HPX_MPI_WITH_FUTURES)
  # setup MPI, if necessary
  if(NOT TARGET Mpi::mpi)
    include(HPX_SetupMPI)
    setup_mpi()
  endif()
  set(additional_dependencies Mpi::mpi)
endif()

# Default location is $HPX_ROOT/libs/mpi_base/include
set(mpi_base_headers
  hpx/mpi_base/mpi.hpp
  hpx/mpi_base/mpi_environment.hpp
)

# Default location is $HPX_ROOT/libs/mpi_base/include_compatibility
set(mpi_base_compat_headers
  hpx/plugins/parcelport/mpi/mpi.hpp
  hpx/plugins/parcelport/mpi/mpi_environment.hpp
)

set(mpi_base_sources
  mpi_environment.cpp
)

include(HPX_AddModule)
add_hpx_module(mpi_base
  COMPATIBILITY_HEADERS ON
  DEPRECATION_WARNINGS
  FORCE_LINKING_GEN
  GLOBAL_HEADER_GEN ON
  SOURCES ${mpi_base_sources}
  HEADERS ${mpi_base_headers}
  COMPAT_HEADERS ${mpi_base_compat_headers}
  DEPENDENCIES
    hpx_config
    hpx_runtime_configuration
    hpx_synchronization
    hpx_util
    ${additional_dependencies}
  CMAKE_SUBDIRS examples tests
)
