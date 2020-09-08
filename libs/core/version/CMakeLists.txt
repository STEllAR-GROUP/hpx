# Copyright (c) 2019 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(version_headers hpx/version.hpp)

set(version_sources version.cpp)

if((HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI) OR HPX_WITH_ASYNC_MPI)
  include(HPX_SetupMPI)
  set(mpi_dependencies Mpi::mpi)
endif()

include(HPX_AddModule)
add_hpx_module(
  core version
  COMPATIBILITY_HEADERS OFF # Added in 1.4.0
  GLOBAL_HEADER_GEN OFF
  SOURCES ${version_sources}
  HEADERS ${version_headers}
  MODULE_DEPENDENCIES hpx_config hpx_config_registry hpx_format hpx_prefix
  DEPENDENCIES Hwloc::hwloc ${mpi_dependencies}
  MODULE_DEPENDENCIES hpx_config hpx_format hpx_prefix
)

target_include_directories(
  hpx_version PRIVATE $<BUILD_INTERFACE:${HPX_BINARY_DIR}>
)
